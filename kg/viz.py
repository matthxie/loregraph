"""Visualization payloads + a self-contained HTML viewer (no build step, no CDN).

Two things to see:
  1. the graph being built — the object-level graph (objects + the derived
     object↔object edges), with a "play" animation that reveals nodes in
     ingestion order;
  2. the traversal path a query takes — seeds, the tag/entity hubs the retriever
     hops through, and the ranked result objects, laid out as a focused subgraph
     and (for BFS) animated hop-by-hop.

`render_html` embeds the data as JSON in a single static .html file. The optional
stdlib server (kg.serve) reuses the same page but adds a live /api/query box.
"""
from __future__ import annotations

import json

import networkx as nx

from .graph import KnowledgeGraph
from .models import EdgeType, NodeType
from .store import GraphStore

_OBJ_EDGES = {EdgeType.SHARED_TAG.value, EdgeType.SHARED_ENTITY.value,
              EdgeType.SIMILAR_TO.value}


# --------------------------------------------------------------------------- #
# layout helpers
# --------------------------------------------------------------------------- #
def _layout(graph: nx.Graph, seed: int = 42) -> dict:
    if graph.number_of_nodes() == 0:
        return {}
    pos = nx.spring_layout(graph, seed=seed, weight="weight",
                           k=1.2 / (graph.number_of_nodes() ** 0.5 or 1), iterations=120)
    xs = [p[0] for p in pos.values()]
    ys = [p[1] for p in pos.values()]
    minx, maxx = min(xs), max(xs)
    miny, maxy = min(ys), max(ys)
    sx = (maxx - minx) or 1.0
    sy = (maxy - miny) or 1.0
    # normalise to [0,1] with a small margin
    return {n: [0.04 + 0.92 * (p[0] - minx) / sx, 0.04 + 0.92 * (p[1] - miny) / sy]
            for n, p in pos.items()}


def _obj_meta(store: GraphStore, oid: str) -> dict:
    n = store.get_node(oid)
    modality = (n.modality.value if n and n.modality else "text")
    label = (n.name or oid)
    return {"id": oid, "label": label[:60], "type": "episode", "modality": modality,
            "tags": (n.tags[:8] if n else [])}


# --------------------------------------------------------------------------- #
# overview: the episode graph + build order
# --------------------------------------------------------------------------- #
def object_subgraph(store: GraphStore) -> nx.Graph:
    G = nx.Graph()
    for n in store.nodes_of_type(NodeType.EPISODE):
        G.add_node(n.id)
    for u, v, d in store.all_edges():
        if not d.get("valid", True) or d["etype"] not in _OBJ_EDGES:
            continue
        if u not in G or v not in G:
            continue
        w = float(d["weight"])
        if G.has_edge(u, v):
            G[u][v]["weight"] += w
            G[u][v]["etypes"].add(d["etype"])
        else:
            G.add_edge(u, v, weight=w, etypes={d["etype"]})
    return G


def graph_payload(store: GraphStore) -> dict:
    G = object_subgraph(store)
    pos = _layout(G)
    nodes = []
    for oid in G.nodes():
        m = _obj_meta(store, oid)
        m["x"], m["y"] = pos.get(oid, [0.5, 0.5])
        m["deg"] = G.degree(oid)
        nodes.append(m)
    edges = []
    for u, v, d in G.edges(data=True):
        edges.append({"s": u, "t": v, "w": round(d["weight"], 3),
                      "etypes": sorted(d["etypes"])})
    # build order: ingestion order (created_at then id), text before images
    order = sorted((store.get_node(oid) for oid in G.nodes()),
                   key=lambda n: (n.created_at or "", n.id))
    build_order = [n.id for n in order]
    return {"nodes": nodes, "edges": edges, "build_order": build_order,
            "stats": store.stats()}


# --------------------------------------------------------------------------- #
# query trace: seeds → hubs → results focused subgraph
# --------------------------------------------------------------------------- #
def query_trace(g: KnowledgeGraph, query: str, mode: str = "bfs",
                k: int = 8, max_hubs: int = 28) -> dict:
    res = g.query(query, mode=mode, k=k)
    if not isinstance(res, list):
        ranked = res.objects
        seeds = res.seeds
    else:  # community/global path has no node-level traversal
        return {"query": query, "mode": "community", "nodes": [], "edges": [],
                "ranked": [], "seeds": [], "hops": [],
                "note": "global/breadth query — routed to community summaries, "
                        "no node traversal to draw."}

    store = g.store
    result_objs = [oid for oid, _ in ranked]
    seed_objs = [s for s in seeds if store.get_node(s)
                 and store.get_node(s).ntype == NodeType.EPISODE]
    obj_set = list(dict.fromkeys(seed_objs + result_objs))  # ordered unique

    # tag hubs that connect these episodes (the things the traversal hops through)
    hub_hits: dict[str, int] = {}
    for oid in obj_set:
        for nbr, d in store.neighbors(oid, etypes={EdgeType.TAGGED_AS}):
            hub_hits[nbr] = hub_hits.get(nbr, 0) + 1
    # also include any tag/entity nodes that were themselves seeds
    seed_hubs = [s for s in seeds if store.get_node(s)
                 and store.get_node(s).ntype in (NodeType.TAG, NodeType.ENTITY)]
    hubs = [h for h, c in sorted(hub_hits.items(), key=lambda kv: -kv[1]) if c >= 2]
    hubs = list(dict.fromkeys(seed_hubs + hubs))[:max_hubs]

    keep = set(obj_set) | set(hubs)
    sub = nx.Graph()
    for nid in keep:
        sub.add_node(nid)
    for oid in obj_set:
        for nbr, d in store.neighbors(oid):
            if nbr in keep and not sub.has_edge(oid, nbr):
                sub.add_edge(oid, nbr, etype=d["etype"], weight=float(d["weight"]))
    # directed entity→entity relationship edges (rev 4 — one parallel edge per
    # relation); aggregate the parallel labels per ordered pair for display
    rel_by_pair: dict[tuple[str, str], list[str]] = {}
    for src in keep:
        sn = store.get_node(src)
        if not sn or sn.ntype != NodeType.ENTITY:
            continue
        for dst, d in store.neighbors(src, etypes={EdgeType.RELATED_TO}, direction="out"):
            if dst not in keep:
                continue
            rn = store.get_node(d.get("rel_tag")) if d.get("rel_tag") else None
            if rn:
                rel_by_pair.setdefault((src, dst), []).append(rn.name)
    for (src, dst), names in rel_by_pair.items():
        sub.add_edge(src, dst, etype="RELATED_TO", weight=1.0,
                     directed=True, dsrc=src, dtgt=dst, rel=", ".join(names))
    pos = _layout(sub, seed=7)

    rank_of = {oid: i + 1 for i, (oid, _) in enumerate(ranked)}
    score_of = {oid: sc for oid, sc in ranked}
    seed_set = set(seeds)
    nodes = []
    for nid in sub.nodes():
        n = store.get_node(nid)
        if n is None:
            continue
        if n.ntype == NodeType.EPISODE:
            entry = _obj_meta(store, nid)
        else:
            entry = {"id": nid, "label": (n.name or nid)[:40],
                     "type": n.ntype.value, "modality": None, "tags": []}
        entry["x"], entry["y"] = pos.get(nid, [0.5, 0.5])
        roles = []
        if nid in seed_set:
            roles.append("seed")
        if nid in rank_of:
            roles.append("result")
            entry["rank"] = rank_of[nid]
            entry["score"] = round(float(score_of[nid]), 4)
        if n.ntype in (NodeType.TAG, NodeType.ENTITY):
            roles.append("hub")
        entry["roles"] = roles
        nodes.append(entry)

    edges = []
    for u, v, d in sub.edges(data=True):
        if d.get("directed"):
            edges.append({"s": d.get("dsrc", u), "t": d.get("dtgt", v),
                          "etype": d["etype"], "directed": True,
                          "rel": d.get("rel", "")})
        else:
            edges.append({"s": u, "t": v, "etype": d["etype"]})

    # BFS hop ordering from the seed objects, for the "watch it traverse" animation
    hops = _bfs_hops(sub, seed_objs or [n for n in sub.nodes()][:1])

    ranked_list = [{"id": oid, "rank": rank_of[oid], "score": round(float(sc), 4),
                    "label": _obj_meta(store, oid)["label"],
                    "modality": _obj_meta(store, oid)["modality"]}
                   for oid, sc in ranked]
    return {"query": query, "mode": mode, "nodes": nodes, "edges": edges,
            "ranked": ranked_list, "seeds": list(seeds), "hops": hops}


# Node types the dashboard's force graph actually draws (episodes + the two anchor kinds).
_DRAWN = {NodeType.EPISODE, NodeType.ENTITY, NodeType.TAG}


def rag_trace_payload(ans, store: GraphStore) -> dict:
    """Map a RagAnswer (kg/rag.py) onto the dashboard's per-query subgraph schema
    {query, mode, ranked, seeds, hops}. There is no per-hop LLM walk to replay, so the
    'hops' are the retrieve-then-read layers PPR produced — seeds → the entity/tag anchors
    it touched → the episodes it surfaced — revealed in order over the ingest graph. Ids
    are filtered to the node types the graph draws so the animation never references a node
    that isn't on screen (e.g. mention seeds)."""
    def is_t(nid, *types) -> bool:
        n = store.get_node(nid)
        return n is not None and n.ntype in types

    seeds = [s for s in ans.seeds if is_t(s, *_DRAWN)]
    episodes = [o for o in ans.object_ids if is_t(o, NodeType.EPISODE)]
    anchors = [t for t in ans.touched if is_t(t, NodeType.ENTITY, NodeType.TAG)]
    seen = set(seeds)
    hops = [list(dict.fromkeys(seeds))]
    layer2 = [a for a in anchors if a not in seen]
    if layer2:
        hops.append(layer2)
        seen |= set(layer2)
    layer3 = [e for e in episodes if e not in seen]
    if layer3:
        hops.append(layer3)
    ranked = []
    for i, oid in enumerate(episodes):
        n = store.get_node(oid)
        ranked.append({"id": oid, "rank": i + 1, "score": "",
                       "label": (n.name or oid) if n else oid,
                       "modality": (n.modality.value if n and n.modality else "text")})
    return {"query": ans.query, "mode": "rag", "nodes": [], "edges": [],
            "ranked": ranked, "seeds": seeds, "hops": [h for h in hops if h]}


def _bfs_hops(graph: nx.Graph, sources: list[str], max_hops: int = 4) -> list[list[str]]:
    if not sources:
        return []
    seen = set(s for s in sources if s in graph)
    frontier = list(seen)
    hops = [frontier]
    for _ in range(max_hops):
        nxt = []
        for nid in frontier:
            for nbr in graph.neighbors(nid):
                if nbr not in seen:
                    seen.add(nbr)
                    nxt.append(nbr)
        if not nxt:
            break
        hops.append(nxt)
        frontier = nxt
    return hops


# --------------------------------------------------------------------------- #
# HTML
# --------------------------------------------------------------------------- #
def render_html(graph: dict, trace: dict | None = None, server: bool = False) -> str:
    payload = {
        "graph": graph,
        "trace": trace,
        "server": server,
    }
    data_json = json.dumps(payload, ensure_ascii=False)
    return _HTML_TEMPLATE.replace("/*__DATA__*/", data_json)


# The viewer: vanilla JS, SVG, no dependencies. Pan/zoom, hover, build animation,
# query traversal highlight + BFS hop animation, results sidebar.
_HTML_TEMPLATE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>kg — knowledge graph viewer</title>
<style>
  :root{ --bg:#0e1116; --panel:#161b22; --line:#30363d; --txt:#e6edf3; --mut:#8b949e;
         --obj-text:#4f8ef7; --obj-image:#2ec27e; --tag:#f5a623; --entity:#b06ff0;
         --seed:#ffd24d; --result:#ff5d8f; --edge:#3a4250; --edge-hi:#ffae57; }
  *{box-sizing:border-box}
  html,body{margin:0;height:100%;background:var(--bg);color:var(--txt);
            font:13px/1.5 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif}
  #app{display:flex;height:100vh}
  #stage{flex:1;position:relative;overflow:hidden}
  svg{width:100%;height:100%;display:block;cursor:grab}
  svg.grab{cursor:grabbing}
  #side{width:330px;background:var(--panel);border-left:1px solid var(--line);
        display:flex;flex-direction:column;overflow:hidden}
  .sec{padding:14px 16px;border-bottom:1px solid var(--line)}
  h1{font-size:15px;margin:0 0 2px}
  h2{font-size:11px;letter-spacing:.08em;text-transform:uppercase;color:var(--mut);margin:0 0 8px}
  .mut{color:var(--mut)}
  input,select,button{font:inherit;color:var(--txt);background:#0d1117;border:1px solid var(--line);
        border-radius:6px;padding:7px 9px}
  input{width:100%}
  button{cursor:pointer;background:#21262d}
  button:hover{border-color:#5a6472}
  button.primary{background:var(--obj-text);border-color:var(--obj-text);color:#06121f;font-weight:600}
  .row{display:flex;gap:8px;align-items:center}
  .row>*{flex:0 0 auto}
  .grow{flex:1 1 auto}
  .legend span{display:inline-flex;align-items:center;gap:6px;margin:3px 10px 3px 0}
  .dot{width:11px;height:11px;border-radius:50%;display:inline-block}
  #results{overflow:auto;flex:1}
  .res{padding:9px 16px;border-bottom:1px solid #1d232b;cursor:pointer}
  .res:hover{background:#1b222c}
  .res .rk{display:inline-block;min-width:20px;height:20px;line-height:20px;text-align:center;
           border-radius:50%;background:var(--result);color:#1a0410;font-weight:700;font-size:11px;margin-right:8px}
  .res .sc{float:right;color:var(--mut);font-variant-numeric:tabular-nums}
  #tip{position:absolute;pointer-events:none;background:#0d1117ee;border:1px solid var(--line);
       border-radius:6px;padding:6px 9px;max-width:260px;font-size:12px;display:none;z-index:5}
  #status{position:absolute;left:12px;bottom:10px;color:var(--mut);font-size:12px;
          background:#0d1117aa;padding:4px 8px;border-radius:6px}
  .pill{font-size:11px;color:var(--mut);background:#0d1117;border:1px solid var(--line);
        border-radius:20px;padding:2px 9px}
  node{}
  circle.node{cursor:pointer;stroke:#0b0e13;stroke-width:1}
  circle.node.dim{opacity:.12}
  line.link{stroke:var(--edge);stroke-opacity:.5}
  line.link.dim{stroke-opacity:.06}
  line.link.hi{stroke:var(--edge-hi);stroke-opacity:.95;stroke-width:2.4}
  line.link.rel{stroke:var(--entity);stroke-opacity:.85;stroke-width:1.8}
  text.lbl{fill:var(--txt);font-size:9px;paint-order:stroke;stroke:#0b0e13;stroke-width:2.5px;pointer-events:none}
  text.rellbl{fill:var(--entity);font-size:8.5px;font-style:italic;paint-order:stroke;stroke:#0b0e13;stroke-width:2.5px;pointer-events:none}
  text.rank{fill:#1a0410;font-weight:700;font-size:11px;text-anchor:middle;dominant-baseline:central;pointer-events:none}
  .ringseed{fill:none;stroke:var(--seed);stroke-width:2.5}
  .ringres{fill:none;stroke:var(--result);stroke-width:2.5}
  .hidden{display:none}
</style>
</head>
<body>
<div id="app">
  <div id="stage">
    <svg id="svg">
      <defs>
        <marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7"
                markerHeight="7" orient="auto-start-reverse">
          <path d="M0,0 L10,5 L0,10 z" fill="var(--entity)"></path>
        </marker>
      </defs>
      <g id="view">
      <g id="links"></g><g id="rings"></g><g id="nodes"></g><g id="labels"></g>
    </g></svg>
    <div id="tip"></div>
    <div id="status"></div>
  </div>
  <div id="side">
    <div class="sec">
      <h1>kg viewer</h1>
      <div class="mut" id="subtitle">knowledge graph</div>
      <div class="legend" style="margin-top:8px">
        <span><i class="dot" style="background:var(--obj-text)"></i>article</span>
        <span><i class="dot" style="background:var(--obj-image)"></i>image</span>
        <span><i class="dot" style="background:var(--tag)"></i>tag</span>
        <span><i class="dot" style="background:var(--entity)"></i>entity</span>
        <span><i class="dot" style="background:var(--seed)"></i>seed</span>
        <span><i class="dot" style="background:var(--result)"></i>result</span>
        <span style="color:var(--entity)">→ relationship (directed)</span>
      </div>
    </div>

    <div class="sec">
      <h2>Build animation</h2>
      <div class="row">
        <button id="play">▶ Play build</button>
        <button id="showall">Show all</button>
        <span class="grow"></span>
        <span class="pill" id="buildcount"></span>
      </div>
      <input id="scrub" type="range" min="0" max="100" value="100" style="margin-top:10px"/>
    </div>

    <div class="sec">
      <h2>Query traversal</h2>
      <input id="q" placeholder="e.g. Canadian football Grey Cup champion"/>
      <div class="row" style="margin-top:8px">
        <select id="mode">
          <option value="bfs">BFS (watch the hops)</option>
          <option value="ppr">PPR seed-and-spread</option>
          <option value="vector">flat vector</option>
        </select>
        <button class="primary grow" id="run">Trace</button>
      </div>
      <div class="mut" id="qnote" style="margin-top:8px"></div>
    </div>

    <div id="results"></div>
  </div>
</div>
<script>
const DATA = /*__DATA__*/;
const NS="http://www.w3.org/2000/svg";
const COLORS={episode_text:"#4f8ef7",episode_image:"#2ec27e",tag:"#f5a623",entity:"#b06ff0",mention:"#7a8699",community:"#9aa4af"};
function colorOf(n){ if(n.type==="episode") return n.modality==="image"?COLORS.episode_image:COLORS.episode_text; return COLORS[n.type]||"#9aa4af"; }
function radius(n){ if(n.type==="episode") return 5+Math.min(7,(n.deg||0)*0.5); return n.type==="tag"?3.5:3; }

const svg=document.getElementById("svg"), view=document.getElementById("view");
const gLinks=document.getElementById("links"), gNodes=document.getElementById("nodes"),
      gLabels=document.getElementById("labels"), gRings=document.getElementById("rings");
const tip=document.getElementById("tip"), statusEl=document.getElementById("status");
let W=svg.clientWidth, H=svg.clientHeight;
function P(x,y){ return [x*W, y*H]; }

// ---- pan/zoom ----
let tx=0,ty=0,scale=1;
function applyView(){ view.setAttribute("transform",`translate(${tx},${ty}) scale(${scale})`); }
svg.addEventListener("wheel",e=>{e.preventDefault();
  const f=e.deltaY<0?1.1:1/1.1; const r=svg.getBoundingClientRect();
  const mx=e.clientX-r.left,my=e.clientY-r.top;
  tx=mx-(mx-tx)*f; ty=my-(my-ty)*f; scale*=f; applyView();},{passive:false});
let drag=null;
svg.addEventListener("mousedown",e=>{drag={x:e.clientX,y:e.clientY,tx,ty};svg.classList.add("grab");});
window.addEventListener("mousemove",e=>{ if(!drag)return; tx=drag.tx+(e.clientX-drag.x); ty=drag.ty+(e.clientY-drag.y); applyView();});
window.addEventListener("mouseup",()=>{drag=null;svg.classList.remove("grab");});

// ---- render a node/edge set ----
let CUR={nodes:[],edges:[]}, NODEBYID={}, elNodes={}, elRings={}, elLinks=[];
function clear(g){ while(g.firstChild) g.removeChild(g.firstChild); }
function render(nodes, edges, opts={}){
  CUR={nodes,edges}; NODEBYID={}; elNodes={}; elRings={}; elLinks=[];
  clear(gLinks); clear(gNodes); clear(gLabels); clear(gRings);
  W=svg.clientWidth; H=svg.clientHeight;
  nodes.forEach(n=>NODEBYID[n.id]=n);
  edges.forEach(e=>{
    const a=NODEBYID[e.s],b=NODEBYID[e.t]; if(!a||!b) return;
    const [x1,y1]=P(a.x,a.y); let [x2,y2]=P(b.x,b.y);
    const ln=document.createElementNS(NS,"line"); ln.setAttribute("class","link");
    if(e.directed){
      // pull the line back to the target's rim so the arrowhead is visible
      const dx=x2-x1, dy=y2-y1, L=Math.hypot(dx,dy)||1, rb=radius(b)+8;
      x2=x2-dx/L*rb; y2=y2-dy/L*rb;
      ln.classList.add("rel"); ln.setAttribute("marker-end","url(#arrow)");
    }
    ln.setAttribute("x1",x1);ln.setAttribute("y1",y1);ln.setAttribute("x2",x2);ln.setAttribute("y2",y2);
    ln._e=e; gLinks.appendChild(ln); elLinks.push(ln);
    if(e.directed && e.rel){
      const t=document.createElementNS(NS,"text"); t.setAttribute("class","rellbl");
      t.setAttribute("x",(x1+x2)/2);t.setAttribute("y",(y1+y2)/2-2);
      t.setAttribute("text-anchor","middle"); t.textContent=e.rel;
      gLabels.appendChild(t);
    }
  });
  nodes.forEach(n=>{
    const [x,y]=P(n.x,n.y);
    const c=document.createElementNS(NS,"circle"); c.setAttribute("class","node");
    c.setAttribute("cx",x);c.setAttribute("cy",y);c.setAttribute("r",radius(n));
    c.setAttribute("fill",colorOf(n)); c._n=n;
    c.addEventListener("mousemove",ev=>showTip(ev,n));
    c.addEventListener("mouseleave",hideTip);
    c.addEventListener("click",()=>focusNode(n.id));
    gNodes.appendChild(c); elNodes[n.id]=c;
    const roles=n.roles||[];
    if(roles.includes("seed")||roles.includes("result")){
      const ring=document.createElementNS(NS,"circle");
      ring.setAttribute("class",roles.includes("result")?"ringres":"ringseed");
      ring.setAttribute("cx",x);ring.setAttribute("cy",y);ring.setAttribute("r",radius(n)+3.5);
      gRings.appendChild(ring); elRings[n.id]=ring;
    }
    if(n.rank){ const t=document.createElementNS(NS,"text"); t.setAttribute("class","rank");
      t.setAttribute("x",x);t.setAttribute("y",y); t.textContent=n.rank; gLabels.appendChild(t); }
    else if(opts.labels && n.type!=="episode" || (opts.labels && n.deg>=opts.labelDeg)){
      const t=document.createElementNS(NS,"text"); t.setAttribute("class","lbl");
      t.setAttribute("x",x+radius(n)+2);t.setAttribute("y",y+3); t.textContent=n.label; gLabels.appendChild(t);
    }
  });
}
function showTip(ev,n){ const r=svg.getBoundingClientRect();
  tip.style.display="block"; tip.style.left=(ev.clientX-r.left+12)+"px"; tip.style.top=(ev.clientY-r.top+12)+"px";
  let h=`<b>${esc(n.label)}</b><br><span class="mut">${n.type}${n.modality?(" · "+n.modality):""}</span>`;
  if(n.score!=null) h+=`<br>rank #${n.rank} · score ${n.score}`;
  if(n.tags&&n.tags.length) h+=`<br><span class="mut">${n.tags.map(esc).join(", ")}</span>`;
  tip.innerHTML=h;
}
function hideTip(){ tip.style.display="none"; }
function esc(s){ return (s+"").replace(/[&<>]/g,c=>({"&":"&amp;","<":"&lt;",">":"&gt;"}[c])); }

function focusNode(id){
  const adj=new Set([id]);
  CUR.edges.forEach(e=>{ if(e.s===id)adj.add(e.t); if(e.t===id)adj.add(e.s); });
  CUR.nodes.forEach(n=>{ elNodes[n.id].classList.toggle("dim",!adj.has(n.id)); });
  elLinks.forEach(ln=>{ const hit=ln._e.s===id||ln._e.t===id; ln.classList.toggle("hi",hit); ln.classList.toggle("dim",!hit); });
}
function clearFocus(){ CUR.nodes.forEach(n=>elNodes[n.id]&&elNodes[n.id].classList.remove("dim"));
  elLinks.forEach(ln=>{ln.classList.remove("hi");ln.classList.remove("dim");}); }
svg.addEventListener("dblclick",clearFocus);

// ============ OVERVIEW (build animation) ============
const G=DATA.graph;
document.getElementById("subtitle").textContent =
  `${G.stats.by_node_type.episode||0} episodes · ${G.edges.length} episode↔episode edges`;
let buildN=G.build_order.length;
function showOverview(limit){
  viewMode="overview";
  const allow=new Set(G.build_order.slice(0,limit));
  const nodes=G.nodes.filter(n=>allow.has(n.id));
  const edges=G.edges.filter(e=>allow.has(e.s)&&allow.has(e.t));
  render(nodes,edges,{labels:true,labelDeg:9});
  document.getElementById("buildcount").textContent=`${nodes.length} / ${G.nodes.length} nodes`;
  statusEl.textContent="overview — scroll to zoom, drag to pan, click a node to isolate, double-click to reset";
}
let timer=null;
function playBuild(){
  if(timer){clearInterval(timer);timer=null;document.getElementById("play").textContent="▶ Play build";return;}
  document.getElementById("play").textContent="⏸ Pause";
  let i=Math.max(1,+document.getElementById("scrub").value/100*buildN|0);
  timer=setInterval(()=>{ i++; if(i>buildN){clearInterval(timer);timer=null;document.getElementById("play").textContent="▶ Play build";}
    document.getElementById("scrub").value=Math.round(i/buildN*100); showOverview(i); },120);
}
document.getElementById("play").onclick=playBuild;
document.getElementById("showall").onclick=()=>{document.getElementById("scrub").value=100;showOverview(buildN);};
document.getElementById("scrub").oninput=e=>showOverview(Math.max(1,Math.round(e.target.value/100*buildN)));

// ============ QUERY TRAVERSAL ============
function renderTrace(tr){
  viewMode="trace"; lastTrace=tr;
  const results=document.getElementById("results");
  if(tr.note){ results.innerHTML=`<div class="sec mut">${esc(tr.note)}</div>`; render([],[]); statusEl.textContent=tr.note; return; }
  render(tr.nodes,tr.edges,{labels:true,labelDeg:0});
  // results sidebar
  results.innerHTML="";
  tr.ranked.forEach(r=>{
    const d=document.createElement("div"); d.className="res"; d.dataset.id=r.id;
    d.innerHTML=`<span class="rk">${r.rank}</span>${esc(r.label)} <span class="sc">${r.score}</span>`;
    d.onmouseenter=()=>focusNode(r.id); d.onmouseleave=clearFocus;
    results.appendChild(d);
  });
  document.getElementById("qnote").textContent =
    `${tr.seeds.length} seeds · ${tr.nodes.length} nodes in subgraph · mode=${tr.mode}`;
  // animate BFS hops
  if(tr.hops&&tr.hops.length){
    CUR.nodes.forEach(n=>elNodes[n.id].classList.add("dim"));
    elLinks.forEach(l=>l.classList.add("dim"));
    let h=0; const shown=new Set();
    const step=()=>{
      if(h>=tr.hops.length){ statusEl.textContent=`traversal complete — ${tr.ranked.length} results ranked`; return; }
      tr.hops[h].forEach(id=>{ shown.add(id); if(elNodes[id])elNodes[id].classList.remove("dim"); });
      elLinks.forEach(l=>{ if(shown.has(l._e.s)&&shown.has(l._e.t)) l.classList.remove("dim"); });
      statusEl.textContent=`hop ${h} — reached ${shown.size} nodes`;
      h++; setTimeout(step,650);
    };
    step();
  } else { statusEl.textContent=`${tr.ranked.length} results`; }
}

async function runQuery(){
  const q=document.getElementById("q").value.trim(); if(!q) return;
  const mode=document.getElementById("mode").value;
  if(DATA.server){
    statusEl.textContent="querying…";
    const r=await fetch(`/api/query?q=${encodeURIComponent(q)}&mode=${mode}`);
    renderTrace(await r.json());
  } else {
    document.getElementById("qnote").innerHTML =
      `Static export — live queries need the server:<br><code>python -m kg serve</code>`;
  }
}
document.getElementById("run").onclick=runQuery;
document.getElementById("q").addEventListener("keydown",e=>{if(e.key==="Enter")runQuery();});

// ---- boot ----
let lastTrace=null, viewMode="overview";
window.addEventListener("resize",()=>{ if(viewMode==="trace"&&lastTrace) renderTrace(lastTrace);
  else showOverview(Math.round(+document.getElementById("scrub").value/100*buildN)||buildN); });
showOverview(buildN);
if(DATA.trace){ lastTrace=DATA.trace; renderTrace(DATA.trace); }
</script>
</body>
</html>
"""
