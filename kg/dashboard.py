"""Test-run dashboard (no build step, no CDN — vanilla JS + SVG, same idiom as viz.py).

A *run index* you click into, then a per-run page with an **Input ⇄ Query** toggle:

  INPUT  — the object graph animating into existence in ingestion order, with
           synchronized charts (cumulative cost/tokens, node/edge growth, avg
           tags-per-object, vocabulary growth, and the doc_frequency of the top
           tags over time — the temporal-tag-change view) plus a per-document card.

  QUERY  — aggregate accuracy/cost panels + a per-query list; click a query to
           replay the agent's traversal subgraph (seeds → touched hubs → cited
           results, BFS hops animated) alongside its answer, citations, accuracy
           and tool-call trace.

`render_run_html(run)` embeds one run's `run.json` into a self-contained .html file
(the static export). `serve(out_dir)` reuses the same page and adds the run index.
"""
from __future__ import annotations

import json
import os
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse


def render_run_html(run: dict, server: bool = False) -> str:
    payload = json.dumps({"run": run, "server": server}, ensure_ascii=False)
    return _RUN_TEMPLATE.replace("/*__DATA__*/", payload)


def render_index_html(runs: list) -> str:
    payload = json.dumps({"runs": runs}, ensure_ascii=False)
    return _INDEX_TEMPLATE.replace("/*__DATA__*/", payload)


# --------------------------------------------------------------------------- #
# stdlib server
# --------------------------------------------------------------------------- #
def _read_json(path: str):
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def make_handler(out_dir: str):
    class Handler(BaseHTTPRequestHandler):
        def _send(self, code, body, ctype):
            if isinstance(body, str):
                body = body.encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            u = urlparse(self.path)
            if u.path in ("/", "/index.html"):
                runs = _read_json(os.path.join(out_dir, "index.json")) or []
                self._send(200, render_index_html(runs), "text/html; charset=utf-8")
            elif u.path == "/run":
                rid = (parse_qs(u.query).get("id", [""])[0])
                run = _read_json(os.path.join(out_dir, rid, "run.json"))
                if not run:
                    self._send(404, "no such run", "text/plain")
                    return
                self._send(200, render_run_html(run, server=True), "text/html; charset=utf-8")
            elif u.path == "/api/runs":
                runs = _read_json(os.path.join(out_dir, "index.json")) or []
                self._send(200, json.dumps(runs), "application/json")
            elif u.path == "/api/run":
                rid = (parse_qs(u.query).get("id", [""])[0])
                run = _read_json(os.path.join(out_dir, rid, "run.json"))
                if not run:
                    self._send(404, "{}", "application/json")
                    return
                self._send(200, json.dumps(run, ensure_ascii=False), "application/json")
            else:
                self._send(404, "not found", "text/plain")

        def log_message(self, *a):
            pass

    return Handler


def serve(out_dir: str = "runs", port: int = 8050) -> None:
    os.makedirs(out_dir, exist_ok=True)
    idx = _read_json(os.path.join(out_dir, "index.json")) or []
    httpd = ThreadingHTTPServer(("127.0.0.1", port), make_handler(out_dir))
    print(f"kg dashboard on  http://127.0.0.1:{port}   ({len(idx)} run(s) in {out_dir}/)")
    if not idx:
        print(f"  no runs yet — `python -m kg testrun` writes one into {out_dir}/")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped.")


# --------------------------------------------------------------------------- #
# Shared CSS / graph + chart JS (used by both templates)
# --------------------------------------------------------------------------- #
_STYLE = r"""
:root{ --bg:#0e1116; --panel:#161b22; --panel2:#1b2129; --line:#30363d; --txt:#e6edf3;
  --mut:#8b949e; --obj-text:#4f8ef7; --obj-image:#2ec27e; --tag:#f5a623; --entity:#b06ff0;
  --rel:#b06ff0; --seed:#ffd24d; --result:#ff5d8f; --cited:#2ec27e; --edge:#3a4250; --edge-hi:#ffae57;
  --ok:#2ec27e; --bad:#ff5d8f; --warn:#f5a623; --accent:#4f8ef7; }
*{box-sizing:border-box}
html,body{margin:0;height:100%;background:var(--bg);color:var(--txt);
  font:13px/1.5 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif}
a{color:var(--accent);text-decoration:none} a:hover{text-decoration:underline}
.mut{color:var(--mut)} .num{font-variant-numeric:tabular-nums}
h1{font-size:16px;margin:0} h2{font-size:11px;letter-spacing:.08em;text-transform:uppercase;color:var(--mut);margin:0 0 8px}
button{font:inherit;color:var(--txt);background:#21262d;border:1px solid var(--line);
  border-radius:6px;padding:6px 12px;cursor:pointer} button:hover{border-color:#5a6472}
button.on{background:var(--accent);border-color:var(--accent);color:#06121f;font-weight:600}
input[type=range]{width:100%}
.pill{font-size:11px;color:var(--mut);background:#0d1117;border:1px solid var(--line);
  border-radius:20px;padding:2px 9px;white-space:nowrap}
.stat{background:var(--panel2);border:1px solid var(--line);border-radius:8px;padding:8px 11px;min-width:0}
.stat .k{font-size:10.5px;color:var(--mut);text-transform:uppercase;letter-spacing:.05em}
.stat .v{font-size:18px;font-weight:600;font-variant-numeric:tabular-nums;margin-top:1px}
.stat .v small{font-size:11px;color:var(--mut);font-weight:400}
.card{background:var(--panel);border:1px solid var(--line);border-radius:10px}
.legend span{display:inline-flex;align-items:center;gap:5px;margin:2px 9px 2px 0;font-size:11.5px}
.dot{width:10px;height:10px;border-radius:50%;display:inline-block}
.bar{height:6px;border-radius:4px;background:#0d1117;overflow:hidden}
.bar>i{display:block;height:100%;background:var(--accent)}
svg .node{stroke:#0b0e13;stroke-width:1;cursor:pointer}
svg .node.dim{opacity:.10}
svg line.link{stroke:var(--edge);stroke-opacity:.5}
svg line.link.dim{stroke-opacity:.05}
svg line.link.hi{stroke:var(--edge-hi);stroke-opacity:.95;stroke-width:2.2}
svg line.link.rel{stroke:var(--rel);stroke-opacity:.85;stroke-width:1.6}
svg text.lbl{fill:var(--txt);font-size:6.5px;text-anchor:middle;paint-order:stroke;stroke:#0b0e13;stroke-width:2.2px;pointer-events:none;opacity:.9}
svg.hidelabels text.lbl{display:none}
svg text.rank{fill:#1a0410;font-weight:700;font-size:11px;text-anchor:middle;dominant-baseline:central;pointer-events:none}
svg .ringseed{fill:none;stroke:var(--seed);stroke-width:2.5}
svg .ringres{fill:none;stroke:var(--result);stroke-width:2.5}
svg .ringcite{fill:none;stroke:var(--cited);stroke-width:3}
svg .ringtrunc{fill:none;stroke:var(--mut);stroke-width:1.6;stroke-dasharray:2,2}
#tip{position:absolute;pointer-events:none;background:#0d1117ee;border:1px solid var(--line);
  border-radius:6px;padding:6px 9px;max-width:280px;font-size:12px;display:none;z-index:9}
.stage{position:relative;overflow:hidden;background:#0b0e13;border-radius:10px}
.stage svg{width:100%;height:100%;display:block;cursor:grab} .stage svg.grab{cursor:grabbing}
.chart text{fill:var(--mut);font-size:9px}
.row{display:flex;gap:10px} .wrap{flex-wrap:wrap}
table{border-collapse:collapse;width:100%;font-size:12px}
th{color:var(--mut);text-align:left;font-weight:500;padding:5px 8px;border-bottom:1px solid var(--line)}
td{padding:5px 8px;border-bottom:1px solid #1d232b}
.qrow{cursor:pointer} .qrow:hover{background:#1b222c} .qrow.sel{background:#1d2530}
.tag{display:inline-block;background:#0d1117;border:1px solid var(--line);border-radius:5px;
  padding:1px 6px;margin:2px 3px 0 0;font-size:11px}
.scroll{overflow:auto}
"""

# Shared JS: a pan/zoom graph widget + a tiny SVG line/bar chart kit.
_GRAPHJS = r"""
const NS="http://www.w3.org/2000/svg";
const COL={episode_text:"#4f8ef7",episode_image:"#2ec27e",tag:"#f5a623",entity:"#b06ff0",
  relation:"#b06ff0",community:"#9aa4af"};
function colorOf(n){ if(n.type==="episode") return n.modality==="image"?COL.episode_image:COL.episode_text;
  return COL[n.type]||"#9aa4af"; }
function radius(n){ if(n.type==="episode") return 4+Math.min(7,(n.deg||0)*0.5); return n.type==="tag"?3.2:2.8; }
function esc(s){ return (s+"").replace(/[&<>]/g,c=>({"&":"&amp;","<":"&lt;",">":"&gt;"}[c])); }
function el(tag,attrs,kids){ const e=document.createElement(tag); for(const k in (attrs||{})){
  if(k==="html")e.innerHTML=attrs[k]; else if(k==="text")e.textContent=attrs[k]; else e.setAttribute(k,attrs[k]); }
  (kids||[]).forEach(c=>e.appendChild(c)); return e; }

// ---- pan/zoom SVG graph widget ----
function makeGraph(stage, tip){
  const svg=document.createElementNS(NS,"svg");
  svg.innerHTML='<defs><marker id="ar" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" fill="#b06ff0"></path></marker></defs>'+
    '<g id="v"><g class="links"></g><g class="rings"></g><g class="nodes"></g><g class="labels"></g></g>';
  stage.appendChild(svg);
  const view=svg.querySelector("#v");
  const gL=svg.querySelector(".links"),gR=svg.querySelector(".rings"),gN=svg.querySelector(".nodes"),gT=svg.querySelector(".labels");
  let tx=0,ty=0,scale=1,W=10,H=10,CUR={nodes:[],edges:[]},elN={},elL=[],NB={};
  let zoomGated=false;
  function apply(){ view.setAttribute("transform",`translate(${tx},${ty}) scale(${scale})`); }
  function updateLabelVis(){ svg.classList.toggle("hidelabels", zoomGated && scale<1.4); }
  // keep label/rank text a constant on-screen size by counter-scaling it against the
  // group zoom, so zooming in doesn't blow the text up.
  function restyle(){ const fs=(11/scale).toFixed(2)+"px", sw=(2.6/scale).toFixed(2)+"px";
    gT.querySelectorAll("text.lbl").forEach(t=>{t.style.fontSize=fs;t.style.strokeWidth=sw;});
    gT.querySelectorAll("text.rank").forEach(t=>{t.style.fontSize=(12/scale).toFixed(2)+"px";}); }
  svg.addEventListener("wheel",e=>{e.preventDefault();const r=svg.getBoundingClientRect();
    // gentle, damped zoom: cap each event's delta (kills trackpad momentum "slipperiness"),
    // map it through exp() for smooth proportional steps, and clamp the absolute scale.
    const dy=Math.max(-40,Math.min(40,e.deltaY||0));
    const ns=Math.max(0.3,Math.min(8,scale*Math.exp(-dy*0.0016))); const f=ns/scale;
    const mx=e.clientX-r.left,my=e.clientY-r.top; tx=mx-(mx-tx)*f; ty=my-(my-ty)*f; scale=ns;
    apply(); updateLabelVis(); restyle();},{passive:false});
  let drag=null;
  svg.addEventListener("mousedown",e=>{drag={x:e.clientX,y:e.clientY,tx,ty};svg.classList.add("grab");});
  window.addEventListener("mousemove",e=>{if(!drag)return;tx=drag.tx+(e.clientX-drag.x);ty=drag.ty+(e.clientY-drag.y);apply();});
  window.addEventListener("mouseup",()=>{drag=null;svg.classList.remove("grab");});
  svg.addEventListener("dblclick",()=>{tx=0;ty=0;scale=1;apply();updateLabelVis();restyle();clearFocus();});
  function clear(g){while(g.firstChild)g.removeChild(g.firstChild);}
  function P(x,y){return [x*W,y*H];}
  function showTip(ev,n){ if(!tip)return; const r=stage.getBoundingClientRect();
    tip.style.display="block"; tip.style.left=(ev.clientX-r.left+12)+"px"; tip.style.top=(ev.clientY-r.top+12)+"px";
    let h=`<b>${esc(n.label||n.id)}</b><br><span class="mut">${n.type}${n.modality?(" · "+n.modality):""}</span>`;
    if(n.rank)h+=`<br>rank #${n.rank}${n.score!=null&&n.score!==""?(" · "+n.score):""}`;
    const roles=n.roles||[];
    if(roles.includes("cited"))h+=`<br><span style="color:var(--cited)">✓ cited in the answer</span>`;
    else if(roles.includes("context"))h+=`<br><span class="mut">in the reader's context</span>`;
    else if(roles.includes("result"))h+=`<br><span class="mut">retrieved, but cut before the reader (rag_context_episodes)</span>`;
    if(n.tags&&n.tags.length)h+=`<br><span class="mut">${n.tags.map(esc).join(", ")}</span>`;
    if(n.snippet){ const short=n.snippet.length>220?n.snippet.slice(0,220)+"…":n.snippet;
      h+=`<div style="margin-top:5px;white-space:normal;font-size:11px;color:#c9d1d9;line-height:1.4">${esc(short)}</div>`; }
    tip.innerHTML=h; }
  function hideTip(){ if(tip)tip.style.display="none"; }
  function focus(id){ const adj=new Set([id]);
    CUR.edges.forEach(e=>{if(e.s===id)adj.add(e.t); if(e.t===id)adj.add(e.s);});
    CUR.nodes.forEach(n=>{const c=elN[n.id]; if(c)c.classList.toggle("dim",!adj.has(n.id));});
    elL.forEach(l=>{const hit=l._e.s===id||l._e.t===id; l.classList.toggle("hi",hit); l.classList.toggle("dim",!hit);}); }
  function clearFocus(){ CUR.nodes.forEach(n=>elN[n.id]&&elN[n.id].classList.remove("dim"));
    elL.forEach(l=>{l.classList.remove("hi");l.classList.remove("dim");}); }
  function render(nodes,edges,opts){ opts=opts||{}; CUR={nodes,edges}; elN={};elL=[];NB={};
    clear(gL);clear(gR);clear(gN);clear(gT); W=svg.clientWidth||stage.clientWidth||600; H=svg.clientHeight||stage.clientHeight||400;
    nodes.forEach(n=>NB[n.id]=n);
    // dedup labels: one per distinct text, placed on its highest-degree node, so the
    // per-paragraph chunks of one article don't stack N identical labels.
    const labelOwner={};
    if(opts.labels){ const best={}; nodes.forEach(n=>{ const lab=(n.label||n.id),d=(n.deg||0);
      if(best[lab]===undefined||d>best[lab]){best[lab]=d;labelOwner[lab]=n.id;} }); }
    edges.forEach(e=>{const a=NB[e.s],b=NB[e.t]; if(!a||!b)return; const[x1,y1]=P(a.x,a.y); let[x2,y2]=P(b.x,b.y);
      const ln=document.createElementNS(NS,"line"); ln.setAttribute("class","link");
      if(e.directed){const dx=x2-x1,dy=y2-y1,L=Math.hypot(dx,dy)||1,rb=radius(b)+8; x2-=dx/L*rb; y2-=dy/L*rb;
        ln.classList.add("rel"); ln.setAttribute("marker-end","url(#ar)");}
      ln.setAttribute("x1",x1);ln.setAttribute("y1",y1);ln.setAttribute("x2",x2);ln.setAttribute("y2",y2);
      ln._e=e; gL.appendChild(ln); elL.push(ln);
      if(e.directed&&e.rel){const t=document.createElementNS(NS,"text");t.setAttribute("class","lbl");
        t.setAttribute("x",(x1+x2)/2);t.setAttribute("y",(y1+y2)/2-2);t.setAttribute("fill","#b06ff0");
        t.setAttribute("text-anchor","middle");t.textContent=e.rel;gT.appendChild(t);} });
    nodes.forEach(n=>{const[x,y]=P(n.x,n.y); const c=document.createElementNS(NS,"circle");c.setAttribute("class","node");
      c.setAttribute("cx",x);c.setAttribute("cy",y);c.setAttribute("r",radius(n));c.setAttribute("fill",colorOf(n));
      c.addEventListener("mousemove",ev=>showTip(ev,n)); c.addEventListener("mouseleave",hideTip);
      c.addEventListener("click",()=>focus(n.id)); gN.appendChild(c); elN[n.id]=c;
      const roles=n.roles||[];
      if(roles.includes("seed")||roles.includes("result")){const ring=document.createElementNS(NS,"circle");
        // consumption > retrieval: an episode's ring shows what the READER did with it
        // (cited > in-context > truncated-away) before falling back to plain seed/result.
        const cls=roles.includes("cited")?"ringcite"
          :roles.includes("context")?"ringres"
          :roles.includes("result")?"ringtrunc"
          :"ringseed";
        ring.setAttribute("class",cls);
        ring.setAttribute("cx",x);ring.setAttribute("cy",y);ring.setAttribute("r",radius(n)+3.5);gR.appendChild(ring);}
      if(n.rank){const t=document.createElementNS(NS,"text");t.setAttribute("class","rank");
        t.setAttribute("x",x);t.setAttribute("y",y);t.textContent=n.rank;gT.appendChild(t);}
      else if(opts.labels&&labelOwner[(n.label||n.id)]===n.id){
        const t=document.createElementNS(NS,"text");t.setAttribute("class","lbl");
        t.setAttribute("x",x);t.setAttribute("y",y+radius(n)+8);t.textContent=(n.label||n.id);gT.appendChild(t);} });
    zoomGated=!!opts.zoomLabels; updateLabelVis(); restyle();
  }
  function setVisible(ids){ // ids = Set of node ids to show (others hidden)
    CUR.nodes.forEach(n=>{const c=elN[n.id]; if(c)c.style.display=ids.has(n.id)?"":"none";});
    elL.forEach(l=>{l.style.display=(ids.has(l._e.s)&&ids.has(l._e.t))?"":"none";});
    gT.querySelectorAll("text").forEach(t=>{}); }
  function dimAll(){ CUR.nodes.forEach(n=>elN[n.id]&&elN[n.id].classList.add("dim")); elL.forEach(l=>l.classList.add("dim")); }
  function undim(ids){ ids.forEach(id=>{const c=elN[id]; if(c)c.classList.remove("dim");});
    elL.forEach(l=>{if(ids.has(l._e.s)&&ids.has(l._e.t))l.classList.remove("dim");}); }
  return {render,setVisible,focus,clearFocus,dimAll,undim,reset:()=>{tx=0;ty=0;scale=1;apply();}};
}

// ---- tiny SVG charts ----
function lineChart(host, series, opts){ opts=opts||{}; const W=opts.w||300,H=opts.h||120,pad=18;
  const n=Math.max(1,(series[0]&&series[0].pts.length)||1);
  let max=opts.max!=null?opts.max:0,min=0;
  series.forEach(s=>s.pts.forEach(v=>{if(v>max)max=v;}));
  if(opts.maxFloor!=null&&max<opts.maxFloor)max=opts.maxFloor; if(max<=0)max=1;
  const X=i=>pad+(i/(Math.max(1,n-1)))*(W-pad-4), Y=v=>H-pad-(v-min)/(max-min||1)*(H-pad-8);
  let svg=`<svg class="chart" viewBox="0 0 ${W} ${H}" width="100%" preserveAspectRatio="none" style="display:block">`;
  svg+=`<line x1="${pad}" y1="${H-pad}" x2="${W}" y2="${H-pad}" stroke="#30363d"/>`;
  svg+=`<line x1="${pad}" y1="8" x2="${pad}" y2="${H-pad}" stroke="#30363d"/>`;
  svg+=`<text x="${pad-2}" y="12" text-anchor="end">${fmtN(max)}</text>`;
  series.forEach(s=>{ let d=""; s.pts.forEach((v,i)=>{ d+=(i?"L":"M")+X(i).toFixed(1)+" "+Y(v).toFixed(1)+" "; });
    svg+=`<path d="${d}" fill="none" stroke="${s.color}" stroke-width="1.6"/>`; });
  svg+=`<line class="cursor" x1="${X(n-1)}" y1="8" x2="${X(n-1)}" y2="${H-pad}" stroke="#e6edf3" stroke-opacity=".35"/>`;
  svg+=`</svg>`;
  host.innerHTML=svg; host._X=X; host._n=n;
  return host; }
function chartCursor(host, i){ const c=host.querySelector(".cursor"); if(c&&host._X){const x=host._X(Math.min(i,host._n-1));
  c.setAttribute("x1",x);c.setAttribute("x2",x);} }
function barChart(host, items, opts){ opts=opts||{}; const max=Math.max(0.001,...items.map(d=>d.v));
  host.innerHTML=items.map(d=>`<div style="margin:4px 0">
    <div class="row" style="justify-content:space-between"><span class="mut" style="font-size:11px">${esc(d.k)}</span>
    <span class="num" style="font-size:11px">${opts.fmt?opts.fmt(d.v):d.v}</span></div>
    <div class="bar"><i style="width:${(d.v/max*100).toFixed(1)}%;background:${d.c||'#4f8ef7'}"></i></div></div>`).join(""); }
function fmtN(v){ if(v>=1e6)return (v/1e6).toFixed(1)+"M"; if(v>=1e3)return (v/1e3).toFixed(1)+"k";
  return (v%1===0)?v:(+v).toFixed(2); }
function fmtUSD(v){ return "$"+(+v||0).toFixed(v<1?4:2); }
function fmtS(v){ v=+v||0; return v>=60?(v/60).toFixed(1)+"m":v>=1?v.toFixed(1)+"s":Math.round(v*1000)+"ms"; }
// profile dict -> sorted barChart items. Values are either bare seconds (per-item compact
// form) or {seconds, calls} (run-level totals); ×N annotates multi-call stages.
function profItems(prof, color){ return Object.entries(prof||{}).map(([k,v])=>{
    const s=(typeof v==="number")?v:(v.seconds||0), c=(v&&v.calls)||0;
    return {k:k+(c>1?" ×"+c:""), v:s, c:color||"#f5a623"}; })
  .filter(d=>d.v>0).sort((a,b)=>b.v-a.v); }
"""

# Canvas force-directed graph — Obsidian-style: live gravity/charge/link physics, draggable
# nodes, grow-in-generation-order, green raw-entry vs blue created nodes, directed arrows +
# relationship edge labels, constant-size deduped node labels, click-to-inspect.
_FORCEJS = r"""
function makeForce(stage, tip, h){
  h=h||{};
  const cv=document.createElement("canvas"); cv.style.cssText="display:block;width:100%;height:100%;cursor:grab";
  stage.appendChild(cv); const ctx=cv.getContext("2d");
  let W=600,H=400,DPR=Math.min(2,window.devicePixelRatio||1);
  function resize(){ W=stage.clientWidth||600; H=stage.clientHeight||400;
    cv.width=W*DPR; cv.height=H*DPR; ctx.setTransform(DPR,0,0,DPR,0,0); }
  resize(); try{ new ResizeObserver(()=>{resize();draw();}).observe(stage); }catch(e){}

  let N=[],E=[],byId={},owner={};
  let tx=0,ty=0,scale=1, step=-1, revN=[],revE=[];
  let sel=null,hover=null,drag=null,pan=null,moved=0;
  let raf=null, running=false, autofit=true, alpha=1, alphaTarget=0;
  let hiSet=null, hiRole={};   // optional highlight overlay (Query "full graph" mode)
  const ROLE_COLOR={seed:"#ffd24d",cited:"#2ec27e",context:"#ff5d8f",touched:"#8b949e"};
  // Obsidian-style CONTINUOUS force sim (the d3-force model Obsidian's Center/Repel/Link/
  // Link-distance controls map to: many-body charge + link springs + weak center, velocity-
  // Verlet integration, alpha cooling). It IDLES at 0 CPU once settled (alpha<A_MIN & not
  // reheated) and REHEATS on drag so neighbours follow the dragged node, then re-cools.
  // Repulsion uses a spatial-hash grid with a distanceMax cutoff (local, ~O(n·k)) so it stays
  // smooth at hundreds–thousands of nodes without an O(n²) all-pairs pass (Barnes-Hut's role).
  const A_MIN=0.001, A_DECAY=0.0228, V_DECAY=0.5;           // ~300-tick cool; 50% velocity kept
  const CHARGE=-160, DMAX=360, DMIN2=1, LINKDIST=46, LINKK=0.9, CENTERK=0.07, TH=1.2;
  const ELABEL={TAGGED_AS:"tagged",MENTIONS:"mentions",SHARED_TAG:"shared tag",
                SHARED_ENTITY:"shared entity",SIMILAR_TO:"similar",HYPERLINKS_TO:"links to"};
  function edgeLabel(e){ if(e.etype==="RELATED_TO") return e.rel||"related to";
    return ELABEL[e.etype]||(e.etype||"").toLowerCase().replace(/_/g," "); }
  const rad=n=>3+Math.min(13,Math.sqrt(n.indeg||0)*2.2);   // grow with incoming edges
  const rnd=()=>Math.random()*2-1;
  function S(wx,wy){ return [wx*scale+tx+W/2, wy*scale+ty+H/2]; }
  function Wld(sx,sy){ return [(sx-tx-W/2)/scale, (sy-ty-H/2)/scale]; }

  function load(nodes,edges){
    // Organic SEED for the live sim: each node starts on a random ring around the first
    // already-placed node it connects to (episodes anchor their own cluster, spread on a
    // ring), so the simulation starts hub-clustered and never from a grid/origin pile-up.
    // Then the continuous force sim (reheat below) takes over and settles it.
    const TAU=6.2832;
    N=nodes.map(n=>Object.assign({},n,{x:0,y:0,vx:0,vy:0,fx:null,fy:null,rad:rad(n),_p:false}));
    byId={}; N.forEach(n=>byId[n.id]=n);
    E=edges.filter(e=>byId[e.s]&&byId[e.t]).map(e=>({etype:e.etype,directed:e.directed,rel:e.rel,a:byId[e.s],b:byId[e.t]}));
    const adj={}, deg={}; N.forEach(n=>{adj[n.id]=[]; deg[n.id]=0;});
    for(const e of E){ adj[e.a.id].push(e.b); adj[e.b.id].push(e.a); deg[e.a.id]++; deg[e.b.id]++; }
    N.forEach(n=>n._k=deg[n.id]||1); E.forEach(e=>{ e.ka=e.a._k; e.kb=e.b._k; });  // link-strength degrees
    const order=N.slice().sort((a,b)=>((a.appear||0)-(b.appear||0))||((b.deg||0)-(a.deg||0)));
    let roots=0;
    for(const n of order){
      let p=null;
      if(!n.raw){ for(const m of adj[n.id]){ if(m._p){ p=m; break; } } }  // episodes anchor own cluster
      if(p){ const a=Math.random()*TAU, r=p.rad+n.rad+20+Math.random()*60; n.x=p.x+Math.cos(a)*r; n.y=p.y+Math.sin(a)*r; }
      else { const a=Math.random()*TAU, r=roots?300*Math.sqrt(roots):0; n.x=Math.cos(a)*r; n.y=Math.sin(a)*r; roots++; }
      n._p=true;
    }
    owner={}; const best={};
    N.forEach(n=>{ const L=n.label||n.id,d=(n.indeg||n.deg||0); if(best[L]===undefined||d>best[L]){best[L]=d;owner[L]=n.id;} });
    step=N.length; revN=N.slice(); revE=E.slice();
    // SYNCHRONOUS WARMUP: run most of the cool off-screen so the graph appears already
    // organized (force-graph's warmupTicks) — instant settled layout, no 5s wait, and not
    // dependent on rAF frame-rate. The live loop then just finishes the cool + handles drag.
    alpha=1; alphaTarget=0; autofit=true;
    for(let i=0;i<250 && alpha>A_MIN;i++) stepPhysics();
    autofit=true; reheat(0);   // brief live finish (frames + settles), then idles; drag reheats it
  }
  // Stepping only changes WHAT is drawn — positions are fixed by the one-time settle.
  function revealStep(i){ step=i; const ok=new Set();
    N.forEach(n=>{ if(n.appear<=i) ok.add(n.id); });
    revN=N.filter(n=>ok.has(n.id));
    revE=E.filter(e=>ok.has(e.a.id)&&ok.has(e.b.id));
    draw();
  }
  function reheat(target){ if(target!=null) alphaTarget=target;
    if(target>0 && alpha<0.3) alpha=0.3;            // wake a settled sim so it reacts to drag
    if(!running){ running=true; raf=requestAnimationFrame(tick); } }
  function easeFit(){ const pts=revN.length?revN:N; if(!pts.length)return; let a=1e9,b=1e9,c=-1e9,d=-1e9;
    for(const n of pts){ if(n.x<a)a=n.x; if(n.x>c)c=n.x; if(n.y<b)b=n.y; if(n.y>d)d=n.y; }
    const bw=Math.max(1,c-a),bh=Math.max(1,d-b),cx=(a+c)/2,cy=(b+d)/2;
    const ts=Math.max(0.1,Math.min(3,Math.min(W*0.88/bw,H*0.88/bh)));
    scale+=(ts-scale)*0.1; tx+=(-cx*ts-tx)*0.1; ty+=(-cy*ts-ty)*0.1; }
  function tick(){ raf=null; if(!running) return;
    stepPhysics(); draw();
    if(alpha<A_MIN && alphaTarget===0){ running=false; autofit=false; }   // settled → idle (0 CPU)
    else raf=requestAnimationFrame(tick); }
  function stepPhysics(){
    alpha += (alphaTarget - alpha) * A_DECAY;
    // 1) many-body CHARGE (repulsion): spatial-hash grid + distanceMax cutoff (local, fast)
    const cell=DMAX, grid={}, dmax2=DMAX*DMAX;
    for(const n of N){ const k=Math.floor(n.x/cell)+","+Math.floor(n.y/cell); (grid[k]||(grid[k]=[])).push(n); }
    for(const n of N){ const cx=Math.floor(n.x/cell),cy=Math.floor(n.y/cell);
      for(let gx=cx-1;gx<=cx+1;gx++)for(let gy=cy-1;gy<=cy+1;gy++){ const arr=grid[gx+","+gy]; if(!arr)continue;
        for(const m of arr){ if(m===n)continue; let dx=m.x-n.x,dy=m.y-n.y,l2=dx*dx+dy*dy;
          if(l2>dmax2)continue;
          if(l2<DMIN2){ dx=(Math.random()-0.5)*1e-2; dy=(Math.random()-0.5)*1e-2; l2=dx*dx+dy*dy+1e-6; }
          const w=CHARGE*alpha/l2; n.vx+=dx*w; n.vy+=dy*w; } } }   // CHARGE<0 ⇒ pushes n away from m
    // 2) LINK springs toward LINKDIST, degree-normalised so hubs stay put (hub-and-spoke look)
    for(const e of E){ const a=e.a,b=e.b;
      let dx=(b.x+b.vx)-(a.x+a.vx), dy=(b.y+b.vy)-(a.y+a.vy), l=Math.sqrt(dx*dx+dy*dy)||1e-6;
      const ll=(l-LINKDIST)/l*alpha*(LINKK/Math.min(e.ka,e.kb)), bias=e.ka/(e.ka+e.kb);
      dx*=ll; dy*=ll; b.vx-=dx*bias; b.vy-=dy*bias; a.vx+=dx*(1-bias); a.vy+=dy*(1-bias); }
    // 3) weak CENTER pull (compactness, no axis/grid) + 4) integrate w/ friction; pin dragged
    const ck=CENTERK*alpha;
    for(const n of N){ n.vx+=(-n.x)*ck; n.vy+=(-n.y)*ck;
      if(n.fx!=null){ n.x=n.fx; n.y=n.fy; n.vx=0; n.vy=0; }
      else { n.vx*=(1-V_DECAY); n.vy*=(1-V_DECAY);
             const sp=Math.hypot(n.vx,n.vy); if(sp>50){ n.vx*=50/sp; n.vy*=50/sp; }   // anti-blowup
             n.x+=n.vx; n.y+=n.vy; } }
    if(autofit) easeFit();
  }

  function focusSet(){ if(!sel)return null; const s=new Set([sel.id]);
    for(const e of E){ if(e.a.id===sel.id)s.add(e.b.id); if(e.b.id===sel.id)s.add(e.a.id); } return s; }
  function draw(){ ctx.clearRect(0,0,W,H); const F=focusSet();
    for(const e of revE){ const [x1,y1]=S(e.a.x,e.a.y); let [x2,y2]=S(e.b.x,e.b.y);
      const sem=e.etype==="RELATED_TO"||e.etype==="MENTIONS"||e.etype==="TAGGED_AS";
      const inc=F&&(e.a.id===sel.id||e.b.id===sel.id);
      const dim=(F&&!(F.has(e.a.id)&&F.has(e.b.id)))||(hiSet&&!(hiSet.has(e.a.id)&&hiSet.has(e.b.id)));
      ctx.strokeStyle=inc?"rgba(176,111,240,0.92)":(dim?"rgba(120,130,150,0.04)":(sem?"rgba(150,160,180,0.42)":"rgba(90,100,120,0.07)"));
      ctx.lineWidth=inc?1.7:(sem?1.1:0.7); ctx.beginPath(); ctx.moveTo(x1,y1);
      if(e.directed){ const dx=x2-x1,dy=y2-y1,L=Math.hypot(dx,dy)||1,rb=e.b.rad*scale+5; x2-=dx/L*rb; y2-=dy/L*rb;
        ctx.lineTo(x2,y2); ctx.stroke();
        if(!dim){ const ah=inc?7:5,ang=Math.atan2(dy,dx); ctx.fillStyle=ctx.strokeStyle; ctx.beginPath();
          ctx.moveTo(x2,y2); ctx.lineTo(x2-ah*Math.cos(ang-0.5),y2-ah*Math.sin(ang-0.5));
          ctx.lineTo(x2-ah*Math.cos(ang+0.5),y2-ah*Math.sin(ang+0.5)); ctx.closePath(); ctx.fill(); }
      } else { ctx.lineTo(x2,y2); ctx.stroke(); } }
    for(const n of revN){ const [x,y]=S(n.x,n.y), r=Math.max(2.5,n.rad*scale),
        dim=(F&&!F.has(n.id))||(hiSet&&!hiSet.has(n.id));
      ctx.beginPath(); ctx.arc(x,y,r,0,6.2832);
      ctx.fillStyle=n.raw?(dim?"rgba(46,194,126,0.16)":"#2ec27e"):(dim?"rgba(79,142,247,0.16)":"#4f8ef7"); ctx.fill();
      if(n===sel){ ctx.strokeStyle="#ffd24d"; ctx.lineWidth=2; ctx.beginPath(); ctx.arc(x,y,r+3,0,6.2832); ctx.stroke(); } }
    // query-highlight overlay: ring the nodes this query touched, colored by what the
    // reader did with them (cited > in-context > seed > merely touched — see ROLE_COLOR)
    if(hiSet){ for(const n of revN){ if(!hiSet.has(n.id))continue; const [x,y]=S(n.x,n.y), r=Math.max(2.5,n.rad*scale);
      const role=hiRole[n.id]||"touched", col=ROLE_COLOR[role]||ROLE_COLOR.touched;
      ctx.beginPath(); ctx.arc(x,y,r+3,0,6.2832); ctx.strokeStyle=col; ctx.lineWidth=role==="cited"?3:2;
      if(role==="touched") ctx.setLineDash([2,2]);
      ctx.stroke(); ctx.setLineDash([]); } }
    // edge labels: ALL of a selected node's connections (focus), plus relations on zoom.
    // Multiple relationships between the SAME pair (e.g. "hosts" + "hosted by", or several
    // parallel rel_tags) are GROUPED by node-pair and STACKED vertically around the midpoint
    // so they read as a list instead of overprinting into a blur.
    ctx.font="10px sans-serif"; ctx.textAlign="center"; ctx.textBaseline="middle";
    const lblGroups={};
    for(const e of revE){ const inc=F&&(e.a.id===sel.id||e.b.id===sel.id);
      const zoomRel=e.etype==="RELATED_TO"&&e.rel&&scale>=TH&&!(F&&!(F.has(e.a.id)&&F.has(e.b.id)));
      if(!inc&&!zoomRel)continue; const lbl=edgeLabel(e); if(!lbl)continue;
      const key=e.a.id<e.b.id?e.a.id+""+e.b.id:e.b.id+""+e.a.id;
      const [x1,y1]=S(e.a.x,e.a.y),[x2,y2]=S(e.b.x,e.b.y);
      let g=lblGroups[key]; if(!g){ g=lblGroups[key]={mx:(x1+x2)/2,my:(y1+y2)/2,inc:false,labels:[]}; }
      if(!g.labels.includes(lbl)) g.labels.push(lbl); if(inc) g.inc=true; }
    for(const key in lblGroups){ const g=lblGroups[key], n=g.labels.length, LH=14;
      for(let i=0;i<n;i++){ const lbl=g.labels[i], yy=g.my+(i-(n-1)/2)*LH, w=ctx.measureText(lbl).width+8;
        ctx.fillStyle="rgba(11,14,19,0.86)"; ctx.fillRect(g.mx-w/2,yy-7,w,13);
        ctx.fillStyle=g.inc?"#d7b6ff":"rgba(176,111,240,0.95)"; ctx.fillText(lbl,g.mx,yy); } }
    ctx.font="11px sans-serif"; ctx.textBaseline="top";
    for(const n of revN){ if(owner[n.label||n.id]!==n.id)continue;
      const show=n===sel||(F&&F.has(n.id))||scale>=TH||(n.indeg||0)>=10; if(!show)continue;
      const [x,y]=S(n.x,n.y), yy=y+n.rad*scale+3; ctx.lineWidth=3; ctx.strokeStyle="rgba(11,14,19,0.9)";
      ctx.strokeText(n.label||n.id,x,yy); ctx.fillStyle=(F&&!F.has(n.id))?"rgba(230,237,243,0.3)":"#e6edf3";
      ctx.fillText(n.label||n.id,x,yy); } }
  function nodeAt(sx,sy){ let best=null,bd=1e9; for(const n of revN){ const [x,y]=S(n.x,n.y), r=Math.max(7,n.rad*scale+4);
    const d=(x-sx)*(x-sx)+(y-sy)*(y-sy); if(d<r*r&&d<bd){bd=d;best=n;} } return best; }

  cv.addEventListener("mousedown",e=>{ const r=cv.getBoundingClientRect(),sx=e.clientX-r.left,sy=e.clientY-r.top;
    moved=0; const n=nodeAt(sx,sy);
    if(n){ drag={n}; }   // nodes are click-to-select only, not draggable — no pin, no reheat
    else { pan={x:e.clientX,y:e.clientY,tx,ty}; } cv.style.cursor="grabbing"; });
  window.addEventListener("mousemove",e=>{ const r=cv.getBoundingClientRect(),sx=e.clientX-r.left,sy=e.clientY-r.top;
    if(drag){ moved+=Math.abs(e.movementX)+Math.abs(e.movementY); }
    else if(pan){ moved+=Math.abs(e.movementX)+Math.abs(e.movementY); autofit=false; tx=pan.tx+(e.clientX-pan.x); ty=pan.ty+(e.clientY-pan.y); draw(); }
    else { const n=nodeAt(sx,sy); cv.style.cursor=n?"pointer":"grab"; if(n!==hover){ hover=n;
      if(n&&tip){ tip.style.display="block"; tip.style.left=(r.left+sx+14+window.scrollX)+"px"; tip.style.top=(r.top+sy+14+window.scrollY)+"px";
        let th="<b>"+esc(n.label||n.id)+"</b><br><span class='mut'>"+n.type+(n.raw?" · raw entry":" · created")+" · "+(n.indeg||0)+" in / deg "+(n.deg||0)+"</span>";
        if(hiSet&&hiSet.has(n.id)){ const role=hiRole[n.id]||"touched",
            txt={seed:"seed (retrieval started here)",cited:"cited in the answer",
                 context:"in the reader's context",touched:"touched, not in the final context"}[role];
          th+="<br><span style='color:"+ROLE_COLOR[role]+"'>"+txt+"</span>"; }
        const snip=(n.meta&&n.meta.snippet)||""; if(snip)th+="<div style='margin-top:5px;white-space:normal;font-size:11px;color:#c9d1d9;line-height:1.4'>"+esc(snip.slice(0,220))+(snip.length>220?"…":"")+"</div>";
        tip.innerHTML=th; }
      else if(tip) tip.style.display="none"; } } });
  window.addEventListener("mouseup",()=>{ if(tip)tip.style.display="none";
    if(drag){ const click=moved<4,n=drag.n; drag=null; cv.style.cursor="grab";
      if(click){ sel=n; if(h.onNode)h.onNode(n); draw(); } }   // click only selects; nodes never move
    else if(pan){ const click=moved<4; pan=null; cv.style.cursor="grab"; if(click){ sel=null; if(h.onBackground)h.onBackground(); draw(); } } });
  cv.addEventListener("wheel",e=>{ e.preventDefault(); autofit=false; const r=cv.getBoundingClientRect(),sx=e.clientX-r.left,sy=e.clientY-r.top;
    const dy=Math.max(-40,Math.min(40,e.deltaY||0)); const [wx,wy]=Wld(sx,sy);
    scale=Math.max(0.15,Math.min(8,scale*Math.exp(-dy*0.0016))); tx=sx-W/2-wx*scale; ty=sy-H/2-wy*scale; draw(); },{passive:false});
  cv.addEventListener("dblclick",()=>{ sel=null; if(h.onBackground)h.onBackground();
    autofit=true; let k=0; (function go(){ easeFit(); draw(); if(++k<26)requestAnimationFrame(go); else autofit=false; })(); });

  return { load, revealStep, draw, recenter(){tx=0;ty=0;scale=1;draw();},
           select(id){sel=byId[id]||null;draw();}, clearSel(){sel=null;draw();},
           count(){return {nodes:revN.length,edges:revE.length};},
           screenOf(id){const n=byId[id];return n?S(n.x,n.y):null;},
           highlight(ids,roleOf){ hiSet=new Set(ids); hiRole=roleOf||{}; draw(); },
           clearHighlight(){ hiSet=null; hiRole={}; draw(); } };
}
"""


# --------------------------------------------------------------------------- #
# Index page
# --------------------------------------------------------------------------- #
_INDEX_TEMPLATE = r"""<!doctype html><html lang="en"><head><meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>kg — test runs</title><style>__STYLE__
#wrap{max-width:1100px;margin:0 auto;padding:26px 22px}
.runs{display:grid;grid-template-columns:repeat(auto-fill,minmax(330px,1fr));gap:14px;margin-top:18px}
.run{padding:15px 16px;display:block;transition:border-color .12s} .run:hover{border-color:#5a6472;text-decoration:none}
.run .grid{display:grid;grid-template-columns:1fr 1fr 1fr;gap:8px;margin-top:12px}
.empty{padding:40px;text-align:center;color:var(--mut)}
</style></head><body><div id="wrap">
<h1>kg · graph test runs</h1>
<div class="mut" id="sub"></div>
<div class="runs" id="runs"></div>
</div><script>
const DATA=/*__DATA__*/;
__GRAPHJS__
const runs=DATA.runs||[];
document.getElementById("sub").textContent=runs.length?`${runs.length} run(s) — newest first`:"";
const host=document.getElementById("runs");
if(!runs.length){host.innerHTML=`<div class="empty card">No runs yet.<br><br>
  <code>python -m kg testrun</code> ingests the temporal dataset and runs the queries,<br>then writes a run here.</div>`;}
runs.forEach(r=>{
  const acc=r.recall_at_k!=null?(r.recall_at_k*100).toFixed(0)+"%":"–";
  const ra=r.response_accuracy!=null?(r.response_accuracy*100).toFixed(0)+"%":"–";
  const tm=r.ingest_seconds!=null?fmtS(r.ingest_seconds)+(r.query_seconds?" + "+fmtS(r.query_seconds):""):"–";
  const a=el("a",{href:"/run?id="+encodeURIComponent(r.run_id),class:"run card"});
  a.innerHTML=`<div class="row" style="justify-content:space-between;align-items:baseline">
    <h1 style="font-size:15px">${esc(r.label||r.run_id)}</h1>
    <span class="pill">${esc((r.backends&&r.backends.extractor)||"")}/${esc((r.backends&&r.backends.agent)||"")}</span></div>
    <div class="mut" style="font-size:11.5px;margin-top:2px">${esc(r.created_at||"")}</div>
    <div class="grid">
      <div class="stat"><div class="k">cost</div><div class="v num">${fmtUSD(r.cost_usd)}</div></div>
      <div class="stat"><div class="k">tokens</div><div class="v num">${fmtN(r.tokens||0)}</div></div>
      <div class="stat"><div class="k">nodes</div><div class="v num">${fmtN(r.nodes||0)}</div></div>
      <div class="stat"><div class="k">recall@k</div><div class="v num">${acc}</div></div>
      <div class="stat"><div class="k">resp acc</div><div class="v num">${ra}</div></div>
      <div class="stat"><div class="k">queries</div><div class="v num">${r.n_queries||0}</div></div>
      <div class="stat"><div class="k">time (ingest + query)</div><div class="v num">${tm}</div></div>
    </div>`;
  host.appendChild(a);
});
</script></body></html>
"""

# --------------------------------------------------------------------------- #
# Run page (Input / Query toggle)
# --------------------------------------------------------------------------- #
_RUN_TEMPLATE = r"""<!doctype html><html lang="en"><head><meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>kg — run</title><style>__STYLE__
#top{display:flex;align-items:center;gap:14px;padding:12px 18px;border-bottom:1px solid var(--line);
  position:sticky;top:0;background:var(--bg);z-index:8;flex-wrap:wrap}
#top .pill{font-size:11px}
.toggle{display:flex;border:1px solid var(--line);border-radius:8px;overflow:hidden}
.toggle button{border:0;border-radius:0;background:#161b22}
.toggle button.on{background:var(--accent);color:#06121f}
#main{padding:14px 18px}
.statbar{display:grid;grid-template-columns:repeat(auto-fit,minmax(120px,1fr));gap:10px;margin-bottom:12px}
.split{display:grid;grid-template-columns:1.55fr 1fr;gap:14px;align-items:stretch}
@media(max-width:980px){.split{grid-template-columns:1fr}}
.panel{padding:12px 14px}
.chartbox{margin-bottom:12px} .chartbox h2{margin-bottom:4px}
.controls{display:flex;align-items:center;gap:10px;margin:10px 0}
.qlayout{display:grid;grid-template-columns:340px 1.4fr;gap:14px;align-items:start}
@media(max-width:980px){.qlayout{grid-template-columns:1fr}}
.kv{display:grid;grid-template-columns:auto 1fr;gap:2px 10px;font-size:12px}
.kv .mut{white-space:nowrap}
.verdict{display:inline-block;padding:1px 7px;border-radius:5px;font-size:11px;font-weight:600}
.v-ok{background:rgba(46,194,126,.15);color:var(--ok)} .v-bad{background:rgba(255,93,143,.15);color:var(--bad)}
.v-mid{background:rgba(245,166,35,.15);color:var(--warn)}
.tracetbl td{font-variant-numeric:tabular-nums;vertical-align:top}
.hh{height:62vh;min-height:420px}
</style></head><body>
<div id="top"></div>
<div id="main">
  <div id="input-view"></div>
  <div id="query-view" style="display:none"></div>
</div>
<div id="tip"></div>
<script>
const DATA=/*__DATA__*/;
__GRAPHJS__
__FORCEJS__
const RUN=DATA.run, ING=RUN.ingest, QRY=RUN.query, PROF=RUN.profile||null,
      tip=document.getElementById("tip");

// ---------- header ----------
(function(){ const t=document.getElementById("top");
  const b=RUN.backends||{};
  t.innerHTML=`<a href="/" title="all runs" style="font-size:18px">&larr;</a>
    <h1>${esc(RUN.label||RUN.run_id)}</h1>
    <span class="pill">extractor: ${esc(b.extractor||"?")}</span>
    <span class="pill">agent: ${esc(b.agent||"?")}</span>
    <span class="pill">${esc(RUN.dataset.input)} · ${RUN.dataset.n_input} episodes</span>
    <span class="pill">${RUN.dataset.n_queries} queries</span>
    <span class="pill num">${fmtUSD(RUN.cost_usd)} · ${fmtN(RUN.tokens)} tok</span>
    <span style="flex:1"></span>
    <div class="toggle"><button id="t-input" class="on">▦ Input</button><button id="t-query">⌕ Query</button></div>`;
  if(!DATA.server){ t.querySelector('a[href="/"]').style.display="none"; }
  document.getElementById("t-input").onclick=()=>swap("input");
  document.getElementById("t-query").onclick=()=>swap("query");
})();
function swap(which){ const iv=which==="input";
  document.getElementById("input-view").style.display=iv?"":"none";
  document.getElementById("query-view").style.display=iv?"none":"";
  document.getElementById("t-input").classList.toggle("on",iv);
  document.getElementById("t-query").classList.toggle("on",!iv);
  if(iv) Input.onShow(); else Query.onShow();
}
function statEl(k,v,sub,tip){ return `<div class="stat"${tip?` data-tip="${esc(tip)}"`:""}><div class="k">${k}</div>
  <div class="v num">${v}${sub?` <small>${sub}</small>`:""}</div></div>`; }
function wireStatTips(host){ host.querySelectorAll(".stat[data-tip]").forEach(s=>{ s.style.cursor="help";
  s.onmouseenter=()=>{ const r=s.getBoundingClientRect(); tip.style.display="block";
    tip.style.left=(r.left+window.scrollX)+"px"; tip.style.top=(r.bottom+window.scrollY+6)+"px";
    tip.innerHTML="<span style='display:inline-block;max-width:250px;white-space:normal;line-height:1.45'>"+esc(s.dataset.tip)+"</span>"; };
  s.onmouseleave=()=>{ tip.style.display="none"; }; }); }

// =================== INPUT VIEW ===================
const Input=(function(){
  const steps=ING.steps, T=ING.totals, G=ING.graph;
  const ingested=steps.filter(s=>s.status==="ingested");
  // forward-fill temporal tag df: pick top-K tags by final df
  const finalDf={}; steps.forEach(s=>(s.tag_df||[]).forEach(d=>{finalDf[d.name]=Math.max(finalDf[d.name]||0,d.df);}));
  const topTags=Object.entries(finalDf).sort((a,b)=>b[1]-a[1]).slice(0,6).map(e=>e[0]);
  const tagSeries=topTags.map((name,idx)=>{ const pts=[]; let cur=0;
    steps.forEach(s=>{ const hit=(s.tag_df||[]).find(d=>d.name===name); if(hit)cur=hit.df; pts.push(cur); });
    return {name,color:["#4f8ef7","#2ec27e","#f5a623","#b06ff0","#ff5d8f","#56d4dd"][idx%6],pts}; });
  // cumulative cost/tokens
  const costPts=[],tokPts=[],nodePts=[],edgePts=[],tagsPerObj=[],vTags=[],vEnt=[],vRel=[];
  let cc=0,ct=0;
  steps.forEach(s=>{ cc+=s.cost_usd||0; ct+=s.tokens||0; costPts.push(+cc.toFixed(6)); tokPts.push(ct);
    nodePts.push(s.nodes); edgePts.push(s.edges); tagsPerObj.push(s.avg_tags_per_object||0);
    vTags.push(s.vocab.tags); vEnt.push(s.vocab.entities); vRel.push(s.vocab.relations); });
  let force=null, idx=steps.length-1, timer=null, charts={};
  const TIP={ documents:"Ingested one at a time. Per-instance mode: each is one LongMemEval instance (its own fresh graph; the structure graph shows a representative one). Shared mode: each is one chat session.",
    nodes:"Total graph nodes: raw entries (green) plus the entities and tags the extractor created (blue).",
    edges:"Total edges: directed TAGGED_AS / MENTIONS / RELATED_TO, plus derived shared-tag / shared-entity / embedding-kNN links.",
    avgtags:"Average canonical tags attached per document — an extraction-density signal.",
    relpair:"Average distinct relationship predicates between a connected entity pair (each RELATED_TO edge carries one).",
    icost:"Total USD spent on extraction LLM calls during ingestion ($0 when offline).",
    itok:"Total input+output tokens spent extracting the graph during ingestion.",
    cache:"Prompt-cache tokens during ingestion, read / write. cache read > 0 means prompt caching is active (the static prefix is reused at ~0.1× input price); both 0 means no caching is in effect.",
    time:"Wall-clock seconds to ingest every document one at a time (includes the per-step re-derive pass)." };
  function build(){
    const host=document.getElementById("input-view");
    host.innerHTML=`
      <div class="statbar">
        ${statEl("documents",T.docs,"ingested",TIP.documents)}
        ${statEl("nodes",fmtN(T.nodes),"","raw+created. "+TIP.nodes)}
        ${statEl("edges",fmtN(T.edges),"",TIP.edges)}
        ${statEl("avg tags/obj",T.avg_tags_per_object,"",TIP.avgtags)}
        ${statEl("rel-tags/pair",T.avg_rel_tags_per_pair,"",TIP.relpair)}
        ${statEl("ingest cost",fmtUSD(T.cost_usd),"",TIP.icost)}
        ${statEl("tokens",fmtN(T.tokens),"",TIP.itok)}
        ${statEl("cache rd/wr",fmtN(T.cache_read||0)+" / "+fmtN(T.cache_write||0),"",TIP.cache)}
        ${statEl("time",T.seconds+"s","",TIP.time)}
      </div>
      <div class="split">
        <div>
          <div class="card stage hh" id="i-stage">
            <div id="i-detail" class="panel" style="display:none;position:absolute;top:10px;right:10px;width:300px;
              max-height:86%;overflow:auto;background:#161b22f2;border:1px solid var(--line);border-radius:10px;z-index:6"></div>
          </div>
          <div class="controls">
            <button id="i-play">▶ Play build</button>
            <button id="i-all">Show all</button>
            <span class="pill" id="i-count"></span>
            <input id="i-scrub" type="range" min="0" max="${steps.length-1}" value="${steps.length-1}" style="flex:1"/>
          </div>
          <div class="legend mut">
            <span><i class="dot" style="background:#2ec27e"></i>raw entry (document / image)</span>
            <span><i class="dot" style="background:#4f8ef7"></i>created (entity / tag)</span>
            <span style="color:#b06ff0">→ relationship (directed, labeled)</span>
            <span class="mut">node size = incoming links · click a node to label its connections · drag empty space to pan · scroll to zoom · dbl-click / tap empty to reset · Space play/pause · A / D step</span>
          </div>
          <div class="card panel" id="i-doc" style="margin-top:12px"></div>
        </div>
        <div>
          <div class="card panel chartbox"><h2>Cumulative cost (USD)</h2><div id="c-cost"></div></div>
          <div class="card panel chartbox"><h2>Cumulative tokens</h2><div id="c-tok"></div></div>
          <div class="card panel chartbox"><h2>Graph size — nodes vs edges</h2><div id="c-graph"></div>
            <div class="legend mut" style="margin-top:4px"><span><i class="dot" style="background:#4f8ef7"></i>nodes</span><span><i class="dot" style="background:#ff5d8f"></i>edges</span></div></div>
          <div class="card panel chartbox"><h2>Vocabulary growth</h2><div id="c-vocab"></div>
            <div class="legend mut" style="margin-top:4px"><span><i class="dot" style="background:#f5a623"></i>tags</span><span><i class="dot" style="background:#b06ff0"></i>entities</span><span><i class="dot" style="background:#2ec27e"></i>relations</span></div></div>
          <div class="card panel chartbox"><h2>Avg tags per object</h2><div id="c-tpo"></div></div>
          <div class="card panel chartbox"><h2>Temporal tags — doc_frequency of top tags over time</h2><div id="c-temporal"></div>
            <div class="legend mut" id="c-temporal-leg" style="margin-top:4px"></div></div>
          <div class="card panel chartbox" id="card-iprof" style="display:none"><h2>⏱ Ingest time by stage</h2><div id="c-iprof"></div>
            <div class="mut" style="font-size:10px;margin-top:4px">ingest.* stages are pipeline wall-clock; extract.* / canon.* are per-call-site
            time (summed across the extraction thread-pool, so they can exceed wall-clock — that is the concurrency working).</div></div>
          <div class="card panel chartbox" id="card-csite" style="display:none"><h2>Cost by call site</h2><div id="c-csite"></div>
            <div class="mut" style="font-size:10px;margin-top:4px">every LLM call site's spend across the whole run (extract = ingestion,
            l3 = canonicalization tie-breaker, rag = production answer calls, judge = eval-only grading).</div></div>
        </div>
      </div>`;
    wireStatTips(host);
    force=makeForce(document.getElementById("i-stage"), tip,
      {onNode:showNodeDetail, onBackground:hideNodeDetail});
    force.load(G.nodes,G.edges); window.__F=force;
    charts.cost=lineChart(document.getElementById("c-cost"),[{color:"#2ec27e",pts:costPts}],{h:110});
    charts.tok=lineChart(document.getElementById("c-tok"),[{color:"#4f8ef7",pts:tokPts}],{h:110});
    charts.graph=lineChart(document.getElementById("c-graph"),[{color:"#4f8ef7",pts:nodePts},{color:"#ff5d8f",pts:edgePts}],{h:110});
    charts.vocab=lineChart(document.getElementById("c-vocab"),[{color:"#f5a623",pts:vTags},{color:"#b06ff0",pts:vEnt},{color:"#2ec27e",pts:vRel}],{h:110});
    charts.tpo=lineChart(document.getElementById("c-tpo"),[{color:"#f5a623",pts:tagsPerObj}],{h:100,maxFloor:1});
    charts.temporal=lineChart(document.getElementById("c-temporal"),tagSeries.length?tagSeries:[{color:"#888",pts:[0]}],{h:130});
    document.getElementById("c-temporal-leg").innerHTML=tagSeries.map(s=>
      `<span><i class="dot" style="background:${s.color}"></i>${esc(s.name)}</span>`).join("")||'<span class="mut">no tag activity</span>';
    if(PROF){
      const it=profItems(PROF.ingest);
      if(it.length){ document.getElementById("card-iprof").style.display="";
        barChart(document.getElementById("c-iprof"),it,{fmt:fmtS}); }
      const cs=Object.entries(PROF.cost_by_site||{}).map(([k,v])=>
        ({k:k+" ×"+v.llm_calls+" ("+fmtN(v.tokens)+" tok)",v:v.cost_usd,c:k==="judge"?"#8b949e":"#2ec27e"}))
        .filter(d=>d.v>0).sort((a,b)=>b.v-a.v);
      if(cs.length){ document.getElementById("card-csite").style.display="";
        barChart(document.getElementById("c-csite"),cs,{fmt:fmtUSD}); }
    }
    document.getElementById("i-scrub").oninput=e=>{stop();setIdx(+e.target.value);};
    document.getElementById("i-all").onclick=()=>{stop();setIdx(steps.length-1);};
    document.getElementById("i-play").onclick=play;
    document.addEventListener("keydown",onKey);
    setIdx(steps.length-1);
  }
  function onKey(e){
    if(document.getElementById("query-view").style.display!=="none")return;   // Input only
    const tag=(e.target.tagName||""); if(tag==="INPUT"||tag==="TEXTAREA")return;
    if(e.code==="Space"){ e.preventDefault(); play(); }
    else if(e.key==="a"||e.key==="A"){ stop(); setIdx(idx-1); }
    else if(e.key==="d"||e.key==="D"){ stop(); setIdx(idx+1); }
  }
  function setIdx(i){ idx=Math.max(0,Math.min(steps.length-1,i));
    document.getElementById("i-scrub").value=idx;
    force.revealStep(idx);
    const shown=G.nodes.reduce((a,n)=>a+(n.appear<=idx?1:0),0);
    document.getElementById("i-count").textContent=`step ${idx+1} / ${steps.length} · ${shown} nodes shown`;
    for(const k in charts) chartCursor(charts[k], idx);
    renderDoc(steps[idx]);
  }
  function showNodeDetail(n){ const d=document.getElementById("i-detail"); d.style.display="";
    const m=n.meta||{}; let h=`<div class="row" style="justify-content:space-between;align-items:baseline">
      <h2 style="margin:0">${esc(n.label||n.id)}</h2>
      <span class="pill" style="background:${n.raw?'rgba(46,194,126,.15)':'rgba(79,142,247,.15)'}">${n.raw?'raw entry':'created'} · ${esc(n.type)}</span></div>`;
    if(n.type==="episode"){
      h+=`<div class="mut" style="font-size:11px;margin-top:2px">${esc(m.modality||"")}${m.created_at?(" · "+esc(m.created_at)):""}</div>`;
      if((m.tags||[]).length) h+=`<div class="mut" style="font-size:11px;margin-top:8px">tags (${m.n_tags})</div><div>${m.tags.map(t=>`<span class="tag">${esc(t)}</span>`).join("")}</div>`;
      if((m.entities||[]).length) h+=`<div class="mut" style="font-size:11px;margin-top:6px">entities</div><div>${m.entities.map(t=>`<span class="tag" style="border-color:#4f8ef7">${esc(t)}</span>`).join("")}</div>`;
      if(m.snippet) h+=`<div class="mut" style="margin-top:8px;font-size:12px;line-height:1.5;white-space:pre-wrap">${esc(m.snippet)}</div>`;
    } else {
      h+=`<div class="kv" style="margin-top:8px"><span class="mut">used by</span><span>${m.df||0} document(s)</span>`;
      if(m.entity_type) h+=`<span class="mut">entity type</span><span>${esc(m.entity_type)}</span>`;
      if((m.aliases||[]).length) h+=`<span class="mut">aliases</span><span>${m.aliases.map(esc).join(", ")}</span>`;
      h+=`</div>`;
    }
    h+=`<div class="mut" style="margin-top:10px;font-size:11px">${n.indeg||0} incoming · degree ${n.deg||0} · click empty space to close</div>`;
    d.innerHTML=h;
  }
  function hideNodeDetail(){ document.getElementById("i-detail").style.display="none"; }
  function renderDoc(s){ const f=s.footprint||{tags:[],entities:[],rel_tags:[]};
    document.getElementById("i-doc").innerHTML=`
      <div class="row" style="justify-content:space-between;align-items:baseline">
        <h2 style="margin:0">doc ${s.i+1} · ${esc(s.modality)}</h2>
        <span class="pill">${esc(s.created_at||"")}</span></div>
      <div style="font-weight:600;margin:6px 0 8px">${esc(s.title)}</div>
      <div class="statbar" style="grid-template-columns:repeat(5,1fr);margin-bottom:8px">
        ${statEl("nodes +",signed(s.node_delta))}
        ${statEl("tokens",fmtN(s.tokens||0))}
        ${statEl("cost",fmtUSD(s.cost_usd||0))}
        ${statEl("llm calls",s.llm_calls||0)}
        ${statEl("time",fmtS(s.seconds||0))}
      </div>
      ${f.tags.length?`<div class="mut" style="font-size:11px">tags</div><div>${f.tags.map(t=>`<span class="tag">${esc(t)}</span>`).join("")}</div>`:""}
      ${f.entities.length?`<div class="mut" style="font-size:11px;margin-top:6px">entities</div><div>${f.entities.map(t=>`<span class="tag" style="border-color:#b06ff0">${esc(t)}</span>`).join("")}</div>`:""}
      ${f.rel_tags.length?`<div class="mut" style="font-size:11px;margin-top:6px">relations</div><div>${f.rel_tags.map(t=>`<span class="tag" style="border-color:#56d4dd">${esc(t)}</span>`).join("")}</div>`:""}
      ${s.profile?`<div class="mut" style="font-size:11px;margin-top:8px">⏱ this step, by stage</div><div id="i-doc-prof"></div>`:""}`;
    if(s.profile) barChart(document.getElementById("i-doc-prof"), profItems(s.profile), {fmt:fmtS});
  }
  function signed(d){ let tot=0; for(const k in (d||{}))tot+=d[k]; return (tot>=0?"+":"")+tot; }
  function play(){ if(timer){stop();return;} document.getElementById("i-play").textContent="⏸ Pause";
    if(idx>=steps.length-1)idx=0;
    timer=setInterval(()=>{ if(idx>=steps.length-1){stop();return;} setIdx(idx+1); },80); }
  function stop(){ if(timer)clearInterval(timer); timer=null; document.getElementById("i-play").textContent="▶ Play build"; }
  let built=false;
  return {onShow(){ if(!built){build();built=true;} }};
})();
Input.onShow();

// =================== QUERY VIEW ===================
const Query=(function(){
  const qs=QRY.queries, T=QRY.totals; let built=false, graph=null, sel=null;
  function avgLat(){ const xs=qs.map(q=>q.seconds).filter(v=>v!=null);
    return xs.length?xs.reduce((a,b)=>a+b,0)/xs.length:null; }
  function build(){
    const host=document.getElementById("query-view");
    const kinds=Object.entries(T.by_kind||{}).map(([k,v])=>({k,v:v.recall_at_k,c:"#4f8ef7"}));
    host.innerHTML=`
      <div class="statbar">
        ${statEl("queries",T.n,"","Eval questions answered via the PPR→RAG ask() path (retrieve-then-read): the non-LLM retriever builds the context, one LLM call answers. No agentic per-hop traversal.")}
        ${statEl("recall@k",pct(T.recall_at_k),"","Mean recall@k — fraction of each question's gold evidence sessions retrieved in the top-k episodes.")}
        ${statEl("MRR",(T.mrr||0).toFixed(3),"","Mean reciprocal rank of the first gold evidence session across questions.")}
        ${statEl("precision@k",T.precision_at_k!=null?pct(T.precision_at_k):"–","gold density","Fraction of the top-k retrieved episodes that are gold evidence. NB the ceiling is n_gold/k — LongMemEval questions carry ~2 gold sessions, so ~0.25 at k=8 is perfect, not poor.")}
        ${statEl("hit rate",pct(T.hit_rate),"","Fraction of questions where at least one gold evidence session was retrieved in the top-k.")}
        ${statEl("cite grounding",pct(T.citation_grounding),"","Mean fraction of gold evidence sessions the agent actually cited (citations ∩ gold).")}
        ${statEl("resp acc",T.response_accuracy!=null?pct(T.response_accuracy):"–","judge","Mean LLM-judge score of the answer text vs the reference answer (live runs only; – when offline).")}
        ${statEl("avg steps",T.avg_steps,"","Average number of tool-call rounds the agent took per question.")}
        ${statEl("avg latency",avgLat()!=null?fmtS(avgLat()):"–","/query","Mean wall-clock seconds per production ask() call (retrieve + context + one LLM answer; excludes the eval-only judge).")}
        ${statEl("query cost (prod)",fmtUSD(T.agent_cost_usd ?? T.cost_usd),"","Production USD: the single PPR→RAG answer call per query, summed across all queries. Excludes the eval-only judge — this is what production actually pays.")}
        ${statEl("eval judge",fmtUSD(T.judge_cost_usd ?? 0),"eval-only","Eval-only USD: the LLM grader that certifies answer correctness during testing. Runs every test round but is NOT paid in production.")}
      </div>
      <div class="qlayout">
        <div>
          <div class="card panel chartbox"><h2>recall@k by question kind</h2><div id="q-kinds"></div></div>
          <div class="card panel chartbox" id="card-qprof" style="display:none"><h2>⏱ Query time by stage</h2><div id="q-prof"></div>
            <div class="mut" style="font-size:10px;margin-top:4px">totals across all queries. judge.llm is eval-only
            (not paid in production); everything else is the live ask() path.</div></div>
          <div class="card scroll" style="max-height:62vh">
            <table><thead><tr><th>id</th><th>kind</th><th title="recall@k — fraction of gold evidence retrieved in the top-k">rec</th><th title="answer correct? green ●=judge correct, red ●=judge incorrect, ◑=gold retrieved (unjudged), ○=miss">ok</th><th class="num">$</th></tr></thead>
            <tbody id="q-list"></tbody></table>
          </div>
        </div>
        <div>
          <div class="row" style="justify-content:space-between;align-items:center;margin-bottom:8px">
            <span class="mut" style="font-size:11px">query graph</span>
            <div class="toggle"><button id="q-m-simple" class="on">Simplified</button><button id="q-m-full">Full graph</button></div>
          </div>
          <div class="card stage hh" id="q-stage" style="position:relative">
            <div id="q-stage-simple" style="position:absolute;inset:0"></div>
            <div id="q-stage-full" style="position:absolute;inset:0;display:none"></div>
          </div>
          <div class="legend mut">
            <span><i class="dot" style="background:#4f8ef7"></i>episode</span>
            <span><i class="dot" style="background:#b06ff0"></i>entity</span>
            <span><i class="dot" style="background:#f5a623"></i>tag</span>
            <span><i class="dot" style="background:transparent;border:2px solid #ffd24d"></i>ring = seed (retrieval started here)</span>
            <span><i class="dot" style="background:transparent;border:2px solid #2ec27e"></i>ring = cited in the answer</span>
            <span><i class="dot" style="background:transparent;border:2px solid #ff5d8f"></i>ring = in the reader's context (number = rank in Simplified)</span>
            <span><i class="dot" style="background:transparent;border:2px dashed #8b949e"></i>ring = retrieved, cut before the reader</span>
            <span class="mut">hover a node for its text · click to highlight connections · drag to pan · scroll/pinch to zoom in for labels · dbl-click to reset</span>
            <span class="mut">Full graph = this query laid over the whole ingest graph (like the Input tab), untouched nodes faded · nodes aren't draggable</span>
          </div>
          <div class="card panel" id="q-detail" style="margin-top:12px"><span class="mut">Select a query to replay its traversal.</span></div>
        </div>
      </div>`;
    wireStatTips(host);
    barChart(document.getElementById("q-kinds"), kinds, {fmt:pct});
    if(PROF){ const qp=profItems(PROF.query);
      if(qp.length){ document.getElementById("card-qprof").style.display="";
        barChart(document.getElementById("q-prof"),qp,{fmt:fmtS}); } }
    const tb=document.getElementById("q-list");
    qs.forEach((q,i)=>{ const tr=el("tr",{class:"qrow",id:"qr-"+i});
      // "ok" dot = did the ANSWER pass. Prefer the LLM judge's correctness verdict (green
      // correct / red incorrect); with no judge, fall back to a half-dot for a retrieval hit
      // (gold found but answer unjudged) vs ○ for a miss — never green, since a retrieval hit
      // alone doesn't mean the answer was right (the bug this fixes: a wrong answer over
      // correctly-retrieved evidence was showing green).
      const j=q.judge, okDot=(j&&!j.error)
        ? (j.correct ? '<span style="color:var(--ok)" title="answer judged correct">●</span>'
                     : '<span style="color:var(--bad)" title="answer judged incorrect">●</span>')
        : (q.hit ? '<span style="color:var(--ok)" title="gold retrieved · answer unjudged">◑</span>'
                 : '<span class="mut" title="gold missed">○</span>');
      tr.innerHTML=`<td>${esc(q.id||("q"+(i+1)))}</td><td class="mut">${esc(q.kind)}</td>
        <td class="num">${(q.recall_at_k*100).toFixed(0)}</td>
        <td>${okDot}</td>
        <td class="num mut">${(q.cost_usd||0)?fmtUSD(q.cost_usd):"–"}</td>`;
      tr.onclick=()=>select(i); tb.appendChild(tr); });
    graph=makeGraph(document.getElementById("q-stage-simple"), tip);
    document.getElementById("q-m-simple").onclick=()=>setMode("simple");
    document.getElementById("q-m-full").onclick=()=>setMode("full");
    if(qs.length)select(0);
  }
  let mode="simple", force=null, forceKey=null;
  function setMode(m){ mode=m;
    document.getElementById("q-m-simple").classList.toggle("on",m==="simple");
    document.getElementById("q-m-full").classList.toggle("on",m==="full");
    document.getElementById("q-stage-simple").style.display=m==="simple"?"":"none";
    document.getElementById("q-stage-full").style.display=m==="full"?"":"none";
    if(m==="full"&&sel!=null) drawFull(qs[sel]);
  }
  function drawFull(q){
    if(!force) force=makeForce(document.getElementById("q-stage-full"), tip);
    // per-instance eval: each question ingested its own fresh graph (q.graph), torn down
    // right after — there's no single persistent graph to reuse, unlike shared-graph mode
    // where every question ran against ING.graph and we only ever load it once.
    const g=q.graph||ING.graph, key=q.graph?("q:"+(q.id||"")):"shared";
    if(key!==forceKey){ force.load(g.nodes||[], g.edges||[]); forceKey=key; }
    const roleOf={};
    (q.touched||[]).forEach(id=>roleOf[id]="touched");
    (q.seeds||[]).forEach(id=>roleOf[id]="seed");
    (q.context_episodes||[]).forEach(id=>roleOf[id]="context");
    (q.citations||[]).forEach(id=>roleOf[id]="cited");
    force.highlight(Object.keys(roleOf), roleOf);
  }
  function select(i){ sel=i; const q=qs[i];
    document.querySelectorAll(".qrow").forEach(r=>r.classList.remove("sel"));
    const r=document.getElementById("qr-"+i); if(r)r.classList.add("sel");
    drawTrace(q); drawDetail(q);
    if(mode==="full") drawFull(q);
  }
  function drawTrace(q){ const sg=q.subgraph||{nodes:[],edges:[],hops:[]};
    graph.reset(); graph.render(sg.nodes,sg.edges,{labels:true,zoomLabels:true});
    if(sg.hops&&sg.hops.length){ graph.dimAll(); let h=0; const shown=new Set();
      (function step(){ if(h>=sg.hops.length)return; sg.hops[h].forEach(id=>shown.add(id));
        graph.undim(shown); h++; setTimeout(step,420); })(); }
  }
  function drawDetail(q){ const j=q.judge;
    let verdict="";
    if(j&&!j.error){ const cls=j.correct?"v-ok":(j.score>=0.5?"v-mid":"v-bad");
      verdict=`<span class="verdict ${cls}">judge: ${j.correct?"correct":"incorrect"} (${j.score})</span>`; }
    const cites=(q.citations||[]).map(c=>`<span class="tag">${esc(c)}</span>`).join("")||'<span class="mut">none</span>';
    const marks=q.gold_marks||(q.gold||[]).map(id=>({id,hit:(q.object_ids||[]).indexOf(id)>=0}));
    const gold=marks.map(m=>`<span class="tag" style="border-color:${m.hit?'#2ec27e':'#ff5d8f'}">${esc(m.id)}</span>`).join("");
    const trace=(q.trace||[]).map(s=>`<tr><td class="mut">${s.step+1}</td><td>${esc(s.tool)}</td>
      <td class="mut">${esc(JSON.stringify(s.input).slice(0,80))}</td><td>${esc(s.result_summary||"")}</td></tr>`).join("");
    document.getElementById("q-detail").innerHTML=`
      <div class="row wrap" style="justify-content:space-between;align-items:baseline">
        <h2 style="margin:0">${esc(q.id||"")} · ${esc(q.kind)} <span class="mut">(${esc(q.difficulty||"")})</span></h2>
        <span class="pill">${q.steps} steps · ${q.stopped} · ${fmtN(q.tokens||0)} tok · ${fmtUSD(q.cost_usd||0)}${q.seconds!=null?` · ${fmtS(q.seconds)}`:""}</span>
      </div>
      <div style="font-weight:600;margin:6px 0">${esc(q.query)}</div>
      <div class="statbar" style="grid-template-columns:repeat(4,1fr);margin:8px 0">
        ${statEl("recall@k",pct(q.recall_at_k))}
        ${statEl("MRR",(q.mrr||0).toFixed(2))}
        ${statEl("grounding",pct(q.citation_grounding))}
        ${statEl("touched",q.n_touched)}
      </div>
      ${verdict?`<div style="margin:4px 0 8px">${verdict}${j&&j.reason?` <span class="mut">${esc(j.reason)}</span>`:""}</div>`:""}
      <div class="kv">
        <span class="mut">answer</span><span>${esc(q.answer||"")}</span>
        ${q.answer_expected?`<span class="mut">expected</span><span class="mut">${esc(q.answer_expected)}</span>`:""}
      </div>
      <div style="margin-top:8px"><span class="mut" style="font-size:11px">gold (green=retrieved)</span><br>${gold}</div>
      <div style="margin-top:6px"><span class="mut" style="font-size:11px">citations</span><br>${cites}</div>
      ${(q.context_episodes||[]).length?`<div style="margin-top:6px"><span class="mut" style="font-size:11px">context — what the reader actually saw (${q.context_episodes.length} episodes${(q.facts||[]).length?`, ${q.facts.length} facts`:""}${q.as_of?`, as-of ${esc(q.as_of.slice(0,10))}`:""})</span><br>${q.context_episodes.map(c=>`<span class="tag" style="border-color:${(q.gold||[]).some(g=>g.includes(c.replace(/^ep_/,'').split('#')[0]))?'#2ec27e':'var(--line)'}">${esc(c)}</span>`).join("")}</div>`:""}
      ${q.profile?`<div style="margin-top:10px"><span class="mut" style="font-size:11px">⏱ this query, by stage (judge.llm is eval-only)</span><div id="qd-prof"></div></div>`:""}
      <div style="margin-top:10px"><span class="mut" style="font-size:11px">tool-call trace</span>
        <div class="scroll" style="max-height:200px;margin-top:4px"><table class="tracetbl">
        <thead><tr><th>#</th><th>tool</th><th>input</th><th>result</th></tr></thead><tbody>${trace}</tbody></table></div></div>`;
    if(q.profile) barChart(document.getElementById("qd-prof"), profItems(q.profile), {fmt:fmtS});
  }
  function pct(v){ return v==null?"–":(v*100).toFixed(0)+"%"; }
  return {onShow(){ if(!built){build();built=true;} }};
})();
</script></body></html>
"""

_RUN_TEMPLATE = (_RUN_TEMPLATE.replace("__STYLE__", _STYLE)
                 .replace("__GRAPHJS__", _GRAPHJS).replace("__FORCEJS__", _FORCEJS))
_INDEX_TEMPLATE = _INDEX_TEMPLATE.replace("__STYLE__", _STYLE).replace("__GRAPHJS__", _GRAPHJS)
