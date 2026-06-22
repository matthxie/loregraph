"""Tiny stdlib HTTP server for the live graph viewer (no Flask, no deps).

    python -m kg serve --store store/kg.db --port 8000

Serves the viewer at /, and answers /api/query?q=...&mode=bfs with a traversal
trace so you can type queries and watch the path the retriever takes.
"""
from __future__ import annotations

import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

from .graph import KnowledgeGraph
from .viz import graph_payload, query_trace, render_html


def make_handler(g: KnowledgeGraph):
    graph = graph_payload(g.store)
    page = render_html(graph, trace=None, server=True).encode("utf-8")

    class Handler(BaseHTTPRequestHandler):
        def _send(self, code, body, ctype):
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            u = urlparse(self.path)
            if u.path in ("/", "/index.html"):
                self._send(200, page, "text/html; charset=utf-8")
            elif u.path == "/api/graph":
                self._send(200, json.dumps(graph).encode(), "application/json")
            elif u.path == "/api/query":
                qs = parse_qs(u.query)
                q = (qs.get("q", [""])[0]).strip()
                mode = qs.get("mode", ["bfs"])[0]
                try:
                    trace = query_trace(g, q, mode=mode) if q else {"note": "empty query"}
                except Exception as e:  # noqa: BLE001 — report, don't 500 silently
                    trace = {"note": f"query failed: {e!r}", "nodes": [], "edges": [],
                             "ranked": [], "seeds": [], "hops": []}
                self._send(200, json.dumps(trace, ensure_ascii=False).encode(),
                           "application/json")
            else:
                self._send(404, b"not found", "text/plain")

        def log_message(self, *a):  # quiet
            pass

    return Handler


def serve(store_path: str, port: int = 8000, config=None) -> None:
    g = KnowledgeGraph.open(store_path, config)
    n = g.stats()["by_node_type"].get("object", 0)
    if n == 0:
        print(f"warning: store {store_path!r} has no objects — run `python -m kg ingest` first")
    httpd = ThreadingHTTPServer(("127.0.0.1", port), make_handler(g))
    print(f"kg viewer on  http://127.0.0.1:{port}   (Ctrl-C to stop)")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped.")
