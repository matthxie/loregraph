"""CSRGraph (kg/csr.py) must be a drop-in for the nx.Graph projection it replaced.

Three contracts:
  1. The PPR operator built from CSR arrays is bit-identical to the one built via
     nx.to_scipy_sparse_array from the same pairs — so exact-PPR scores (and therefore
     every banked ranking) are unchanged.
  2. The push backend on CSR matches the push backend on nx (same fixed point, eps).
  3. The nx-compatible surface (len/in/iter, neighbors, degree, edges, nodes) agrees
     with a real nx.Graph built from the same pairs.

Run: python -m pytest tests/test_csr_parity.py -q
"""
from __future__ import annotations

import networkx as nx
import numpy as np

from kg.csr import CSRGraph
from kg.retrieval import local_push_ppr, personalized_pagerank


def _pairs(n=60, seed=11):
    rng = np.random.default_rng(seed)
    ids = sorted(f"n{i:03d}" for i in range(n))
    pairs = {}
    for _ in range(n * 3):
        a, b = rng.integers(0, n, 2)
        if a == b:
            continue
        u, v = sorted((ids[a], ids[b]))
        pairs[(u, v)] = float(rng.random()) + 0.05
    ids.append("dangling")            # isolated row: exercises the dangling path
    return ids, pairs


def _both(n=60, seed=11):
    ids, pairs = _pairs(n, seed)
    G = nx.Graph()
    G.add_nodes_from(ids)
    for (u, v), w in pairs.items():
        G.add_edge(u, v, weight=w)
    return CSRGraph.from_pairs(ids, pairs), G


def test_surface_matches_networkx():
    C, G = _both()
    assert len(C) == len(G)
    assert set(C) == set(G)
    assert set(C.nodes()) == set(G.nodes())
    assert C.number_of_nodes() == G.number_of_nodes()
    assert C.number_of_edges() == G.number_of_edges()
    assert ("n000" in C) == ("n000" in G)
    assert "absent" not in C
    for nid in list(C)[:20]:
        assert sorted(C.neighbors(nid)) == sorted(G.neighbors(nid))
        assert C.degree(nid) == G.degree(nid)
        assert C[nid].keys() == G.adj[nid].keys()
        for v in C[nid]:
            assert abs(C[nid][v]["weight"] - G.adj[nid][v]["weight"]) < 1e-15
    assert set(C.edges()) == set(G.edges())
    ew_c = {tuple(sorted((u, v))): d["weight"] for u, v, d in C.edges(data=True)}
    ew_g = {tuple(sorted((u, v))): d["weight"] for u, v, d in G.edges(data=True)}
    assert ew_c == ew_g


def test_exact_ppr_identical_on_csr_and_nx():
    """Contract 1: same scores to float equality — the byte-identical-rankings claim."""
    C, G = _both()
    rng = np.random.default_rng(3)
    pers = {f"n{i:03d}": float(rng.random()) + 0.01 for i in range(0, 60, 4)}
    on_csr = personalized_pagerank(C, alpha=0.5, personalization=pers, max_iter=200)
    on_nx = personalized_pagerank(G, alpha=0.5, personalization=pers, max_iter=200)
    assert set(on_csr) == set(on_nx)
    assert all(on_csr[n] == on_nx[n] for n in on_nx), \
        max(abs(on_csr[n] - on_nx[n]) for n in on_nx)
    # and both still agree with the reference implementation
    ref = nx.pagerank(G, alpha=0.5, personalization=pers, weight="weight", max_iter=200)
    assert all(abs(on_csr[n] - ref[n]) < 1e-10 for n in ref)


def test_push_ppr_matches_across_representations():
    """Push is eps-approximate, and CSR iterates neighbors in index order while nx
    iterates in insertion order — so the two runs make float-level different score
    estimates. The contract is that BOTH sit within the push guarantee of the exact
    solution and produce the same ranking, not that they agree bit-for-bit."""
    C, G = _both()
    pers = {"n000": 2.0, "n007": 1.0, "n042": 0.5, "dangling": 0.25}
    on_csr = local_push_ppr(C, alpha=0.5, personalization=pers, eps=1e-9)
    on_nx = local_push_ppr(G, alpha=0.5, personalization=pers, eps=1e-9)
    exact = personalized_pagerank(G, alpha=0.5, personalization=pers)
    assert set(on_csr) == set(on_nx)
    assert all(abs(on_csr[n] - exact[n]) < 1e-5 for n in on_csr)
    assert all(abs(on_nx[n] - exact[n]) < 1e-5 for n in on_nx)
    # ranking parity at retrieval resolution
    top_c = sorted(on_csr, key=lambda x: (-on_csr[x], x))[:20]
    top_n = sorted(on_nx, key=lambda x: (-on_nx[x], x))[:20]
    assert top_c == top_n


def test_empty_and_edgeless():
    E = CSRGraph.from_pairs([], {})
    assert len(E) == 0 and E.number_of_edges() == 0 and list(E.edges()) == []
    assert personalized_pagerank(E, alpha=0.5, personalization={"x": 1.0}) == {}
    assert local_push_ppr(E, alpha=0.5, personalization={"x": 1.0}) == {}
    lone = CSRGraph.from_pairs(["a", "b"], {})
    assert "a" in lone and lone.degree("a") == 0 and list(lone.neighbors("a")) == []
    ppr = personalized_pagerank(lone, alpha=0.5, personalization={"a": 1.0})
    assert set(ppr) == {"a", "b"}


def test_to_networkx_roundtrip():
    C, G = _both()
    R = C.to_networkx()
    assert set(R.nodes()) == set(G.nodes())
    assert set(R.edges()) == set(G.edges())
    for u, v, d in R.edges(data=True):
        assert abs(d["weight"] - G[u][v]["weight"]) < 1e-15
