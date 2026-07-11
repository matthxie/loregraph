"""local_push_ppr (config.ppr_backend="push") must match the exact global solver.

Same fixed point — x = (1-alpha)·s + alpha·x·W with dangling mass teleporting to the
personalization — approximated to eps·weighted-degree per node. At retrieval resolution
that means: same node ranking, scores within a small absolute tolerance, and it must
never explore work proportional to graph size (locality is the whole point).
"""
from __future__ import annotations

import networkx as nx

from kg.retrieval import local_push_ppr, personalized_pagerank


def _weighted_graph(n=120, seed=7) -> nx.Graph:
    import random
    rng = random.Random(seed)
    G = nx.Graph()
    G.add_nodes_from(f"n{i}" for i in range(n))
    for i in range(n):
        for _ in range(3):
            j = rng.randrange(n)
            if j != i:
                G.add_edge(f"n{i}", f"n{j}", weight=rng.uniform(0.1, 2.0))
    G.add_node("dangling-a")          # isolated: exercises the teleport-to-seeds path
    G.add_node("dangling-b")
    return G


def test_push_matches_global_ranking_and_scores():
    G = _weighted_graph()
    pers = {"n0": 2.0, "n7": 1.0, "n42": 0.5, "dangling-a": 0.25}
    alpha = 0.5
    exact = personalized_pagerank(G, alpha=alpha, personalization=pers)
    approx = local_push_ppr(G, alpha=alpha, personalization=pers, eps=1e-9)
    # every node the push assigns mass must agree with the exact score
    assert approx, "push returned nothing"
    for nid, sc in approx.items():
        assert abs(sc - exact[nid]) < 1e-5, (nid, sc, exact[nid])
    # the exact top-20 ranking is reproduced
    top_exact = sorted(exact, key=lambda x: (-exact[x], x))[:20]
    top_push = sorted(approx, key=lambda x: (-approx.get(x, 0.0), x))[:20]
    assert top_exact == top_push


def test_push_is_local():
    """On a graph with a far-away component, push must not touch it at all."""
    G = _weighted_graph()
    # a second component the seeds cannot reach
    for i in range(50):
        G.add_edge(f"far{i}", f"far{i+1}", weight=1.0)
    out = local_push_ppr(G, alpha=0.5, personalization={"n0": 1.0}, eps=1e-8)
    assert out
    assert not any(nid.startswith("far") for nid in out)


def test_push_deterministic_and_handles_empty():
    G = _weighted_graph()
    pers = {"n3": 1.0, "n9": 3.0}
    a = local_push_ppr(G, alpha=0.5, personalization=pers, eps=1e-8)
    b = local_push_ppr(G, alpha=0.5, personalization=pers, eps=1e-8)
    assert a == b
    assert local_push_ppr(nx.Graph(), alpha=0.5, personalization={"x": 1.0}) == {}
    assert local_push_ppr(G, alpha=0.5, personalization={"absent": 1.0}) == {}
