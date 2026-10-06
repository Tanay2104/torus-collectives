import io
from collections import defaultdict, deque
from contextlib import redirect_stdout

from msccl.language import MSCCLProgram, Buffer, chunk, XML, Check
from msccl.language.collectives import AllGather
from msccl.topologies import Topology


def topology_from_trees(Ts, n, bw=1):
    links = [[0] * n for _ in range(n)]
    for ps in Ts.values():
        for p in ps:
            a, b = p[0][0], p[0][1]
            links[a][b] = links[b][a] = bw
    return Topology("tree_union", links)


def _bfs_edges(root, edges, heavy_first=True):
    """BFS edge order. heavy_first: visit the deepest child first"""
    kids = defaultdict(list)
    for a, b in edges:
        kids[a].append(b)

    if heavy_first:
        height = {}

        def h(u):
            height[u] = 1 + max((h(c) for c in kids.get(u, [])), default=0)
            return height[u]

        h(root)
        for cs in kids.values():
            cs.sort(key=lambda c: -height[c])  # stable: ties keep their existing order
    out, dq = [], deque([root])
    while dq:
        u = dq.popleft()
        for c in kids[u]:
            out.append((u, c))
            dq.append(c)
    return out


def allgather_trees(
    Ts,
    k,
    n,
    P,
    instances=1,
    proto="Simple",
    topology=None,
    name=None,
    heavy_first=True,
    chan_of=None,
):
    per_root = defaultdict(list)
    for u, i in sorted(Ts):
        per_root[u].append(i)
    assert len(per_root) == n and all(v == list(range(k)) for v in per_root.values()), (
        "need trees (u,0..k-1) per root"
    )

    trees = {}
    for key, ps in sorted(Ts.items()):
        assert all(p[0][2] is None for p in ps), (
            "per-edge channel hints are not supported"
        )
        trees[key] = _bfs_edges(key[0], [(p[0][0], p[0][1]) for p in ps], heavy_first)
    tree_order = list(trees)

    q = k * P  # chunks per shard
    name = name or f"allgather_trees_k{k}_P{P}_inst{instances}"
    with MSCCLProgram(
        name,
        topology or topology_from_trees(Ts, n),
        AllGather(n, q, True),
        instances,
        protocol=proto,
    ):
        for s in range(P):  # pipeline
            for u, i in tree_order:
                bfs = trees[u, i]
                idx = (
                    u * q + i * P + s
                )  # in-place: own shard already sits in the output buffer
                ref = {u: chunk(u, Buffer.output, idx)}
                for a, b in bfs:  # BFS: parent already holds the chunk
                    ref[b] = ref[a].copy(
                        b, Buffer.output, idx, ch=(chan_of(u, i, s) if chan_of else -1)
                    )  # ch=-1: compiler assigns/reuses
        buf = io.StringIO()
        with redirect_stdout(buf):
            XML()
        ok = Check()
    return buf.getvalue(), ok
