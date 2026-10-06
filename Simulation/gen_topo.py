#!/usr/bin/env python3
import argparse
from math import prod


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dims", type=int, nargs="+", required=True)
    p.add_argument("--bandwidths", type=str, nargs="+", required=True)
    p.add_argument("--latencies", type=str, nargs="+")
    p.add_argument("--output", default="topo.txt")
    args = p.parse_args()

    dims = args.dims
    if len(dims) != len(args.bandwidths):
        p.error("--dims and --bandwidths must have same length")
    latencies = args.latencies or ["100"] * len(dims)
    if len(latencies) != len(dims):
        p.error("--latencies must have same length as --dims")

    total_nodes = prod(dims)
    strides = []
    s = 1
    for n in reversed(dims):
        strides.insert(0, s)
        s *= n

    links = []
    for i, (n, bw, lat) in enumerate(zip(dims, args.bandwidths, latencies)):
        if n == 1:
            continue
        stride = strides[i]
        for node in range(total_nodes):
            coord = (node // stride) % n
            if coord < n - 1:
                neighbor = node + stride
            else:
                neighbor = node - (n - 1) * stride
            links.append(f"{node} {neighbor} {bw}GBps {lat}ns 0")

    with open(args.output, "w") as f:
        f.write(f"{total_nodes} 0 {len(links)}\n")
        f.write("\n".join(links) + "\n")


if __name__ == "__main__":
    main()
