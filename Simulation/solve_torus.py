import argparse
from torus.torus import solve_torus, torus_topology, to_msccl_xml
from ForestColl.pipeline_allgather import optimal_pipeline_spanning_trees


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dims", type=int, nargs="+", required=True)
    p.add_argument(
        "--bandwidths",
        type=float,
        nargs="+",
        required=True,
        help="Per-dimension bandwidth in GBps",
    )
    p.add_argument("--prefix", default="torus")
    p.add_argument(
        "--lowerer",
        choices=("forestcoll", "msccldsl"),
        default="forestcoll",
        help="XML lowering used for both generated schedules",
    )
    args = p.parse_args()

    dims = tuple(args.dims)
    if len(dims) != len(args.bandwidths):
        p.error("--dims and --bandwidths must have same length")
    arc_capacities = tuple(args.bandwidths)

    # ForestColl
    G, compute_nodes = torus_topology(dims, arc_capacities)
    (U_fc, k_fc), (Ts_fc, Cs_fc) = optimal_pipeline_spanning_trees(G, compute_nodes)
    use_msccldsl = args.lowerer == "msccldsl"
    xml_fc = to_msccl_xml(
        Ts_fc,
        Cs_fc,
        k_fc,
        dims,
        msccl=use_msccldsl,
        coll="allgather",
        ninstance=1,
    )
    with open(f"{args.prefix}-forestcoll.xml", "w") as f:
        f.write(xml_fc)
    #
    # SPST
    (U_spst, k_spst), (Ts_spst, Cs_spst) = solve_torus(dims, arc_capacities)
    xml_spst = to_msccl_xml(
        Ts_spst,
        Cs_spst,
        k_spst,
        dims,
        msccl=use_msccldsl,
        coll="allgather",
    )
    with open(f"{args.prefix}-spst.xml", "w") as f:
        f.write(xml_spst)
    print(f"SPST: U={U_spst}, k={k_spst}")
    print(f"ForestColl: U={U_fc}, k={k_fc}")


if __name__ == "__main__":
    main()
