# Torus collective experiments

Compare ForestColl with the SPST torus solver. Since ForestColl's to_xml is only for small topologies and uses too many TBs/Channels, we also test/dev a custom MSCCLang based lowerer. `run.sh` generates schedules and runs ASTRA-Sim.

 ForestColl schedules can exceed realistic GPU TB/channel limits, and the DSL-to-Chakra converter is unfinished. `results/provisional/` is old output if we increase allowed TBs in ForestColl to_xml, not reliable performance evidence.

## Setup and run

`dependencies.lock` is a pin list of clone path, URL, commit, and role. `setup-deps.sh` checks out each repository at that path and revision, then applies the tracked `rcs` compatibility patch to the local Collective API checkout. It does not install packages or build ASTRA-Sim. ASTRA-Sim brings its own pinned Chakra and ns-3 submodules. Its `A+B` pin is a reproducible merge of two public commits.

```sh
./setup-deps.sh
python3 -m venv .venv
.venv/bin/pip install -e ./msccl-tools networkx protobuf
```

Build the simulator. Python 3.14 also needs the ns-3 wrapper patch.

```sh
./apply-patches.sh python314   # Python 3.14 only
bash Simulation/astra-sim/build/astra_ns3/build.sh --compile
./run.sh --dims "2 2" --bandwidths "50 50" --latencies "100 100" --coll-size 1
# Try the experimental MSCCL DSL lowerer for the torus solver's schedule:
./run.sh --dims "2 2" --bandwidths "50 50" --latencies "100 100" --coll-size 1 --lowerer msccldsl
```

`--lowerer msccldsl` selects the experimental MSCCL DSL lowerer for both the torus solver's and ForestColl's schedules. `--lowerer forestcoll` uses the hand-written ForestColl XML lowering for both. The run script checks for converter support before starting and exits with a setup instruction if it is missing. Setup applies the Collective API `rcs` compatibility patch and a small ASTRA-Sim reentrancy patch; ASTRA-Sim's upstream NS-3 concurrent-receive fix is included in the pinned revision. These simulations remain proof of concept, not reliable performance results: ForestColl can exceed GPU threadblock/channel limits, and the DSL-to-Chakra path is still under development. The two rows compare different schedules under the selected lowerer; they do not isolate lowerer performance. Each run stays under `runs/run-*` with schedules, generated configs, traces, simulator logs, and scratch output. Runs time out after 120 seconds by default; set `SIM_TIMEOUT_SECONDS` to change that.
