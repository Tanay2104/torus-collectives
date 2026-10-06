# Local patches

`setup-deps.sh` applies the `rcs` and `astra-reentrancy` patches after checking out the pinned dependencies. They modify the local dependency clones; they do not change upstream Git history. The ns-3 Python 3.14 patch is optional and must be applied manually.

| Patch | Target | Purpose |
| --- | --- | --- |
| `collectiveapi-rcs.patch` | Collective API converter | Converts MSCCL `recv_copy_send` steps into dependent receive and send nodes. Without it, those steps are dropped and simulations can hang. |
| `astra-custom-reentrancy.patch` | ASTRA-Sim custom collective runner | Prevents synchronous receive callbacks from re-entering node issuance. It refreshes the ready-node set after each issue and delays collective exit until the outer issue loop returns. This fixes duplicate node issuance and the ring-4 DSL completion crash observed in testing. |
| `ns3-python314.patch` | ASTRA-Sim's ns-3 submodule | Adjusts the ns-3 launcher for Python 3.14. Apply only when building with Python 3.14. |

ASTRA-Sim is pinned to a revision containing upstream [PR #383](https://github.com/astra-sim/astra-sim/pull/383), which queues pending receives that share a `(tag, source, destination)` key. That fixes a separate lost-callback hang; it does not fix the custom-runner reentrancy issue by itself.

To apply the optional Python patch:

```sh
./apply-patches.sh python314
```

The DSL paths were rerun after applying both required patches on 2×2, 2×3, and 1D rings of 4, 8, and 16 NPUs. All completed in those tested configurations.
