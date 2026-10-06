from __future__ import annotations


from dataclasses import dataclass
from fractions import Fraction
from itertools import permutations, product
import math
from typing import Iterable, Sequence

Coordinate = tuple[int, ...]
Edge = tuple[Coordinate, Coordinate]
Order = tuple[int, ...]
TieDirections = tuple[bool, ...]

__all__ = [
    "solve_torus",
    "torus_topology",
    "to_msccl_xml",
    "TorusModel",
    "Coordinate",
    "Edge",
]


def _outgoing_arcs_per_vertex(ring_size: int) -> int:
    """Return the number of distinct directed torus arcs per vertex."""
    return 1 if ring_size == 2 else 2


@dataclass(frozen=True)
class TorusModel:
    """A canonical directed torus with one homogeneous capacity per dimension."""

    nodes: tuple[Coordinate, ...]
    dimensions: tuple[int, ...]
    arc_capacities: tuple[Fraction | float | int, ...]

    @property
    def node_count(self) -> int:
        return len(self.nodes)

    @property
    def dimension_budgets(self) -> tuple[Fraction, ...]:
        """Usable outgoing capacity at one vertex in every dimension."""
        return tuple(
            outgoing_arcs * as_exact_fraction(capacity)
            for outgoing_arcs, capacity in zip(
                (_outgoing_arcs_per_vertex(size) for size in self.dimensions),
                self.arc_capacities,
            )
        )

    def edges(self) -> list[tuple[Coordinate, Coordinate, Fraction | float | int]]:
        """Return all directed torus arcs as (src, dst, capacity) tuples."""
        edge_list = []
        for u in self.nodes:
            for dim, ring_size in enumerate(self.dimensions):
                cap = self.arc_capacities[dim]
                # +1 direction
                v_plus = list(u)
                v_plus[dim] = (u[dim] + 1) % ring_size
                edge_list.append((u, tuple(v_plus), cap))
                # -1 direction (distinct arc when ring_size > 2)
                if ring_size > 2:
                    v_minus = list(u)
                    v_minus[dim] = (u[dim] - 1) % ring_size
                    edge_list.append((u, tuple(v_minus), cap))
        return edge_list

    def to_networkx(self):
        """Construct a NetworkX DiGraph representing this torus topology."""
        import networkx as nx

        G = nx.DiGraph()
        G.add_nodes_from(self.nodes)
        for u, v, cap in self.edges():
            # Convert Fraction or numeric capacity to float/int if suitable for networkx
            c = (
                float(cap)
                if isinstance(cap, Fraction) and cap.denominator != 1
                else (int(cap) if isinstance(cap, Fraction) else cap)
            )
            G.add_edge(u, v, capacity=c)
        return G


def torus_topology(
    dimensions: Sequence[int],
    arc_capacities: Sequence[Fraction | float | int | str],
):
    """Build a NetworkX DiGraph and compute_nodes list for a torus topology."""
    canonical_dimensions = tuple(dimensions)
    if not canonical_dimensions or any(size < 2 for size in canonical_dimensions):
        raise ValueError(
            "Every torus dimension must have an integer ring size of at least two."
        )
    if len(arc_capacities) != len(canonical_dimensions):
        raise ValueError("Provide exactly one directed-arc capacity per dimension.")
    exact_capacities = tuple(as_exact_fraction(capacity) for capacity in arc_capacities)
    model = TorusModel(
        nodes=tuple(product(*(range(size) for size in canonical_dimensions))),
        dimensions=canonical_dimensions,
        arc_capacities=exact_capacities,
    )
    return model.to_networkx(), list(model.nodes)


def to_msccl_xml(
    Ts: dict,
    Cs: dict,
    k: int,
    dimensions: Sequence[int] | None = None,
    msccl=False,
    coll: str = "allgather",
    ninstance: int = 1,
    inplace: bool = True,
) -> str:
    """Convert schedule trees (Ts, Cs, k) to an MSCCL-IR XML string."""
    import xml.etree.ElementTree as ET
    from xml.dom import minidom
    from ForestColl.to_xml import (
        construct_algo_allgather,
        construct_algo_allreduce,
        construct_algo_reduce_scatter,
    )
    from torus.msccldsl import allgather_trees

    def _to_rank(node):
        if not isinstance(node, tuple):
            return int(node)
        if dimensions is None:
            raise ValueError(
                "dimensions must be provided when tree coordinates are tuples."
            )
        rank = 0
        for x, n in zip(node, dimensions):
            rank = rank * n + x
        return rank

    msccl_Ts = {}
    msccl_Cs = {}
    tree_idx_counter = {}
    nodes = set()

    for (u, i), paths in sorted(Ts.items()):
        u_rank = _to_rank(u)
        nodes.add(u_rank)
        c_count = Cs[u, i]
        mapped_paths = []
        for path in paths:
            src = _to_rank(path[0][0])
            dst = _to_rank(path[-1][-1])
            nodes.add(src)
            nodes.add(dst)
            ch = path[0][2] if len(path[0]) > 2 else None
            mapped_paths.append([(src, dst, ch)])

        for _ in range(c_count):
            idx = tree_idx_counter.get(u_rank, 0)
            msccl_Ts[u_rank, idx] = mapped_paths
            msccl_Cs[u_rank, idx] = 1
            tree_idx_counter[u_rank] = idx + 1

    node_list = sorted(nodes)
    if coll == "allgather":
        if msccl is False:
            elem = construct_algo_allgather(
                msccl_Ts, msccl_Cs, k, node_list, ninstance=ninstance, inplace=inplace
            )
        else:
            elem, _ = allgather_trees(msccl_Ts, k, int(len(msccl_Ts) / k), 1)
    elif coll == "allreduce":
        elem = construct_algo_allreduce(
            msccl_Ts, msccl_Cs, k, node_list, ninstance=ninstance, inplace=inplace
        )
    elif coll == "reduce_scatter":
        elem = construct_algo_reduce_scatter(
            msccl_Ts,
            msccl_Cs,
            k,
            node_list,
            ninstance=ninstance,
            is_amd=False,
            inplace=inplace,
        )
    else:
        raise ValueError(f"Unsupported collective: {coll}")

    s = None
    if msccl is False:
        s = ET.tostring(elem, "utf-8")
    else:
        s = elem
    s = minidom.parseString(s).toprettyxml(indent="  ")
    return "\n".join(s.split("\n")[1:])


@dataclass(frozen=True)
class OrderPacking:
    """The LP weight and dimension-load vector for one dimension order."""

    order: Order
    weight: Fraction
    dimension_loads: tuple[int, ...]


@dataclass(frozen=True)
class WeightedOrderTemplate:
    """One concrete tree family after choosing every antipodal direction."""

    order: Order
    tie_directions: TieDirections
    weight: Fraction


@dataclass(frozen=True)
class TorusPackingPlan:
    """The optimal fractional packing, ready for schedule lowering."""

    per_root_rate: Fraction
    order_packings: tuple[OrderPacking, ...]
    active_templates: tuple[WeightedOrderTemplate, ...]
    order_count: int
    template_count: int
    cut_upper_bound: Fraction

    @property
    def aggregate_bandwidth(self) -> Fraction:
        """ForestColl-compatible algorithmic bandwidth: ``N * k / U``."""
        # The caller can also read this as ``cut_upper_bound``; the solver
        # explicitly verifies that the two quantities agree.
        return self.cut_upper_bound


@dataclass(frozen=True)
class TorusSchedule:
    """An exact direct-edge schedule, in ForestColl's ``Ts``/``Cs`` format."""

    U: Fraction
    k: int
    trees: dict[tuple[Coordinate, int], list[list[Edge]]]
    chunk_counts: dict[tuple[Coordinate, int], int]

    @property
    def Ts(self) -> dict[tuple[Coordinate, int], list[list[Edge]]]:
        """ForestColl's conventional name for ``trees``."""
        return self.trees

    @property
    def Cs(self) -> dict[tuple[Coordinate, int], int]:
        """ForestColl's conventional name for ``chunk_counts``."""
        return self.chunk_counts

    @property
    def per_root_rate(self) -> Fraction:
        return Fraction(self.k, 1) / self.U

    def as_tuple(
        self,
    ) -> tuple[
        tuple[Fraction, int],
        tuple[
            dict[tuple[Coordinate, int], list[list[Edge]]],
            dict[tuple[Coordinate, int], int],
        ],
    ]:
        """Return exactly ``((U, k), (Ts, Cs))``."""
        return (self.U, self.k), (self.Ts, self.Cs)


class TorusPackingSolver:
    """Solve and materialize the dimension-order construction for one torus."""

    def __init__(self, model: TorusModel) -> None:
        self.model = model
        self.orders: tuple[Order, ...] = tuple(
            permutations(range(len(model.dimensions)))
        )
        self.tie_variants: tuple[TieDirections, ...] = tuple(
            antipodal_tie_variants(model.dimensions)
        )

    def solve(self, *, max_orders: int = 720) -> TorusPackingPlan:
        """Return the optimum order weights and their concrete tie variants."""
        self._check_enumeration_limit(max_orders)

        order_packings = self._solve_compact_order_lp()
        per_root_rate = sum((packing.weight for packing in order_packings), Fraction())
        if per_root_rate <= 0:
            raise RuntimeError("The dimension-order LP found no positive packing.")

        aggregate_cut_bound = self.coordinate_cut_upper_bound()
        aggregate_rate = self.model.node_count * per_root_rate
        if aggregate_rate != aggregate_cut_bound:
            raise RuntimeError(
                "The order LP did not attain the coordinate-cut optimum: "
                f"LP={aggregate_rate}, cut={aggregate_cut_bound}."
            )

        return TorusPackingPlan(
            per_root_rate=per_root_rate,
            order_packings=order_packings,
            active_templates=tuple(self._split_antipodal_ties(order_packings)),
            order_count=len(self.orders),
            template_count=len(self.orders) * len(self.tie_variants),
            cut_upper_bound=aggregate_cut_bound,
        )

    def build_schedule(
        self, *, max_orders: int = 720, fixed_k: int | None = None
    ) -> TorusSchedule:
        """Build the final chunk schedule for this torus."""
        return self.to_schedule(self.solve(max_orders=max_orders), fixed_k=fixed_k)

    def coordinate_cut_upper_bound(self) -> Fraction:
        """Return the aggregate coordinate-cut bound from Section 2.1."""
        bounds: list[Fraction] = []
        dimension_count = len(self.model.dimensions)
        for mask in range(1, 1 << dimension_count):
            selected_dimensions = [
                dimension
                for dimension in range(dimension_count)
                if mask & (1 << dimension)
            ]
            required_crossings = (
                math.prod(
                    self.model.dimensions[dimension]
                    for dimension in selected_dimensions
                )
                - 1
            )
            available_capacity = self.model.node_count * sum(
                self.model.dimension_budgets[dimension]
                for dimension in selected_dimensions
            )
            bounds.append(available_capacity / required_crossings)
        return min(bounds)

    def build_tree_edges(
        self, root: Coordinate, template: WeightedOrderTemplate
    ) -> list[Edge]:
        """Build the out-arborescence for ``root`` and one concrete template."""
        if root not in self.model.nodes:
            raise ValueError("The tree root must be a torus coordinate.")

        edges: list[Edge] = []
        coordinate_ranges = (range(size) for size in self.model.dimensions)
        for child in product(*coordinate_ranges):
            if child == root:
                continue
            parent = self._parent_towards_root(child, root, template)
            edges.append((parent, child))
        return edges

    def _parent_towards_root(
        self, child: Coordinate, root: Coordinate, template: WeightedOrderTemplate
    ) -> Coordinate:
        for dimension in template.order:
            displacement_from_root = (
                child[dimension] - root[dimension]
            ) % self.model.dimensions[dimension]
            if displacement_from_root:
                direction_to_parent = shortest_direction_towards_root(
                    displacement_from_root,
                    self.model.dimensions[dimension],
                    template.tie_directions[dimension],
                )
                return step_in_dimension(
                    child,
                    dimension,
                    direction_to_parent,
                    self.model.dimensions[dimension],
                )
        raise ValueError("Only the root may have no parent in a torus tree.")

    def _check_enumeration_limit(self, max_orders: int) -> None:
        if len(self.orders) > max_orders:
            raise ValueError(
                f"This torus has {len(self.orders)} dimension orders; "
                f"the configured LP limit is {max_orders}."
            )

    def _solve_compact_order_lp(self) -> tuple[OrderPacking, ...]:
        """Solve the PDF's logical LP, one variable for every order."""
        loads_by_order = tuple(
            dimension_loads_for_order(self.model.dimensions, order)
            for order in self.orders
        )
        loads_by_dimension = tuple(
            tuple(loads[dimension] for loads in loads_by_order)
            for dimension in range(len(self.model.dimensions))
        )
        weights = maximize_rational_linear_program(
            loads_by_dimension, self.model.dimension_budgets
        )

        return tuple(
            OrderPacking(
                order=order,
                weight=weight,
                dimension_loads=loads_by_order[order_index],
            )
            for order_index, (order, weight) in enumerate(zip(self.orders, weights))
            if weight > 0
        )

    def to_schedule(
        self, plan: TorusPackingPlan, *, fixed_k: int | None = None
    ) -> TorusSchedule:
        """Scale exact template weights into integer ``Ts``/``Cs``."""
        if not plan.active_templates:
            raise ValueError("Cannot lower a torus plan without active templates.")

        if fixed_k is None:
            quantum = rational_hcf(
                template.weight for template in plan.active_templates
            )
            if quantum <= 0:
                raise ValueError("Every active template must have a positive weight.")
            U = Fraction(1, 1) / quantum
            chunk_counts_per_template = tuple(
                _integer_multiple(template.weight, quantum)
                for template in plan.active_templates
            )
            k = sum(chunk_counts_per_template)
            if Fraction(k, 1) / U != plan.per_root_rate:
                raise RuntimeError(
                    "Template scaling did not preserve the per-root rate."
                )
        else:
            if not isinstance(fixed_k, int) or fixed_k <= 0:
                raise ValueError("fixed_k must be a positive integer.")
            template_count = len(plan.active_templates)
            if fixed_k < template_count:
                raise ValueError(
                    f"fixed_k={fixed_k} is below the {template_count} active "
                    "dimension-order/antipodal-tie templates this torus "
                    "needs; every template must receive at least one chunk "
                    "to preserve per-arc load balance. Raise fixed_k."
                )
            chunk_counts_per_template = self._apportion_fixed_k(plan, fixed_k)
            k = fixed_k
            U = self._achieved_U(plan, chunk_counts_per_template)

        trees: dict[tuple[Coordinate, int], list[list[Edge]]] = {}
        chunk_counts: dict[tuple[Coordinate, int], int] = {}
        for root in self.model.nodes:
            for index, (template, chunk_count) in enumerate(
                zip(plan.active_templates, chunk_counts_per_template)
            ):
                key = root, index
                trees[key] = [[edge] for edge in self.build_tree_edges(root, template)]
                chunk_counts[key] = chunk_count
            if (
                sum(
                    chunk_counts[root, index]
                    for index in range(len(chunk_counts_per_template))
                )
                != k
            ):
                raise RuntimeError("A root did not receive exactly k chunks.")
        return TorusSchedule(U, k, trees, chunk_counts)

    def _apportion_fixed_k(
        self, plan: TorusPackingPlan, fixed_k: int
    ) -> tuple[int, ...]:
        """Split ``fixed_k`` chunks across active templates, proportional to"""
        templates = plan.active_templates
        template_count = len(templates)
        total_weight = sum((template.weight for template in templates), Fraction())
        remaining = fixed_k - template_count  # after the guaranteed 1 each
        shares = [
            remaining * (template.weight / total_weight) for template in templates
        ]
        base = [int(share) for share in shares]  # floor: shares are >= 0
        remainders = [share - whole for share, whole in zip(shares, base)]
        leftover = remaining - sum(base)
        # Largest remainder first; break ties by template index for determinism.
        priority = sorted(range(template_count), key=lambda i: (-remainders[i], i))
        for i in priority[:leftover]:
            base[i] += 1
        return tuple(1 + whole for whole in base)

    def _achieved_U(
        self, plan: TorusPackingPlan, chunk_counts_per_template: tuple[int, ...]
    ) -> Fraction:
        """Recompute the true achievable ``U`` for a specific integer chunk"""
        loads_by_order = {
            packing.order: packing.dimension_loads for packing in plan.order_packings
        }
        budgets = self.model.dimension_budgets
        worst = Fraction(0)
        for dimension, budget in enumerate(budgets):
            used = sum(
                count * loads_by_order[template.order][dimension]
                for template, count in zip(
                    plan.active_templates, chunk_counts_per_template
                )
            )
            worst = max(worst, Fraction(used) / budget)
        if worst <= 0:
            raise RuntimeError("Achieved rate computation collapsed to zero.")
        return worst

    def _split_antipodal_ties(
        self, order_packings: Iterable[OrderPacking]
    ) -> Iterable[WeightedOrderTemplate]:
        """Average all antipodal choices to balance every physical arc."""
        for packing in order_packings:
            for tie_directions in self.tie_variants:
                yield WeightedOrderTemplate(
                    order=packing.order,
                    tie_directions=tie_directions,
                    weight=packing.weight / len(self.tie_variants),
                )


def solve_torus(
    dimensions: Sequence[int],
    arc_capacities: Sequence[Fraction | float | int | str],
    fixed_k: int | None = None,
) -> tuple[
    tuple[Fraction, int],
    tuple[
        dict[tuple[Coordinate, int], list[list[Edge]]],
        dict[tuple[Coordinate, int], int],
    ],
]:
    """Return a schedule for a torus described by its dimensions."""
    canonical_dimensions = tuple(dimensions)
    if not canonical_dimensions or any(size < 2 for size in canonical_dimensions):
        raise ValueError(
            "Every torus dimension must have an integer ring size of at least two."
        )
    if len(arc_capacities) != len(canonical_dimensions):
        raise ValueError("Provide exactly one directed-arc capacity per dimension.")
    exact_capacities = tuple(as_exact_fraction(capacity) for capacity in arc_capacities)
    if any(capacity <= 0 for capacity in exact_capacities):
        raise ValueError("Every directed-arc capacity must be positive.")
    model = TorusModel(
        nodes=tuple(product(*(range(size) for size in canonical_dimensions))),
        dimensions=canonical_dimensions,
        arc_capacities=exact_capacities,
    )
    return TorusPackingSolver(model).build_schedule(fixed_k=fixed_k).as_tuple()


def dimension_loads_for_order(
    dimensions: tuple[int, ...], order: Order
) -> tuple[int, ...]:
    """Return ``L_i(order)`` for the first-differing-coordinate tree."""
    loads = [0] * len(dimensions)
    later_dimension_product = 1
    for dimension in reversed(order):
        loads[dimension] = (dimensions[dimension] - 1) * later_dimension_product
        later_dimension_product *= dimensions[dimension]
    return tuple(loads)


def as_exact_fraction(value: Fraction | float | int | str) -> Fraction:
    """Interpret an integer or finite decimal as an exact rational quantity."""
    if isinstance(value, Fraction):
        return value
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("Torus capacities must be finite.")
    try:
        # ``str`` preserves the intended decimal (0.1 means 1/10), rather
        # than importing a float's irrelevant binary representation.
        return Fraction(str(value))
    except (TypeError, ValueError, ZeroDivisionError) as error:
        raise ValueError(f"Torus capacity {value!r} is not rational.") from error


def maximize_rational_linear_program(
    A: Sequence[Sequence[int | Fraction]],
    b: Sequence[int | Fraction],
    objective: Sequence[int | Fraction] | None = None,
) -> tuple[Fraction, ...]:
    """Solve a feasible bounded LP exactly with rational primal simplex."""
    row_count = len(A)
    if row_count != len(b) or row_count == 0:
        raise ValueError("LP must have one budget for every constraint row.")
    variable_count = len(A[0])
    if variable_count == 0 or any(len(row) != variable_count for row in A):
        raise ValueError("LP matrix rows must have a common positive length.")
    if objective is None:
        objective = (1,) * variable_count
    if len(objective) != variable_count:
        raise ValueError("LP objective length must match the number of variables.")

    tableau = [
        [as_exact_fraction(value) for value in row]
        + [
            Fraction(int(row_index == column_index))
            for column_index in range(row_count)
        ]
        + [as_exact_fraction(b[row_index])]
        for row_index, row in enumerate(A)
    ]
    tableau.append(
        [-as_exact_fraction(value) for value in objective]
        + [Fraction()] * row_count
        + [Fraction()]
    )
    basis = [variable_count + row_index for row_index in range(row_count)]

    while True:
        objective_row = tableau[-1]
        entering = next(
            (
                column
                for column in range(variable_count + row_count)
                if objective_row[column] < 0
            ),
            None,
        )
        if entering is None:
            break
        candidates = [
            (tableau[row][-1] / tableau[row][entering], basis[row], row)
            for row in range(row_count)
            if tableau[row][entering] > 0
        ]
        if not candidates:
            raise RuntimeError("The torus LP is unexpectedly unbounded.")
        _, _, leaving = min(candidates)
        _pivot_tableau(tableau, leaving, entering)
        basis[leaving] = entering

    solution = [Fraction()] * variable_count
    for row, basic_variable in enumerate(basis):
        if basic_variable < variable_count:
            solution[basic_variable] = tableau[row][-1]
    return tuple(solution)


def maximize_linear_program(
    A: Sequence[Sequence[int | Fraction]],
    b: Sequence[int | Fraction],
    objective: Sequence[int | Fraction] | None = None,
) -> tuple[Fraction, ...]:
    """Compatibility name for the exact rational LP solver."""
    return maximize_rational_linear_program(A, b, objective)


def _pivot_tableau(tableau: list[list[Fraction]], row: int, column: int) -> None:
    pivot = tableau[row][column]
    tableau[row] = [value / pivot for value in tableau[row]]
    for other_row, values in enumerate(tableau):
        if other_row == row:
            continue
        factor = values[column]
        if factor:
            tableau[other_row] = [
                value - factor * pivot_value
                for value, pivot_value in zip(values, tableau[row])
            ]


def rational_hcf(weights: Iterable[Fraction]) -> Fraction:
    """Return the greatest positive rational quantum dividing every weight."""
    values = tuple(weights)
    if not values or any(value <= 0 for value in values):
        raise ValueError("The rational HCF requires positive weights.")
    denominator = math.lcm(*(value.denominator for value in values))
    common_numerator = math.gcd(
        *(value.numerator * (denominator // value.denominator) for value in values)
    )
    return Fraction(common_numerator, denominator)


def _integer_multiple(value: Fraction, quantum: Fraction) -> int:
    multiple = value / quantum
    if multiple.denominator != 1 or multiple <= 0:
        raise ValueError("A weight is not an integer multiple of the quantum.")
    return multiple.numerator


def template_label(template: WeightedOrderTemplate) -> str:
    """Return a compact, stable display label for a concrete tree family."""
    signs = "".join("+" if direction else "-" for direction in template.tie_directions)
    return "(" + ", ".join(map(str, template.order)) + ") tie=" + signs


def antipodal_tie_variants(dimensions: tuple[int, ...]) -> Iterable[TieDirections]:
    """Enumerate only the directions that differ at an even-ring antipode."""
    return product(
        *[
            (False, True) if size > 2 and size % 2 == 0 else (False,)
            for size in dimensions
        ]
    )


def shortest_direction_towards_root(
    displacement_from_root: int, ring_size: int, choose_positive_at_tie: bool
) -> int:
    """Choose the signed one-hop move from a vertex towards the root."""
    if (
        ring_size > 2
        and ring_size % 2 == 0
        and displacement_from_root == ring_size // 2
    ):
        return 1 if choose_positive_at_tie else -1
    return -1 if displacement_from_root <= ring_size // 2 else 1


def step_in_dimension(
    node: Coordinate, dimension: int, direction: int, ring_size: int
) -> Coordinate:
    """Return the torus neighbour reached by one signed coordinate step."""
    stepped = list(node)
    stepped[dimension] = (stepped[dimension] + direction) % ring_size
    return tuple(stepped)
