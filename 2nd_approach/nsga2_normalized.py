from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from pymoo.core.problem import Problem
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize
from pymoo.core.sampling import Sampling
from pymoo.core.mutation import Mutation
from pymoo.core.crossover import Crossover
from pymoo.core.repair import Repair

from pyvrp import read, Model
from pyvrp.stop import MaxRuntime

import obj1
import obj2


TYPE_LABELS = {0: "diesel", 1: "clean"}
DIESEL = 0
CLEAN = 1
PENALTY = 1e12


# ============================================================
# USER SETTINGS
# ============================================================

INSTANCE_PATH = "instances/vrp/X-n115-k10.vrp"

TOTAL_FLEET = 30
N_DIESEL = TOTAL_FLEET // 2
N_CLEAN = TOTAL_FLEET - N_DIESEL

MAX_ROUTES = TOTAL_FLEET

POP_SIZE = 300
N_GEN = 300
SEED = 0

USE_WEIGHTED_SUM_SEEDS = True
N_WEIGHTED_SEEDS = 80
WEIGHTED_SEED_RUNTIME = 5
SCALE = 1000

RANDOM_FRACTION = 0.02

# Cache weighted-sum PyVRP seed chromosomes locally so repeated runs do not
# recompute them. Delete this directory or set FORCE_RECOMPUTE_WEIGHTED_CACHE=True
# when you change objective parameters and want a fresh reference front.
WEIGHTED_CACHE_DIR = "weighted_sum_cache"
FORCE_RECOMPUTE_WEIGHTED_CACHE = False


# ============================================================
# BASIC HELPERS
# ============================================================

def demand_at_customer(customer_id, data):
    d = data.clients()[customer_id - 1].delivery

    if isinstance(d, (list, tuple, np.ndarray)):
        return int(d[0]) if len(d) > 0 else 0

    return int(d)


def route_demand(route, data):
    return sum(demand_at_customer(c, data) for c in route)


def compute_distance(routes, data):
    D = data.distance_matrix(0)
    total = 0.0

    for route in routes:
        visits = [0] + list(route) + [0]

        for a, b in zip(visits[:-1], visits[1:]):
            total += D[a][b]

    return float(total)


def is_capacity_feasible(assignments, data, Q_by_type):
    for route, vt_idx in assignments:
        vehicle_name = TYPE_LABELS[int(vt_idx)]
        capacity = Q_by_type[vehicle_name]

        if route_demand(route, data) > capacity:
            return False

    return True


def is_fleet_feasible(assignments, n_diesel, n_clean):
    diesel_used = sum(1 for _, vt in assignments if int(vt) == DIESEL)
    clean_used = sum(1 for _, vt in assignments if int(vt) == CLEAN)

    return diesel_used <= n_diesel and clean_used <= n_clean


def pareto_filter(points):
    points = list(set(points))
    front = []

    for p in points:
        dominated = False

        for q in points:
            if q == p:
                continue

            if q[0] <= p[0] and q[1] <= p[1] and (q[0] < p[0] or q[1] < p[1]):
                dominated = True
                break

        if not dominated:
            front.append(p)

    return sorted(front, key=lambda x: x[0])


# ============================================================
# CHROMOSOME HELPERS
# ============================================================

def repair_permutation(order, n_customers):
    order = [int(v) for v in order]

    valid = set(range(1, n_customers + 1))
    seen = set()
    repaired = []

    missing = [c for c in range(1, n_customers + 1) if c not in order]
    missing_idx = 0

    for v in order:
        if v in valid and v not in seen:
            repaired.append(v)
            seen.add(v)
        else:
            repaired.append(missing[missing_idx])
            seen.add(missing[missing_idx])
            missing_idx += 1

    return np.array(repaired, dtype=int)


def clean_splits(split_gene, n_customers):
    active = []

    for s in split_gene:
        s = int(s)

        if 1 <= s <= n_customers - 1:
            active.append(s)

    return sorted(set(active))


def repair_vehicle_type_for_route(route, vt_idx, data, Q_by_type):
    """
    If a route is assigned to clean but exceeds clean capacity,
    switch it to diesel if diesel capacity allows it.
    """
    demand = route_demand(route, data)

    clean_cap = Q_by_type["clean"]
    diesel_cap = Q_by_type["diesel"]

    vt_idx = int(np.clip(vt_idx, 0, 1))

    if vt_idx == CLEAN and demand > clean_cap:
        if demand <= diesel_cap:
            return DIESEL

    return vt_idx


def decode_chromosome(x, problem):
    x = np.round(x).astype(int)

    n = problem.n_customers
    m = problem.max_routes

    order = repair_permutation(x[:n], n)

    split_gene = x[n:n + m - 1]
    type_gene = x[n + m - 1:n + m - 1 + m]
    type_gene = np.clip(type_gene, 0, 1)

    active_splits = clean_splits(split_gene, n)

    routes = []
    start = 0

    for s in active_splits:
        route = list(order[start:s])

        if route:
            routes.append(route)

        start = s

    final_route = list(order[start:])

    if final_route:
        routes.append(final_route)

    if len(routes) > m:
        routes = routes[:m]

    assignments = []

    for idx, route in enumerate(routes):
        vt = int(type_gene[idx]) if idx < len(type_gene) else DIESEL

        vt = repair_vehicle_type_for_route(
            route,
            vt,
            problem.vrp_data,
            problem.Q_by_type,
        )

        assignments.append((route, vt))

    return routes, assignments


def routes_to_chromosome(routes, type_gene, problem):
    n = problem.n_customers
    m = problem.max_routes

    routes = [list(r) for r in routes if len(r) > 0]

    order = []
    splits = []
    count = 0

    for idx, route in enumerate(routes):
        order.extend(route)
        count += len(route)

        if idx < len(routes) - 1:
            splits.append(count)

    order = repair_permutation(order, n)

    while len(splits) < m - 1:
        splits.append(-1)

    splits = splits[:m - 1]

    type_gene = list(type_gene)

    while len(type_gene) < m:
        type_gene.append(DIESEL)

    type_gene = np.clip(type_gene[:m], 0, 1)

    return np.concatenate([
        np.array(order, dtype=int),
        np.array(splits, dtype=int),
        np.array(type_gene, dtype=int),
    ])


def solution_to_chromosome(solution, problem):
    routes = []
    type_gene = []

    for route in solution.routes():
        visits = list(route.visits())

        if len(visits) == 0:
            continue

        routes.append(visits)

        vt = int(route.vehicle_type())

        if vt not in [0, 1]:
            raise ValueError(f"Unexpected vehicle type index: {vt}")

        type_gene.append(vt)

    return routes_to_chromosome(routes, type_gene, problem)


# ============================================================
# GREEDY VEHICLE-TYPE IMPROVEMENT
# ============================================================

def co2_for_single_route(route, vehicle_name, data, Q_by_type,
                         f0, alpha, beta, gamma, mult):
    """
    Computes true load-dependent CO2 for one route and one vehicle type.
    """
    D = data.distance_matrix(0)
    Qk = float(Q_by_type[vehicle_name])

    visits = [0] + list(route) + [0]
    remaining = float(route_demand(route, data))

    total = 0.0

    for a, b in zip(visits[:-1], visits[1:]):
        dij = float(D[a][b])

        load_ratio = 0.0 if Qk <= 0 else remaining / Qk
        fuel = f0 * (1.0 + alpha * (load_ratio ** beta)) * mult[vehicle_name]

        total += dij * gamma * fuel

        if b != 0:
            remaining -= demand_at_customer(b, data)

    return float(total)


def assign_best_vehicle_types(routes, problem):
    """
    For fixed routes, assign clean vehicles to the routes where they save
    the most CO2, while respecting:
        - clean capacity,
        - diesel capacity,
        - number of clean vehicles,
        - number of diesel vehicles.

    This avoids wasting NSGA-II search on dominated vehicle-type assignments.
    """
    assignments = []
    clean_candidates = []

    for idx, route in enumerate(routes):
        demand = route_demand(route, problem.vrp_data)

        if demand > problem.Q_by_type["diesel"]:
            return None

        diesel_co2 = co2_for_single_route(
            route,
            "diesel",
            problem.vrp_data,
            problem.Q_by_type,
            problem.f0,
            problem.alpha,
            problem.beta,
            problem.gamma,
            problem.mult,
        )

        assignments.append((route, DIESEL))

        if demand <= problem.Q_by_type["clean"]:
            clean_co2 = co2_for_single_route(
                route,
                "clean",
                problem.vrp_data,
                problem.Q_by_type,
                problem.f0,
                problem.alpha,
                problem.beta,
                problem.gamma,
                problem.mult,
            )

            saving = diesel_co2 - clean_co2
            clean_candidates.append((saving, idx))

    clean_candidates.sort(reverse=True)

    clean_used = 0

    for _, idx in clean_candidates:
        if clean_used >= problem.n_clean:
            break

        route, _ = assignments[idx]
        assignments[idx] = (route, CLEAN)
        clean_used += 1

    diesel_used = len(assignments) - clean_used

    if clean_used > problem.n_clean:
        return None

    if diesel_used > problem.n_diesel:
        return None

    return assignments


# ============================================================
# REPAIR
# ============================================================

class VRPRepair(Repair):

    def _do(self, problem, X, **kwargs):
        Y = X.copy()

        n = problem.n_customers
        m = problem.max_routes

        for i in range(len(Y)):
            x = np.round(Y[i]).astype(int)

            order = repair_permutation(x[:n], n)

            raw_splits = x[n:n + m - 1]
            active_splits = clean_splits(raw_splits, n)

            split_gene = active_splits[:m - 1]

            while len(split_gene) < m - 1:
                split_gene.append(-1)

            type_gene = x[n + m - 1:n + m - 1 + m]
            type_gene = np.clip(type_gene, 0, 1)

            Y[i] = np.concatenate([
                order,
                np.array(split_gene, dtype=int),
                np.array(type_gene, dtype=int),
            ])

        return Y


# ============================================================
# PROBLEM
# ============================================================

class VRPProblem(Problem):

    def __init__(
        self,
        data,
        n_customers,
        max_routes,
        Q_by_type,
        params,
        n_diesel,
        n_clean,
    ):
        self.vrp_data = data
        self.n_customers = n_customers
        self.max_routes = max_routes
        self.Q_by_type = Q_by_type

        self.n_diesel = n_diesel
        self.n_clean = n_clean

        self.f0, self.alpha, self.beta, self.gamma, self.mult = params

        # Normalization bounds are filled after weighted-sum seeds are evaluated.
        # Until then, _evaluate returns raw values.
        self.norm_ready = False
        self.d_min = 0.0
        self.d_range = 1.0
        self.c_min = 0.0
        self.c_range = 1.0

        self.order_len = n_customers
        self.split_len = max_routes - 1
        self.type_len = max_routes

        n_var = self.order_len + self.split_len + self.type_len

        xl = np.concatenate([
            np.ones(self.order_len),
            np.full(self.split_len, -1),
            np.zeros(self.type_len),
        ])

        xu = np.concatenate([
            np.full(self.order_len, n_customers),
            np.full(self.split_len, n_customers - 1),
            np.ones(self.type_len),
        ])

        super().__init__(
            n_var=n_var,
            n_obj=2,
            xl=xl,
            xu=xu,
            vtype=int,
        )

    def set_normalization_bounds(self, reference_points):
        """
        Uses PyVRP weighted-sum seed points as the scale for NSGA-II.
        NSGA-II still evaluates true distance/CO2, but selection happens in
        normalized objective space so neither objective dominates numerically.
        """
        if not reference_points:
            self.norm_ready = False
            return

        arr = np.array(reference_points, dtype=float)
        self.d_min = float(np.min(arr[:, 0]))
        self.c_min = float(np.min(arr[:, 1]))
        self.d_range = max(float(np.max(arr[:, 0]) - self.d_min), 1e-9)
        self.c_range = max(float(np.max(arr[:, 1]) - self.c_min), 1e-9)
        self.norm_ready = True

    def normalize_objectives(self, distance, co2):
        if not self.norm_ready:
            return float(distance), float(co2)

        return (
            (float(distance) - self.d_min) / self.d_range,
            (float(co2) - self.c_min) / self.c_range,
        )

    def _evaluate(self, X, out, *args, **kwargs):
        F = []
        F_raw = []

        for x in X:
            try:
                routes, _ = decode_chromosome(x, self)

                if len(routes) == 0 or len(routes) > self.max_routes:
                    F.append([PENALTY, PENALTY])
                    F_raw.append([PENALTY, PENALTY])
                    continue

                assignments = assign_best_vehicle_types(routes, self)

                if assignments is None:
                    F.append([PENALTY, PENALTY])
                    F_raw.append([PENALTY, PENALTY])
                    continue

                if not is_capacity_feasible(assignments, self.vrp_data, self.Q_by_type):
                    F.append([PENALTY, PENALTY])
                    F_raw.append([PENALTY, PENALTY])
                    continue

                if not is_fleet_feasible(assignments, self.n_diesel, self.n_clean):
                    F.append([PENALTY, PENALTY])
                    F_raw.append([PENALTY, PENALTY])
                    continue

                distance = compute_distance(routes, self.vrp_data)

                co2 = obj2.true_co2(
                    assignments,
                    self.vrp_data,
                    TYPE_LABELS,
                    self.Q_by_type,
                    self.f0,
                    self.alpha,
                    self.beta,
                    self.gamma,
                    self.mult,
                )

                f1, f2 = self.normalize_objectives(distance, co2)
                F.append([f1, f2])
                F_raw.append([distance, co2])

            except Exception:
                F.append([PENALTY, PENALTY])
                F_raw.append([PENALTY, PENALTY])

        out["F"] = np.array(F)
        out["F_raw"] = np.array(F_raw)


# ============================================================
# WEIGHTED-SUM SEEDS
# ============================================================

def build_model_two_types(data, n_diesel, n_clean, Q_diesel, Q_clean):
    model = Model.from_data(data)

    vt0 = model.vehicle_types[0]

    model.vehicle_types[0] = vt0.replace(
        num_available=n_diesel,
        capacity=[Q_diesel],
        name="diesel",
    )

    model.add_vehicle_type(
        num_available=n_clean,
        capacity=Q_clean,
        name="clean",
    )

    return model


def solve_weighted_surrogate(
    data,
    w_dist,
    w_co2,
    n_diesel,
    n_clean,
    Q_diesel,
    Q_clean,
    f0,
    alpha,
    beta,
    gamma,
    mult,
    runtime_seconds,
):
    model = build_model_two_types(
        data,
        n_diesel=n_diesel,
        n_clean=n_clean,
        Q_diesel=Q_diesel,
        Q_clean=Q_clean,
    )

    l_nom = 0.50

    fbar = {
        k: f0 * (1.0 + alpha * (l_nom ** beta)) * mult[k]
        for k in mult
    }

    co2_per_dist = {
        k: gamma * fbar[k]
        for k in mult
    }

    unit_diesel = int(round(SCALE * (w_dist + w_co2 * co2_per_dist["diesel"])))
    unit_clean = int(round(SCALE * (w_dist + w_co2 * co2_per_dist["clean"])))

    vt_d = model.vehicle_types[0]
    vt_c = model.vehicle_types[1]

    model.vehicle_types[0] = vt_d.replace(
        fixed_cost=0,
        unit_distance_cost=unit_diesel,
    )

    model.vehicle_types[1] = vt_c.replace(
        fixed_cost=0,
        unit_distance_cost=unit_clean,
    )

    res = model.solve(
        stop=MaxRuntime(runtime_seconds),
        display=False,
    )

    return res.best


def weighted_seed_weights():
    return [
        (1.0 - w, w)
        for w in np.linspace(0.05, 0.95, N_WEIGHTED_SEEDS)
    ]


def weighted_cache_metadata(problem):
    return {
        "version": 1,
        "instance_path": INSTANCE_PATH,
        "n_customers": problem.n_customers,
        "max_routes": problem.max_routes,
        "n_diesel": problem.n_diesel,
        "n_clean": problem.n_clean,
        "q_diesel": int(problem.Q_by_type["diesel"]),
        "q_clean": int(problem.Q_by_type["clean"]),
        "f0": float(problem.f0),
        "alpha": float(problem.alpha),
        "beta": float(problem.beta),
        "gamma": float(problem.gamma),
        "mult": {k: float(v) for k, v in sorted(problem.mult.items())},
        "n_weighted_seeds": int(N_WEIGHTED_SEEDS),
        "weighted_seed_runtime": float(WEIGHTED_SEED_RUNTIME),
        "scale": int(SCALE),
        "weights": [[float(a), float(b)] for a, b in weighted_seed_weights()],
    }


def weighted_cache_path(problem):
    metadata = weighted_cache_metadata(problem)
    payload = json.dumps(metadata, sort_keys=True).encode("utf-8")
    digest = hashlib.sha256(payload).hexdigest()[:16]

    instance_name = Path(INSTANCE_PATH).stem.replace("/", "_")
    cache_dir = Path(WEIGHTED_CACHE_DIR)
    cache_dir.mkdir(parents=True, exist_ok=True)

    return cache_dir / f"weighted_seeds_{instance_name}_{digest}.npz"


def load_weighted_seed_cache(problem):
    path = weighted_cache_path(problem)

    if FORCE_RECOMPUTE_WEIGHTED_CACHE or not path.exists():
        return None

    try:
        data = np.load(path, allow_pickle=True)
        chromosomes = data["chromosomes"].astype(int)
        points = [tuple(map(float, row)) for row in data["points"]]
        weights = [tuple(map(float, row)) for row in data["weights"]]

        if len(chromosomes) == 0 or len(points) == 0:
            print(f"Weighted-sum cache exists but is empty, ignoring: {path}")
            return None

        print(f"\nLoaded {len(chromosomes)} weighted-sum seed chromosomes from cache:")
        print(path)

        return chromosomes, points, weights

    except Exception as e:
        print("Could not load weighted-sum cache, recomputing:", type(e).__name__, e)
        return None


def save_weighted_seed_cache(problem, chromosomes, points, weights):
    path = weighted_cache_path(problem)

    metadata = weighted_cache_metadata(problem)
    np.savez_compressed(
        path,
        chromosomes=np.array(chromosomes, dtype=int),
        points=np.array(points, dtype=float),
        weights=np.array(weights, dtype=float),
        metadata=json.dumps(metadata, sort_keys=True),
    )

    print(f"Saved weighted-sum seed cache to:\n{path}")


def make_weighted_sum_seed_chromosomes_cached(problem):
    """
    Returns weighted-sum seed chromosomes and their true objective values.

    The first run solves the weighted PyVRP problems. Later runs load the
    converted chromosomes directly from disk, so NSGA-II can start immediately.
    """
    cached = load_weighted_seed_cache(problem)

    if cached is not None:
        chromosomes, points, weights = cached
        return list(chromosomes), points, pareto_filter(points)

    chromosomes = []
    points = []
    successful_weights = []
    weights = weighted_seed_weights()

    print("\nCreating weighted-sum seed solutions...")

    for i, (w_dist, w_co2) in enumerate(weights, start=1):
        try:
            sol = solve_weighted_surrogate(
                problem.vrp_data,
                w_dist=w_dist,
                w_co2=w_co2,
                n_diesel=problem.n_diesel,
                n_clean=problem.n_clean,
                Q_diesel=problem.Q_by_type["diesel"],
                Q_clean=problem.Q_by_type["clean"],
                f0=problem.f0,
                alpha=problem.alpha,
                beta=problem.beta,
                gamma=problem.gamma,
                mult=problem.mult,
                runtime_seconds=WEIGHTED_SEED_RUNTIME,
            )

            x = solution_to_chromosome(sol, problem)
            metrics = chromosome_to_metrics(x, problem)

            if metrics is None:
                print(f"Weighted seed {i} skipped: infeasible after decoding")
                continue

            distance, co2, _, _ = metrics
            chromosomes.append(x)
            points.append((distance, co2))
            successful_weights.append((w_dist, w_co2))

            print(
                f"Seed {i:2d}: "
                f"w_dist={w_dist:.2f}, w_co2={w_co2:.2f}, "
                f"distance={distance}, co2={co2}"
            )

        except Exception as e:
            print(f"Weighted seed {i} skipped:", type(e).__name__, e)

    print(f"Created {len(chromosomes)} weighted-sum seed chromosomes.\n")

    if chromosomes:
        save_weighted_seed_cache(problem, chromosomes, points, successful_weights)

    return chromosomes, points, pareto_filter(points)


# ============================================================
# REFERENCE FRONT FROM PYVRP WEIGHTED-SUM SEEDS
# ============================================================

def seed_solutions_to_chromosomes_and_points(seed_solutions, problem):
    """
    Converts PyVRP seeds to chromosomes and evaluates them with the same true
    distance and load-dependent CO2 used by NSGA-II.
    """
    chromosomes = []
    points = []

    for sol in seed_solutions:
        try:
            x = solution_to_chromosome(sol, problem)
            metrics = chromosome_to_metrics(x, problem)

            if metrics is None:
                continue

            distance, co2, _, _ = metrics
            chromosomes.append(x)
            points.append((distance, co2))

        except Exception as e:
            print("Reference seed skipped:", type(e).__name__, e)

    ref_front = pareto_filter(points)

    return chromosomes, points, ref_front


# ============================================================
# SAMPLING
# ============================================================

def create_random_individual(problem):
    n = problem.n_customers
    m = problem.max_routes

    order = np.random.permutation(np.arange(1, n + 1))

    num_routes = np.random.randint(1, m + 1)

    if num_routes == 1:
        splits = []
    else:
        splits = sorted(np.random.choice(
            np.arange(1, n),
            size=num_routes - 1,
            replace=False,
        ))

    while len(splits) < m - 1:
        splits.append(-1)

    # Vehicle types are optimized by assign_best_vehicle_types() => keeping this neutral
    type_gene = np.zeros(m, dtype=int)

    return np.concatenate([
        order,
        np.array(splits, dtype=int),
        type_gene,
    ])


def make_seed_variant(seed, problem):
    x = np.round(seed).astype(int).copy()

    n = problem.n_customers
    m = problem.max_routes

    order = x[:n].copy()
    splits = x[n:n + m - 1].copy()
    type_gene = x[n + m - 1:n + m - 1 + m].copy()

    r = np.random.random()

    if r < 0.40:
        a, b = np.random.choice(n, size=2, replace=False)
        order[a], order[b] = order[b], order[a]

    elif r < 0.75:
        a, b = sorted(np.random.choice(n, size=2, replace=False))
        order[a:b + 1] = order[a:b + 1][::-1]

    else:
        a, b = np.random.choice(n, size=2, replace=False)
        val = order[a]
        order = np.delete(order, a)
        order = np.insert(order, b, val)

    active_splits = clean_splits(splits, n)

    if active_splits and np.random.random() < 0.50:
        idx = np.random.randint(len(active_splits))
        shift = np.random.choice([-2, -1, 1, 2])
        active_splits[idx] = int(np.clip(active_splits[idx] + shift, 1, n - 1))
        active_splits = sorted(set(active_splits))

    if len(active_splits) < m - 1 and np.random.random() < 0.05:
        candidate = np.random.randint(1, n)

        if candidate not in active_splits:
            active_splits.append(candidate)
            active_splits = sorted(active_splits)

    if active_splits and np.random.random() < 0.05:
        idx = np.random.randint(len(active_splits))
        active_splits.pop(idx)

    new_splits = active_splits[:m - 1]

    while len(new_splits) < m - 1:
        new_splits.append(-1)

    # No need to mutate vehicle type genes: they are greedily optimized during evaluation.

    x_new = np.concatenate([
        order,
        np.array(new_splits, dtype=int),
        type_gene,
    ])

    return VRPRepair()._do(problem, x_new.reshape(1, -1))[0]


class VRPSampling(Sampling):

    def __init__(self, seed_solutions=None, seed_chromosomes=None, random_fraction=0.10):
        super().__init__()
        self.seed_solutions = seed_solutions or []
        self.seed_chromosomes = seed_chromosomes or []
        self.random_fraction = random_fraction

    def _do(self, problem, n_samples, **kwargs):
        pop = []
        seed_chromosomes = []

        for seed_x in self.seed_chromosomes:
            try:
                seed_chromosomes.append(
                    VRPRepair()._do(problem, np.asarray(seed_x).reshape(1, -1))[0]
                )
            except Exception as e:
                print("Cached chromosome seed skipped:", type(e).__name__, e)

        for seed_solution in self.seed_solutions:
            try:
                seed_x = solution_to_chromosome(seed_solution, problem)
                seed_chromosomes.append(seed_x)
            except Exception as e:
                print("Seed solution skipped:", type(e).__name__, e)

        pop.extend(seed_chromosomes)

        target_seed_count = int(n_samples * (1.0 - self.random_fraction))

        while len(pop) < target_seed_count and seed_chromosomes:
            seed = seed_chromosomes[np.random.randint(len(seed_chromosomes))]
            pop.append(make_seed_variant(seed, problem))

        while len(pop) < n_samples:
            pop.append(create_random_individual(problem))

        pop = np.array(pop[:n_samples])
        return VRPRepair()._do(problem, pop)


# ============================================================
# CROSSOVER
# ============================================================

def order_crossover(p1, p2):
    n = len(p1)

    a, b = sorted(np.random.choice(n, size=2, replace=False))

    child = np.full(n, -1, dtype=int)
    child[a:b + 1] = p1[a:b + 1]

    fill = [v for v in p2 if v not in child]

    idx = 0

    for i in range(n):
        if child[i] == -1:
            child[i] = fill[idx]
            idx += 1

    return child


class VRPCrossover(Crossover):

    def __init__(self, prob=0.9):
        super().__init__(2, 2)
        self.prob = prob

    def _do(self, problem, X, **kwargs):
        n_matings = X.shape[1]
        Y = np.full((2, n_matings, problem.n_var), -1, dtype=float)

        n = problem.n_customers
        m = problem.max_routes

        for k in range(n_matings):
            p1 = np.round(X[0, k]).astype(int)
            p2 = np.round(X[1, k]).astype(int)

            if np.random.random() > self.prob:
                Y[0, k] = p1
                Y[1, k] = p2
                continue

            p1_order = repair_permutation(p1[:n], n)
            p2_order = repair_permutation(p2[:n], n)

            p1_splits = p1[n:n + m - 1]
            p2_splits = p2[n:n + m - 1]

            p1_types = np.clip(p1[n + m - 1:n + m - 1 + m], 0, 1)
            p2_types = np.clip(p2[n + m - 1:n + m - 1 + m], 0, 1)

            c1_order = order_crossover(p1_order, p2_order)
            c2_order = order_crossover(p2_order, p1_order)

            c1_splits = p1_splits.copy() if np.random.random() < 0.5 else p2_splits.copy()
            c2_splits = p2_splits.copy() if np.random.random() < 0.5 else p1_splits.copy()

            mask_types = np.random.random(m) < 0.5
            c1_types = np.where(mask_types, p1_types, p2_types)
            c2_types = np.where(mask_types, p2_types, p1_types)

            Y[0, k] = np.concatenate([c1_order, c1_splits, c1_types])
            Y[1, k] = np.concatenate([c2_order, c2_splits, c2_types])

        return VRPRepair()._do(problem, Y.reshape(-1, problem.n_var)).reshape(
            2, n_matings, problem.n_var
        )


# ============================================================
# MUTATION
# ============================================================

class VRPMutation(Mutation):

    def __init__(
        self,
        prob=0.45,
        swap_prob=0.20,
        inversion_prob=0.25,
        insertion_prob=0.20,
        split_move_prob=0.20,
        split_add_prob=0.02,
        split_remove_prob=0.02,
        type_prob=0.08,
    ):
        super().__init__()

        self.prob = prob
        self.swap_prob = swap_prob
        self.inversion_prob = inversion_prob
        self.insertion_prob = insertion_prob
        self.split_move_prob = split_move_prob
        self.split_add_prob = split_add_prob
        self.split_remove_prob = split_remove_prob
        self.type_prob = type_prob

    def _do(self, problem, X, **kwargs):
        Y = X.copy()

        n = problem.n_customers
        m = problem.max_routes

        for i in range(len(Y)):
            if np.random.random() > self.prob:
                continue

            x = np.round(Y[i]).astype(int)

            order = x[:n].copy()
            splits = x[n:n + m - 1].copy()
            type_gene = x[n + m - 1:n + m - 1 + m].copy()

            if np.random.random() < self.swap_prob:
                a, b = np.random.choice(n, size=2, replace=False)
                order[a], order[b] = order[b], order[a]

            if np.random.random() < self.inversion_prob:
                a, b = sorted(np.random.choice(n, size=2, replace=False))
                order[a:b + 1] = order[a:b + 1][::-1]

            if np.random.random() < self.insertion_prob:
                a, b = np.random.choice(n, size=2, replace=False)
                val = order[a]
                order = np.delete(order, a)
                order = np.insert(order, b, val)

            active_splits = clean_splits(splits, n)

            if active_splits and np.random.random() < self.split_move_prob:
                idx = np.random.randint(len(active_splits))
                shift = np.random.choice([-3, -2, -1, 1, 2, 3])
                active_splits[idx] = int(np.clip(active_splits[idx] + shift, 1, n - 1))
                active_splits = sorted(set(active_splits))

            if len(active_splits) < m - 1 and np.random.random() < self.split_add_prob:
                candidate = np.random.randint(1, n)

                if candidate not in active_splits:
                    active_splits.append(candidate)
                    active_splits = sorted(active_splits)

            if active_splits and np.random.random() < self.split_remove_prob:
                idx = np.random.randint(len(active_splits))
                active_splits.pop(idx)

            new_splits = active_splits[:m - 1]

            while len(new_splits) < m - 1:
                new_splits.append(-1)

            for j in range(m):
                if np.random.random() < self.type_prob:
                    type_gene[j] = 1 - int(np.clip(type_gene[j], 0, 1))

            Y[i] = np.concatenate([
                order,
                np.array(new_splits, dtype=int),
                type_gene,
            ])

        return VRPRepair()._do(problem, Y)


# ============================================================
# FINAL METRICS / DEBUG
# ============================================================

def chromosome_to_metrics(x, problem):
    routes, _ = decode_chromosome(x, problem)

    if len(routes) == 0 or len(routes) > problem.max_routes:
        return None

    assignments = assign_best_vehicle_types(routes, problem)

    if assignments is None:
        return None

    if not is_capacity_feasible(assignments, problem.vrp_data, problem.Q_by_type):
        return None

    if not is_fleet_feasible(assignments, problem.n_diesel, problem.n_clean):
        return None

    distance = compute_distance(routes, problem.vrp_data)

    co2 = obj2.true_co2(
        assignments,
        problem.vrp_data,
        TYPE_LABELS,
        problem.Q_by_type,
        problem.f0,
        problem.alpha,
        problem.beta,
        problem.gamma,
        problem.mult,
    )

    return distance, co2, routes, assignments


def print_solution_check(name, solution, problem):
    try:
        seed_x = solution_to_chromosome(solution, problem)
    except Exception as e:
        print(f"\n{name}: could not convert to chromosome")
        print(type(e).__name__, e)
        return

    metrics = chromosome_to_metrics(seed_x, problem)

    print(f"\n{name}")

    if metrics is None:
        print("Could not evaluate solution.")
        return

    distance, co2, routes, assignments = metrics

    print("Decoded routes:", len(routes))
    print("Decoded distance:", distance)
    print("Decoded CO2 after greedy type assignment:", co2)

    diesel_used = sum(1 for _, vt in assignments if vt == DIESEL)
    clean_used = sum(1 for _, vt in assignments if vt == CLEAN)

    print("Diesel used:", diesel_used)
    print("Clean used:", clean_used)

    for i, (route, vt_idx) in enumerate(assignments):
        vehicle_name = TYPE_LABELS[int(vt_idx)]
        demand = route_demand(route, problem.vrp_data)
        capacity = problem.Q_by_type[vehicle_name]

        print(
            f"Route {i + 1:2d}: "
            f"type={vehicle_name:6s}, "
            f"demand={demand:4d}, "
            f"capacity={capacity:4d}, "
            f"feasible={demand <= capacity}"
        )


# ============================================================
# MAIN
# ============================================================

def main():
    np.random.seed(SEED)

    data = read(INSTANCE_PATH, round_func="round")

    n_customers = len(data.clients())
    max_routes = MAX_ROUTES

    f0, alpha, beta, gamma = 0.30, 0.20, 1.0, 2.61

    mult = {
        "diesel": 1.00,
        "clean": 0.45,
    }

    base_cap = Model.from_data(data).vehicle_types[0].capacity[0]

    Q_by_type = {
        "diesel": base_cap,
        "clean": int(0.9 * base_cap),
    }

    print("Instance:", INSTANCE_PATH)
    print("Customers:", n_customers)
    print("Diesel capacity:", Q_by_type["diesel"])
    print("Clean capacity:", Q_by_type["clean"])
    print("Diesel vehicles:", N_DIESEL)
    print("Clean vehicles:", N_CLEAN)

    problem = VRPProblem(
        data=data,
        n_customers=n_customers,
        max_routes=max_routes,
        Q_by_type=Q_by_type,
        params=(f0, alpha, beta, gamma, mult),
        n_diesel=N_DIESEL,
        n_clean=N_CLEAN,
    )

    seed_solutions = []

    try:
        res_seed_dist, _ = obj1.obj1(
            data,
            f0,
            alpha,
            beta,
            gamma,
            mult,
            0.5,
            10,
            10,
            Q_by_type,
        )

        seed_solutions.append(res_seed_dist.best)
        print_solution_check("Distance seed", res_seed_dist.best, problem)

    except Exception as e:
        print("Distance seed skipped:", type(e).__name__, e)

    try:
        res_seed_co2, _ = obj2.obj2(
            data,
            f0,
            alpha,
            beta,
            gamma,
            mult,
            0.5,
            10,
            10,
            Q_by_type,
        )

        seed_solutions.append(res_seed_co2.best)
        print_solution_check("CO2 seed", res_seed_co2.best, problem)

    except Exception as e:
        print("CO2 seed skipped:", type(e).__name__, e)


    weighted_seed_chromosomes = []
    weighted_reference_points = []
    weighted_reference_front = []

    if USE_WEIGHTED_SUM_SEEDS:
        (
            weighted_seed_chromosomes,
            weighted_reference_points,
            weighted_reference_front,
        ) = make_weighted_sum_seed_chromosomes_cached(problem)

    endpoint_chromosomes, endpoint_points, _ = (
        seed_solutions_to_chromosomes_and_points(seed_solutions, problem)
    )

    seed_chromosomes = endpoint_chromosomes + list(weighted_seed_chromosomes)
    reference_points = endpoint_points + weighted_reference_points
    reference_front = pareto_filter(reference_points)

    problem.set_normalization_bounds(reference_front if reference_front else reference_points)

    print("\nPyVRP weighted-sum reference front, evaluated with true objectives:")
    for p in reference_front:
        print(p)
    print("Reference nondominated points:", len(reference_front))

    algorithm = NSGA2(
        pop_size=POP_SIZE,
        sampling=VRPSampling(
            seed_solutions=seed_solutions,
            seed_chromosomes=seed_chromosomes,
            random_fraction=RANDOM_FRACTION,
        ),
        crossover=VRPCrossover(prob=0.9),
        mutation=VRPMutation(
            prob=0.25,
            swap_prob=0.10,
            inversion_prob=0.15,
            insertion_prob=0.35,
            split_move_prob=0.30,
            split_add_prob=0.005,
            split_remove_prob=0.005,
            type_prob=0.00,
        ),
        repair=VRPRepair(),
        eliminate_duplicates=True,
    )

    res = minimize(
        problem,
        algorithm,
        ("n_gen", N_GEN),
        seed=0,
        verbose=True,
    )

    points = []
    detailed = []

    for x in res.X:
        metrics = chromosome_to_metrics(x, problem)

        if metrics is None:
            continue

        distance, co2, routes, assignments = metrics

        points.append((distance, co2))
        detailed.append((distance, co2, routes, assignments))

    front = pareto_filter(points)

    print("\nFinal Pareto front:")
    for p in front:
        print(p)

    print("\nNumber of nondominated points:", len(front))

    if reference_front:
        print("\nDistance of NSGA-II front points to nearest PyVRP reference point:")
        ref = np.array(reference_front, dtype=float)
        for p in front:
            pn = np.array(problem.normalize_objectives(p[0], p[1]), dtype=float)
            refn = np.array([problem.normalize_objectives(r[0], r[1]) for r in ref])
            nearest = float(np.min(np.linalg.norm(refn - pn, axis=1)))
            print(f"point={p}, nearest_reference_distance={nearest:.6f}")

    print("\nDetailed nondominated solutions:")
    for dist, co2 in front:
        for d in detailed:
            if d[0] == dist and d[1] == co2:
                _, _, routes, assignments = d
                diesel_used = sum(1 for _, vt in assignments if vt == DIESEL)
                clean_used = sum(1 for _, vt in assignments if vt == CLEAN)

                print(
                    f"distance={dist}, co2={co2}, "
                    f"routes={len(routes)}, diesel={diesel_used}, clean={clean_used}"
                )
                break


if __name__ == "__main__":
    main()