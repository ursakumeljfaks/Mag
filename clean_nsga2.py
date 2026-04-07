from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
from pyvrp import Model, read
from pyvrp.stop import MaxRuntime
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.algorithms.moo.moead import MOEAD
from pymoo.core.mutation import Mutation
from pymoo.core.problem import ElementwiseProblem
from pymoo.core.sampling import Sampling
from pymoo.operators.crossover.ox import OrderCrossover
from pymoo.optimize import minimize



@dataclass(frozen=True)
class VehicleType:
    name: str
    capacity: float
    max_vehicles: int
    co2_multiplier: float


@dataclass
class GreenInstance:
    distance_matrix: np.ndarray
    demands: np.ndarray
    depot_index: int
    fleet: List[VehicleType]
    f0: float = 0.30
    alpha: float = 0.20
    beta: float = 1.0
    gamma: float = 2.61


def _extract_customer_demands(data) -> np.ndarray:
    demands = []

    for client in data.clients():
        delivery = getattr(client, "delivery", 0)

        if isinstance(delivery, (list, tuple, np.ndarray)):
            demand = float(np.sum(delivery))
        else:
            demand = float(delivery)

        demands.append(demand)

    return np.asarray(demands, dtype=float)


def create_green_instance(
    vrp_path: str | Path,
    fleet: List[VehicleType],
    profile: int = 0,) -> GreenInstance:
    
    data = read(vrp_path, round_func="round")
    D = np.asarray(data.distance_matrix(profile), dtype=float)
    demands = _extract_customer_demands(data)

    return GreenInstance(
        distance_matrix=D,
        demands=demands,
        depot_index=0,
        fleet=fleet,
    )


def build_model_two_types(data, n_diesel, n_clean, Q_diesel=0, Q_clean=0):
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


def pyvrp_solution_to_list_of_lists(sol) -> List[List[int]]:
    routes = []
    for r in sol.routes():
        routes.append(list(r.visits()))
    return routes


def pyvrp_solution_to_permutation(sol) -> np.ndarray:
    routes = pyvrp_solution_to_list_of_lists(sol)
    perm = [cust for route in routes for cust in route]
    return np.asarray(perm, dtype=int)


def _extract_best_pyvrp_solution(result):
    best = getattr(result, "best", None)

    if callable(best):
        best = best()

    if best is None:
        best = getattr(result, "solution", None)

    if callable(best):
        best = best()

    if best is None and hasattr(result, "routes"):
        best = result

    if best is None:
        raise RuntimeError("Could not extract a PyVRP solution from model.solve().")

    return best


def get_pyvrp_initial_permutation(
    vrp_path: str | Path,
    diesel_capacity: float,
    diesel_count: int,
    clean_capacity: float,
    clean_count: int,
    seed: int = 0,
    max_runtime: float = 1.0,) -> np.ndarray:

    data = read(vrp_path, round_func="round")
    model = build_model_two_types(
        data,
        n_diesel=diesel_count,
        n_clean=clean_count,
        Q_diesel=diesel_capacity,
        Q_clean=clean_capacity,
    )

    result = model.solve(stop=MaxRuntime(max_runtime), seed=seed)
    best = _extract_best_pyvrp_solution(result)
    perm_1_based = pyvrp_solution_to_permutation(best)

    if perm_1_based.size == 0:
        raise RuntimeError("PyVRP returned an empty initial solution.")

    return perm_1_based


def decode_permutation_to_routes(perm: np.ndarray, inst: GreenInstance):
    perm = np.asarray(perm, dtype=int)
    n = len(perm)

    if n == 0:
        return []

    D = inst.distance_matrix
    demands = inst.demands
    depot = inst.depot_index
    fleet = inst.fleet

    def route_co2_with_type(customers: List[int], vehicle_type_idx: int) -> float:
        vt = fleet[vehicle_type_idx]
        Qk = vt.capacity
        multk = vt.co2_multiplier

        seq = [depot] + customers + [depot]
        remaining_load = float(sum(demands[c - 1] for c in customers))
        total = 0.0

        for t in range(len(seq) - 1):
            a, b = seq[t], seq[t + 1]
            dist = D[a, b]

            load_ratio = remaining_load / Qk if Qk > 0 else 0.0
            fuel_per_dist = inst.f0 * (1.0 + inst.alpha * (load_ratio ** inst.beta)) * multk
            total += dist * inst.gamma * fuel_per_dist

            if b != depot:
                remaining_load -= demands[b - 1]

        return float(total)

    def feasible_types_for_load(load: float) -> List[int]:
        return [k for k, vt in enumerate(fleet) if load <= vt.capacity]

    V = np.full(n + 1, np.inf, dtype=float)
    P = np.full(n + 1, -1, dtype=int)
    V[0] = 0.0

    for i in range(1, n + 1):
        load = 0.0
        cost = 0.0

        for j in range(i, n + 1):
            cust_j = int(perm[j - 1])
            load += float(demands[cust_j - 1])

            if i == j:
                cost = D[depot, cust_j] + D[cust_j, depot]
            else:
                cust_prev = int(perm[j - 2])
                cost = cost - D[cust_prev, depot] + D[cust_prev, cust_j] + D[cust_j, depot]

            if not feasible_types_for_load(load):
                break

            new_cost = V[i - 1] + cost
            if new_cost < V[j]:
                V[j] = new_cost
                P[j] = i - 1

    segments = []
    j = n
    while j > 0:
        i = P[j]
        if i < 0:
            segments = [(k, k + 1) for k in range(n)]
            break

        segments.append((i, j))
        j = i

    segments.reverse()

    vehicles_used = [0] * len(fleet)
    routes = []

    for i, j in segments:
        customers = perm[i:j].tolist()
        load = float(sum(demands[c - 1] for c in customers))
        feasible = feasible_types_for_load(load)

        available_feasible = [
            k for k in feasible if vehicles_used[k] < fleet[k].max_vehicles
        ]

        if available_feasible:
            assigned_type = min(
                available_feasible,
                key=lambda k: route_co2_with_type(customers, k),
            )
        elif feasible:
            assigned_type = min(
                feasible,
                key=lambda k: route_co2_with_type(customers, k),
            )
        else:
            assigned_type = int(np.argmax([vt.capacity for vt in fleet]))

        vehicles_used[assigned_type] += 1
        routes.append({
            "vehicle_type": assigned_type,
            "customers": customers,
        })

    return routes


def route_distance(route: Dict[str, Any], inst: GreenInstance) -> float:
    seq = [inst.depot_index] + route["customers"] + [inst.depot_index]
    D = inst.distance_matrix
    return float(sum(D[seq[i], seq[i + 1]] for i in range(len(seq) - 1)))


def route_co2(route: Dict[str, Any], inst: GreenInstance) -> float:
    vt = inst.fleet[route["vehicle_type"]]
    Qk = vt.capacity
    multk = vt.co2_multiplier

    customers = route["customers"]
    seq = [inst.depot_index] + customers + [inst.depot_index]
    D = inst.distance_matrix

    remaining_load = float(sum(inst.demands[c - 1] for c in customers))
    total = 0.0

    for i in range(len(seq) - 1):
        a, b = seq[i], seq[i + 1]
        d = D[a, b]

        load_ratio = remaining_load / Qk if Qk > 0 else 0.0
        fuel_per_dist = inst.f0 * (1.0 + inst.alpha * (load_ratio ** inst.beta)) * multk
        total += d * inst.gamma * fuel_per_dist

        if b != inst.depot_index:
            remaining_load -= inst.demands[b - 1]

    return float(total)


def evaluate_routes(routes: List[Dict[str, Any]], inst: GreenInstance):
    total_distance = sum(route_distance(r, inst) for r in routes)
    total_co2 = sum(route_co2(r, inst) for r in routes)

    used = [0] * len(inst.fleet)
    for r in routes:
        used[r["vehicle_type"]] += 1

    g_fleet = [used[k] - inst.fleet[k].max_vehicles for k in range(len(inst.fleet))]

    cap_violation = 0.0
    served = []
    for r in routes:
        load = sum(inst.demands[c - 1] for c in r["customers"])
        cap = inst.fleet[r["vehicle_type"]].capacity
        cap_violation += max(0.0, load - cap)
        served.extend(r["customers"])

    n = len(inst.demands)
    unique_served = len(set(served))
    duplicate_count = max(0, len(served) - unique_served)
    missed_count = max(0, n - unique_served)
    g_service = duplicate_count + missed_count

    G = g_fleet + [cap_violation, g_service]
    return total_distance, total_co2, G


class SwapMutation(Mutation):
    def __init__(self, prob: float = 0.2):
        super().__init__()
        self.prob = prob

    def _do(self, problem, X, **kwargs):
        Y = X.copy()

        for i in range(len(Y)):
            if np.random.random() < self.prob:
                a, b = np.random.choice(problem.n_var, size=2, replace=False)
                Y[i, a], Y[i, b] = Y[i, b], Y[i, a]

        return Y


class PyVRPInitialSampling(Sampling):
    def __init__(self, base_perm_1_based: np.ndarray, seed: int = 0):
        super().__init__()
        self.base_perm_0_based = np.asarray(base_perm_1_based, dtype=int) - 1
        self.seed = seed

    def _do(self, problem, n_samples, **kwargs):
        rng = np.random.default_rng(self.seed)
        X = np.empty((n_samples, problem.n_var), dtype=int)

        X[0] = self.base_perm_0_based.copy()

        for i in range(1, n_samples):
            perm = self.base_perm_0_based.copy()
            n_swaps = max(1, problem.n_var // 20)

            for _ in range(n_swaps):
                a, b = rng.choice(problem.n_var, size=2, replace=False)
                perm[a], perm[b] = perm[b], perm[a]

            X[i] = perm

        return X


class GreenVRPProblem(ElementwiseProblem):
    def __init__(self, inst: GreenInstance):
        self.inst = inst
        self.n_customers = len(inst.demands)

        super().__init__(
            n_var=self.n_customers,
            n_obj=2,
            #n_ieq_constr=len(inst.fleet) + 2, # for nsga2 uncomment this and comment th enext line
            n_ieq_constr=0,  # <--- Change this to 0 for MOEA/D
            xl=0,
            xu=self.n_customers - 1,
            vtype=int,
        )

    def _evaluate(self, x, out, *args, **kwargs):
        perm = np.asarray(x, dtype=int) + 1
        routes = decode_permutation_to_routes(perm, self.inst)
        total_distance, total_co2, G = evaluate_routes(routes, self.inst)

        # out["F"] = [total_distance, total_co2]
        # out["G"] = G

        ## COMMENT this and uncomment above two lines to use nsga2
        # --- PENALTY LOGIC ---
        # Sum up all positive violations (where used > max_vehicles, etc.)
        total_violation = sum(max(0, g) for g in G)
        
        if total_violation > 0:
            # We add a "Big M" penalty (10^6) to make infeasible solutions undesirable
            # This forces MOEA/D to find feasible routes
            penalty_value = 1e6 + (total_violation * 1000)
            f1 = total_distance + penalty_value
            f2 = total_co2 + penalty_value
        else:
            f1 = total_distance
            f2 = total_co2

        # IMPORTANT: Only return F. Do NOT set out["G"]
        out["F"] = [f1, f2]


def solve_green_nsga2(
    vrp_path: str | Path,
    diesel_capacity: float,
    diesel_count: int,
    clean_capacity: float,
    clean_count: int,
    clean_multiplier: float = 0.45,
    diesel_multiplier: float = 1.0,
    pop_size: int = 300,
    n_gen: int = 300,
    seed: int = 0,
    pyvrp_runtime: float = 2.0,
):
    fleet = [
        VehicleType(
            name="diesel",
            capacity=diesel_capacity,
            max_vehicles=diesel_count,
            co2_multiplier=diesel_multiplier,
        ),
        VehicleType(
            name="clean",
            capacity=clean_capacity,
            max_vehicles=clean_count,
            co2_multiplier=clean_multiplier,
        ),
    ]

    inst = create_green_instance(vrp_path, fleet=fleet)
    problem = GreenVRPProblem(inst)

    initial_perm = get_pyvrp_initial_permutation(
        vrp_path=vrp_path,
        diesel_capacity=diesel_capacity,
        diesel_count=diesel_count,
        clean_capacity=clean_capacity,
        clean_count=clean_count,
        seed=seed,
        max_runtime=pyvrp_runtime,
    )

    algorithm = NSGA2(
        pop_size=pop_size,
        sampling=PyVRPInitialSampling(initial_perm, seed=seed),
        crossover=OrderCrossover(prob=0.8),
        mutation=SwapMutation(prob=0.2),
        #mutation=Mutation(prob=0.2),
        eliminate_duplicates=True,
    )

    res = minimize(
        problem,
        algorithm,
        termination=("n_gen", n_gen),
        seed=seed,
        verbose=True,
    )

    return inst, problem, res, initial_perm


def decode_solution(x: np.ndarray, inst: GreenInstance):
    routes = decode_permutation_to_routes(np.asarray(x, dtype=int) + 1, inst)
    dist, co2, G = evaluate_routes(routes, inst)
    return {
        "routes": routes,
        "distance": dist,
        "co2": co2,
        "constraints": G,
    }

#### MOEAD
from pymoo.util.ref_dirs import get_reference_directions
from pymoo.algorithms.moo.moead import MOEAD

def solve_green_moead(
    vrp_path: str | Path,
    diesel_capacity: float,
    diesel_count: int,
    clean_capacity: float,
    clean_count: int,
    clean_multiplier: float = 0.45,
    diesel_multiplier: float = 1.0,
    # For 2 objectives, n_partitions + 1 = Population Size
    n_partitions: int = 99,   # 2 objectives => 100 reference directions
    n_gen: int = 300,
    seed: int = 0,
    pyvrp_runtime: float = 2.0,):
    
    fleet = [
        VehicleType("diesel", diesel_capacity, diesel_count, diesel_multiplier),
        VehicleType("clean", clean_capacity, clean_count, clean_multiplier),
    ]
    inst = create_green_instance(vrp_path, fleet=fleet)
    problem = GreenVRPProblem(inst)

    initial_perm = get_pyvrp_initial_permutation(
        vrp_path, diesel_capacity, diesel_count, clean_capacity, clean_count, seed, pyvrp_runtime
    )

    # 1. Generate Reference Directions
    # "das-dennis" with 2 objectives and n_partitions=99 creates 100 subproblems
    ref_dirs = get_reference_directions("das-dennis", 2, n_partitions=n_partitions)

    algorithm = MOEAD(
        ref_dirs,
        n_neighbors=10,  # Size of the weight-vector neighborhood, 10% of the pop_size/n_partitions
        prob_neighbor_mating=0.4,#0.7,   # Probability of mating within neighborhood
        sampling=PyVRPInitialSampling(initial_perm, seed=seed),
        crossover=OrderCrossover(prob=0.8),
        mutation=SwapMutation(prob=0.5),
    )

    res = minimize(
        problem,
        algorithm,
        termination=("n_gen", n_gen),
        seed=seed,
        verbose=True,
    )

    return inst, problem, res, initial_perm


if __name__ == "__main__":
    vrp_file = "instances/vrp/X-n106-k14.vrp"

    # inst, problem, res, initial_perm = solve_green_nsga2(
    #     vrp_path=vrp_file,
    #     diesel_capacity=600,
    #     diesel_count=10,
    #     clean_capacity=int(600 * 0.9),
    #     clean_count=10,
    #     clean_multiplier=0.45,
    #     diesel_multiplier=1.0,
    #     pop_size=5000,
    #     n_gen=5000,
    #     seed=0,
    #     pyvrp_runtime=2.0,
    # )
    inst, problem, res, initial_perm = solve_green_moead(
        vrp_path=vrp_file,
        diesel_capacity=600,
        diesel_count=10,
        clean_capacity=int(600 * 0.9),
        clean_count=10,
        clean_multiplier=0.45,
        diesel_multiplier=1.0,
        n_partitions=699,  # 2 objectives => 100 reference directions
        n_gen=700,
        seed=0,
        pyvrp_runtime=2.0,
    )

    print("\nInitial PyVRP customer order:")
    print(initial_perm)

    #np.set_printoptions(threshold=np.inf)
    print("\nPareto objective values [distance, co2]:")
    print(res.F)

    # print(f"\nAll {len(res.X)} Pareto solutions:")
    # for sol_idx, x in enumerate(res.X, start=1):
    #     solution = decode_solution(x, inst)
    #     print(f"\n--- Solution {sol_idx} ---")
    #     print("Distance:", solution["distance"])
    #     print("CO2:", solution["co2"])
    #     print("Constraint violations:", solution["constraints"])
    #     print("Routes:")
    #     for idx, r in enumerate(solution["routes"], start=1):
    #         vt_name = inst.fleet[r["vehicle_type"]].name
    #         print(f"  Route {idx:02d} [{vt_name}]: {r['customers']}")
