from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Any
import numpy as np
from pyvrp import read
from pymoo.core.problem import ElementwiseProblem
from pymoo.core.mutation import Mutation
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.operators.sampling.rnd import PermutationRandomSampling
from pymoo.operators.crossover.ox import OrderCrossover
from pymoo.optimize import minimize

# classes for clean code
@dataclass(frozen=True)
class VehicleType:
    name: str # deisel or clean
    capacity: float # how much load it can carry
    max_vehicles: int # how many vehicles of this type are available
    co2_multiplier: float  # diesel=1.0, clean=0.45

@dataclass
class GreenInstance:
    distance_matrix: np.ndarray      # full matrix including depot
    demands: np.ndarray              # customers only; shape (n,)
    depot_index: int                 # 0
    fleet: List[VehicleType]
    f0: float = 0.30                # base fuel consumption per unit distance
    alpha: float = 0.20
    beta: float = 1.0
    gamma: float = 2.61             # kg CO2/liter

def _extract_customer_demands(data) -> np.ndarray:
    """
    Extract client delivery demands.
    """
    demands = []

    for client in data.clients():
        delivery = getattr(client, "delivery", 0)

        if isinstance(delivery, (list, tuple, np.ndarray)):
            demand = float(np.sum(delivery))
        else:
            demand = float(delivery)

        demands.append(demand)

    return np.asarray(demands, dtype=float)

# data = read("instances/vrp/X-n106-k14.vrp")
# demands = _extract_customer_demands(data)
# print("Num customers:", len(demands))
# print("First 10 demands:", demands[:10])
# print("Total demand:", demands.sum())
# D = data.distance_matrix(0)
# print("Shape:", D.shape)
# print("First row:", D[0][:10])
# print("Distance depot → customer 1:", D[0, 1])
# print("Distance customer 1 → customer 2:", D[1, 2])

def create_green_instance(vrp_path: str | Path, fleet: List[VehicleType], profile: int = 0,) -> GreenInstance:
    """
    Creates a GreenInstance from a VRP file, given a fleet configuration.
    """
    data = read(vrp_path)

    D = np.asarray(data.distance_matrix(profile), dtype=float)
    demands = _extract_customer_demands(data)

    return GreenInstance(distance_matrix=D, demands=demands, depot_index=0, fleet=fleet,)

# Chromosome decoding
import numpy as np
from typing import List

def decode_permutation_to_routes(perm: np.ndarray, inst: GreenInstance):
    """
    Split-based decoder.
    """
    perm = np.asarray(perm, dtype=int)
    n = len(perm)

    if n == 0:
        return []

    D = inst.distance_matrix
    demands = inst.demands
    depot = inst.depot_index
    fleet = inst.fleet

    # Route CO2 for a specific vehicle type
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

    # feasible vehicle types for a given load
    def feasible_types_for_load(load: float) -> List[int]:
        return [k for k, vt in enumerate(fleet) if load <= vt.capacity]

    # split DP
    # V[j] = best split cost for first j customers in perm
    # P[j] = predecessor index of j in the shortest path
    V = np.full(n + 1, np.inf, dtype=float)
    P = np.full(n + 1, -1, dtype=int)

    V[0] = 0.0

    # Split using incremental route-cost update
    for i in range(1, n + 1):
        load = 0.0
        cost = 0.0

        for j in range(i, n + 1):
            cust_j = int(perm[j - 1])
            load += float(demands[cust_j - 1])

            # Incremental distance update as in the split algorithm:
            # first customer: depot -> s_i -> depot
            # extension: remove previous last->depot, add prev->new and new->depot
            if i == j:
                cost = D[depot, cust_j] + D[cust_j, depot]
            else:
                cust_prev = int(perm[j - 2])
                cost = cost - D[cust_prev, depot] + D[cust_prev, cust_j] + D[cust_j, depot]

            # At least one vehicle type can carry the accumulated load
            if not feasible_types_for_load(load):
                break

            # Relax arc (i-1, j)
            new_cost = V[i - 1] + cost
            if new_cost < V[j]:
                V[j] = new_cost
                P[j] = i - 1

    # Recover split from predecessor vector P
    segments = []
    j = n
    while j > 0:
        i = P[j]
        if i < 0:
            # Should not happen unless something is badly infeasible.
            # Fallback: one customer per route from the beginning.
            segments = [(k, k + 1) for k in range(n)]
            break

        segments.append((i, j))
        j = i

    segments.reverse()

    # Assign vehicle types to the decoded routes
    vehicles_used = [0] * len(fleet)
    routes = []

    for i, j in segments:
        customers = perm[i:j].tolist()
        load = float(sum(demands[c - 1] for c in customers))

        feasible = feasible_types_for_load(load)

        available_feasible = [
            k for k in feasible
            if vehicles_used[k] < fleet[k].max_vehicles
        ]

        if available_feasible:
            assigned_type = min(
                available_feasible,
                key=lambda k: route_co2_with_type(customers, k)
            )
        elif feasible:
            # Feasible by capacity, but fleet count exhausted.
            assigned_type = min(
                feasible,
                key=lambda k: route_co2_with_type(customers, k)
            )
        else:
            # No type can carry the route: fallback to largest vehicle.
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
    """
    Load-dependent CO2:
        f_ij^k = f0 * (1 + alpha * (load_ratio ** beta)) * mult_k
        CO2 = sum d_ij * gamma * f_ij^k

    """
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

    # constraint violations g(x) <= 0
    used = [0] * len(inst.fleet)
    for r in routes:
        used[r["vehicle_type"]] += 1

    g_fleet = [
        used[k] - inst.fleet[k].max_vehicles
        for k in range(len(inst.fleet))
    ]

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
    # the same as in the paper
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

class GreenVRPProblem(ElementwiseProblem):

    def __init__(self, inst: GreenInstance):
        self.inst = inst
        self.n_customers = len(inst.demands)

        # permutation of customer ids 1..n
        super().__init__(
            n_var=self.n_customers,
            n_obj=2,
            n_ieq_constr=len(inst.fleet) + 2,
            xl=0,
            xu=self.n_customers - 1,
            vtype=int,
        )

    def _evaluate(self, x, out, *args, **kwargs):
        perm = np.asarray(x, dtype=int) + 1

        routes = decode_permutation_to_routes(perm, self.inst)
        total_distance, total_co2, G = evaluate_routes(routes, self.inst)

        out["F"] = [total_distance, total_co2]
        out["G"] = G

def solve_green_nsga2(
    vrp_path: str | Path,
    diesel_capacity: float,
    diesel_count: int,
    clean_capacity: float,
    clean_count: int,
    clean_multiplier: float = 0.45,
    diesel_multiplier: float = 1.0,
    pop_size: int = 100,
    n_gen: int = 300,
    seed: int = 1,
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

    algorithm = NSGA2(
        pop_size=pop_size,
        sampling=PermutationRandomSampling(),
        crossover=OrderCrossover(prob=0.8),
        mutation=SwapMutation(prob=0.2),
        eliminate_duplicates=True,
    )

    res = minimize(
        problem,
        algorithm,
        termination=("n_gen", n_gen),
        seed=seed,
        verbose=True,
    )

    return inst, problem, res



def decode_solution(x: np.ndarray, inst: GreenInstance):
    routes = decode_permutation_to_routes(np.asarray(x, dtype=int) + 1, inst)
    dist, co2, G = evaluate_routes(routes, inst)
    return {
        "routes": routes,
        "distance": dist,
        "co2": co2,
        "constraints": G,
    }


# Example

if __name__ == "__main__":
    vrp_file = "instances/vrp/X-n106-k14.vrp"

    inst, problem, res = solve_green_nsga2(
        vrp_path=vrp_file,
        diesel_capacity=600,
        diesel_count=10,
        clean_capacity=540,
        clean_count=10,
        clean_multiplier=0.45,
        diesel_multiplier=1.0,
        pop_size=300,
        n_gen=500,
        seed=42,
    )

    print("\nPareto objective values [distance, co2]:")
    print(res.F)

    print("\nOne Pareto solution:")
    one = decode_solution(res.X[0], inst)
    print("Distance:", one["distance"])
    print("CO2:", one["co2"])
    print("Constraint violations:", one["constraints"])
    print("Routes:")
    for idx, r in enumerate(one["routes"], start=1):
        vt_name = inst.fleet[r["vehicle_type"]].name
        print(f"  Route {idx:02d} [{vt_name}]: {r['customers']}")

import matplotlib.pyplot as plt

F = res.F
plt.scatter(F[:, 0], F[:, 1])
plt.xlabel("Total distance")
plt.ylabel("CO$_2$ emissions")
plt.title(f"Distance vs CO$_2$ ({vrp_file.removesuffix(".vrp")})")
plt.show()

