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
    Builds routes in permutation order while respecting the remaining fleet.
    A route is extended only as long as at least one still-available vehicle
    type can serve its load. When that is no longer true, the current route is
    closed and assigned to the greenest feasible available vehicle.
    """
    if len(perm) == 0:
        return []

    fleet = inst.fleet
    n_types = len(fleet)

    # Greener first; tie-break by smaller capacity first.
    fleet_order = sorted(
        range(n_types),
        key=lambda k: (fleet[k].co2_multiplier, fleet[k].capacity)
    )

    vehicles_used = [0] * n_types
    routes = []

    # def route_load(route):
    #     return sum(inst.demands[cust - 1] for cust in route)

    def feasible_vehicle_types(load):
        """
        Returns vehicle type indices that:
        - still have availability
        - can carry the given load
        Ordered from greenest to least green.
        """
        feasible = []
        for k in fleet_order:
            vt = fleet[k]
            if vehicles_used[k] < vt.max_vehicles and load <= vt.capacity:
                feasible.append(k)
        return feasible

    def assign_vehicle(load):
        """
        Assign greenest feasible available vehicle to a route load.
        Returns vehicle type index or None if no feasible type exists.
        """
        feasible = feasible_vehicle_types(load)
        return feasible[0] if feasible else None

    current_route = []
    current_load = 0.0

    for cust in perm:
        cust = int(cust)
        demand = float(inst.demands[cust - 1])

        # Candidate load if we append this customer.
        new_load = current_load + demand

        # Can some remaining vehicle still serve this enlarged route?
        if feasible_vehicle_types(new_load):
            current_route.append(cust)
            current_load = new_load
            continue

        # Otherwise, close the current route first.
        if not current_route:
            # Single customer already infeasible for all remaining vehicles.
            # Assign fallback to largest-capacity vehicle type and let penalties
            # handle infeasibility later.
            fallback = int(np.argmax([vt.capacity for vt in fleet]))
            vehicles_used[fallback] += 1
            routes.append({
                "vehicle_type": fallback,
                "customers": [cust],
            })
            current_route = []
            current_load = 0.0
            continue

        assigned_type = assign_vehicle(current_load)

        if assigned_type is None:
            # Should be rare: the route was feasible earlier, but fleet got
            # exhausted in a way that leaves no legal assignment now.
            fallback = int(np.argmax([vt.capacity for vt in fleet]))
            assigned_type = fallback

        vehicles_used[assigned_type] += 1
        routes.append({
            "vehicle_type": assigned_type,
            "customers": current_route,
        })

        # Start a new route with current customer.
        current_route = [cust]
        current_load = demand

    # Close the final route.
    if current_route:
        assigned_type = assign_vehicle(current_load)

        if assigned_type is None:
            fallback = int(np.argmax([vt.capacity for vt in fleet]))
            assigned_type = fallback

        vehicles_used[assigned_type] += 1
        routes.append({
            "vehicle_type": assigned_type,
            "customers": current_route,
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
        n_gen=300,
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
plt.ylabel("Total CO2")
plt.title("NSGA-II Pareto front")
plt.show()

