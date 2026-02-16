import numpy as np
import matplotlib.pyplot as plt

from pyvrp import read, Model
from pyvrp.stop import MaxRuntime

INSTANCE_PATH = "X-n106-k14.vrp"
RUNTIME_SECONDS = 30     
SEED = 1

VEH_PENALTY_MULT = 3.0
SCALE = 1000              

# w1 = total distance weight for obj1=total distance
# w2 = weight for obj2=max distance for one vehicle
weights = [(1 - w, w) for w in np.linspace(0.05, 0.95, 10)]


def solve_with_costs(data, fixed_cost, unit_distance_cost):
    """Builds a model, sets vehicle costs, solves, returns best solution"""
    model = Model.from_data(data)

    vt0 = model.vehicle_types[0]
    model.vehicle_types[0] = vt0.replace(
        fixed_cost=int(fixed_cost),
        unit_distance_cost=int(unit_distance_cost),
    )

    res = model.solve(
        stop=MaxRuntime(RUNTIME_SECONDS),
        seed=SEED,
        collect_stats=False,
        display=False,
    )
    return res.best


def metrics_from_solution(sol):
    """Returns (total_distance, max_route_distance, vehicles_used)"""
    total_dist = sol.distance()
    routes = sol.routes()
    vehicles = len(routes)

    if vehicles == 0:
        return total_dist, 0.0, 0

    route_dists = np.array([r.distance() for r in routes], dtype=float)
    max_route_dist = float(route_dists.max())
    return total_dist, max_route_dist, vehicles


data = read(INSTANCE_PATH, round_func="round")

sol0 = solve_with_costs(data, fixed_cost=0, unit_distance_cost=1)
base_total, base_maxr, base_veh = metrics_from_solution(sol0)

distance_scale = base_total / max(1, base_veh)

print("Baseline (distance-only)")
print("->total distance:", base_total)
print("->max route dist:", base_maxr)
print("->vehicles used :", base_veh)
print("->distance_scale:", distance_scale)


records = []  # (w1, w2, vehicles, total_dist, max_route_dist)

for w1, w2 in weights:
    unit_distance_cost = max(1, int(round(SCALE * w1)))

    fixed_cost = int(round(SCALE * w2 * VEH_PENALTY_MULT * distance_scale))

    sol = solve_with_costs(data, fixed_cost=fixed_cost, unit_distance_cost=unit_distance_cost)

    total_dist, max_route_dist, vehicles = metrics_from_solution(sol)
    records.append((w1, w2, vehicles, total_dist, max_route_dist))

    print(f"w1={w1:.2f}, w2={w2:.2f} -> veh={vehicles:3d}, "
          f"total={total_dist:6d}, max_route={max_route_dist:7.0f}")

# plt.figure()
# xs = [r[3] for r in records]  # total distance
# ys = [r[4] for r in records]  # max route distance
# plt.scatter(xs, ys)

# plt.xlabel("obj1: Total distance by all vehicles")
# plt.ylabel("obj2: Max route distance by one vehicle")
# plt.title("Final solutions")
# plt.grid(True, linestyle="--", alpha=0.4)
# plt.show()

plt.figure()

green_x, green_y = [], []
blue_x, blue_y = [], []

for w1, w2, vehicles, total_dist, max_route_dist in records:
    if 0.0 <= w1 <= 0.5:
        green_x.append(total_dist)
        green_y.append(max_route_dist)
    else:
        blue_x.append(total_dist)
        blue_y.append(max_route_dist)


plt.scatter(blue_x, blue_y, color="blue", label="w1 > 0.5")
plt.scatter(green_x, green_y, color="green", label="0 ≤ w1 ≤ 0.5")

plt.xlabel("obj1: Total distance by all vehicles")
plt.ylabel("obj2: Max route distance by one vehicle")
plt.title("Final solutions")
plt.legend()
plt.grid(True, linestyle="--", alpha=0.4)
plt.show(block=False)


def pareto_efficient(records):
    """
    Returns the subset of records that are Pareto-efficient for:
      obj1 = total_dist (min)
      obj2 = max_route_dist (min)
    """
    objs = np.array([[r[3], r[4]] for r in records], dtype=float)

    n = len(records)
    is_efficient = np.ones(n, dtype=bool)

    for i in range(n):
        if not is_efficient[i]:
            continue
        # A point j dominates i if:
        #   objs[j] <= objs[i] in both objectives AND strictly < in at least one
        dominates_i = np.all(objs <= objs[i], axis=1) & np.any(objs < objs[i], axis=1)
        dominates_i[i] = False  # ignore self
        if np.any(dominates_i):
            is_efficient[i] = False

    pareto = [records[i] for i in range(n) if is_efficient[i]]
    return pareto

pareto_records = pareto_efficient(records)

print("\nPareto-efficient solutions:")
for w1, w2, veh, total_dist, max_route_dist in pareto_records:
    print(f"w1={w1:.2f}, w2={w2:.2f} -> veh={veh:3d}, "
          f"total={total_dist:6d}, max_route={max_route_dist:7.0f}")

plt.figure()

all_x = [r[3] for r in records]
all_y = [r[4] for r in records]
plt.scatter(all_x, all_y, alpha=0.3, label="All solutions")

px = [r[3] for r in pareto_records]
py = [r[4] for r in pareto_records]
plt.scatter(px, py, color="red", label="Pareto-efficient")

plt.xlabel("obj1: Total distance by all vehicles")
plt.ylabel("obj2: Max route distance by one vehicle")
plt.title("Pareto-efficient solutions")
plt.legend()
plt.grid(True, linestyle="--", alpha=0.4)
plt.show()
