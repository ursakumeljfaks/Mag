import numpy as np
import matplotlib.pyplot as plt

from pyvrp import read, Model
from pyvrp.stop import MaxRuntime


# pre-req. from pdf
f0 = 0.30
alpha = 0.20
beta = 1.0
gamma = 2.61
mult = {"diesel": 1.00, "clean": 0.45}
l_nom = 0.50

# Weighted-Sum Scalarization and Surrogate Objective
fbar = {k: f0 * (1 + alpha * (l_nom ** beta)) * mult[k] for k in mult}
co2_per_dist = {k: gamma * fbar[k] for k in mult}  # kg CO2 per unit distance

INSTANCE_PATH = "instances/vrp/X-n115-k10.vrp"
RUNTIME_SECONDS = 5
SCALE = 1000

weights = [(1-w, w) for w in np.linspace(0.05, 0.95, 300)] 

data = read(INSTANCE_PATH, round_func="round")


def build_model_two_types(data, n_diesel, n_clean, Q_diesel=0, Q_clean=0):
    model = Model.from_data(data)

    # first type of vehicle is diesel
    vt0 = model.vehicle_types[0]
    model.vehicle_types[0] = vt0.replace(num_available=n_diesel, capacity=[Q_diesel], name="diesel")

    # second type is clean
    model.add_vehicle_type(num_available=n_clean, capacity=Q_clean, name="clean")

    return model


def solve_weighted_surrogate(data, w_dist, w_co2, n_diesel, n_clean, Q_diesel, Q_clean):
    model = build_model_two_types(data, n_diesel=n_diesel, n_clean=n_clean, Q_diesel=Q_diesel, Q_clean=Q_clean)

    # cost_k = w_dist * distance + w_co2 * (co2_per_dist_k * distance) = distance * (w_dist + w_co2 * co2_per_dist_k)
    unit_diesel = int(round(SCALE * (w_dist + w_co2 * co2_per_dist["diesel"])))
    unit_clean  = int(round(SCALE * (w_dist + w_co2 * co2_per_dist["clean"])))

    vt_d = model.vehicle_types[0]
    vt_c = model.vehicle_types[1]
    model.vehicle_types[0] = vt_d.replace(fixed_cost=0, unit_distance_cost=max(1, unit_diesel))
    model.vehicle_types[1] = vt_c.replace(fixed_cost=0, unit_distance_cost=max(1, unit_clean))

    res = model.solve(stop=MaxRuntime(RUNTIME_SECONDS), display=False) 
    return res.best


def demand_at_visit_id(vid, clients):
    # route.visits() returns customer ids 1..N
    idx = vid - 1
    d = clients[idx].delivery
    if isinstance(d, (list, tuple, np.ndarray)):
        return int(d[0]) if len(d) > 0 else 0
    return int(d)

def true_co2_of_solution(sol, data, type_labels, Q_by_type, f0, alpha, beta, gamma, mult):
    total_co2 = 0.0
    clients = data.clients()
    D = data.distance_matrix(0)

    for route in sol.routes():
        vt_idx = route.vehicle_type()
        k = type_labels[vt_idx]  # diesel or clean

        Qk_raw = Q_by_type[k]
        Qk = float(Qk_raw[0]) if isinstance(Qk_raw, (list, tuple, np.ndarray)) else float(Qk_raw)

        visits_only = list(route.visits())          # customer ids 1..N
        visits = [0] + visits_only + [0]            # include depot 0 for distance matrix

        demand = sum(demand_at_visit_id(i, clients) for i in visits_only)
        remaining = float(demand)

        for a, b in zip(visits[:-1], visits[1:]):
            dij = float(D[a][b])

            l_ij = 0.0 if Qk <= 0 else remaining / Qk
            f_ij = f0 * (1.0 + alpha * (l_ij ** beta)) * mult[k]
            total_co2 += dij * gamma * f_ij

            if b != 0:  # b is a customer id
                remaining -= demand_at_visit_id(b, clients)

    return float(total_co2)

def pareto_front(points):
    # points = [(dist, co2, ...), ...]
    nondominated = []

    for i, p in enumerate(points):
        dist_i, co2_i = p[:2]
        dominated = False

        for j, q in enumerate(points):
            if i == j:
                continue

            dist_j, co2_j = q[:2]

            if (dist_j <= dist_i and co2_j <= co2_i) and (dist_j < dist_i or co2_j < co2_i):
                dominated = True
                break

        if not dominated:
            nondominated.append(p)

    return sorted(nondominated, key=lambda x: x[0])

# Example

total_fleet = 20
n_diesel = total_fleet // 2
n_clean = total_fleet - n_diesel

type_labels = {0: "diesel", 1: "clean"}

base_cap = Model.from_data(data).vehicle_types[0].capacity
print(f"Base capacity: {base_cap[0]}")
Q_by_type = {"diesel": base_cap[0], "clean": int(0.9 * base_cap[0])}  

records = []  # (w_dist, w_co2, dist, true_co2)

for i, (w_dist, w_co2) in enumerate(weights, start=1):
    sol = solve_weighted_surrogate(data, w_dist, w_co2, n_diesel, n_clean, Q_diesel=Q_by_type["diesel"], Q_clean=Q_by_type["clean"])
    dist = sol.distance()
    co2 = true_co2_of_solution(sol, data, type_labels, Q_by_type, f0=f0, alpha=alpha, beta=beta, gamma=gamma, mult=mult)

    records.append((w_dist, w_co2, dist, co2))
    #points = [(r[2], r[3], r[0], r[1]) for r in records]  # (dist, co2, w_dist, w_co2)
    #front = pareto_front(points)

    # for dist, co2, w_dist, w_co2 in front:
    #     print(f"Solution {i}: dist={dist}, co2={co2:.2f}, weights=({w_dist:.2f}, {w_co2:.2f})")
    print(f"Solution {i}: w_dist={w_dist:.2f}, w_co2={w_co2:.2f} -> dist={dist}, trueCO2={co2:.2f}")

points = [(r[2], r[3], r[0], r[1]) for r in records]
front = pareto_front(points) # (dist, co2, w_dist, w_co2)

for i, (dist, co2, w_dist, w_co2) in enumerate(front, start=1):
    print(f"Pareto {i}: dist={dist}, co2={co2:.2f}, weights=({w_dist:.2f}, {w_co2:.2f})")


xs = [r[2] for r in records]
ys = [r[3] for r in records]
front = pareto_front([(r[2], r[3], r[0], r[1]) for r in records])
fx = [p[0] for p in front]
fy = [p[1] for p in front]

plt.scatter(xs, ys, alpha=0.4, label="All solutions")
# non-dominated points
plt.scatter(fx, fy, color="red", s=60, label="Non-dominated")
# line showing the Pareto front
# plt.plot(fx, fy, color="red", linewidth=2)

# plt.scatter(xs, ys)
plt.xlabel("Total distance")
plt.ylabel("CO$_2$ emissions")
plt.title(f"Distance vs CO$_2$ ({INSTANCE_PATH.removesuffix(".vrp")})")
plt.grid(True, alpha=0.3)
plt.show()

print(front)


# Representing the best optimal solution in bi-objetive space
def true_co2_of_best_routes(best_routes, data, Q_diesel, f0, alpha, beta, gamma, mult):
    """almost the same as true_co2_of_solution but takes best_routes instead of sol and assumes all vehicles are diesel for simplicity"""
    total_co2 = 0.0
    clients = data.clients()
    D = data.distance_matrix(0)

    for route in best_routes:
        visits = [0] + route + [0]   # depot + customers + depot

        demand = sum(demand_at_visit_id(i, clients) for i in route)
        remaining = float(demand)

        for a, b in zip(visits[:-1], visits[1:]):
            dij = float(D[a][b])

            l_ij = 0.0 if Q_diesel <= 0 else remaining / Q_diesel
            f_ij = f0 * (1.0 + alpha * (l_ij ** beta)) * mult["diesel"]
            total_co2 += dij * gamma * f_ij

            if b != 0:
                remaining -= demand_at_visit_id(b, clients)

    return float(total_co2)

# # X-n106-14
# best_routes = [
#     [54, 77, 83, 2, 17, 69, 105, 64, 94, 82],
#     [57, 63, 31, 40, 13, 85, 21, 74],
#     [5, 44, 15, 76, 51, 100, 45, 96],
#     [71, 49, 99, 11, 34, 72, 29],
#     [53, 103, 101, 93, 97, 43, 87, 60],
#     [67, 48, 8, 36, 91, 88, 58, 14],
#     [1, 104, 12, 9, 84, 75, 7],
#     [62, 26, 90, 24, 102, 92, 52],
#     [28, 66, 46, 79, 37, 27, 19, 78],
#     [22, 33],
#     [56, 20, 16, 10, 41, 18, 86],
#     [59, 89, 30, 42, 23, 3, 32, 73, 35],
#     [39, 50, 81, 6, 70, 4, 98, 55],
#     [68, 95, 38, 65, 80, 61, 47, 25],
# ]
# best_dist = 26362  # known best distance
# best_co2 = true_co2_of_best_routes(
#     best_routes,
#     data,
#     Q_diesel=Q_by_type["diesel"],
#     f0=f0,
#     alpha=alpha,
#     beta=beta,
#     gamma=gamma,
#     mult=mult,
# )

# print(f"Best known solution -> dist={best_dist}, CO2={best_co2:.2f}")
# # Output: Best known solution -> dist=26362, CO2=22630.28

# # X-n110-13
# best_routes = [
#     [27, 59, 95, 17, 31, 28, 39, 50, 52],
#     [58, 12, 41, 21, 75, 88, 96, 54, 68],
#     [19, 63, 72, 10, 86, 100, 62, 8, 49],
#     [23, 13, 57, 107, 82, 48, 47, 40],
#     [70, 44, 81, 94, 37, 7, 35, 73],
#     [79, 32, 74, 78, 29, 16, 90, 1],
#     [91, 38, 108, 3, 14],
#     [105, 20, 92, 97, 42, 106, 67, 34, 45],
#     [5, 46, 89, 51, 85, 11, 77, 6],
#     [98, 66, 53, 15, 26, 87, 84, 18, 109],
#     [56, 102, 33, 9, 101, 4, 24, 103, 25, 69],
#     [71, 93, 104, 60, 61, 22, 83, 76, 55, 30],
#     [99, 65, 36, 64, 43, 2, 80],
# ]
# best_dist = 14971  # known best distance
# best_co2 = true_co2_of_best_routes(
#     best_routes,
#     data,
#     Q_diesel=Q_by_type["diesel"],
#     f0=f0,
#     alpha=alpha,
#     beta=beta,
#     gamma=gamma,
#     mult=mult,
# )

# print(f"Best known solution -> dist={best_dist}, CO2={best_co2:.2f}")
# # Output: Best known solution -> dist=14971, CO2=12973.87

# X-n115-10
best_routes = [
    [10],
    [59, 42, 35, 99, 49, 79, 47, 109, 18, 93, 3, 32, 15],
    [16, 5, 6],
    [89, 17, 66, 106, 73, 87, 86, 60, 48, 108, 19, 51, 25, 2, 91, 41, 104],
    [72, 102, 24, 71, 76, 55, 80, 97, 38, 77, 58, 113, 12, 45, 96, 22, 23, 65, 36],
    [13, 70, 56, 31, 11, 81, 33, 61, 103, 69],
    [85, 74, 1, 7, 63, 64],
    [46, 26, 34, 98, 37, 90, 39, 67, 40, 43, 68, 14, 8, 83, 75, 84, 112, 110, 111],
    [52, 21, 82, 88, 105, 27, 44, 30, 114, 54, 9, 78, 53, 29, 50],
    [95, 107, 57, 92, 4, 101, 20, 94, 62, 100, 28]
]
best_dist = 12747  # known best distance
best_co2 = true_co2_of_best_routes(
    best_routes,
    data,
    Q_diesel=Q_by_type["diesel"],
    f0=f0,
    alpha=alpha,
    beta=beta,
    gamma=gamma,
    mult=mult,
)

print(f"Best known solution -> dist={best_dist}, CO2={best_co2:.2f}")
# Output: Best known solution -> dist=12747, CO2=10951.48
