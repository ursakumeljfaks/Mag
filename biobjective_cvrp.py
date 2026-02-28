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

INSTANCE_PATH = "X-n106-k14.vrp"
RUNTIME_SECONDS = 30
SEED = 1
SCALE = 1000

weights = [(1-w, w) for w in np.linspace(0.05, 0.95, 15)] 

data = read(INSTANCE_PATH, round_func="round")



def build_model_two_types(data, n_diesel, n_clean, Q_diesel=0, Q_clean=0):
    model = Model.from_data(data)

    # first type of vehicle is diesel
    model.add_vehicle_type(num_available=n_diesel, capacity=Q_diesel)

    # second type is clean
    model.add_vehicle_type(num_available=n_clean, capacity=Q_clean)

    return model


def solve_weighted_surrogate(data, w_dist, w_co2, n_diesel, n_clean):
    model = build_model_two_types(data, n_diesel=n_diesel, n_clean=n_clean)

    # cost_k = w_dist * distance + w_co2 * (co2_per_dist_k * distance) = distance * (w_dist + w_co2 * co2_per_dist_k)
    unit_diesel = int(round(SCALE * (w_dist + w_co2 * co2_per_dist["diesel"])))
    unit_clean  = int(round(SCALE * (w_dist + w_co2 * co2_per_dist["clean"])))

    vt_d = model.vehicle_types[0]
    vt_c = model.vehicle_types[1]
    model.vehicle_types[0] = vt_d.replace(fixed_cost=0, unit_distance_cost=max(1, unit_diesel))
    model.vehicle_types[1] = vt_c.replace(fixed_cost=0, unit_distance_cost=max(1, unit_clean))

    res = model.solve(stop=MaxRuntime(RUNTIME_SECONDS), seed=SEED, display=False)
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

# Example

total_fleet = 20
n_diesel = total_fleet // 2
n_clean = total_fleet - n_diesel

type_labels = {0: "diesel", 1: "clean"}

base_cap = Model.from_data(data).vehicle_types[0].capacity
Q_by_type = {"diesel": base_cap, "clean": base_cap}

records = []  # (w_dist, w_co2, dist, true_co2)

for w_dist, w_co2 in weights:
    sol = solve_weighted_surrogate(data, w_dist, w_co2, n_diesel, n_clean)
    dist = sol.distance()
    co2 = true_co2_of_solution(sol, data, type_labels, Q_by_type, f0=f0, alpha=alpha, beta=beta, gamma=gamma, mult=mult)

    records.append((w_dist, w_co2, dist, co2))
    print(f"w_dist={w_dist:.2f}, w_co2={w_co2:.2f} -> dist={dist}, trueCO2={co2:.2f}")


xs = [r[2] for r in records]
ys = [r[3] for r in records]
plt.scatter(xs, ys)
plt.xlabel("Total distance")
plt.ylabel("CO2 emissions")
plt.title("Distance vs CO2")
plt.grid(True, alpha=0.3)
plt.show()



