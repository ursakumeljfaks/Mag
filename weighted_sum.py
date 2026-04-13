import numpy as np
import matplotlib.pyplot as plt
import time
from pyvrp import read, Model
from pyvrp.stop import MaxRuntime


# ============================================================
# PARAMETERS
# ============================================================

f0 = 0.30
alpha = 0.20
beta = 1.0
gamma = 2.61
mult = {"diesel": 1.00, "clean": 0.45}
l_nom = 0.50

INSTANCE_PATH = "instances/vrp/X-n110-k13.vrp"
RUNTIME_SECONDS = 5
SCALE = 1000

# weights for weighted-sum scalarization
weights = [(1 - w, w) for w in np.linspace(0.05, 0.95, 300)]


# ============================================================
# PRECOMPUTED SURROGATE CO2 PER DISTANCE
# ============================================================

fbar = {k: f0 * (1 + alpha * (l_nom ** beta)) * mult[k] for k in mult}
co2_per_dist = {k: gamma * fbar[k] for k in mult}


data = read(INSTANCE_PATH, round_func="round")


# ============================================================
# MODEL BUILDING
# ============================================================

def build_model_two_types(data, n_diesel, n_clean, Q_diesel=0, Q_clean=0):
    model = Model.from_data(data)

    # Replace original vehicle type with diesel
    vt0 = model.vehicle_types[0]
    model.vehicle_types[0] = vt0.replace(
        num_available=n_diesel,
        capacity=[Q_diesel],
        name="diesel",
    )

    # Add clean vehicle type
    model.add_vehicle_type(
        num_available=n_clean,
        capacity=Q_clean,
        name="clean",
    )

    return model


def solve_weighted_surrogate(data, w_dist, w_co2, n_diesel, n_clean, Q_diesel, Q_clean):
    model = build_model_two_types(
        data,
        n_diesel=n_diesel,
        n_clean=n_clean,
        Q_diesel=Q_diesel,
        Q_clean=Q_clean,
    )

    # surrogate weighted cost per unit distance
    unit_diesel = int(round(SCALE * (w_dist + w_co2 * co2_per_dist["diesel"])))
    unit_clean = int(round(SCALE * (w_dist + w_co2 * co2_per_dist["clean"])))

    vt_d = model.vehicle_types[0]
    vt_c = model.vehicle_types[1]

    model.vehicle_types[0] = vt_d.replace(
        fixed_cost=0,
        unit_distance_cost=max(1, unit_diesel),
    )
    model.vehicle_types[1] = vt_c.replace(
        fixed_cost=0,
        unit_distance_cost=max(1, unit_clean),
    )

    res = model.solve(stop=MaxRuntime(RUNTIME_SECONDS), display=False)
    return res, res.best


# ============================================================
# DEMAND / TRUE CO2 EVALUATION
# ============================================================

def demand_at_visit_id(vid, clients):
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
        vehicle_name = type_labels[vt_idx]

        Qk_raw = Q_by_type[vehicle_name]
        Qk = float(Qk_raw[0]) if isinstance(Qk_raw, (list, tuple, np.ndarray)) else float(Qk_raw)

        visits_only = list(route.visits())
        visits = [0] + visits_only + [0]

        demand = sum(demand_at_visit_id(i, clients) for i in visits_only)
        remaining = float(demand)

        for a, b in zip(visits[:-1], visits[1:]):
            dij = float(D[a][b])

            l_ij = 0.0 if Qk <= 0 else remaining / Qk
            f_ij = f0 * (1.0 + alpha * (l_ij ** beta)) * mult[vehicle_name]
            total_co2 += dij * gamma * f_ij

            if b != 0:
                remaining -= demand_at_visit_id(b, clients)

    return float(total_co2)


# ============================================================
# PARETO HELPERS
# ============================================================

def pareto_mask(points):
    """
    points: list of (dist, co2)
    returns a boolean mask: True if nondominated
    """
    n = len(points)
    mask = [True] * n

    for i in range(n):
        di, ci = points[i]

        for j in range(n):
            if i == j:
                continue

            dj, cj = points[j]

            if (dj <= di and cj <= ci) and (dj < di or cj < ci):
                mask[i] = False
                break

    return mask


# ============================================================
# PRINT ROUTE INFORMATION
# ============================================================

def print_solution_routes(sol, type_labels):
    print("\nChosen solution routes:")
    for idx, route in enumerate(sol.routes(), start=1):
        vt_idx = route.vehicle_type()
        veh_name = type_labels[vt_idx]
        visits = list(route.visits())

        print(
            f"Route {idx:2d} | "
            f"vehicle={veh_name:6s} | "
            f"distance={route.distance()} | "
            f"visits={visits}"
        )


# ============================================================
# PARETO PLOT
# ============================================================

def plot_pareto(records):
    xs = [r["dist"] for r in records]
    ys = [r["co2"] for r in records]

    is_nd = pareto_mask(list(zip(xs, ys)))

    feasible_x = [x for x, keep in zip(xs, is_nd) if not keep]
    feasible_y = [y for y, keep in zip(ys, is_nd) if not keep]

    pareto_x = [x for x, keep in zip(xs, is_nd) if keep]
    pareto_y = [y for y, keep in zip(ys, is_nd) if keep]

    plt.figure(figsize=(8, 6))
    plt.scatter(feasible_x, feasible_y, alpha=0.4, label="Other feasible solutions")
    plt.scatter(pareto_x, pareto_y, color="red", label="Pareto front")
    plt.xlabel("Total distance")
    plt.ylabel("CO$_2$ emissions")
    plt.title("Bi-objective space")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.show()


# ============================================================
# COORDINATES FROM VRP FILE
# ============================================================

def read_vrp_coordinates(instance_path):
    coords = {}
    in_section = False

    with open(instance_path, "r") as f:
        for line in f:
            line = line.strip()

            if line == "NODE_COORD_SECTION":
                in_section = True
                continue

            if line == "DEMAND_SECTION":
                break

            if in_section and line:
                parts = line.split()
                if len(parts) >= 3:
                    node_id = int(parts[0])
                    x = float(parts[1])
                    y = float(parts[2])
                    coords[node_id] = (x, y)

    return coords


# ============================================================
# ROUTE PLOT WITH VEHICLE TYPES
# ============================================================

def plot_solution_with_vehicle_types(sol, instance_path, type_labels):
    coords = read_vrp_coordinates(instance_path)

    plt.figure(figsize=(8, 6))

    depot_x, depot_y = coords[1]
    plt.scatter(depot_x, depot_y, marker="*", s=350, color="red", label="Depot", zorder=5)

    color_map = {
        "diesel": "tab:blue",
        "clean": "tab:green",
    }


    shown_labels = set()

    for r_idx, route in enumerate(sol.routes(), start=1):
        vt_idx = route.vehicle_type()
        veh_name = type_labels[vt_idx]

        cust_nodes = [v + 1 for v in route.visits()]
        full_nodes = [1] + cust_nodes + [1]

        xs = [coords[n][0] for n in full_nodes]
        ys = [coords[n][1] for n in full_nodes]

        label = veh_name if veh_name not in shown_labels else None
        shown_labels.add(veh_name)

        plt.plot(
            xs,
            ys,
            color=color_map.get(veh_name, "gray"),
            linewidth=2,
            alpha=0.9,
            label=label,
        )

        plt.scatter(
            [coords[n][0] for n in cust_nodes],
            [coords[n][1] for n in cust_nodes],
            color=color_map.get(veh_name, "gray"),
            s=18,
            alpha=0.9,
        )

        # route index near first customer
        if cust_nodes:
            x0, y0 = coords[cust_nodes[0]]
            plt.text(x0, y0, str(r_idx), fontsize=9)

    plt.title("Chosen solution with vehicle types")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.axis("equal")
    plt.show()


# ============================================================
# MAIN EXAMPLE RUN
# ============================================================

def main():
    total_fleet = 20
    n_diesel = total_fleet // 2
    n_clean = total_fleet - n_diesel

    type_labels = {0: "diesel", 1: "clean"}

    base_cap = Model.from_data(data).vehicle_types[0].capacity
    print(f"Base capacity: {base_cap[0]}")

    Q_by_type = {
        "diesel": base_cap[0],
        "clean": int(0.9 * base_cap[0]),
    }

    records = []
    start_time = time.time()

    for i, (w_dist, w_co2) in enumerate(weights, start=1):
        res, sol = solve_weighted_surrogate(
            data,
            w_dist,
            w_co2,
            n_diesel,
            n_clean,
            Q_diesel=Q_by_type["diesel"],
            Q_clean=Q_by_type["clean"],
        )

        dist = sol.distance()
        co2 = true_co2_of_solution(
            sol,
            data,
            type_labels,
            Q_by_type,
            f0=f0,
            alpha=alpha,
            beta=beta,
            gamma=gamma,
            mult=mult,
        )

        records.append({
            "w_dist": w_dist,
            "w_co2": w_co2,
            "dist": dist,
            "co2": co2,
            "res": res,
            "sol": sol,
        })

        print(
            f"Solution {i:2d}: "
            f"w_dist={w_dist:.2f}, w_co2={w_co2:.2f} "
            f"-> dist={dist}, trueCO2={co2:.2f}"
        )

    end_time = time.time()
    print(f"\nTotal runtime: {end_time - start_time:.2f} seconds")

    plot_pareto(records)

    is_nd = pareto_mask([(r["dist"], r["co2"]) for r in records])
    pareto_records = [r for r, keep in zip(records, is_nd) if keep]

    # example: minimum CO2 among Pareto solutions
    chosen = min(pareto_records, key=lambda r: r["co2"])

    print(
        f"\nChosen Pareto solution:"
        f"\nweights = ({chosen['w_dist']:.2f}, {chosen['w_co2']:.2f})"
        f"\ndistance = {chosen['dist']}"
        f"\nCO2 = {chosen['co2']:.2f}"
    )

    print_solution_routes(chosen["sol"], type_labels)

    plot_solution_with_vehicle_types(
        chosen["sol"],
        INSTANCE_PATH,
        type_labels,
    )


if __name__ == "__main__":
    main()