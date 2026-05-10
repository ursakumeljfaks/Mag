import numpy as np
from pyvrp import Model, ProblemData, VehicleType, read, Solution
from pyvrp.stop import MaxRuntime

RUNTIME = 5
INSTANCE_PATH = "instances/vrp/X-n106-k14.vrp"

#########
# MAIN FUNCTIONS
#########

## functions for calculating true CO2
def demand_at_visit_id(vid, clients):
    idx = vid - 1
    d = clients[idx].delivery
    if isinstance(d, (list, tuple, np.ndarray)):
        return int(d[0]) if len(d) > 0 else 0
    return int(d)

def true_co2(sol_or_assignments, data, type_labels, Q_by_type, f0, alpha, beta, gamma, mult):
    total_co2 = 0.0
    clients = data.clients()
    D = data.distance_matrix(0)

    if hasattr(sol_or_assignments, "routes"):
        # PyVRP Solution
        iterator = [
            (list(route.visits()), route.vehicle_type())
            for route in sol_or_assignments.routes()
        ]
    else:
        # Custom assignments for variants
        iterator = sol_or_assignments

    for visits_only, vt_idx in iterator:
        k = type_labels[vt_idx]
        Qk = float(Q_by_type[k])

        visits = [0] + visits_only + [0]
        demand = sum(demand_at_visit_id(i, clients) for i in visits_only)
        remaining = float(demand)

        for a, b in zip(visits[:-1], visits[1:]):
            dij = float(D[a][b])
            l_ij = 0.0 if Qk <= 0 else remaining / Qk
            f_ij = f0 * (1.0 + alpha * (l_ij ** beta)) * mult[k]
            total_co2 += dij * gamma * f_ij

            if b != 0:
                remaining -= demand_at_visit_id(b, clients)

    return float(total_co2)

## objective 2 function for surrogate CO2 (using precomputed matrices)
def obj2(raw_data, f0, alpha, beta, gamma, mult, l_nom, num_available_diesel, num_available_clean, Q_by_type):
    base_dist = raw_data.distance_matrix(0)
    #Q_by_type = {"diesel": base_cap, "clean": int(0.9 * base_cap)}
    
    # creating two distance matrices, for diesel and clean, by applying the surrogate CO2 formula to the base distance matrix
    co2_matrices = []
    for k in ["diesel", "clean"]:
        factor = gamma * f0 * (1 + alpha * (l_nom ** beta)) * mult[k]
        co2_matrices.append((base_dist * factor).astype(int))

    # creating the ProblemData with the two matrices and profiles
    data = ProblemData(
        clients=raw_data.clients(),
        depots=raw_data.depots(),
        vehicle_types=[
            VehicleType(num_available=num_available_diesel, capacity=[Q_by_type["diesel"]], profile=0, name="diesel"),
            VehicleType(num_available=num_available_clean, capacity=[Q_by_type["clean"]], profile=1, name="clean")
        ],
        distance_matrices=co2_matrices,
        duration_matrices=[raw_data.duration_matrix(0)] * 2
    )

    model = Model.from_data(data)
    res = model.solve(stop=MaxRuntime(RUNTIME), display=False)
    
    # calculating true CO2 of the solution
    type_labels = {0: "diesel", 1: "clean"}
    actual_co2 = true_co2(res.best, data, type_labels, Q_by_type, f0, alpha, beta, gamma, mult)
    
    return res, actual_co2


#########
# FUNCTIONS FOR GENERATING MULTIPLE VARIANTS
#########

def extract_routes(solution):
    return [list(route.visits()) for route in solution.routes()]

def is_feasible(visits, vt_idx, data, Q_by_type, type_labels):
    clients = data.clients()
    demand = sum(demand_at_visit_id(i, clients) for i in visits)

    k = type_labels[vt_idx]
    return demand <= Q_by_type[k]

def random_assignment(routes, data, Q_by_type, type_labels):
    assignment = []

    for visits in routes:
        vt_idx = np.random.choice(list(type_labels.keys()))

        if not is_feasible(visits, vt_idx, data, Q_by_type, type_labels):
            return None  

        assignment.append((visits, vt_idx))

    return assignment


class VariantResult:
    def __init__(self, assignments):
        self.assignments = assignments

    def routes(self):
        return self.assignments
    
def generate_variants(best_solution, data, number_variants, Q_by_type, type_labels, f0, alpha, beta, gamma, mult, max_attempts=1000):
    routes = extract_routes(best_solution)
    variants = []

    attempts = 0
    while len(variants) < number_variants and attempts < max_attempts:
        attempts += 1

        assign = random_assignment(routes, data, Q_by_type, type_labels)
        if assign is None:
            continue

        co2 = true_co2(assign, data, type_labels, Q_by_type, f0, alpha, beta, gamma, mult)

        res_like = VariantResult(assign)
        variants.append((res_like, co2))

    return variants


######
# MAIN
######

def main():
    raw_data = read(INSTANCE_PATH, round_func="round")
    f0, alpha, beta, gamma = 0.30, 0.20, 1.0, 2.61
    mult = {"diesel": 1.00, "clean": 0.45}
    l_nom = 0.50
    base_cap = Model.from_data(raw_data).vehicle_types[0].capacity[0]
    Q_by_type = {"diesel": base_cap, "clean": int(0.9 * base_cap)}
    type_labels = {0: "diesel", 1: "clean"}
    
    n_diesel = 10
    n_clean = 10

    res, actual_co2 = obj2(raw_data, f0, alpha, beta, gamma, mult, l_nom, n_diesel, n_clean, Q_by_type)
    
    print("Best solution found:")
    print(res.best)
    print(f"True CO2 of the solution: {actual_co2}")

    variants = generate_variants(res.best, raw_data, number_variants=10, Q_by_type=Q_by_type, type_labels=type_labels, f0=f0, alpha=alpha, beta=beta, gamma=gamma, mult=mult)
    for i, (res_like, co2) in enumerate(variants, 1):
        print(f"\nVariant {i}: CO2 = {co2:.2f}")
        for visits, vt_idx in res_like.routes():
            print(f"  [{type_labels[vt_idx]}] {visits}")

if __name__ == "__main__":    
    main()