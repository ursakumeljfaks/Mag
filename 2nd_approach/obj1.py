import numpy as np
from pyvrp import Model, ProblemData, VehicleType, read, Solution
from pyvrp.stop import MaxRuntime
import obj2

RUNTIME = 5
INSTANCE_PATH = "instances/vrp/X-n106-k14.vrp"

#########
# MAIN FUNCTIONS
#########

def obj1(raw_data, f0, alpha, beta, gamma, mult, l_nom, num_available_diesel, num_available_clean, Q_by_type):
    model = Model.from_data(raw_data)

    # first type of vehicle is diesel
    vt0 = model.vehicle_types[0]
    model.vehicle_types[0] = vt0.replace(num_available=num_available_diesel, capacity=[Q_by_type["diesel"]], name="diesel")

    # second type is clean
    model.add_vehicle_type(num_available=num_available_clean, capacity=[Q_by_type["clean"]], name="clean")
    res = model.solve(stop=MaxRuntime(RUNTIME), display=False)
    return res, res.best.distance_cost()

class VariantResult:
    def __init__(self, assignments):
        self.assignments = assignments

    def routes(self):
        return self.assignments
    
def extract_routes(solution):
    return [list(route.visits()) for route in solution.routes()]

def generate_distance_variants(best_solution, number_variants, type_labels):
    routes = extract_routes(best_solution)
    variants = []

    base_cost = best_solution.distance_cost()

    for _ in range(number_variants):
        assignment = []

        for visits in routes:
            vt_idx = np.random.choice(list(type_labels.keys()))
            assignment.append((visits, vt_idx))

        res_like = VariantResult(assignment)
        variants.append((res_like, base_cost))

    return variants

def main():
    raw_data = read(INSTANCE_PATH, round_func="round")
    f0, alpha, beta, gamma = 0.30, 0.20, 1.0, 2.61
    mult = {"diesel": 1.00, "clean": 0.45}
    l_nom = 0.50
    base_cap = Model.from_data(raw_data).vehicle_types[0].capacity[0]
    Q_by_type = {"diesel": base_cap, "clean": int(0.9 * base_cap)}
    #type_labels = {0: "diesel", 1: "clean"}
    
    n_diesel = 10
    n_clean = 10
    res, best_cost = obj1(raw_data, f0, alpha, beta, gamma, mult, l_nom, n_diesel, n_clean, Q_by_type)
    print("Best solution found:")
    print(best_cost)
    print(res.best)

    type_labels = {0: "diesel", 1: "clean"}
    variants = generate_distance_variants(best_solution=res.best, number_variants=10, type_labels=type_labels)

    for i, (res_like, cost) in enumerate(variants, 1):
        print(f"\nVariant {i}: Cost = {cost}")

        for visits, vt_idx in res_like.routes():
            print(f"  [{type_labels[vt_idx]}] {visits}")

    actual_co2 = obj2.true_co2(res.best, raw_data, type_labels, Q_by_type, f0, alpha, beta, gamma, mult)
    print(f"\nTrue CO2 of the best solution: {actual_co2:.2f}")


if __name__ == "__main__":    
    main()

