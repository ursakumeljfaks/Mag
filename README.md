# Weighted-sum Green CVRP workflow

This implementation is provided in the file `weighted-sum.py`. It solves a bi-objective Green CVRP with two vehicle types (`diesel` and `clean`) using a weighted-sum approach on top of PyVRP.

The code follows this workflow:

1. Define global parameters for fuel consumption, CO2 emissions, fleet composition, runtime, and weighted-sum coefficients.
2. Read a VRP instance from file.
3. Build a PyVRP model with two vehicle types.
4. For each weight pair `(w_dist, w_co2)`:
   - convert the bi-objective problem into a single surrogate routing cost
   - solve the resulting VRP with PyVRP
   - evaluate the true load-dependent CO2 emissions of the obtained solution
   - store distance, CO2, and solution information
5. Identify nondominated solutions to approximate the Pareto front.
6. Plot the objective-space trade-off.
7. Select one Pareto solution and visualize its routes, colored by vehicle type.
