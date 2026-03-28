import matplotlib.pyplot as plt

# weighted sum solution points (green)
points = [
(26727, 18544.307215499994),
(26762, 17816.8803795),
(26839, 17293.83311699999),
(26952, 17129.332386),
(26990, 16668.834425999994),
(27003, 16614.413577000014),
(27145, 16458.252448500007),
(27193, 15571.131587999993),
(27223, 15545.139250500004),
(27231, 15492.437347500003),
(27261, 15166.310800499987),
(27351, 15152.802223499999),
(27388, 14849.727718499998),
(28533, 14176.829915999993),
(28535, 14134.011560999992),
(28542, 14115.495437999994),
(28563, 14109.187067999994),
]
xs = [p[0] for p in points]
ys = [p[1] for p in points]

# NSGA2 points (blue)
nsga_points = [
(31141.0, 22354.096419), 
(31171.0, 22348.732347), 
(31275.0, 21538.662993), 
(31278.0, 21537.943155), 
(31292.0, 21404.4942465), 
(31293.0, 21173.27787), 
(31300.0, 21164.789628), 
(31341.0, 20066.113782), 
(31526.0, 20046.161898), 
(31527.0, 19982.2582665), 
(32293.0, 19794.2967675), 
(32301.0, 19794.2388255), 
(32302.0, 19051.621137), 
(32306.0, 19043.592777), 
(32324.0, 17136.0211635), 
(32352.0, 16925.417523), 
(32402.0, 16641.589158)]

nsga_x = [p[0] for p in nsga_points]
nsga_y = [p[1] for p in nsga_points]

# best-known solution (red)
best_dist = 26362
best_co2 = 22630.28

plt.scatter(xs, ys, color="green", label="Weighted-sum")
plt.scatter(nsga_x, nsga_y, color="blue", label="NSGA-II")
plt.scatter([best_dist], [best_co2], color="red", label="Best distance solution")

plt.xlabel("Total distance")
plt.ylabel("CO$_2$ emissions")
plt.title("Distance vs CO$_2$")
plt.grid(True, alpha=0.3)
plt.legend()

plt.show()