import re
import numpy as np

def extract_pareto_fronts_from_txt(txt_path):
    with open(txt_path, "r", encoding="utf-8") as f:
        text = f.read()

    run_blocks = re.split(r"\r?\n\s*\d+\s+RUN\s*\r?\n", text)

    all_fronts = []

    for block in run_blocks:
        if "Pareto objective values [distance, co2]:" not in block:
            continue

        part = block.split("Pareto objective values [distance, co2]:", 1)[1]

        # keep only the matrix of pareto values
        if "All 20 Pareto solutions:" in part:
            part = part.split("All 20 Pareto solutions:", 1)[0]

        pairs = re.findall(r"\[\s*([0-9.eE+-]+)\s+([0-9.eE+-]+)\s*\]", part)

        front = [(float(a), float(b)) for a, b in pairs]

        if front:
            all_fronts.append(front)

    return all_fronts

def change_fronts(front):
    pts = [(p[0], p[1]) for p in front]
    return sorted(set(pts), key=lambda t: t[0])

def ref_point(list_fronts):
    max_dist = max(p[0] for front in list_fronts for p in front)
    max_co2 = max(p[1] for front in list_fronts for p in front)
    return (max_dist * 1.05, max_co2 * 1.05)

def hypervolume(front, ref_pt):
    sorted_front = sorted(front, key=lambda p: p[0])
    hv = 0.0
    prev_y = ref_pt[1]

    for dist, co2 in sorted_front:
        width = ref_pt[0] - dist
        height = prev_y - co2
        hv += width * height
        prev_y = co2

    return hv


all_fronts = extract_pareto_fronts_from_txt("statistics_and_evaluation/nsga2_eval.txt")
print(f'Number of runs found: {len(all_fronts)}')
clean_fronts1 = [change_fronts(front) for front in all_fronts[40:]]
rp = ref_point(clean_fronts1)
hv_values = [hypervolume(front, rp) for front in clean_fronts1]
# sort indices by hypervolume
sorted_indices = sorted(range(len(hv_values)), key=lambda i: hv_values[i])
median_pos = len(hv_values) // 2
median_idx = sorted_indices[median_pos]
median_hv = hv_values[median_idx]
median_front = clean_fronts1[median_idx]

print(median_front) # use this to plot it in the same plot as the weighted-sum
print(median_hv)
