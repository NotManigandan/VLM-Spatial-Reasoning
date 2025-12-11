import json
import pickle
import numpy as np

def summarize_results(json_path):
    # Load the saved result dict
    with open(json_path, "r") as f:
        result = json.load(f)

    eps = 1e-6

    # Overall accuracy
    # ref: OmniSpatial code
    overall = sum(result["Total"]) / (len(result["Total"]) + eps) * 100
    print("\n======= FINAL =======")
    print(f"Overall: {overall:.2f}% (N={len(result['Total'])})")

    # Per-task and per-subtask accuracies
    for task in [k for k in result if k not in {"Total", "Processed"}]:
        task_total = result[task]["Total"]
        task_acc = sum(task_total) / (len(task_total) + eps) * 100
        print(f"{task}: {task_acc:.2f}% (N={len(task_total)})")

        for sub in result[task]:
            if sub == "Total":
                continue
            sub_vals = result[task][sub]
            sub_acc = sum(sub_vals) / (len(sub_vals) + eps) * 100
            print(f"    {sub}: {sub_acc:.2f}% (N={len(sub_vals)})")

summarize_results("/home/ubuntu/OmniSpatial/vlms_eval/result/UCSC-VLAA/VLAA-Thinker-Qwen2.5VL-3B/results_0_k_3_2025-12-07T07-57-49.json")


def summarize_latency(pkl_path):
    # Load list of latency measurements
    with open(pkl_path, "rb") as f:
        LATENCY = pickle.load(f)

    LATENCY = np.array(LATENCY)

    print("\n======= LATENCY =======")
    print(f"Count: {len(LATENCY)}")
    print(f"Average latency: {np.mean(LATENCY):.3f} seconds")
    print(f"Max latency: {np.max(LATENCY):.3f} seconds")
    print(f"Min latency: {np.min(LATENCY):.3f} seconds")
    print(f"Median latency: {np.median(LATENCY):.3f} seconds")
    print(f"Std dev: {np.std(LATENCY):.3f} seconds")

# usage:
summarize_latency("/home/ubuntu/OmniSpatial/vlms_eval/result/UCSC-VLAA/VLAA-Thinker-Qwen2.5VL-3B/latency_0_k_3_2025-12-07T07-57-49.pkl")
