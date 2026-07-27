# evaluation/report_generator.py
import json, datetime
import matplotlib.pyplot as plt

def save_report(all_results, out_dir="./reports"):
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M")

    # JSON dump
    with open(f"{out_dir}/benchmark_{timestamp}.json", "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    # Bar chart — triage accuracy per corruption type
    types = list({r["corruption"] for r in all_results})
    accs  = []
    for t in types:
        subset = [r for r in all_results if r["corruption"] == t]
        accs.append(sum(r["predicted"] == r["expected"] for r in subset) / len(subset) * 100)

    plt.figure(figsize=(8, 4))
    plt.bar(types, accs, color=["tomato" if a < 90 else "seagreen" for a in accs])
    plt.axhline(90, color="gray", linestyle="--", label="90% target")
    plt.ylabel("Triage Accuracy (%)")
    plt.title("Validation Agent — Benchmark Results")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{out_dir}/benchmark_{timestamp}.png", dpi=120)
    plt.show()
    print(f"Saved → {out_dir}/benchmark_{timestamp}.json/.png")
