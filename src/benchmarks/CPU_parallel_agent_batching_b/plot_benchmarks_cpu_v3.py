import os
import pandas as pd
import matplotlib.pyplot as plt

os.makedirs("figures", exist_ok=True)

INPUT = "results/benchmarks_cpu_v3_threads.csv"
PREFIX = "cpu_v3_threads"
TITLE_PREFIX = "CPU V3"

df = pd.read_csv(INPUT)

print("Columns in CSV:")
print(df.columns.tolist())

group_cols = ["model", "version", "N", "T", "threads"]

agg = (
    df.groupby(group_cols)
    .agg(
        avg_step_ms_mean=("avg_step_ms", "mean"),
        avg_step_ms_std=("avg_step_ms", "std"),

        counting_mean=("counting_avg_step_ms", "mean"),
        probabilities_mean=("probabilities_avg_step_ms", "mean"),
        updating_mean=("updating_avg_step_ms", "mean"),
        order_generation_mean=("order_generation_avg_step_ms", "mean"),
        clearing_mean=("clearing_avg_step_ms", "mean"),
        price_update_mean=("price_update_avg_step_ms", "mean"),
        output_mean=("output_avg_step_ms", "mean"),

        counting_std=("counting_avg_step_ms", "std"),
        updating_std=("updating_avg_step_ms", "std"),
        order_generation_std=("order_generation_avg_step_ms", "std"),
        clearing_std=("clearing_avg_step_ms", "std"),
    )
    .reset_index()
)

# ------------------------------------------------
# 1. Average step time vs N, one curve per thread count
# ------------------------------------------------

plt.figure(figsize=(7, 4.5))

for threads, sub in agg.groupby("threads"):
    sub = sub.sort_values("N")

    plt.errorbar(
        sub["N"],
        sub["avg_step_ms_mean"],
        yerr=sub["avg_step_ms_std"],
        marker="o",
        capsize=3,
        label=f"{threads} thread(s)",
    )

plt.xscale("log")
plt.yscale("log")
plt.xlabel("Number of agents N")
plt.ylabel("Average time per step (ms)")
plt.title(f"{TITLE_PREFIX}: average step time by thread count")
plt.grid(True, which="both", linestyle="--", linewidth=0.5)
plt.legend()
plt.tight_layout()
plt.savefig(f"figures/{PREFIX}_avg_step_time.png", dpi=300)
plt.close()

# ------------------------------------------------
# 2. Speedup vs N, relative to 1 thread
# ------------------------------------------------

speedup_rows = []

for (model, version, T), sub in agg.groupby(["model", "version", "T"]):
    sub = sub.copy()

    baseline = (
        sub[sub["threads"] == 1][["N", "avg_step_ms_mean"]]
        .rename(columns={"avg_step_ms_mean": "baseline_avg_step_ms"})
    )

    merged = sub.merge(baseline, on="N", how="left")
    merged["speedup"] = (
        merged["baseline_avg_step_ms"] / merged["avg_step_ms_mean"]
    )

    speedup_rows.append(merged)

speedup_df = pd.concat(speedup_rows, ignore_index=True)

plt.figure(figsize=(7, 4.5))

for threads, sub in speedup_df.groupby("threads"):
    sub = sub.sort_values("N")

    plt.plot(
        sub["N"],
        sub["speedup"],
        marker="o",
        label=f"{threads} thread(s)",
    )

plt.axhline(1.0, linestyle="--", linewidth=1)
plt.xscale("log")
plt.xlabel("Number of agents N")
plt.ylabel("Speedup relative to 1 thread")
plt.title(f"{TITLE_PREFIX}: speedup by thread count")
plt.grid(True, which="both", linestyle="--", linewidth=0.5)
plt.legend()
plt.tight_layout()
plt.savefig(f"figures/{PREFIX}_speedup_vs_N.png", dpi=300)
plt.close()

# ------------------------------------------------
# 3. Speedup vs thread count, one curve per N
# ------------------------------------------------

plt.figure(figsize=(7, 4.5))

for N, sub in speedup_df.groupby("N"):
    sub = sub.sort_values("threads")

    plt.plot(
        sub["threads"],
        sub["speedup"],
        marker="o",
        label=f"N={N}",
    )

plt.axhline(1.0, linestyle="--", linewidth=1)
plt.xlabel("Number of threads")
plt.ylabel("Speedup relative to 1 thread")
plt.title(f"{TITLE_PREFIX}: speedup vs number of threads")
plt.grid(True, linestyle="--", linewidth=0.5)
plt.legend(fontsize=8)
plt.tight_layout()
plt.savefig(f"figures/{PREFIX}_speedup_vs_threads.png", dpi=300)
plt.close()

# ------------------------------------------------
# 4. Runtime decomposition for largest N, one bar per thread count
# ------------------------------------------------

largest_N = agg["N"].max()
plot_df = agg[agg["N"] == largest_N].sort_values("threads").copy()

components = [
    "counting_mean",
    "updating_mean",
    "order_generation_mean",
    "clearing_mean",
]

labels = [
    "Counting",
    "Type updating",
    "Order generation",
    "Clearing",
]

x = range(len(plot_df))
bottom = [0.0] * len(plot_df)

plt.figure(figsize=(7, 4.5))

for comp, label in zip(components, labels):
    values = plot_df[comp].values

    plt.bar(
        x,
        values,
        bottom=bottom,
        label=label,
    )

    bottom = [b + v for b, v in zip(bottom, values)]

plt.xticks(x, plot_df["threads"].astype(str))
plt.xlabel("Number of threads")
plt.ylabel("Average time per step (ms)")
plt.title(f"{TITLE_PREFIX}: runtime decomposition at N={largest_N}")
plt.legend()
plt.tight_layout()
plt.savefig(f"figures/{PREFIX}_runtime_decomposition_largest_N.png", dpi=300)
plt.close()

# ------------------------------------------------
# 5. Non-clearing decomposition for largest N
# ------------------------------------------------

small_components = [
    "counting_mean",
    "probabilities_mean",
    "updating_mean",
    "order_generation_mean",
    "price_update_mean",
    "output_mean",
]

small_labels = [
    "Counting",
    "Probabilities",
    "Type updating",
    "Order generation",
    "Price update",
    "Output",
]

bottom = [0.0] * len(plot_df)

plt.figure(figsize=(7, 4.5))

for comp, label in zip(small_components, small_labels):
    values = plot_df[comp].values

    plt.bar(
        x,
        values,
        bottom=bottom,
        label=label,
    )

    bottom = [b + v for b, v in zip(bottom, values)]

plt.xticks(x, plot_df["threads"].astype(str))
plt.xlabel("Number of threads")
plt.ylabel("Average time per step (ms)")
plt.title(f"{TITLE_PREFIX}: non-clearing decomposition at N={largest_N}")
plt.legend(fontsize=8)
plt.tight_layout()
plt.savefig(f"figures/{PREFIX}_non_clearing_decomposition_largest_N.png", dpi=300)
plt.close()

# ------------------------------------------------
# 6. Clearing share for largest N
# ------------------------------------------------

plot_df["clearing_share"] = (
    plot_df["clearing_mean"] / plot_df["avg_step_ms_mean"]
)

plt.figure(figsize=(7, 4.5))

plt.plot(
    plot_df["threads"],
    100 * plot_df["clearing_share"],
    marker="o",
)

plt.xlabel("Number of threads")
plt.ylabel("Clearing share of step time (%)")
plt.title(f"{TITLE_PREFIX}: clearing share at N={largest_N}")
plt.grid(True, linestyle="--", linewidth=0.5)
plt.tight_layout()
plt.savefig(f"figures/{PREFIX}_clearing_share_largest_N.png", dpi=300)
plt.close()

# ------------------------------------------------
# 7. Save aggregated tables
# ------------------------------------------------

agg.to_csv(f"results/benchmarks_{PREFIX}_aggregated.csv", index=False)
speedup_df.to_csv(f"results/benchmarks_{PREFIX}_speedup.csv", index=False)

print("Saved figures:")
print(f"  figures/{PREFIX}_avg_step_time.png")
print(f"  figures/{PREFIX}_speedup_vs_N.png")
print(f"  figures/{PREFIX}_speedup_vs_threads.png")
print(f"  figures/{PREFIX}_runtime_decomposition_largest_N.png")
print(f"  figures/{PREFIX}_non_clearing_decomposition_largest_N.png")
print(f"  figures/{PREFIX}_clearing_share_largest_N.png")
print("Saved aggregated tables:")
print(f"  results/benchmarks_{PREFIX}_aggregated.csv")
print(f"  results/benchmarks_{PREFIX}_speedup.csv")