import os
import pandas as pd
import matplotlib.pyplot as plt

os.makedirs("figures", exist_ok=True)

df = pd.read_csv("results/benchmarks_sequential.csv")

print("Columns in CSV:")
print(df.columns.tolist())

group_cols = ["model", "version", "N", "T"]

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
# Graph 1: Average step time vs number of agents
# ------------------------------------------------

plt.figure(figsize=(7, 4.5))

for (model, version), sub in agg.groupby(["model", "version"]):
    sub = sub.sort_values("N")
    label = f"{model} {version}"

    plt.errorbar(
        sub["N"],
        sub["avg_step_ms_mean"],
        yerr=sub["avg_step_ms_std"],
        marker="o",
        capsize=3,
        label=label,
    )

plt.xscale("log")
plt.yscale("log")
plt.xlabel("Number of agents N")
plt.ylabel("Average time per step (ms)")
plt.title("Sequential baseline: average step time")
plt.grid(True, which="both", linestyle="--", linewidth=0.5)
plt.legend()
plt.tight_layout()
plt.savefig("figures/sequential_avg_step_time.png", dpi=300)
plt.close()

# ------------------------------------------------
# Graph 2: Runtime decomposition by component
# ------------------------------------------------

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

plot_df = agg.sort_values("N").copy()

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

plt.xticks(x, plot_df["N"].astype(str))
plt.xlabel("Number of agents N")
plt.ylabel("Average time per step (ms)")
plt.title("Sequential baseline: runtime decomposition")
plt.legend()
plt.tight_layout()
plt.savefig("figures/sequential_runtime_decomposition.png", dpi=300)
plt.close()

# ------------------------------------------------
# Graph 3: Runtime decomposition, excluding clearing
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

plt.xticks(x, plot_df["N"].astype(str))
plt.xlabel("Number of agents N")
plt.ylabel("Average time per step (ms)")
plt.title("Sequential baseline: non-clearing runtime decomposition")
plt.legend(fontsize=8)
plt.tight_layout()
plt.savefig("figures/sequential_non_clearing_decomposition.png", dpi=300)
plt.close()

# ------------------------------------------------
# Graph 4: Clearing share of total step time
# ------------------------------------------------

plot_df["clearing_share"] = (
    plot_df["clearing_mean"] / plot_df["avg_step_ms_mean"]
)

plt.figure(figsize=(7, 4.5))
plt.plot(
    plot_df["N"],
    100 * plot_df["clearing_share"],
    marker="o",
)

plt.xscale("log")
plt.xlabel("Number of agents N")
plt.ylabel("Clearing share of step time (%)")
plt.title("Sequential baseline: clearing share")
plt.grid(True, which="both", linestyle="--", linewidth=0.5)
plt.tight_layout()
plt.savefig("figures/sequential_clearing_share.png", dpi=300)
plt.close()

# ------------------------------------------------
# Save aggregated table
# ------------------------------------------------

agg.to_csv("results/benchmarks_sequential_aggregated.csv", index=False)

print("Saved figures:")
print("  figures/sequential_avg_step_time.png")
print("  figures/sequential_runtime_decomposition.png")
print("  figures/sequential_non_clearing_decomposition.png")
print("  figures/sequential_clearing_share.png")
print("Saved aggregated table:")
print("  results/benchmarks_sequential_aggregated.csv")