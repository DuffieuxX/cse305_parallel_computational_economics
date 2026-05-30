import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("benchmarks/gpu_results.csv")

for mech in df["mechanism"].unique():
    sub = df[df["mechanism"] == mech]
    plt.plot(sub["N"], sub["avg_step_ms"], marker="o", label=mech)

plt.xlabel("Number of agents N")
plt.ylabel("Average runtime per step (ms)")
plt.xscale("log")
plt.legend()
plt.tight_layout()
plt.savefig("benchmarks/gpu_runtime_by_N.pdf")
plt.show()