import subprocess
import re
import csv
import os

NVCC = "/usr/local/cuda/bin/nvcc"

mechanisms = {
    "histogram": {
        "file": "CUDA/market.cu",
        "Ns": [1000, 5000, 10000, 50000, 100000, 1000000],
    },
    "ordered": {
        "file": "CUDA/market_ordered.cu",
        "Ns": [1000, 5000, 10000, 20000],
    },
    "fcfs": {
        "file": "CUDA/market_FCFS.cu",
        "Ns": [1000, 5000, 10000,50000],
    },
}

os.makedirs("CUDA/benchmarks", exist_ok=True)

with open("CUDA/benchmarks/gpu_results.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["mechanism", "N", "runtime_ms", "avg_step_ms"])

    for mechanism, config in mechanisms.items():
        market_file = config["file"]

        for N in config["Ns"]:
            compile_cmd = [
                NVCC,
                "-arch=sm_60",
                "-std=c++17",
                "-I/usr/local/cuda/include",
                "-ICUDA",
                f"-DN_AGENTS={N}",
                "CUDA/main.cu",
                "CUDA/model.cu",
                market_file,
                "-o",
                "sim_cuda",
            ]

            subprocess.run(compile_cmd, check=True)

            out = subprocess.check_output(["./sim_cuda"], text=True)

            runtime = float(re.search(r"CUDA runtime:\s+([0-9.]+)", out).group(1))
            avg = float(re.search(r"CUDA average step runtime:\s+([0-9.]+)", out).group(1))

            writer.writerow([mechanism, N, runtime, avg])
            print(mechanism, N, avg)