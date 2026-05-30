import subprocess
import re
import csv

NVCC = "/usr/local/cuda/bin/nvcc"

mechanisms = {
    "histogram": "CUDA/market.cu",
    "fcfs": "CUDA/market_FCFS.cu",
    "sort": "CUDA/market_sort.cu",
}

Ns = [1000, 5000, 10000, 50000, 100000]

with open("benchmarks/gpu_results.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["mechanism", "N", "runtime_ms", "avg_step_ms"])

    for mech, market_file in mechanisms.items():
        for N in Ns:
            cmd_compile = [
                NVCC, "-arch=sm_60", "-std=c++17",
                "-I/usr/local/cuda/include", "-ICUDA",
                f"-DN_AGENTS={N}",
                "CUDA/main.cu", "CUDA/model.cu", market_file,
                "-o", "sim_cuda"
            ]

            subprocess.run(cmd_compile, check=True)

            out = subprocess.check_output(["./sim_cuda"], text=True)

            runtime = float(re.search(r"CUDA runtime:\s+([0-9.]+)", out).group(1))
            avg = float(re.search(r"CUDA average step runtime:\s+([0-9.]+)", out).group(1))

            writer.writerow([mech, N, runtime, avg])
            print(mech, N, avg)