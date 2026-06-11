#include "market.hpp"

#include <chrono>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <vector>

int main() {
    std::filesystem::create_directories("results");

    const std::vector<int> N_grid = {
        10000,
        50000,
        100000,
        500000
    };

    const std::vector<int> thread_grid = {
        1,
        2,
        3,
        4
    };

    const int T = 30;
    const int runs = 3;
    const int base_seed = 42;

    const std::string output_path = "results/benchmarks_cpu_v4_threads.csv";

    std::ofstream out(output_path);

    if (!out.is_open()) {
        std::cerr << "Error: could not open " << output_path << "\n";
        return 1;
    }

    out << "model,"
        << "version,"
        << "N,"
        << "T,"
        << "threads,"
        << "run,"
        << "wall_ms,"
        << "avg_step_ms,"
        << "counting_avg_step_ms,"
        << "probabilities_avg_step_ms,"
        << "updating_avg_step_ms,"
        << "order_generation_avg_step_ms,"
        << "clearing_avg_step_ms,"
        << "price_update_avg_step_ms,"
        << "output_avg_step_ms"
        << "\n";

    for (int N : N_grid) {
        for (int threads : thread_grid) {
            for (int r = 0; r < runs; ++r) {

                Params params;

                params.N = N;
                params.T = T;
                params.seed = base_seed + r;
                params.nb_threads = threads;

                params.write_output = false;
                params.verbose = false;

                auto start = std::chrono::high_resolution_clock::now();

                SimTimes times = run_simulation(params);

                auto end = std::chrono::high_resolution_clock::now();

                double wall_ms = std::chrono::duration<double, std::milli>(
                    end - start
                ).count();

                double denom = static_cast<double>(params.T);

                out << "CPU,"
                    << "V4,"
                    << N << ","
                    << T << ","
                    << threads << ","
                    << r << ","
                    << wall_ms << ","
                    << wall_ms / denom << ","
                    << times.counting / denom << ","
                    << times.probabilities / denom << ","
                    << times.updating / denom << ","
                    << times.order_generation / denom << ","
                    << times.clearing / denom << ","
                    << times.price_update / denom << ","
                    << times.output / denom
                    << "\n";

                std::cout << "N=" << N
                          << " threads=" << threads
                          << " run=" << r + 1
                          << " avg_step_ms=" << wall_ms / denom
                          << "\n";
            }
        }
    }

    out.close();

    std::cout << "Saved " << output_path << "\n";

    return 0;
}