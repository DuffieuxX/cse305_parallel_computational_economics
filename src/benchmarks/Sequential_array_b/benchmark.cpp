#include "market.hpp"

#include <chrono>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <vector>

int main() {
    std::filesystem::create_directories("results");

    std::vector<int> agent_grid = {
        10000,
        100000,
        500000
    };

    std::vector<int> step_grid = {
        40
    };

    int repeats = 1;
    int base_seed = 42;

    std::ofstream out("results/benchmarks_sequential.csv");

    if (!out.is_open()) {
        std::cerr << "Error: could not open results/benchmarks_sequential.csv\n";
        return 1;
    }

    out << "model,"
        << "version,"
        << "N,"
        << "T,"
        << "run,"
        << "seed,"
        << "total_ms,"
        << "avg_step_ms,"
        << "counting_total_ms,"
        << "probabilities_total_ms,"
        << "updating_total_ms,"
        << "order_generation_total_ms,"
        << "clearing_total_ms,"
        << "price_update_total_ms,"
        << "output_total_ms,"
        << "counting_avg_step_ms,"
        << "probabilities_avg_step_ms,"
        << "updating_avg_step_ms,"
        << "order_generation_avg_step_ms,"
        << "clearing_avg_step_ms,"
        << "price_update_avg_step_ms,"
        << "output_avg_step_ms,"
        << "other_total_ms,"
        << "other_avg_step_ms"
        << "\n";

    for (int T : step_grid) {
        for (int N : agent_grid) {
            for (int r = 0; r < repeats; ++r) {
                Params params;

                params.N = N;
                params.T = T;
                params.seed = base_seed + r;

                // Pure compute benchmark. Set true only when benchmarking CSV I/O.
                params.write_output = false;
                params.verbose = false;

                auto start = std::chrono::high_resolution_clock::now();
                SimTimes times = run_simulation(params);
                auto end = std::chrono::high_resolution_clock::now();

                double total_ms =
                    std::chrono::duration<double, std::milli>(end - start).count();

                double measured_ms =
                    times.counting
                    + times.probabilities
                    + times.updating
                    + times.order_generation
                    + times.clearing
                    + times.price_update
                    + times.output;

                double other_ms = total_ms - measured_ms;
                double T_double = static_cast<double>(params.T);

                out << "sequential,"
                    << "V1,"
                    << N << ","
                    << T << ","
                    << r + 1 << ","
                    << params.seed << ","
                    << total_ms << ","
                    << total_ms / T_double << ","
                    << times.counting << ","
                    << times.probabilities << ","
                    << times.updating << ","
                    << times.order_generation << ","
                    << times.clearing << ","
                    << times.price_update << ","
                    << times.output << ","
                    << times.counting / T_double << ","
                    << times.probabilities / T_double << ","
                    << times.updating / T_double << ","
                    << times.order_generation / T_double << ","
                    << times.clearing / T_double << ","
                    << times.price_update / T_double << ","
                    << times.output / T_double << ","
                    << other_ms << ","
                    << other_ms / T_double
                    << "\n";

                std::cout << "Done: sequential"
                          << " N=" << N
                          << " T=" << T
                          << " run=" << r + 1
                          << " total_ms=" << total_ms
                          << " avg_step_ms=" << total_ms / T_double
                          << " counting=" << times.counting / T_double
                          << " probabilities=" << times.probabilities / T_double
                          << " updating=" << times.updating / T_double
                          << " order_generation=" << times.order_generation / T_double
                          << " clearing=" << times.clearing / T_double
                          << " price_update=" << times.price_update / T_double
                          << " output=" << times.output / T_double
                          << "\n";
            }
        }
    }

    out.close();

    std::cout << "Benchmark completed. Results written to results/benchmarks_sequential.csv\n";

    return 0;
}
