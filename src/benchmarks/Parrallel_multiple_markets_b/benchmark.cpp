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
        100000
    };

    const std::vector<int> thread_grid = {
        1,
        2,
        4
    };

    const int T = 20;
    const int runs = 4;
    const int base_seed = 42;

    const std::string output_path =
        "results/benchmarks_parallel_multiple_markets_exact.csv";

    std::ofstream out(output_path);

    if (!out.is_open()) {
        std::cerr << "Error: could not open " << output_path << "\n";
        return 1;
    }

    out << "model,"
        << "version,"
        << "N,"
        << "markets,"
        << "T,"
        << "threads,"
        << "run,"
        << "seed,"
        << "wall_ms,"
        << "avg_step_ms,"
        << "counting_total_ms,"
        << "probabilities_total_ms,"
        << "updating_total_ms,"
        << "market_processing_total_ms,"
        << "price_update_total_ms,"
        << "output_total_ms,"
        << "counting_avg_step_ms,"
        << "probabilities_avg_step_ms,"
        << "updating_avg_step_ms,"
        << "market_processing_avg_step_ms,"
        << "price_update_avg_step_ms,"
        << "output_avg_step_ms"
        << "\n";

    for (int N : N_grid) {
        for (int threads : thread_grid) {
            for (int r = 0; r < runs; ++r) {

                Params params;

                params.N = N;
                params.T = T;
                params.nb_threads = threads;
                params.seed = base_seed + r;

                params.write_output = false;
                params.verbose = false;

                params.rebuild_derived();

                auto start = std::chrono::high_resolution_clock::now();

                SimTimes times = run_simulation(params);

                auto end = std::chrono::high_resolution_clock::now();

                double wall_ms =
                    std::chrono::duration<double, std::milli>(
                        end - start
                    ).count();

                double denom = static_cast<double>(params.T);

                out << "CPU,"
                    << "Parallel multiple markets exact,"
                    << params.N << ","
                    << params.nb_markets << ","
                    << params.T << ","
                    << params.nb_threads << ","
                    << r << ","
                    << params.seed << ","
                    << wall_ms << ","
                    << wall_ms / denom << ","
                    << times.counting << ","
                    << times.probabilities << ","
                    << times.updating << ","
                    << times.market_processing << ","
                    << times.price_update << ","
                    << times.output << ","
                    << times.counting / denom << ","
                    << times.probabilities / denom << ","
                    << times.updating / denom << ","
                    << times.market_processing / denom << ","
                    << times.price_update / denom << ","
                    << times.output / denom
                    << "\n";

                std::cout << "Done: parallel multiple markets exact"
                          << " markets=" << params.nb_markets
                          << " N=" << params.N
                          << " threads=" << params.nb_threads
                          << " T=" << params.T
                          << " run=" << r + 1
                          << " avg_step_ms=" << wall_ms / denom
                          << " counting=" << times.counting / denom
                          << " probabilities=" << times.probabilities / denom
                          << " updating=" << times.updating / denom
                          << " market_processing=" << times.market_processing / denom
                          << " price_update=" << times.price_update / denom
                          << " output=" << times.output / denom
                          << "\n";
            }
        }
    }

    out.close();

    std::cout << "Benchmark completed. Results written to "
              << output_path << "\n";

    return 0;
}