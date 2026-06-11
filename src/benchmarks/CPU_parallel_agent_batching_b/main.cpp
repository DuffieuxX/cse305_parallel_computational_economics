#include "market.hpp"
#include <chrono>

int main(int argc, char* argv[]) {
    std::filesystem::create_directories("results");
    Params params;
    bool time_exec = false;
    int nb_runs = 1;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--agents" && i + 1 < argc) {
            params.N = std::stoi(argv[++i]);
        }
        else if (arg == "--threads" && i + 1 < argc) {
            params.nb_threads = std::stoi(argv[++i]);
        }
        else if (arg == "--steps" && i + 1 < argc) {
            params.T = std::stoi(argv[++i]);
        }
        else if (arg == "--seed" && i + 1 < argc) {
            params.seed = static_cast<unsigned>(std::stoul(argv[++i]));
        }
        else if (arg == "--output" && i + 1 < argc) {
            params.output = argv[++i];
        }
        else if (arg == "--sigma-pf" && i + 1 < argc) {
            params.sigma_pf = std::stod(argv[++i]);
        }
        else if (arg == "--time") {
            time_exec = true;
        }
        else if (arg == "--runs" && i + 1 < argc) {
            nb_runs = std::stoi(argv[++i]);
        }
        else {
            std::cerr << "Unknown or incomplete argument: " << arg << "\n";
            return 1;
        }
    }

    if (time_exec) {
    double total_time = 0.0;
    double total_counting = 0.0;
    double total_probabilities = 0.0;
    double total_updating = 0.0;
    double total_order_generation = 0.0;
    double total_clearing = 0.0;
    double total_price_update = 0.0;
    double total_output = 0.0;

    for (int r = 0; r < nb_runs; r++) {
        auto start = std::chrono::high_resolution_clock::now();

        SimTimes times = run_simulation(params);

        auto end = std::chrono::high_resolution_clock::now();

        double ms = std::chrono::duration<double, std::milli>(
            end - start
        ).count();

        total_time += ms;
        total_counting += times.counting;
        total_probabilities += times.probabilities;
        total_updating += times.updating;
        total_order_generation += times.order_generation;
        total_clearing += times.clearing;
        total_price_update += times.price_update;
        total_output += times.output;

        std::cout << "Run " << r + 1 << ": " << ms << " ms"
                  << "  (counting=" << times.counting << " ms"
                  << "  probabilities=" << times.probabilities << " ms"
                  << "  updating=" << times.updating << " ms"
                  << "  order_generation=" << times.order_generation << " ms"
                  << "  clearing=" << times.clearing << " ms"
                  << "  price_update=" << times.price_update << " ms"
                  << "  output=" << times.output << " ms)\n";
    }

    if (nb_runs > 1) {
        std::cout << "\n=== Averages over " << nb_runs << " runs ===\n";
        std::cout << "Total            : " << total_time / nb_runs << " ms\n";
        std::cout << "Counting         : " << total_counting / nb_runs << " ms\n";
        std::cout << "Probabilities    : " << total_probabilities / nb_runs << " ms\n";
        std::cout << "Updating         : " << total_updating / nb_runs << " ms\n";
        std::cout << "Order generation : " << total_order_generation / nb_runs << " ms\n";
        std::cout << "Clearing         : " << total_clearing / nb_runs << " ms\n";
        std::cout << "Price update     : " << total_price_update / nb_runs << " ms\n";
        std::cout << "Output           : " << total_output / nb_runs << " ms\n";
    }

    } else {
        run_simulation(params);
    }

    return 0;
}