#include "market.hpp"

int main(int argc, char* argv[]) {
    std::filesystem::create_directories("results");
    Params params;
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--agents" && i + 1 < argc) {
            params.N = std::stoi(argv[++i]);
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
        else {
            std::cerr << "Unknown or incomplete argument: " << arg << "\n";
            return 1;
        }
    }
    run_simulation(params);
    return 0;
}