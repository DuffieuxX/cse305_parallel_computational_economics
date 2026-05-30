#include "market.hpp"
#include <cuda_runtime.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <iostream>
#include <random>

int main() {
    ParamsGPU params;

    MarketGPU market{params.p0, params.p0, params.pf0};
    AgentGPU* agents = new AgentGPU[params.N];

    std::mt19937 rng(params.seed);
    std::uniform_real_distribution<double> u01(0.0, 1.0);
    std::normal_distribution<double> aggr_dist(0.0, params.sigma_aggressiveness);
    std::normal_distribution<double> pf_noise_dist(0.0, params.sigma_pf);

    int n_plus = params.N / 3;
    int n_minus = params.N / 3;

    for (int i = 0; i < params.N; ++i) {
        agents[i].type = (i < n_plus) ? 0 : (i < n_plus + n_minus) ? 1 : 2;
        agents[i].cash = params.initial_cash;
        agents[i].inventory = params.initial_inventory;
        agents[i].chartist_order_size = 0.2 * params.order_size + 1.8 * params.order_size * u01(rng);
        agents[i].aggressiveness = std::abs(aggr_dist(rng));
    }

    std::shuffle(agents, agents + params.N, rng);

    CountsGPU counts = count_agents_host(agents, params.N);

    CudaMarketContext ctx;
    initialize_cuda_market(ctx, agents, params.N);

    auto start = std::chrono::high_resolution_clock::now();

    for (int t = 0; t < params.T; ++t) {
        market.pf *= std::exp(pf_noise_dist(rng));

        ProbsGPU probs = compute_probabilities_host(params, market, counts);
        double new_price = run_cuda_clearing_step(ctx, counts, params, market, probs, static_cast<unsigned long long>(params.seed + t));

        market.p_prev = market.p;
        market.p = new_price;
    }

    cudaDeviceSynchronize();

    auto end = std::chrono::high_resolution_clock::now();
    double elapsed_ms = std::chrono::duration<double, std::milli>(end - start).count();

    std::cout << "CUDA final price: " << market.p << std::endl;
    std::cout << "CUDA final fundamental value: " << market.pf << std::endl;
    std::cout << "CUDA runtime: " << elapsed_ms << " ms" << std::endl;
    std::cout << "CUDA average step runtime: " << elapsed_ms / static_cast<double>(params.T) << " ms" << std::endl;
    std::cout << "Final counts: optimists=" << counts.nb_optimists
              << ", pessimists=" << counts.nb_pessimists
              << ", fundamentalists=" << counts.nb_fundamentalists
              << std::endl;

    finalize_cuda_market(ctx);
    delete[] agents;
    return 0;
}