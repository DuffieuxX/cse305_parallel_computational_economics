#include "market.hpp"
#include <cuda_runtime.h>
#include <cmath>

__device__
int price_to_bin(double price, const ParamsGPU& params, int K) {
    int bin = static_cast<int>((price - params.price_min) / params.tick_size);
    if (bin < 0) return 0;
    if (bin >= K) return K - 1;
    return bin;
}

//one thread reads one order and adds its quantity to a price bin
__global__
void histogram_orders_kernel(const OrderGPU* orders, double* demand_bins, double* supply_bins, int N, ParamsGPU params, int K) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;

    OrderGPU o = orders[i];
    if (o.side == 0 || o.quantity <= 0.0) return;

    int bin = price_to_bin(o.price, params, K);

    if (o.side == 1) atomicAdd(&demand_bins[bin], o.quantity);
    else if (o.side == -1) atomicAdd(&supply_bins[bin], o.quantity);
}


// (GPU kernel) scans price bins + computes the clearing price
__global__
void compute_histogram_clearing_kernel(
    const double* demand_bins,
    const double* supply_bins,
    double* cum_demand,
    double* cum_supply,
    int K,
    ParamsGPU params,
    double previous_price,
    int* clearing_bin,
    double* clearing_price,
    double* matched_volume,
    double* total_demand,
    double* total_supply
) {
    double running_supply = 0.0;
    for (int i = 0; i < K; ++i) {
        running_supply += supply_bins[i];
        cum_supply[i] = running_supply;
    }

    double running_demand = 0.0;
    for (int i = K - 1; i >= 0; --i) {
        running_demand += demand_bins[i];
        cum_demand[i] = running_demand;
    }

    int best_bin = -1;
    double best_volume = 0.0;

    for (int i = 0; i < K; ++i) {
        double executable = fmin(cum_demand[i], cum_supply[i]);
        if (executable > best_volume) {
            best_volume = executable;
            best_bin = i;
        }
    }

    if (best_bin >= 0 && best_volume > 0.0) {
        *clearing_bin = best_bin;
        *clearing_price = params.price_min + best_bin * params.tick_size;
        *matched_volume = best_volume;
        *total_demand = cum_demand[best_bin];
        *total_supply = cum_supply[best_bin];
    } else {
        *clearing_bin = -1;
        *clearing_price = previous_price;
        *matched_volume = 0.0;
        *total_demand = 0.0;
        *total_supply = 0.0;
    }
}

// (GPU kernel) proportional settlement at the clearing price
__global__
void settle_histogram_trades_kernel(
    AgentGPU* agents,
    const OrderGPU* orders,
    int N,
    double* clearing_price,
    double* matched_volume,
    double* total_demand,
    double* total_supply
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;

    OrderGPU o = orders[i];
    if (o.side == 0 || o.quantity <= 0.0) return;

    double price = *clearing_price;
    double V = *matched_volume;

    if (V <= 0.0) return;

    if (o.side == 1 && o.price >= price && *total_demand > 0.0) {
        double q = o.quantity * V / (*total_demand);
        agents[o.agent_id].cash -= q * price;
        agents[o.agent_id].inventory += q;
    } else if (o.side == -1 && o.price <= price && *total_supply > 0.0) {
        double q = o.quantity * V / (*total_supply);
        agents[o.agent_id].cash += q * price;
        agents[o.agent_id].inventory -= q;
    }
}

// (GPU kernel) counts how many agents are optimists, pessimists, and fundamentalists
__global__
void count_agents_kernel(const AgentGPU* agents, int N, int* counts) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;

    atomicAdd(&counts[agents[i].type], 1);
}

// #1 (host side) allocate device memory and copy initial agents to GPU
void initialize_cuda_market(CudaMarketContext& ctx, AgentGPU* h_agents, int N) {
    ctx.N = N;

    ParamsGPU default_params;
    ctx.K = static_cast<int>((default_params.price_max - default_params.price_min) / default_params.tick_size) + 1;

    cudaMalloc(&ctx.d_agents, N * sizeof(AgentGPU));
    cudaMalloc(&ctx.d_orders, N * sizeof(OrderGPU));

    cudaMalloc(&ctx.d_demand_bins, ctx.K * sizeof(double));
    cudaMalloc(&ctx.d_supply_bins, ctx.K * sizeof(double));
    cudaMalloc(&ctx.d_cum_demand, ctx.K * sizeof(double));
    cudaMalloc(&ctx.d_cum_supply, ctx.K * sizeof(double));

    cudaMalloc(&ctx.d_counts, 3 * sizeof(int));
    cudaMalloc(&ctx.d_clearing_bin, sizeof(int));

    cudaMalloc(&ctx.d_clearing_price, sizeof(double));
    cudaMalloc(&ctx.d_matched_volume, sizeof(double));
    cudaMalloc(&ctx.d_total_demand, sizeof(double));
    cudaMalloc(&ctx.d_total_supply, sizeof(double));

    cudaMemcpy(ctx.d_agents, h_agents, N * sizeof(AgentGPU), cudaMemcpyHostToDevice);
}

// #2 (host side) reset arrays, launch kernels in order, synchronize, copy back small outputs
double run_cuda_clearing_step(
    CudaMarketContext& ctx,
    CountsGPU& counts,
    const ParamsGPU& params,
    const MarketGPU& market,
    const ProbsGPU& probs,
    unsigned long long seed
) {
    const int threads_per_block = 256;
    const int blocks_per_grid = (ctx.N + threads_per_block - 1) / threads_per_block;

    // Reset device arrays for this timestep.
    cudaMemset(ctx.d_demand_bins, 0, ctx.K * sizeof(double));
    cudaMemset(ctx.d_supply_bins, 0, ctx.K * sizeof(double));
    cudaMemset(ctx.d_counts, 0, 3 * sizeof(int));

    // Ordered kernel launches.
    update_agents_kernel<<<blocks_per_grid, threads_per_block>>>(ctx.d_agents, ctx.N, probs, seed);
    generate_orders_kernel<<<blocks_per_grid, threads_per_block>>>(ctx.d_agents, ctx.d_orders, ctx.N, params, market);

    histogram_orders_kernel<<<blocks_per_grid, threads_per_block>>>(ctx.d_orders,ctx.d_demand_bins,ctx.d_supply_bins,ctx.N,params,ctx.K);

    compute_histogram_clearing_kernel<<<1, 1>>>(ctx.d_demand_bins,ctx.d_supply_bins,ctx.d_cum_demand,
        ctx.d_cum_supply,
        ctx.K,
        params,
        market.p,
        ctx.d_clearing_bin,
        ctx.d_clearing_price,
        ctx.d_matched_volume,
        ctx.d_total_demand,
        ctx.d_total_supply
    );

    settle_histogram_trades_kernel<<<blocks_per_grid, threads_per_block>>>(
        ctx.d_agents,
        ctx.d_orders,
        ctx.N,
        ctx.d_clearing_price,
        ctx.d_matched_volume,
        ctx.d_total_demand,
        ctx.d_total_supply
    );

    count_agents_kernel<<<blocks_per_grid, threads_per_block>>>(ctx.d_agents, ctx.N, ctx.d_counts);

    cudaDeviceSynchronize();

    // copy back small aggregate outputs to host
    double h_clearing_price = market.p;
    int h_counts[3];

    cudaMemcpy(&h_clearing_price, ctx.d_clearing_price, sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_counts, ctx.d_counts, 3 * sizeof(int), cudaMemcpyDeviceToHost);

    counts.nb_optimists = h_counts[0];
    counts.nb_pessimists = h_counts[1];
    counts.nb_fundamentalists = h_counts[2];

    return h_clearing_price;
}

// #3 (host side) free device memory
void finalize_cuda_market(CudaMarketContext& ctx) {
    cudaFree(ctx.d_agents);
    cudaFree(ctx.d_orders);

    cudaFree(ctx.d_demand_bins);
    cudaFree(ctx.d_supply_bins);
    cudaFree(ctx.d_cum_demand);
    cudaFree(ctx.d_cum_supply);

    cudaFree(ctx.d_counts);
    cudaFree(ctx.d_clearing_bin);

    cudaFree(ctx.d_clearing_price);
    cudaFree(ctx.d_matched_volume);
    cudaFree(ctx.d_total_demand);
    cudaFree(ctx.d_total_supply);

    ctx = CudaMarketContext{};
}