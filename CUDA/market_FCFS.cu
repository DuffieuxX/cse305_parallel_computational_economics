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

__device__
double bin_to_price(int bin, const ParamsGPU& params) {
    return params.price_min + bin * params.tick_size;
}

__device__
double atomic_take(double* address, double requested) {
    unsigned long long int* addr_as_ull = reinterpret_cast<unsigned long long int*>(address);
    unsigned long long int old = *addr_as_ull;
    unsigned long long int assumed;

    do {
        assumed = old;
        double available = __longlong_as_double(assumed);
        double taken = fmin(available, requested);
        double remaining = available - taken;

        old = atomicCAS(
            addr_as_ull,
            assumed,
            __double_as_longlong(remaining)
        );

        if (old == assumed) return taken;

    } while (true);
}

// Parallel GPU kernel: one thread reads one order and adds its quantity to a price bin.
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


__global__
void settle_fcfs_bins_kernel(
    AgentGPU* agents,
    const OrderGPU* orders,
    double* demand_bins,
    double* supply_bins,
    int N,
    ParamsGPU params,
    int K,
    double* traded_value,
    double* traded_volume
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;

    OrderGPU o = orders[i];
    if (o.side == 0 || o.quantity <= 0.0) return;

    int limit_bin = price_to_bin(o.price, params, K);
    double remaining = o.quantity;
    double executed = 0.0;
    double value = 0.0;

    if (o.side == 1) {
        // buyer consume sell liquidity from cheapest asks up to the buyer's limit price.
        for (int b = 0; b <= limit_bin && remaining > 0.0; ++b) {
            double q = atomic_take(&supply_bins[b], remaining);
            if (q > 0.0) {
                double trade_price = bin_to_price(b, params);
                remaining -= q;
                executed += q;
                value += q * trade_price;
            }
        }

        if (executed > 0.0) {
            agents[o.agent_id].cash -= value;
            agents[o.agent_id].inventory += executed;
            atomicAdd(traded_value, value);
            atomicAdd(traded_volume, executed);
        }
    }

    else if (o.side == -1) {
        // celler consume buy liquidity from highest bids down to the seller's limit price.
        for (int b = K - 1; b >= limit_bin && remaining > 0.0; --b) {
            double q = atomic_take(&demand_bins[b], remaining);
            if (q > 0.0) {
                double trade_price = bin_to_price(b, params);
                remaining -= q;
                executed += q;
                value += q * trade_price;
            }
        }

        if (executed > 0.0) {
            agents[o.agent_id].cash += value;
            agents[o.agent_id].inventory -= executed;
            atomicAdd(traded_value, value);
            atomicAdd(traded_volume, executed);
        }
    }
}

__global__
void compute_fcfs_price_kernel(double* traded_value, double* traded_volume, double* clearing_price, double previous_price) {
    double volume = *traded_volume;

    if (volume > 0.0) {
        *clearing_price = (*traded_value) / volume;
    } else {
        *clearing_price = previous_price;
    }
}

__global__
void count_agents_kernel(const AgentGPU* agents, int N, int* counts) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;

    atomicAdd(&counts[agents[i].type], 1);
}

// #1 Host-side setup: allocate device memory and copy initial agents to GPU.
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

// #2 (host side) reset arrays, launch kernels in order, synchronize, copy back small outputs.
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

    cudaMemset(ctx.d_demand_bins, 0, ctx.K * sizeof(double));
    cudaMemset(ctx.d_supply_bins, 0, ctx.K * sizeof(double));
    cudaMemset(ctx.d_counts, 0, 3 * sizeof(int));

    cudaMemset(ctx.d_matched_volume, 0, sizeof(double)); // traded volume
    cudaMemset(ctx.d_total_demand, 0, sizeof(double));   // traded value

    update_agents_kernel<<<blocks_per_grid, threads_per_block>>>(ctx.d_agents, ctx.N, probs, seed);
    generate_orders_kernel<<<blocks_per_grid, threads_per_block>>>(ctx.d_agents, ctx.d_orders, ctx.N, params, market);

    histogram_orders_kernel<<<blocks_per_grid, threads_per_block>>>(ctx.d_orders, ctx.d_demand_bins, ctx.d_supply_bins, ctx.N, params, ctx.K);

    settle_fcfs_bins_kernel<<<blocks_per_grid, threads_per_block>>>(
        ctx.d_agents,
        ctx.d_orders,
        ctx.d_demand_bins,
        ctx.d_supply_bins,
        ctx.N,
        params,
        ctx.K,
        ctx.d_total_demand,
        ctx.d_matched_volume
    );

    compute_fcfs_price_kernel<<<1, 1>>>(
        ctx.d_total_demand,
        ctx.d_matched_volume,
        ctx.d_clearing_price,
        market.p
    );

    count_agents_kernel<<<blocks_per_grid, threads_per_block>>>(ctx.d_agents, ctx.N, ctx.d_counts);

    cudaDeviceSynchronize();

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