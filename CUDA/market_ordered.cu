#include "cuda_market.hpp"
#include <cuda_runtime.h>
#include <cmath>

// (parallel GPU kernel) separates generated orders into bid and ask arrays.
// atomicAdd gives each valid order a unique position in the corresponding array.
__global__
void split_orders_kernel(const OrderGPU* orders, OrderGPU* bids, OrderGPU* asks, int* nb_bids, int* nb_asks, int N) {
    
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (i >= N){
        return;
    }

    OrderGPU o = orders[i];

    if (o.side == 1 && o.quantity > 0.0) {
        bids[atomicAdd(nb_bids, 1)] = o;
    } else if (o.side == -1 && o.quantity > 0.0) {
        asks[atomicAdd(nb_asks, 1)] = o;
    }
}

__device__
void swap_orders(OrderGPU& a, OrderGPU& b) {
    OrderGPU tmp = a;
    a = b;
    b = tmp;
}

// (single-thread GPU kernel) sorts bids/asks and computes the batch clearing price (sequential for now)
__global__
void compute_clearing_price_kernel(OrderGPU* bids, OrderGPU* asks, int* nb_bids, int* nb_asks, int* nb_trades, double* clearing_price, double previous_price) {
    int B = *nb_bids, A = *nb_asks;

    for (int i = 0; i < B; ++i)
        for (int j = i + 1; j < B; ++j)
            if (bids[j].price > bids[i].price) swap_orders(bids[i], bids[j]);

    for (int i = 0; i < A; ++i)
        for (int j = i + 1; j < A; ++j)
            if (asks[j].price < asks[i].price) swap_orders(asks[i], asks[j]);

    int M = (B < A) ? B : A;
    int K = 0;

    while (K < M && bids[K].price >= asks[K].price) ++K;

    *nb_trades = K;
    *clearing_price = (K > 0)
        ? 0.5 * (bids[K - 1].price + asks[K - 1].price)
        : previous_price;
}

// (parallel GPU kernel) one thread settles one matched bid-ask pair.
__global__
void settle_trades_kernel(AgentGPU* agents, const OrderGPU* bids, const OrderGPU* asks, int* nb_trades, double* clearing_price) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= *nb_trades) return;

    OrderGPU bid = bids[i], ask = asks[i];
    double price = *clearing_price;
    double q = fmin(bid.quantity, ask.quantity);

    agents[bid.agent_id].cash -= q * price;
    agents[bid.agent_id].inventory += q;

    agents[ask.agent_id].cash += q * price;
    agents[ask.agent_id].inventory -= q;
}

// (host side) allocates memory, launches kernels in order,
// copies updated agents and clearing price back to the CPU.
double run_cuda_clearing_step(AgentGPU* h_agents, int N, const ParamsGPU& params, const MarketGPU& market, const ProbsGPU& probs, unsigned long long seed) {
    
    const int threads_per_block = 256;
    const int blocks_per_grid = (N + threads_per_block - 1) / threads_per_block;
    
    AgentGPU* d_agents = nullptr;
    OrderGPU* d_orders = nullptr;
    OrderGPU* d_bids = nullptr;
    OrderGPU* d_asks = nullptr;

    int* d_nb_bids = nullptr;
    int* d_nb_asks = nullptr;
    int* d_nb_trades = nullptr;
    double* d_clearing_price = nullptr;

    cudaMalloc(&d_agents, N * sizeof(AgentGPU));
    cudaMalloc(&d_orders, N * sizeof(OrderGPU));
    cudaMalloc(&d_bids, N * sizeof(OrderGPU));
    cudaMalloc(&d_asks, N * sizeof(OrderGPU));
    cudaMalloc(&d_nb_bids, sizeof(int));
    cudaMalloc(&d_nb_asks, sizeof(int));
    cudaMalloc(&d_nb_trades, sizeof(int));
    cudaMalloc(&d_clearing_price, sizeof(double));

    cudaMemcpy(d_agents, h_agents, N * sizeof(AgentGPU), cudaMemcpyHostToDevice);
    cudaMemset(d_nb_bids, 0, sizeof(int));
    cudaMemset(d_nb_asks, 0, sizeof(int));
    cudaMemset(d_nb_trades, 0, sizeof(int));

    update_agents_kernel<<<blocks_per_grid, threads_per_block>>>(d_agents, N, probs, seed);
    generate_orders_kernel<<<blocks_per_grid, threads_per_block>>>(d_agents, d_orders, N, params, market);
    split_orders_kernel<<<blocks_per_grid, threads_per_block>>>(d_orders, d_bids, d_asks, d_nb_bids, d_nb_asks, N);
    compute_clearing_price_kernel<<<1, 1>>>(d_bids, d_asks, d_nb_bids, d_nb_asks, d_nb_trades, d_clearing_price, market.p);
    settle_trades_kernel<<<blocks_per_grid, threads_per_block>>>(d_agents, d_bids, d_asks, d_nb_trades, d_clearing_price);

    cudaDeviceSynchronize();
    
    double h_clearing_price = market.p;

    cudaMemcpy(&h_clearing_price, d_clearing_price, sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_agents, d_agents, N * sizeof(AgentGPU), cudaMemcpyDeviceToHost);

    cudaFree(d_agents);
    cudaFree(d_orders);
    cudaFree(d_bids);
    cudaFree(d_asks);
    cudaFree(d_nb_bids);
    cudaFree(d_nb_asks);
    cudaFree(d_nb_trades);
    cudaFree(d_clearing_price);

    return h_clearing_price;
}