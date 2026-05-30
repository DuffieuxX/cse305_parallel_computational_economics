#ifndef CUDA_MARKET_HPP
#define CUDA_MARKET_HPP
#ifndef N_AGENTS

#define N_AGENTS 1000

#endif
struct CountsGPU {
    int nb_optimists = 0;
    int nb_pessimists = 0;
    int nb_fundamentalists = 0;
};

struct ParamsGPU {
    int N = N_AGENTS;
    int T = 1000;
    int seed = 42;
    double dt = 1.0;
    double p0 = 100.0;
    double pf0 = 100.0;
    double initial_cash = 10000.0;
    double initial_inventory = 100.0;
    double price_min = 50.0;
    double price_max = 200.0;
    double tick_size = 0.01;


    double sigma_pf = 0.005;
    double nu_opinion = 0.05;
    double switching_speed = 0.01;
    double a_herding = 1.0;
    double a_trend = 1.0;
    double order_size = 10.0;
    double sigma_aggressiveness = 0.02;
    double m_min = 0.005;
    double gamma_f = 2.0;

};

struct MarketGPU {
    double p;
    double p_prev;
    double pf;
};

struct ProbsGPU {
    double plus_to_minus = 0.0;
    double plus_to_fund = 0.0;
    double minus_to_plus = 0.0;
    double minus_to_fund = 0.0;
    double fund_to_plus = 0.0;
    double fund_to_minus = 0.0;
};

struct AgentGPU {
    int type; // 0 = Optimist, 1 = Pessimist, 2 = Fundamentalist
    double cash;
    double inventory;
    double chartist_order_size;
    double aggressiveness;
};

struct OrderGPU {
    int agent_id;
    int side; // 1 = buy, -1 = sell, 0 = no order
    double quantity;
    double price;
};


// CUDA device memory context
struct CudaMarketContext {
    AgentGPU* d_agents = nullptr;
    OrderGPU* d_orders = nullptr;

    double* d_demand_bins = nullptr;
    double* d_supply_bins = nullptr;
    double* d_cum_demand = nullptr;
    double* d_cum_supply = nullptr;
    int* d_counts = nullptr;
    int* d_clearing_bin = nullptr;
    double* d_clearing_price = nullptr;
    double* d_matched_volume = nullptr;
    double* d_total_demand = nullptr;
    double* d_total_supply = nullptr;

    int N = 0;
    int K = 0;
};

void initialize_cuda_market(CudaMarketContext& ctx, AgentGPU* h_agents, int N);
double run_cuda_clearing_step(CudaMarketContext& ctx, CountsGPU& counts, const ParamsGPU& params, const MarketGPU& market, const ProbsGPU& probs, unsigned long long seed);
void finalize_cuda_market(CudaMarketContext& ctx);

// declaration of host-side model functions
CountsGPU count_agents_host(const AgentGPU* agents, int N);
double sentiment_host(const CountsGPU& counts);
double switch_probability_host(double nu, double U, double dt);
ProbsGPU compute_probabilities_host(const ParamsGPU& params, const MarketGPU& market, const CountsGPU& counts);

#ifdef __CUDACC__

__global__
void update_agents_kernel(
    AgentGPU* agents,
    int N,
    ProbsGPU probs,
    unsigned long long seed
);

__global__
void generate_orders_kernel(
    AgentGPU* agents,
    OrderGPU* orders,
    int N,
    ParamsGPU params,
    MarketGPU market
);

#endif

#endif // CUDA_MARKET_HPP