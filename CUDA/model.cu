#include "market.hpp"
#include <cmath>
#include <algorithm>

// (device side) generate uniform random nb in [0,1)
__device__
double device_uniform01(unsigned long long x) {
    x ^= x >> 12;
    x ^= x << 25;
    x ^= x >> 27;
    unsigned long long z = x * 2685821657736338717ULL;
    return (z >> 11) * (1.0 / 9007199254740992.0);
}


// (parallel GPU kernel) 1 thread generates at most 1 order per agent
// reproduces Agent::new_order (\src) logic
__global__
void generate_orders_kernel(AgentGPU* agents, OrderGPU* orders, int N, ParamsGPU params, MarketGPU market) {
    
    int i = blockIdx.x* blockDim.x + threadIdx.x;
    
    if (i >= N){
        return;
    }

    AgentGPU& a = agents[i];

    orders[i].agent_id = i;
    orders[i].side = 0;
    orders[i].quantity = 0.0;
    orders[i].price = 0.0;

    if (a.type == 0) { // Optimist buys
        
        double limit_price = market.p * (1.0 + a.aggressiveness);
        double quantity = fmin(a.chartist_order_size, a.cash / limit_price);

        if (quantity > 0.0) {
            orders[i].side = 1;
            orders[i].quantity = quantity;
            orders[i].price = limit_price;
        }
    }
    else if (a.type == 1) { // Pessimist sells
        double limit_price = market.p * (1.0 - a.aggressiveness);
        double quantity = fmin(a.chartist_order_size, a.inventory);

        if (quantity > 0.0) {
            orders[i].side = -1;
            orders[i].quantity = quantity;
            orders[i].price = limit_price;
        }
    }
    else { // Fundamentalist

        double m_t = (market.pf - market.p)/market.p;
        double fundamentalist_order_size = a.chartist_order_size * (1.0 + params.gamma_f * fabs(m_t));

        if (m_t >= params.m_min) {
    
            double limit_price = market.pf * (1.0 + a.aggressiveness / 4.0);
            double quantity = fmin(fundamentalist_order_size, a.cash / limit_price);
            
            if (quantity > 0.0) {
                orders[i].side = 1;
                orders[i].quantity = quantity;
                orders[i].price = limit_price;
            }
        }
        else if (m_t <= -params.m_min) {
            
            double limit_price = market.pf * (1.0 - a.aggressiveness / 4.0);
            double quantity = fmin(fundamentalist_order_size, a.inventory);

            if (quantity > 0.0) {
                orders[i].side = -1;
                orders[i].quantity = quantity;
                orders[i].price = limit_price;
            }
        }
    }
}

// (host side) counts agent types on the CPU
CountsGPU count_agents_host(const AgentGPU* agents, int N) {
    CountsGPU counts;
    counts.nb_optimists = 0;
    counts.nb_pessimists = 0;
    counts.nb_fundamentalists = 0;

    for (int i = 0; i < N; ++i) {
        if (agents[i].type == 0) {
            counts.nb_optimists++;
        } else if (agents[i].type == 1) {
            counts.nb_pessimists++;
        } else {
            counts.nb_fundamentalists++;
        }
    }
    return counts;
}

// (host side) compute sentiment x 
double sentiment_host(const CountsGPU& counts) {
    
    int nb_chartist = counts.nb_optimists + counts.nb_pessimists;

    if (nb_chartist == 0) {
        return 0.0;
    }

    return static_cast<double>(counts.nb_optimists - counts.nb_pessimists) / static_cast<double>(nb_chartist);
}

// (host side) compute switching probabilities
double switch_probability_host(double nu, double U, double dt) {
    
    double p = nu * std::exp(U) * dt;

    if (!std::isfinite(p)) {
        return 1.0;
    }
    if (p < 0.0){
        return 0.0;
    } 
    if (p > 1.0){
        return 1.0;
    }
    return p;
}


// (host side) computes transition probabilities before launching CUDA kernels
ProbsGPU compute_probabilities_host(const ParamsGPU& params, const MarketGPU& market, const CountsGPU& counts) {
    
    ProbsGPU probs;

    double r = std::log(market.p / market.p_prev);
    double x = sentiment_host(counts);
    double mispricing = std::log(market.pf / market.p);

    // within chartists: optimists <-> pessimists
    double U_minus_to_plus = params.a_herding * x + params.a_trend * r;
    double U_plus_to_minus = -U_minus_to_plus;

    probs.minus_to_plus = switch_probability_host(params.nu_opinion, U_minus_to_plus, params.dt);
    probs.plus_to_minus = switch_probability_host(params.nu_opinion, U_plus_to_minus, params.dt);

    // between chartists and fundamentalists
    double pi_plus = r;
    double pi_minus = -r;
    double pi_fund = std::abs(mispricing);

    probs.fund_to_plus = switch_probability_host(params.switching_speed, pi_plus - pi_fund, params.dt);
    probs.plus_to_fund = switch_probability_host(params.switching_speed, pi_fund - pi_plus, params.dt);
    probs.fund_to_minus = switch_probability_host(params.switching_speed, pi_minus - pi_fund, params.dt);
    probs.minus_to_fund = switch_probability_host(params.switching_speed, pi_fund - pi_minus, params.dt);

    return probs;
}


// (parallel GPU kernel) 1 thread = 1 agent's type udpate with transition probabilities
__global__
void update_agents_kernel(AgentGPU* agents, int N, ProbsGPU probs, unsigned long long seed){
    
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (i >= N){
        return;
    }

    double u = device_uniform01(seed + 1315423911ULL * i);
    int type = agents[i].type;
    
    if (type == 0) { // Optimist
        if (u < probs.plus_to_minus) {
            agents[i].type = 1;
        } else if (u < probs.plus_to_minus + probs.plus_to_fund) {
            agents[i].type = 2;
        }
    }
    else if (type == 1) { // Pessimist
        if (u < probs.minus_to_plus) {
            agents[i].type = 0;
        } else if (u < probs.minus_to_plus + probs.minus_to_fund) {
            agents[i].type = 2;
        }
    }
    else { // Fundamentalist
        if (u < probs.fund_to_plus) {
            agents[i].type = 0;
        } else if (u < probs.fund_to_plus + probs.fund_to_minus) {
            agents[i].type = 1;
        }
    }
}
