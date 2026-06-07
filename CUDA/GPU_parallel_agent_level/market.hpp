#ifndef MARKET_HPP
#define MARKET_HPP
#ifndef __CUDACC__ //
#define __host__
#define __device__
#define __global__
#endif

#include <cmath>
#include <fstream>
#include <iostream>
#include <random>
#include <string>
#include <vector>
#include <algorithm>
#include <filesystem>
#include<thread>
#include <limits>
#include <climits>
#include <chrono>
#include <stdexcept>

// #1 Data structure 
struct Params {

    int N = 10000; //number of agents 
    int T = 1000; // number of time periods

    int nb_threads=10;

    int chunk_size = N/nb_threads;
    
    int seed = 42;
    double dt = 1.0; //  period increments for strategy re-evaluation 
    double p0 = 100.0; // initial price 
    double pf0 = 100.0; // initial fundamental value
    double initial_cash=10000.0;//average initial cash of agents 
    double initial_inventory=100.0;//average initial inventory of agents 

    double sigma_pf = 0.005; // variance of log return of fundamental value
    double nu_opinion = 0.05; //frequency of re-evaluating opinion for chartists
    double switching_speed = 0.01; // baseline speed of switching between chartist and fundamentalist strategies
    double a_herding = 1.0; //influence of herding on chartist behaviour
    double a_trend = 1.0; // influence of temporary price trend on chartist behaviour 
    double order_size = 10.0; // average individual order size scale
    double sigma_aggressiveness=0.02; // variance of aggressiveness
    double gamma_f=2.0; // sensitivity of fundamentalist order size to mispricing
    double m_min=0.005; // minimum threshold on m_t for fundamentalists to trade

    std::string output = "results/simulation.csv";
};

// #2 same as Params for GPU
struct GpuParams {
    int N;
    int T;
    int seed;

    double dt;
    double p0;
    double pf0;
    double initial_cash;
    double initial_inventory;

    double sigma_pf;
    double nu_opinion;
    double switching_speed;
    double a_herding;
    double a_trend;
    double order_size;
    double sigma_aggressiveness;
    double gamma_f;
    double m_min;
};

inline GpuParams make_gpu_params(const Params& params) {
    GpuParams gpu_params;

    gpu_params.N = params.N;
    gpu_params.T = params.T;
    gpu_params.seed = params.seed;

    gpu_params.dt = params.dt;
    gpu_params.p0 = params.p0;
    gpu_params.pf0 = params.pf0;
    gpu_params.initial_cash = params.initial_cash;
    gpu_params.initial_inventory = params.initial_inventory;

    gpu_params.sigma_pf = params.sigma_pf;
    gpu_params.nu_opinion = params.nu_opinion;
    gpu_params.switching_speed = params.switching_speed;
    gpu_params.a_herding = params.a_herding;
    gpu_params.a_trend = params.a_trend;
    gpu_params.order_size = params.order_size;
    gpu_params.sigma_aggressiveness = params.sigma_aggressiveness;
    gpu_params.gamma_f = params.gamma_f;
    gpu_params.m_min = params.m_min;

    return gpu_params;
}

struct Market {
    double p;
    double p_prev;
    double pf;
};

struct Order{
    bool buy; 
    double quantity;
    double price; 
    int agent_id; 
    
    Order() = default; //constructor
    
    __host__ __device__
    Order(bool buy, double quantity, double price, int agent_id): buy(buy), quantity(quantity), price(price), agent_id(agent_id) {}

};


struct Counts {
    int nb_optimists = 0; 
    int nb_pessimists = 0;
    int nb_fundamentalists = 0;
};

struct Agent{
    enum class Agent_type {Optimist,Pessimist,Fundamentalist};
    Agent_type type; //wether agent is optimist, pessimist or fundamentalist
    int agent_id;
    double cash; //quantity of cash of the agent >=0 (no borrowing)
    double asset_inventory; // quantity of asset held by client >=0 (no shorting)
    double chartist_order_size; // order size when the agent is a chartist 
    double aggressiveness; // aggressiveness of agent when deciding order price 
    
    Agent() = default; //constructor to allocate array
    Agent(Agent_type type, int agent_id, Params& params, std::mt19937& rng);
    
    __host__ __device__
    Order make_order(Market market, GpuParams params) const {
        if (this->type == Agent::Agent_type::Optimist) {
            double limit_price = market.p * (1.0 + this->aggressiveness);
            double quantity = fmin(this->chartist_order_size, this->cash/limit_price);
            bool buy = true;

            return Order(buy, quantity, limit_price, this->agent_id);
        }
        else if (this->type == Agent::Agent_type::Pessimist) {
            double limit_price = market.p * (1.0 - this->aggressiveness);
            double quantity = fmin(this->chartist_order_size, this->asset_inventory);
            bool buy = false;

            return Order(buy, quantity, limit_price, this->agent_id);
        }
        else {
            double m_t = (market.pf - market.p) / market.p;

            if (m_t >= params.m_min) {
                bool buy = true;
                double limit_price = market.pf * (1.0 + this->aggressiveness / 4.0);
                double fundamentalist_order_size =
                    this->chartist_order_size * (1.0 + params.gamma_f * fabs(m_t));
                double quantity = fmin(fundamentalist_order_size, this->cash / limit_price);

                return Order(buy, quantity, limit_price, this->agent_id);
            }

            if (m_t <= -params.m_min) {
                bool buy = false;
                double limit_price = market.pf * (1.0 - this->aggressiveness / 4.0);
                double fundamentalist_order_size =
                    this->chartist_order_size * (1.0 + params.gamma_f * fabs(m_t));
                double quantity = fmin(fundamentalist_order_size, this->asset_inventory);

                return Order(buy, quantity, limit_price, this->agent_id);
            }
            else {
                return Order(true, 0.0, 0.0, this->agent_id);
            }
        }
    }
};


struct Probs {
    double plus_to_minus = 0.0;
    double plus_to_fund = 0.0;

    double minus_to_plus = 0.0;
    double minus_to_fund = 0.0;

    double fund_to_plus = 0.0;
    double fund_to_minus = 0.0;
};

struct Order_book {
    std::vector<Order*> bids;
    std::vector<Order*> asks;

    double volume_weighted_sum=0;
    double volume=0;

    Order_book( Params& params);
    ~Order_book();

    void add_order(Agent* agents, Order* new_order);
};

struct SimTimes {
    double counting = 0;
    double updating = 0;
    double adding   = 0;
};


// main simulation loop (declarations)
SimTimes run_simulation( Params& params);

// Utility function (declarations)
double uniform(std::mt19937& rng, double min, double max);
double normal(std::mt19937& rng, double mean, double sigma_squared);
double half_normal(std::mt19937& rng, double mean, double sigma_squared);
double switch_probability(double nu, double U, double dt);
double uniform01(std::mt19937& rng);

// Model function (declarations)
Agent* initialize_agents( Params& params, std::mt19937& rng);
Counts count_agents_gpu(Agent* agents, Params& params);
void update_agents_gpu( Params& params, GpuParams& gpu_params, Agent* agents, Probs& probs, int t );
void add_all_new_orders_gpu( Market& market, Params& params, GpuParams& gpu_params, Agent* agents, Order* orders, Order_book& order_book );
double sentiment( Counts& counts);
Probs compute_probabilities(Params& params, Market& market, Counts& counts);
double update_price(Order_book& order_book);

// output functions (declarations)
void write_header(std::ofstream& out);
void write_row(std::ofstream& out, int t, Market market, double new_price, double epsilon, Counts counts);

#endif