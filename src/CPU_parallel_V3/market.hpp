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

// #1 Data structure 
struct Params {

    int N = 100000; //number of agents 
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

// Forward declarations
struct Order;
struct Market;
struct Counts;

struct Agent{
    enum class Agent_type {Optimist,Pessimist,Fundamentalist};
    Agent_type type; //wether agent is optimist, pessimist or fundamentalist
    int agent_id;
    double cash; //quantity of cash of the agent >=0 (no borrowing)
    double asset_inventory; // quantity of asset held by client >=0 (no shorting)
    double chartist_order_size; // order size when the agent is a chartist 
    double aggressiveness; // aggressiveness of agent when deciding order price 
    
    Agent(Agent_type type,int agent_id,Params& params,std::mt19937& rng);
    
    Order new_order(const Market& market, const Params& params) const;
};


struct Counts {
    int nb_optimists = 0; 
    int nb_pessimists = 0;
    int nb_fundamentalists = 0;
};

struct Market {
    double p;
    double p_prev;
    double pf;
};

struct Probs {
    double plus_to_minus = 0.0;
    double plus_to_fund = 0.0;

    double minus_to_plus = 0.0;
    double minus_to_fund = 0.0;

    double fund_to_plus = 0.0;
    double fund_to_minus = 0.0;
};


struct Order{
    bool buy; 
    double quantity;
    double price; 
    int agent_id; 
    Order(bool buy, double quantity, double price, int agent_id);
};

struct Order_book {
    std::vector<Order> order_storage; //will store the real orders

    std::vector<Order*> bids;
    std::vector<Order*> asks;

    std::size_t bid_head = 0;
    std::size_t ask_head = 0;

    double volume_weighted_sum = 0;
    double volume = 0;

    Order_book(Params& params);

    void add_order(std::vector<Agent*>& agents, const Order& new_order);
};

struct SimTimes {
    double counting = 0;
    double updating = 0;
    double adding   = 0;
};

// declarations of output functions
void write_header(std::ofstream& out);
void write_row(std::ofstream& out, int t, Market market, double new_pric, double epsilon, Counts counts);

// declarations of main simulation loop
SimTimes run_simulation( Params& params);

// Utility function declarations
double uniform(std::mt19937& rng, double min, double max);
double normal(std::mt19937& rng, double mean, double sigma_squared);
double half_normal(std::mt19937& rng, double mean, double sigma_squared);
double switch_probability(double nu, double U, double dt);
double uniform01(std::mt19937& rng);

// Model function declarations
std::vector<Agent*> initialize_agents( Params& params, std::mt19937& rng);
Counts count_agents( std::vector<Agent*>& agents, Params& params);
double sentiment( Counts& counts);
Probs compute_probabilities(Params& params, Market& market, Counts& counts);
Counts update_agents(Params& params, std::vector<Agent*>& agents, Probs& probs, std::mt19937& rng);
double update_price(Order_book& order_book);