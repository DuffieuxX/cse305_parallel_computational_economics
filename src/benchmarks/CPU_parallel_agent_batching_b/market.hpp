#include <cmath>
#include <fstream>
#include <iostream>
#include <random>
#include <string>
#include <vector>
#include <algorithm>
#include <filesystem>
#include <thread>
#include <limits>
#include <chrono>

// #1 Data structure 
struct Params {

    int N = 100000; // number of agents
    int T = 1000;   // number of time periods

    int nb_threads = 1;
    int chunk_size = 0;

    int seed = 42;
    double dt = 1.0;
    double p0 = 100.0;
    double pf0 = 100.0;
    double initial_cash = 10000.0;
    double initial_inventory = 100.0;

    double sigma_pf = 0.005;
    double nu_opinion = 0.05;
    double switching_speed = 0.01;
    double a_herding = 1.0;
    double a_trend = 1.0;
    double order_size = 10.0;
    double sigma_aggressiveness = 0.02;
    double gamma_f = 2.0;
    double m_min = 0.005;

    std::string output = "results/simulation.csv";

    bool write_output = true;
    bool verbose = true;
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

    Order_book(Params& params, int nb_agents=0);

    void add_order(std::vector<Agent*>& agents, const Order& new_order);
};

struct SimTimes {
    double counting = 0.0;
    double probabilities = 0.0;
    double updating = 0.0;
    double order_generation = 0.0;
    double clearing = 0.0;
    double price_update = 0.0;
    double output = 0.0;
};




// declarations of output functions
void write_header(std::ofstream& out);

void write_row(
    std::ofstream& out,
    int t,
    Market market,
    double new_price,
    double epsilon,
    Counts counts
);

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

Probs compute_probabilities(
     Params& params,
     Market& market,
     Counts& counts
);

void update_agents(
    Params& params,
    std::vector<Agent*>& agents,
     Probs& probs,
    std::mt19937& rng
);

double update_price(Order_book& order_book);

void generate_all_new_orders_thread(
    int index_start,
    int index_end,
    const Market& market,
    const Params& params,
    std::vector<Agent*>& agents,
    std::vector<Order>& local_orders
);

std::vector<Order> generate_all_new_orders(
    Market& market,
    Params& params,
    std::vector<Agent*>& agents
);

void clear_all_orders(
    std::vector<Agent*>& agents,
    Order_book& order_book,
    const std::vector<Order>& orders
);