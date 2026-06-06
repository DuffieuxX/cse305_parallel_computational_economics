#include <cmath>
#include <fstream>
#include <iostream>
#include <random>
#include <string>
#include <vector>
#include <algorithm>
#include <filesystem>
#include <limits>
#include<thread>

// #1 Data structure 
struct Params {
    int nb_markets=1;
    int nb_threads=10; 
        
    int N = 100000; //number of agents 
    int T = 1000; // number of time periods
    int seed = 42;
    std::vector<int> vector_seeds;
    double dt = 1.0; //  period increments for strategy re-evaluation 

    double p0=100.00; // average initial price
    std::vector<double> vector_p0; // vector of initial prices 

    double pf0=100.00;//average initial fundamental value
    std::vector<double> vector_pf0; // vector of initial fundamental values
    
    double initial_cash=10000.0;//average initial cash of agents 
    double asset_inventory=100.0; //average initial asset inventory
    std::vector<double> vector_initial_inventory; //average initial inventory of agents per asset 

    double sigma_pf =0.005; //average variance of log return of fundamental value
    std::vector<double> vector_sigma_pf; // variance of log return of fundamental value
    
    double nu_opinion = 0.05; //frequency of re-evaluating opinion for chartists
    double switching_speed = 0.01; // baseline speed of switching between chartist and fundamentalist strategies
    double a_herding = 1.0; //influence of herding on chartist behaviour
    double a_trend = 1.0; // influence of temporary price trend on chartist behaviour 
    double order_size = 10.0; // average individual order size scale
    double sigma_aggressiveness=0.02; // variance of aggressiveness
    double gamma_f=2.0; // sensitivity of fundamentalist order size to mispricing
    double m_min=0.005; // minimum threshold on m_t for fundamentalists to trade

    std::string output = "results/simulation.csv";

    Params(){
        std::vector<double> vector_p0(this->nb_markets,this->p0);
        this->vector_p0=vector_p0;

        std::vector<double> vector_pf0(this->nb_markets,this->pf0);
        this->vector_pf0=vector_pf0;

        std::vector<double> vector_sigma_pf(this->nb_markets,this->sigma_pf);
        this->vector_sigma_pf=vector_sigma_pf;

        std::vector<double> vector_initial_inventory(this->nb_markets,this->asset_inventory);
        this->vector_initial_inventory=vector_initial_inventory;



        this->vector_seeds.resize(this->nb_threads);
        for(int i=0;i<this->nb_threads;i++){
            this->vector_seeds[i]=this->seed+1+i;
        }
    };
    
};

// Forward declarations
struct Order;
struct global_Market;
struct Market;
struct Counts;

struct Agent{
    enum class Agent_type {Optimist,Pessimist,Fundamentalist};
    std::vector<Agent_type> vector_type; //wether agent is optimist, pessimist or fundamentalist
    int agent_id;
    double cash; //quantity of cash of the agent >=0 (no borrowing)
    std::mutex cash_lock;
 
    std::vector<double> asset_inventory; // quantity of asset held by client >=0 (no shorting)
    double chartist_order_size; // order size when the agent is a chartist 
    double aggressiveness; // aggressiveness of agent when deciding order price 
    
    Agent(int agent_id,Params& params,std::mt19937& rng);
    
    Order new_order(Market& market, Params& params);
};


struct Counts {
    int nb_optimists = 0; 
    int nb_pessimists = 0;
    int nb_fundamentalists = 0;
    int asset_id;
    Counts()=default;
    Counts(int asset_int);
};

struct Market {
    double p;
    double p_prev;
    double pf;
    int asset_id;
    Market()=default;
    Market(double p, double p_prev, double pf, int asset_id);
};

struct Probs {
    double plus_to_minus = 0.0;
    double plus_to_fund = 0.0;

    double minus_to_plus = 0.0;
    double minus_to_fund = 0.0;

    double fund_to_plus = 0.0;
    double fund_to_minus = 0.0;

    int asset_id;
    Probs()=default;
    Probs(int asset_id);
};


struct Order{
    bool buy; 
    double quantity;
    double price; 
    int agent_id; 
    Order(bool buy, double quantity, double price, int agent_id);
};


struct Order_book {
    std::vector<Order> order_storage;
    std::vector<Order*> bids;
    std::vector<Order*> asks;

    std::size_t bid_head = 0;
    std::size_t ask_head = 0;

    int asset_id;

    double volume_weighted_sum=0;
    double volume=0;

    Order_book()=default;
    Order_book( Params& params,int asset_id);

    void add_order(std::vector<Agent*>& agents, const Order& order);

    void reset();
    
};

struct SimTimes {
    double counting = 0;
    double updating = 0;
    double adding   = 0;
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
std::vector<Agent*> initialize_agents( Params& params, std::mt19937& rng, std::vector<std::mt19937>& vector_rng);
std::vector<Counts>  count_agents( std::vector<Agent*>& agents,Params& params);
std::vector<double> sentiment( std::vector<Counts>& vector_counts, Params& params);
std::vector<Probs> compute_probabilities(Params& params, std::vector<Market>& vector_markets, std::vector<Counts>& vector_counts);
std::vector<Counts> update_agents(Params& params, std::vector<Agent*>& agents, std::vector<Probs>& vector_probs, std::vector<std::mt19937>& vector_rng);
std::vector<double> update_prices(std::vector<Order_book>& order_book, Params& params, std::vector<Market>& vector_markets);