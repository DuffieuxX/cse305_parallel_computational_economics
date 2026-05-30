#include <cmath>
#include <fstream>
#include <iostream>
#include <random>
#include <string>
#include <vector>
#include <cmath>
#include <algorithm>
#include <filesystem>

// #1 Data structure 


struct Params {

    int N = 10000; //number of agents 
    int T = 1000; // number of time periods
    int seed = 42;
    
    double dt = 1.0; //  period increments for strategy re-evaluation 

    double p0 = 100.0; // initial price 
    double pf0 = 100.0; // initial fundamental value

    double initial_cash=10000.0;//average initial cash of agents 
    double initial_inventory=100.0;//average initial inventory of agents 

    double sigma_pf = 0.005; // variance of log return of fundamental value
    double beta = 0.01; // 

    double nu_opinion = 0.02; //frequency of re-evaluating opinion for chartists
    double nu_strategy = 0.01; //frequency of re-evaluating strategy (switches btw chartists and fundamentalists)

    double a_herding = 1.5; //influence of herding on chartist behaviour
    double a_trend = 2.0; // influence of temporary price trend on chartist behaviour 

    double r_alt = 0.0; //return of alternative investments for fundamentalists 
    double discount = 0.5; // discount on profits from arbitrage for fundamentalists
    double profit_sensitivity = 10.0; // profit sensitivity of fundamentalists for strategy switching


    double order_size_min_chartist=5.0; //minimum order size of chartists
    double order_size_max_chartist=15.0; //maximium order size of chartists 
    
    double sigma_aggressiveness=0.02; // variance of aggressiveness

    double m_min=0.005; // minimum threshold on m_t for fundamentalists to trade
    double gamma_f=1.0; //sensitivity of order size of fundamentalists on m_t


    std::string output = "results/simulation.csv";
};



double uniform(std::mt19937& rng, double min, double max) {
    std::uniform_real_distribution<double> dist(min, max);
    return dist(rng);
}

double normal(std::mt19937& rng, double mean, double sigma_squared) {
    std::normal_distribution<double> dist(mean, sigma_squared);
    return dist(rng);
}

double half_normal(std::mt19937& rng, double mean, double sigma_squared) {
    std::normal_distribution<double> dist(mean, sigma_squared);
    return std::abs(dist(rng));
}

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
double fundamentalist_order_size; //order size when agent is fundamentalist
double aggressiveness; // aggressiveness of agent when deciding order price 

Agent(Agent_type type,int agent_id,Params params,std::mt19937& rng);

Order* new_order(Market market, Params params);
};



Agent::Agent(Agent_type type,int agent_id,Params params,std::mt19937& rng):
type(type),
agent_id(agent_id),
cash(params.initial_cash),
asset_inventory(params.initial_inventory),
chartist_order_size(uniform(rng,params.order_size_min_chartist,params.order_size_max_chartist)),
aggressiveness(half_normal(rng,0,params.sigma_aggressiveness))
{}


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
    Order* next; 

    Order(bool buy, double quantity, double price, int agent_id): buy(buy), quantity(quantity), price(price), agent_id(agent_id), next(nullptr) {}
};

struct Order_book {
    Order* bid_head;
    Order* ask_head;

    double volume_weighted_sum=0;
    double volume=0;

    Order_book();
    ~Order_book();
    void add_order(std::vector<Agent*> agents,Agent* agent, Market market, Params params);
};

Order_book::Order_book() {
    this->bid_head= new Order(false,0,std::numeric_limits<double>::max(),-1);
    this->ask_head=new Order(false,0,std::numeric_limits<double>::min(),-1);
}

Order_book::~Order_book() {
    Order* current = bid_head;
    while(current != nullptr){
        Order* next = current->next;
        delete current;
        current = next;
    }

    current = ask_head;
    while(current != nullptr){
        Order* next = current->next;
        delete current;
        current = next;
    }
}



Order* Agent::new_order(Market market, Params params){

    if(this->type==Agent::Agent_type::Optimist){
        double limit_price =market.p*(1+this->aggressiveness);
        double quantity =std::min(this->chartist_order_size, this->cash/limit_price);
        bool buy=true;

        Order* new_order = new Order(buy, quantity, limit_price,this->agent_id);
        return new_order;

    }

    else if(this->type==Agent::Agent_type::Pessimist){
        double limit_price =market.p*(1-this->aggressiveness);
        double quantity =std::min(this->chartist_order_size, this->asset_inventory);
        bool buy=false;
        Order* new_order = new Order(buy, quantity, limit_price,this->agent_id);
        return new_order;

    }

    else{
        double m_t= (market.pf-market.p)/market.p;

        if(m_t>=params.m_min){
            bool buy=true;
            double limit_price= market.pf*(1+this->aggressiveness/4);
            this->fundamentalist_order_size = this->chartist_order_size*(1+params.gamma_f*std::abs(m_t));
            double quantity = std::min(this->fundamentalist_order_size, this->cash/limit_price);

            Order* new_order= new Order(buy,quantity,limit_price,this->agent_id);
            return new_order;
        }


        if(m_t<=-params.m_min){
            bool buy=false;
            double limit_price= market.pf*(1-this->aggressiveness/4);
            this->fundamentalist_order_size = this->chartist_order_size*(1+params.gamma_f*std::abs(m_t));
            double quantity = std::min(this->fundamentalist_order_size, this->asset_inventory);

            Order* new_order= new Order(buy,quantity,limit_price,this->agent_id);
            return new_order;
        }

        if(m_t< std::abs(params.m_min)){
            Order* new_order= new Order(true,0,0,this->agent_id);
            return new_order;

        }
    }

    }





void Order_book::add_order(std::vector<Agent*> agents, Agent* agent, Market market, Params params){
    Order* new_order = agent->new_order(market,params);
    bool buy =new_order->buy;
    

    if(buy){
        
        Order* previous_ask = this->ask_head;
        Order* current_ask=previous_ask->next;
        
        while(new_order->quantity>0 && current_ask!=nullptr){

        while(current_ask->price>new_order->price && current_ask->next!=nullptr){
        previous_ask=current_ask;
        current_ask=current_ask->next;
        }

        if(current_ask->price<=new_order->price){
            double price_trade = current_ask->price;
            double quantity_trade = std::min(current_ask->quantity, new_order->quantity);

            new_order->quantity -= quantity_trade;
            current_ask-> quantity -= quantity_trade; 

            
            
            agents[current_ask->agent_id]->asset_inventory-=quantity_trade;
            agents[current_ask->agent_id]->cash+=quantity_trade*price_trade;

            agent->asset_inventory+=quantity_trade;
            agent->cash-=quantity_trade*price_trade;

            this->volume+=quantity_trade;
            this->volume_weighted_sum+=quantity_trade * price_trade;


            if(current_ask->quantity==0){
                previous_ask->next=current_ask->next;
                current_ask=current_ask->next;
                }    
            }

        else{
            break;
        }

        }

       

        if(new_order->quantity>0){

        Order* previous_bid = this->bid_head;
        Order* current_bid=previous_bid->next;

        if (current_bid==nullptr){
            previous_bid->next=new_order;
            return;
        }

        while(current_bid->price>new_order->price && current_bid->next!=nullptr){
        previous_bid=current_bid;
        current_bid=current_bid->next;
        }

        if(current_bid->price<=new_order->price){
            previous_bid->next=new_order;
            new_order->next=current_bid;
        }

        else{
    
            current_bid->next=new_order;
            
        }
        }



    }


if(!buy){
    Order* previous_bid = this->bid_head;
    Order* current_bid = previous_bid->next;

    while(new_order->quantity > 0 && current_bid != nullptr){
        while(current_bid->price < new_order->price && current_bid->next != nullptr){
            previous_bid = current_bid;
            current_bid = current_bid->next;
        }
        if(current_bid->price >= new_order->price){
            double price_trade = current_bid->price;
            double quantity_trade = std::min(current_bid->quantity, new_order->quantity);

            new_order->quantity -= quantity_trade;
            current_bid->quantity -= quantity_trade;

            agents[current_bid->agent_id]->asset_inventory += quantity_trade;
            agents[current_bid->agent_id]->cash -= quantity_trade * price_trade;

            agent->asset_inventory -= quantity_trade;
            agent->cash += quantity_trade * price_trade;

            this->volume+=quantity_trade;
            this->volume_weighted_sum+=quantity_trade * price_trade;


            if(current_bid->quantity == 0){
                previous_bid->next = current_bid->next;
                current_bid = current_bid->next;
            }

        } else {
            break;
        }
    }

    if(new_order->quantity > 0){
        Order* previous_ask = this->ask_head;
        Order* current_ask = previous_ask->next;

        if (current_ask==nullptr){
            previous_ask->next=new_order;
            return;
        }

        while(current_ask->price < new_order->price && current_ask->next != nullptr){
            previous_ask = current_ask;
            current_ask = current_ask->next;
        }
        if(current_ask->price >= new_order->price){
            previous_ask->next = new_order;
            new_order->next = current_ask;
        } else {
            current_ask->next = new_order;
        }
    }
}

}





// #2 Utility functions

double switch_probability(double nu, double U, double dt) {
    return nu * std::exp(U) * dt;
}

double uniform01(std::mt19937& rng) {
    
    std::uniform_real_distribution<double> dist(0.0, 1.0);
    
    return dist(rng);
}

// #3 initialization and counting (n_+(t), n_-(t), n_f(t) and x_t)

std::vector<Agent*> initialize_agents(const Params& params, std::mt19937& rng) {
    
    std::vector<Agent*> agents;
    agents.reserve(params.N);
    int n_plus = params.N / 3;
    int n_minus = params.N / 3;
    int n_fund = params.N - n_plus - n_minus;

    for (int i = 0; i < n_plus; ++i) {
        Agent* new_agent = new Agent(Agent::Agent_type::Optimist,i,params,rng);
        agents.push_back(new_agent);
    }

    for (int i = n_plus; i < n_plus+n_minus; ++i) {
        Agent* new_agent = new Agent(Agent::Agent_type::Pessimist,i,params,rng);
        agents.push_back(new_agent);
    }

    for (int i = n_plus+n_minus; i <n_plus+n_minus+n_fund; ++i) {
        Agent* new_agent = new Agent(Agent::Agent_type::Fundamentalist,i,params,rng);
        agents.push_back(new_agent);
    }
    std::shuffle(agents.begin(), agents.end(), rng);
    return agents;
}

Counts count_agents(const std::vector<Agent*>& agents) {
    Counts counts;
    for (Agent* agent : agents) {
        if (agent->type == Agent::Agent_type::Optimist) {
            counts.nb_optimists++;
        } else if (agent->type == Agent::Agent_type::Pessimist) {
            counts.nb_pessimists++;
        } else {
            counts.nb_fundamentalists++;
        }
    }
    return counts;
}

double sentiment(const Counts& counts) {
    int noise_traders = counts.nb_optimists + counts.nb_pessimists;
    if (noise_traders == 0) {
        return 0.0;
    }
    return static_cast<double>(counts.nb_optimists- counts.nb_pessimists) / static_cast<double>(noise_traders);
}


// #4 Computing Transition probabilities

Probs compute_probabilities(const Params& params, const Market& market, const Counts& counts) {
    Probs probs;
    double r = std::log(market.p / market.p_prev);
    double x = sentiment(counts);
    double mispricing = std::log(market.pf / market.p);

    // within noise traders changes
    double U_minus_to_plus = params.a_herding * x + params.a_trend * r;
    double U_plus_to_minus = -U_minus_to_plus;
    
    probs.minus_to_plus = switch_probability(params.nu_opinion, U_minus_to_plus, params.dt);
    probs.plus_to_minus = switch_probability(params.nu_opinion, U_plus_to_minus, params.dt);

    // noise traders to fundamentalists
    double pi_plus = r;
    double pi_minus = params.r_alt - r;
    double pi_fund = params.discount * std::abs(mispricing);
    double sensitivity = params.profit_sensitivity;
    
    probs.fund_to_plus = switch_probability(params.nu_strategy, sensitivity * (pi_plus - pi_fund), params.dt);
    probs.plus_to_fund = switch_probability(params.nu_strategy, sensitivity * (pi_fund - pi_plus), params.dt);
    probs.fund_to_minus = switch_probability(params.nu_strategy, sensitivity * (pi_minus - pi_fund), params.dt);

    probs.minus_to_fund = switch_probability(params.nu_strategy, sensitivity * (pi_fund - pi_minus), params.dt);

    return probs;
}

// #5 Update agents
void update_agents(std::vector<Agent*>& agents, const Probs& probs, std::mt19937& rng) {
    
    for (Agent* agent : agents) {
        
        double u = uniform01(rng);
        
        if (agent->type == Agent::Agent_type::Optimist) {
            
            if (u < probs.plus_to_minus) {
                agent->type = Agent::Agent_type::Pessimist;
            } 
            else if (u < probs.plus_to_minus + probs.plus_to_fund) {
                agent->type = Agent::Agent_type::Fundamentalist;
            }
        }
        else if (agent->type == Agent::Agent_type::Pessimist) {
            
            if (u < probs.minus_to_plus) {
                agent->type = Agent::Agent_type::Optimist;
            } 
            else if (u < probs.minus_to_plus + probs.minus_to_fund) {
                agent->type = Agent::Agent_type::Fundamentalist;
            }
        }
        else if (agent->type == Agent::Agent_type::Fundamentalist) {
            
            if (u < probs.fund_to_plus) {
                agent->type = Agent::Agent_type::Optimist;
            } 
            else if (u < probs.fund_to_plus + probs.fund_to_minus) {
                agent->type = Agent::Agent_type::Pessimist;
            }
        }
    }
}



// #6 Price update

double update_price(Order_book& order_book) {
    if(order_book.volume==0){
        return 100;
    }

    std::cerr<<"Volume traded= "<<order_book.volume<<"\n";
    return order_book.volume_weighted_sum/order_book.volume;



}









// #7 Output (time,price,fundamental_value,log_return,fundamental_shock,log_mispricing,sentiment,optimists,pessimists,fundamentalists,excess_demand)

void write_header(std::ofstream& out) {
    
    out << "time,"
        << "price,"
        << "fundamental_value,"
        << "log_return,"
        << "fundamental_shock,"
        << "log_mispricing,"
        << "sentiment,"
        << "optimists,"
        << "pessimists,"
        << "fundamentalists,"
        << "\n";
}

void write_row(
    std::ofstream& out,
    int t,
    const Market& market,
    double new_price,
    double epsilon,
    const Counts& counts) {
    double log_return = std::log(new_price / market.p);
    double log_mispricing = std::log(market.pf / market.p);
    double x = sentiment(counts);
    out << t << ","
        << new_price << ","
        << market.pf << ","
        << log_return << ","
        << epsilon << ","
        << log_mispricing << ","
        << x << ","
        << counts.nb_optimists<< ","
        << counts.nb_pessimists << ","
        << counts.nb_fundamentalists << ",";
}

// #8 Simulation loop

void run_simulation(const Params& params) {
    
    std::mt19937 rng(params.seed);
    std::normal_distribution<double> normal_dist(0.0, params.sigma_pf);

    std::vector<Agent*> agents = initialize_agents(params, rng);

    Market market;
    market.p = params.p0;
    market.p_prev = params.p0;
    market.pf = params.pf0;
    std::ofstream out(params.output);

    if (!out.is_open()) {
        std::cerr << "Error: could not open output file " << params.output << "\n";
        return;
    }

    write_header(out);

    for (int t = 0; t < params.T; ++t) {
        Order_book order_book =Order_book();
        double epsilon = normal_dist(rng);
        market.pf *= std::exp(epsilon);
        
        Counts counts_before = count_agents(agents);

        Probs probs = compute_probabilities(params, market, counts_before);

        update_agents(agents, probs, rng);

        Counts counts_after = count_agents(agents);

        for (int i=0; i<params.N;i++){
            order_book.add_order( agents, agents[i], market, params);
        }
        double new_price = update_price(std::ref(order_book));

        if (!std::isfinite(new_price) || new_price <= 0.0) {
            std::cerr << "Numerical error at time step " << t << "\n";
            std::cerr<<new_price;
            break;
        }

        write_row(out, t, market, new_price, epsilon, counts_after);

        market.p_prev = market.p;
        market.p = new_price;

        


    }
    for(Agent* a : agents) delete a;

    out.close();

    std::cout << "Simulation completed.\n";
    std::cout << "Output written to: " << params.output << "\n";
}


// #9 Main function
int main(int argc, char* argv[]) {
    std::filesystem::create_directories("results");
    Params params;
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--agents" && i + 1 < argc) {
            params.N = std::stoi(argv[++i]);
        }
        else if (arg == "--steps" && i + 1 < argc) {
            params.T = std::stoi(argv[++i]);
        } 
        else if (arg == "--seed" && i + 1 < argc) {
            params.seed = static_cast<unsigned>(std::stoul(argv[++i]));
        } 
        else if (arg == "--output" && i + 1 < argc) {
            params.output = argv[++i];
        } 
        else if (arg == "--beta" && i + 1 < argc) {
            params.beta = std::stod(argv[++i]);
        } 
        else if (arg == "--sigma-pf" && i + 1 < argc) {
            params.sigma_pf = std::stod(argv[++i]);
        } 
        else {
            std::cerr << "Unknown or incomplete argument: " << arg << "\n";
            return 1;
        }
    }
    run_simulation(params);
    return 0;
}