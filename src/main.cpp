#include <cmath>
#include <fstream>
#include <iostream>
#include <random>
#include <string>
#include <vector>


// #1 Data structure 

enum class Agent {
    Optimist,
    Pessimist,
    Fundamentalist
};

struct Params {
    int N = 10000;
    int T = 1000;
    unsigned seed = 42;

    double dt = 1.0;

    double p0 = 100.0;
    double pf0 = 100.0;

    double sigma_pf = 0.001;
    double beta = 0.01;

    double q_noise = 1.0;
    double q_fund = 5.0;

    double nu_opinion = 0.02;
    double nu_strategy = 0.01;

    double a_herding = 1.0;
    double a_trend = 100.0;

    double r_alt = 0.0;
    double discount = 0.5;
    double profit_sensitivity = 100.0;

    std::string output = "results/simulation.csv";
};

struct Counts {
    int plus = 0;
    int minus = 0;
    int fund = 0;
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



// #2 Utility functions

double switch_probability(double nu, double U, double dt) {
    return nu * std::exp(U) * dt;
}


double uniform01(std::mt19937& rng) {
    
    std::uniform_real_distribution<double> dist(0.0, 1.0);
    
    return dist(rng);
}

// #3 initialization and counting (n_+(t), n_-(t), n_f(t) and x_t)

std::vector<Agent> initialize_agents(const Params& params, std::mt19937& rng) {
    
    std::vector<Agent> agents;
    agents.reserve(params.N);

    int n_plus = params.N / 3;
    int n_minus = params.N / 3;
    int n_fund = params.N - n_plus - n_minus;

    for (int i = 0; i < n_plus; ++i) {
        agents.push_back(Agent::Optimist);
    }

    for (int i = 0; i < n_minus; ++i) {
        agents.push_back(Agent::Pessimist);
    }

    for (int i = 0; i < n_fund; ++i) {
        agents.push_back(Agent::Fundamentalist);
    }

    std::shuffle(agents.begin(), agents.end(), rng);

    return agents;
}



Counts count_agents(const std::vector<Agent>& agents) {
    
    Counts counts;

    for (Agent agent : agents) {
        if (agent == Agent::Optimist) {
            counts.plus++;
        } else if (agent == Agent::Pessimist) {
            counts.minus++;
        } else {
            counts.fund++;
        }
    }

    return counts;
}

double sentiment(const Counts& counts) {
    
    int noise_traders = counts.plus + counts.minus;

    if (noise_traders == 0) {
        return 0.0;
    }

    return static_cast<double>(counts.plus - counts.minus) / static_cast<double>(noise_traders);
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
void update_agents(std::vector<Agent>& agents, const Probs& probs, std::mt19937& rng) {
    
    for (Agent& agent : agents) {
        
        double u = uniform01(rng);
        
        if (agent == Agent::Optimist) {
            
            if (u < probs.plus_to_minus) {
                agent = Agent::Pessimist;
            } 
            else if (u < probs.plus_to_minus + probs.plus_to_fund) {
                agent = Agent::Fundamentalist;
            }
        }
        else if (agent == Agent::Pessimist) {
            
            if (u < probs.minus_to_plus) {
                agent = Agent::Optimist;
            } 
            else if (u < probs.minus_to_plus + probs.minus_to_fund) {
                agent = Agent::Fundamentalist;
            }
        }
        else if (agent == Agent::Fundamentalist) {
            
            if (u < probs.fund_to_plus) {
                agent = Agent::Optimist;
            } 
            else if (u < probs.fund_to_plus + probs.fund_to_minus) {
                agent = Agent::Pessimist;
            }
        }
    }
}



// #6 Excess demand and price update

double excess_demand(const Params& params, const Market& market, const Counts& counts) {
    
    double noise_demand = params.q_noise * static_cast<double>(counts.plus - counts.minus);
    double fundamentalist_demand = params.q_fund * static_cast<double>(counts.fund) * std::log(market.pf / market.p);

    return noise_demand + fundamentalist_demand;
}

double update_price(const Params& params, double price, double excess_demand_value) {
    return price * std::exp( params.beta * excess_demand_value / static_cast<double>(params.N));
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
        << "excess_demand\n";
}

void write_row(
    std::ofstream& out,
    int t,
    const Market& market,
    double new_price,
    double epsilon,
    const Counts& counts,
    double ED
) {
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
        << counts.plus << ","
        << counts.minus << ","
        << counts.fund << ","
        << ED << "\n";
}


// #8 Simulation loop

void run_simulation(const Params& params) {
    
    std::mt19937 rng(params.seed);
    std::normal_distribution<double> normal_dist(0.0, params.sigma_pf);

    std::vector<Agent> agents = initialize_agents(params, rng);

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

        double epsilon = normal_dist(rng);
        market.pf *= std::exp(epsilon);
        
        Counts counts_before = count_agents(agents);

        Probs probs = compute_probabilities(params, market, counts_before);

        update_agents(agents, probs, rng);

        Counts counts_after = count_agents(agents);

        double ED = excess_demand(params, market, counts_after);

        double new_price = update_price(params, market.p, ED);

        if (!std::isfinite(new_price) || new_price <= 0.0) {
            std::cerr << "Numerical error at time step " << t << "\n";
            break;
        }

        write_row(out, t, market, new_price, epsilon, counts_after, ED);

        market.p_prev = market.p;
        market.p = new_price;
    }

    out.close();

    std::cout << "Simulation completed.\n";
    std::cout << "Output written to: " << params.output << "\n";
}


// #9 Main function
int main(int argc, char* argv[]) {
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