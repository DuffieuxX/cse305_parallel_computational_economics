#include "market.hpp"

// #3 initialization and counting (n_+(t), n_-(t), n_f(t) and x_t)
std::vector<Agent*> initialize_agents( Params& params, std::mt19937& rng) {
    
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
    for(int i = 0; i < params.N; ++i){
        agents[i]->agent_id = i;
    }
    return agents;
}

Counts count_agents( std::vector<Agent*>& agents) {
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

double sentiment( Counts& counts) {
    int noise_traders = counts.nb_optimists + counts.nb_pessimists;
    if (noise_traders == 0) {
        return 0.0;
    }
    return static_cast<double>(counts.nb_optimists- counts.nb_pessimists) / static_cast<double>(noise_traders);
}


// #4 Computing Transition probabilities
Probs compute_probabilities( Params& params,  Market& market,  Counts& counts) {
    Probs probs;
    double r = std::log(market.p / market.p_prev);
    double x = sentiment(counts);
    double mispricing = std::log(market.pf / market.p);

    // within noise traders changes
    double U_minus_to_plus = params.a_herding * x + params.a_trend * r;
    double U_plus_to_minus = -U_minus_to_plus;
    
    probs.minus_to_plus = switch_probability(params.nu_opinion, U_minus_to_plus, params.dt);
    probs.plus_to_minus = switch_probability(params.nu_opinion, U_plus_to_minus, params.dt);

    // strategy payoffs: chartists vs fundamentalists
    double pi_plus = r;
    double pi_minus = -r;
    double pi_fund = std::abs(mispricing);
    
    probs.fund_to_plus = switch_probability(params.switching_speed, pi_plus - pi_fund, params.dt);
    probs.plus_to_fund = switch_probability(params.switching_speed, pi_fund - pi_plus, params.dt);
    probs.fund_to_minus = switch_probability(params.switching_speed, pi_minus - pi_fund, params.dt);
    probs.minus_to_fund = switch_probability(params.switching_speed, pi_fund - pi_minus, params.dt);
    return probs;
}

// #5 Update agents
Counts update_agents(Params& params, std::vector<Agent*>& agents, Probs& probs, std::mt19937& rng) {
    Counts counts_after;
    
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
        // Count final type immediately after update
        if (agent->type == Agent::Agent_type::Optimist) {
            counts_after.nb_optimists++;
        }
        else if (agent->type == Agent::Agent_type::Pessimist) {
            counts_after.nb_pessimists++;
        }
        else {
            counts_after.nb_fundamentalists++;
        }
    }
    return counts_after;
}


//6 New order generation and sequential clearing:
std::vector<Order> generate_all_new_orders(
    Market& market,
    Params& params,
    std::vector<Agent*>& agents
) {
    std::vector<Order> orders;
    orders.reserve(params.N);

    for (int i = 0; i < params.N; i++) {
        Order new_order = agents[i]->new_order(market, params);

        if (new_order.quantity != 0.0) {
            orders.push_back(new_order);
        }
    }

    return orders;
}

void clear_all_orders(
    std::vector<Agent*>& agents,
    Order_book& order_book,
    const std::vector<Order>& orders
) {
    for (const Order& order : orders) {
        order_book.add_order(agents, order);
    }
}

void add_all_new_orders(
    Market& market,
    Params& params,
    std::vector<Agent*>& agents,
    Order_book& order_book,
    SimTimes& simtimes
) {
    auto t_gen0 = std::chrono::high_resolution_clock::now();

    std::vector<Order> orders = generate_all_new_orders(
        market,
        params,
        agents
    );

    simtimes.order_generation += std::chrono::duration<double, std::milli>(
        std::chrono::high_resolution_clock::now() - t_gen0
    ).count();

    auto t_clear0 = std::chrono::high_resolution_clock::now();

    clear_all_orders(
        agents,
        order_book,
        orders
    );

    simtimes.clearing += std::chrono::duration<double, std::milli>(
        std::chrono::high_resolution_clock::now() - t_clear0
    ).count();
}


// #7 Price update

double update_price(Order_book& order_book, const Market& market) {
    if (order_book.volume == 0.0) {
        return market.p;
    }

    return order_book.volume_weighted_sum / order_book.volume;
}


// Simulation loop
SimTimes run_simulation(Params& params) {
    
    std::mt19937 rng(params.seed);
    std::normal_distribution<double> normal_dist(0.0, params.sigma_pf);

    std::vector<Agent*> agents = initialize_agents(params, rng);

    Market market;
    market.p = params.p0;
    market.p_prev = params.p0;
    market.pf = params.pf0;

    std::ofstream out;

    if (params.write_output) {
        out.open(params.output);

        if (!out.is_open()) {
            std::cerr << "Error: could not open output file " << params.output << "\n";
            throw std::runtime_error("Could not open output file");
        }

        write_header(out);
    }

    SimTimes simtimes;

    for (int t = 0; t < params.T; ++t) {
        if (params.verbose) {
            std::cerr << "Period t= " << t << "\n";
        }

        Order_book order_book(params);

        double epsilon = normal_dist(rng);
        market.pf *= std::exp(epsilon);

        auto t_count0 = std::chrono::high_resolution_clock::now();
        Counts counts_before = count_agents(agents);
        simtimes.counting += std::chrono::duration<double, std::milli>(
            std::chrono::high_resolution_clock::now() - t_count0
        ).count();

        auto t_prob0 = std::chrono::high_resolution_clock::now();
        Probs probs = compute_probabilities(params, market, counts_before);
        simtimes.probabilities += std::chrono::duration<double, std::milli>(
            std::chrono::high_resolution_clock::now() - t_prob0
        ).count();

        auto t_update0 = std::chrono::high_resolution_clock::now();
        Counts counts_after = update_agents(params, agents, probs, rng);
        simtimes.updating += std::chrono::duration<double, std::milli>(
            std::chrono::high_resolution_clock::now() - t_update0
        ).count();

        add_all_new_orders(market, params, agents, order_book, simtimes);

        auto t_price0 = std::chrono::high_resolution_clock::now();
        double new_price = update_price(order_book, market);
        simtimes.price_update += std::chrono::duration<double, std::milli>(
            std::chrono::high_resolution_clock::now() - t_price0
        ).count();

        if (!std::isfinite(new_price) || new_price <= 0.0) {
            std::cerr << "Numerical error at time step " << t << "\n";
            std::cerr << "new_price = " << new_price << "\n";
            break;
        }

        auto t_output0 = std::chrono::high_resolution_clock::now();
        if (params.write_output) {
            write_row(out, t, market, new_price, epsilon, counts_after);
        }
        simtimes.output += std::chrono::duration<double, std::milli>(
            std::chrono::high_resolution_clock::now() - t_output0
        ).count();

        market.p_prev = market.p;
        market.p = new_price;

        if (params.verbose) {
            int active_orders = static_cast<int>(order_book.order_storage.size());
            std::cerr << "t=" << t << " active_orders=" << active_orders << "\n";
        }
    }

    for (Agent* a : agents) {
        delete a;
    }

    if (params.write_output) {
        out.close();
    }

    if (params.verbose) {
        std::cout << "Simulation completed.\n";
    }

    return simtimes;
}
