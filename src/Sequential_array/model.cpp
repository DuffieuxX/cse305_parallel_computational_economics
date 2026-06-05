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


//6 New order generation: 
void add_all_new_orders(Market& market, Params& params, std::vector<Agent*>& agents, Order_book& order_book){
    for (int i = 0; i < params.N; i++) {
        Order new_order = agents[i]->new_order(market, params);
        order_book.add_order(agents, new_order);
    }
}



// #7 Price update

double update_price(Order_book& order_book) {
    if(order_book.volume==0){
        return 100;
    }

    return order_book.volume_weighted_sum/order_book.volume;

}


// Simulation loop
SimTimes run_simulation( Params& params) {
    
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
        try{
            throw 1 ;
        }

        catch(int errorCode){
            std::cout<<"Error:"<<errorCode;
        } 
    }

    write_header(out);

    double time_counting  = 0;
    double time_updating  = 0;
    double time_adding = 0;

    for (int t = 0; t < params.T; ++t) {
        std::cerr<<"Period t= "<<t<<"\n";
        Order_book order_book = Order_book(params);
        double epsilon = normal_dist(rng);
        market.pf *= std::exp(epsilon);
        
        auto t0 = std::chrono::high_resolution_clock::now();
        Counts counts_before = count_agents(agents);
        time_counting += std::chrono::duration<double, std::milli>(
            std::chrono::high_resolution_clock::now() - t0).count();

        Probs probs = compute_probabilities(params, market, counts_before);

        auto t1 = std::chrono::high_resolution_clock::now();
        Counts counts_after = update_agents(params, agents, probs, rng);
        time_updating += std::chrono::duration<double, std::milli>(
            std::chrono::high_resolution_clock::now() - t1).count();

        auto t3 = std::chrono::high_resolution_clock::now();    
        add_all_new_orders(market, params, agents, order_book);
        time_adding += std::chrono::duration<double, std::milli>(
            std::chrono::high_resolution_clock::now() - t3).count();

        double new_price = update_price(order_book);

        if (!std::isfinite(new_price) || new_price <= 0.0) {
            std::cerr << "Numerical error at time step " << t << "\n";
            std::cerr<<new_price;
            break;
        }

        write_row(out, t, market, new_price, epsilon, counts_after);

        market.p_prev = market.p;
        market.p = new_price;


        int active_orders = 0;

    active_orders += order_book.order_storage.size();
std::cerr << "t=" << t << " active_orders=" << active_orders << "\n";
    }
    for(Agent* a : agents) delete a;

    out.close();
    std::cout << "Simulation completed.\n";


    SimTimes simtimes;
    simtimes.adding=time_adding;
    simtimes.counting=time_counting;
    simtimes.updating=time_updating;
    
    return simtimes;
    
}
        
        

        
     
        

       


        


  

 