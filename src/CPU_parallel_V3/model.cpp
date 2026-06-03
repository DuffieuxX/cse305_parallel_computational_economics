#include "market.hpp"

// #3 initialization and counting (n_+(t), n_-(t), n_f(t) and x_t)

//Parallel

void initialize_agent_thread(int index_start, int index_end, int n_plus, int n_minus, int n_fund, std::vector<Agent*>& agents, Params& params,unsigned int seed_thread  ){

    std::mt19937 rng_thread (seed_thread);

    for(int i=index_start;i< std::min(index_end,n_plus);i++){

        Agent* new_agent = new Agent(Agent::Agent_type::Optimist,i,params,rng_thread);
        agents[i] =(new_agent);
    }

    for( int i = std::max (index_start,n_plus) ;i< std::min(n_plus+n_minus,index_end);i++){
        Agent* new_agent = new Agent(Agent::Agent_type::Pessimist,i,params,rng_thread);
        agents[i] =(new_agent);
    }


    for(int i= std::max(index_start, n_plus+n_minus); i< index_end;i++){
        Agent* new_agent = new Agent(Agent::Agent_type::Fundamentalist,i,params,rng_thread);
        agents[i] =(new_agent);
    }
}


std::vector<Agent*> initialize_agents( Params& params,std::mt19937& rng) {

    int nb_threads =params.nb_threads;
    int chunk_size = params.chunk_size;
    std::vector<std::thread> vector_threads(nb_threads);

    std::vector<Agent*> agents(params.N);
    

    int n_plus = params.N / 3;
    int n_minus = params.N / 3;
    int n_fund = params.N - n_plus - n_minus;


    for(int i=0; i<nb_threads;i++){
        int index_start= i*chunk_size;
        int index_end = index_start+chunk_size;

        if(i==nb_threads-1){
            index_end=params.N;
        }

        unsigned int seed_thread = rng();
        

        vector_threads[i]= std::thread(initialize_agent_thread, index_start, index_end,n_plus,n_minus,n_fund, std::ref(agents), std::ref (params), seed_thread);

    }

    for(int i=0; i<nb_threads;i++){
        vector_threads[i].join();
    }

    std::shuffle(agents.begin(), agents.end(), rng);
    for(int i = 0; i < params.N; ++i){
        agents[i]->agent_id = i;
    }
    return agents;
}

//Parallel

void count_agents_thread(int index_start, int index_end,std::vector<Agent*>& agents, int& counter_optimist, int & counter_pessimist, int& counter_fundamentalist ){

    for (int i= index_start; i<index_end;i++) {
        Agent* agent =agents[i];

        if (agent->type == Agent::Agent_type::Optimist) {
            counter_optimist++;
        } else if (agent->type == Agent::Agent_type::Pessimist) {
            counter_pessimist++;
        } else {
            counter_fundamentalist++;
        }
    }

}


Counts count_agents( std::vector<Agent*>& agents, Params& params) {

    int nb_threads =params.nb_threads;
    int chunk_size = params.chunk_size;
    std::vector<std::thread> vector_threads(nb_threads);
    std::vector<int> counter_optimist(nb_threads,0);
    std::vector<int> counter_pessimist(nb_threads,0);
    std::vector<int> counter_fundamentalist(nb_threads,0);

    Counts counts;
    

    for(int i=0; i<nb_threads;i++){
        int index_start= i*chunk_size;
        int index_end = index_start+chunk_size;

        if(i==nb_threads-1){
            index_end=params.N;
        }        
        vector_threads[i]= std::thread(count_agents_thread, index_start, index_end,std::ref(agents), std::ref(counter_optimist[i]),std::ref(counter_pessimist[i]),std::ref(counter_fundamentalist[i]));

    }

     for(int i=0; i<nb_threads;i++){
        vector_threads[i].join();
        counts.nb_optimists+=counter_optimist[i];
        counts.nb_pessimists+=counter_pessimist[i];
        counts.nb_fundamentalists+=counter_fundamentalist[i];
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
//Parallel

void update_agents_thread(int index_start, int index_end, std::vector<Agent*>& agents, Probs& probs, unsigned int seed_thread){

    std::mt19937 rng(seed_thread);
    

    for(int i=index_start;i<index_end;i++){

        double u = uniform01(rng);
            
        Agent* agent = agents[i];
        
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

    }}
    




void update_agents( Params& params, std::vector<Agent*>& agents, Probs& probs, std::mt19937& rng) {

    int nb_threads =params.nb_threads;
    int chunk_size = params.chunk_size;
    std::vector<std::thread> vector_threads(nb_threads);

    for(int i=0; i<nb_threads;i++){
        int index_start= i*chunk_size;
        int index_end = index_start+chunk_size;

        if(i==nb_threads-1){
            index_end=params.N;
        }

        unsigned int seed_thread = rng();
        

        vector_threads[i]= std::thread(update_agents_thread, index_start, index_end, std::ref(agents),std::ref(probs), seed_thread);

    }

    for(int i=0; i<nb_threads;i++){
        vector_threads[i].join();
    }


    
    
}

//6 New orders

void add_all_new_orders_thread(int index_start, int index_end, const Market& market, const Params& params, std::vector<Agent*>& agents, std::vector<Order>& new_orders_in_thread) {
    for (int i = index_start; i < index_end; i++) {
        
        Order new_order = agents[i]->new_order(market, params);
        
        if (new_order.quantity != 0.0) {
            new_orders_in_thread.push_back(new_order);
        }
    }
}

void add_all_new_orders(Market& market, Params& params, std::vector<Agent*>& agents, Order_book& order_book){

    int nb_threads =params.nb_threads;
    int chunk_size = params.chunk_size;
    std::vector<std::thread> vector_threads(nb_threads);
    std::vector<std::vector<Order>> vector_new_orders_thread(nb_threads);

    for(int i=0; i<nb_threads;i++){
        int index_start= i*chunk_size;
        int index_end = index_start+chunk_size;

        if(i==nb_threads-1){
            index_end=params.N;
        }   
        vector_threads[i]= std::thread(add_all_new_orders_thread,index_start,index_end, std::ref(market),  std::ref(params), std::ref(agents), std::ref(vector_new_orders_thread[i]));
    }

    for(int i=0; i<nb_threads;i++){
        vector_threads[i].join();
        for(const Order& new_order : vector_new_orders_thread[i]){
            order_book.add_order(agents, new_order);
        }
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
//Parallel
SimTimes run_simulation( Params& params) {
    
    params.nb_threads = std::max(1, std::min(params.nb_threads, params.N));
    params.chunk_size = (params.N + params.nb_threads - 1) / params.nb_threads;
    
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

    double time_counting  = 0;
    double time_updating  = 0;
    double time_adding = 0;

    write_header(out);
    for (int t = 0; t < params.T; ++t) {
        std::cerr<<"Period t= "<<t<<"\n";
        Order_book order_book =Order_book(params);
        double epsilon = normal_dist(rng);
        market.pf *= std::exp(epsilon);
        
        auto t0 = std::chrono::high_resolution_clock::now();
        Counts counts_before = count_agents(agents,params);
        time_counting += std::chrono::duration<double, std::milli>(
            std::chrono::high_resolution_clock::now() - t0).count();

        
        Probs probs = compute_probabilities(params, market, counts_before);
        

        auto t1 = std::chrono::high_resolution_clock::now();
        update_agents(params,agents, probs, rng);
        time_updating += std::chrono::duration<double, std::milli>(
            std::chrono::high_resolution_clock::now() - t1).count();


        auto t2 = std::chrono::high_resolution_clock::now();
        Counts counts_after = count_agents(agents,params);
        time_counting += std::chrono::duration<double, std::milli>(
            std::chrono::high_resolution_clock::now() - t2).count();



        auto t3 = std::chrono::high_resolution_clock::now();
        add_all_new_orders(market, params, agents, order_book);
        double new_price = update_price(order_book);
        time_adding += std::chrono::duration<double, std::milli>(
            std::chrono::high_resolution_clock::now() - t3).count();

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
   
    

    SimTimes simtimes;
    simtimes.adding=time_adding;
    simtimes.counting=time_counting;
    simtimes.updating=time_updating;
    
    return simtimes;


}