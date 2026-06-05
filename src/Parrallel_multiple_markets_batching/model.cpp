#include "market.hpp"

// #3 initialization and counting (n_+(t), n_-(t), n_f(t) and x_t)


std::vector<std::mt19937>  initialize_vector_rng(Params& params){
    std::vector<std::mt19937> vector_rng(params.nb_threads);
    for(int i=0;i<params.nb_threads;i++){
        vector_rng[i]=std::mt19937(params.vector_seeds[i]);
    }
    return vector_rng;
}

std::vector<std::normal_distribution<double>> initialize_vector_normal_dist(Params& params){
    std::vector<std::normal_distribution<double>> vector_normal_dist(params.nb_markets);
    for(int i=0; i<params.nb_markets;i++){
        vector_normal_dist[i] = std::normal_distribution<double>(0.0, params.vector_sigma_pf[i]);
    }
    return vector_normal_dist;
}

void initialize_agents_per_market(std::vector<Agent*>& agents,int asset_id ,Params& params, std::mt19937& rng_market) {
    std::vector<int> indices(params.N);
    std::iota(indices.begin(), indices.end(), 0);
    std::shuffle(indices.begin(), indices.end(), rng_market);
    
    int n_plus = params.N / 3;
    int n_minus = params.N / 3;
    int n_fund = params.N - n_plus - n_minus;

    for (int i = 0; i < n_plus; ++i) {
        agents[indices[i]]->vector_type[asset_id]=Agent::Agent_type::Optimist;
    }

    for (int i = n_plus; i < n_plus+n_minus; ++i) {
        agents[indices[i]]->vector_type[asset_id]=Agent::Agent_type::Pessimist;
    }

    for (int i = n_plus+n_minus; i <n_plus+n_minus+n_fund; ++i) {
        agents[indices[i]]->vector_type[asset_id]=Agent::Agent_type::Fundamentalist;
    }
}

std::vector<Agent*> initialize_agents(Params& params, std::mt19937& init_rng, std::vector<std::mt19937>& vector_rng) {
    
    std::vector<Agent*> agents;
    agents.reserve(params.N);
    std::vector<std::thread> vector_threads(params.nb_markets);

    for (int i=0;i<params.N;i++){
        agents.push_back(new Agent(i, params, init_rng));
    }

    for(int i=0;i<params.nb_markets;i++){
        vector_threads[i] = std::thread (initialize_agents_per_market, std::ref(agents),i , std::ref(params), std::ref(vector_rng[i]));
        }

    for(int i=0; i<params.nb_markets; i++){
        vector_threads[i].join();
    }
    return agents;
}


std::vector<Order_book> initialize_vector_order_books(Params& params){
    std::vector<Order_book> vector_order_books(params.nb_markets);
    for(int i=0;i<params.nb_markets;i++){
        vector_order_books[i]=Order_book(params,i);
    }
    return vector_order_books;
}

void reset_vector_order_books(Params& params, std::vector<Order_book>& vector_order_books){
    for(int i=0;i<params.nb_markets;i++){
        vector_order_books[i].reset();
    }
}


std::vector<Market> initialize_vector_markets (Params& params){
    std::vector<Market> vector_markets(params.nb_markets);
    for(int i=0;i<params.nb_markets;i++){
        vector_markets[i]=Market(params.p0,params.p0,params.pf0,i);
    }
    return vector_markets;
}

    

void update_fundamental_value(Params& params, std::vector<Market>& vector_markets,std::vector<std::normal_distribution<double>>&  vector_normal_dist, std::vector<std::mt19937>& vector_rng){

    for(int i=0;i<params.nb_markets;i++){
        double epsilon = vector_normal_dist[i](vector_rng[i]);
        vector_markets[i].pf *= std::exp(epsilon);
    }
    return;
}



void count_agents_thread(Params& params,int index_start, int index_end,std::vector<Agent*>& agents, std::vector<int>& counter_optimist, std::vector<int>& counter_pessimist, std::vector<int>& counter_fundamentalist ){

     std::vector<int> thread_counter_optimist(params.nb_markets);
     std::vector<int> thread_counter_pessimist(params.nb_markets);
     std::vector<int> thread_counter_fundamentalist(params.nb_markets);


    for(int asset_id=0;asset_id<params.nb_markets;asset_id++){
        for (int i= index_start; i<index_end;i++) {

            if (agents[i]->vector_type[asset_id] == Agent::Agent_type::Optimist) {
                thread_counter_optimist[asset_id]++;
            } else if (agents[i]->vector_type[asset_id] == Agent::Agent_type::Pessimist) {
                thread_counter_pessimist[asset_id]++;
            } else {
                thread_counter_fundamentalist[asset_id]++;
            }
        }
    }  
    counter_optimist=thread_counter_optimist;
    counter_pessimist=thread_counter_pessimist;
    counter_fundamentalist=thread_counter_fundamentalist;


}

std::vector<Counts> count_agents( std::vector<Agent*>& agents, Params& params) {

    int nb_threads =params.nb_threads;
    int chunk_size = params.N/nb_threads;
    std::vector<std::thread> vector_threads(nb_threads);
    
    std::vector<Counts> vector_counts(params.nb_markets);
    for(int i=0; i<params.nb_markets;i++){
        vector_counts[i]=Counts(i);
    }
    std::vector<std::vector<int>> counter_optimist_per_market(params.nb_threads, std::vector<int> (params.nb_markets,0));
    std::vector<std::vector<int>> counter_pessimist_per_market(params.nb_threads, std::vector<int> (params.nb_markets,0));
    std::vector<std::vector<int>> counter_fundamentalist_per_market(params.nb_threads, std::vector<int> (params.nb_markets,0));

    for(int i=0; i<nb_threads;i++){
        int index_start= i*chunk_size;
        int index_end = index_start+chunk_size;

        if(i==nb_threads-1){
            index_end=params.N;
        }        
        vector_threads[i]= std::thread(count_agents_thread,std::ref(params), index_start, index_end,std::ref(agents), std::ref(counter_optimist_per_market[i]),std::ref(counter_pessimist_per_market[i]),std::ref(counter_fundamentalist_per_market[i]));
    }

     for(int i=0; i<nb_threads;i++){
        vector_threads[i].join();

        for(int asset_id=0; asset_id<params.nb_markets; asset_id++){
            vector_counts[asset_id].nb_optimists+=counter_optimist_per_market[i][asset_id];
            vector_counts[asset_id].nb_pessimists+=counter_pessimist_per_market[i][asset_id];
            vector_counts[asset_id].nb_fundamentalists+=counter_fundamentalist_per_market[i][asset_id];
        }
     }
    return vector_counts;
}



void sentiment_per_market(double& sentiment, Counts& counts) {
    int noise_traders = counts.nb_optimists + counts.nb_pessimists;
    if (noise_traders == 0) {
        sentiment=0;
    }
    else{sentiment=(counts.nb_optimists- counts.nb_pessimists) / static_cast<double>(noise_traders);}
}


std::vector<double> sentiment(std::vector<Counts>& vector_counts, Params& params){
    std::vector<double> vector_sentiment(params.nb_markets);
    for(int i=0;i<params.nb_markets;i++){
        sentiment_per_market(vector_sentiment[i], vector_counts[i]);
    }
    return vector_sentiment;
}


// #4 Computing Transition probabilities

void compute_probabilities_per_market(Probs& probs, Params& params,  Market& market,  Counts& counts) {
    
    double r = std::log(market.p / market.p_prev);
    double x;
    sentiment_per_market(x,counts);
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

    return ;
}

std::vector<Probs> compute_probabilities( Params& params,  std::vector<Market>& vector_markets,  std::vector<Counts>& vector_counts) {
    std::vector<Probs> vector_probs (params.nb_markets);

    for(int i=0;i<params.nb_markets;i++){
        vector_probs[i]=Probs(i);
        compute_probabilities_per_market(vector_probs[i], params,  vector_markets[i],  vector_counts[i]);
    }
    return vector_probs;
}





// #5 Update agents


void update_agents_thread(Params& params,int index_start, int index_end,std::vector<Agent*>& agents, std::vector<int>& counter_optimist, std::vector<int>& counter_pessimist, std::vector<int>& counter_fundamentalist, std::vector<Probs>& vector_probs, std::mt19937& rng_thread){

     std::vector<int> thread_counter_optimist(params.nb_markets);
     std::vector<int> thread_counter_pessimist(params.nb_markets);
     std::vector<int> thread_counter_fundamentalist(params.nb_markets);


    for (int i=index_start; i<index_end;i++) {

        for(int asset_id =0; asset_id<params.nb_markets;asset_id++){
            double u = uniform01(rng_thread);
            Agent* agent= agents[i];
            Probs probs =vector_probs[asset_id];

            if (agent->vector_type[asset_id] == Agent::Agent_type::Optimist) {
                if (u < probs.plus_to_minus) {
                    agent->vector_type[asset_id] = Agent::Agent_type::Pessimist;
                } 
                else if (u < probs.plus_to_minus + probs.plus_to_fund) {
                    agent->vector_type[asset_id] = Agent::Agent_type::Fundamentalist;
                }
            }
            else if (agent->vector_type[asset_id] == Agent::Agent_type::Pessimist) {
                
                if (u < probs.minus_to_plus) {
                    agent->vector_type[asset_id] = Agent::Agent_type::Optimist;
                } 
                else if (u < probs.minus_to_plus + probs.minus_to_fund) {
                    agent->vector_type[asset_id] = Agent::Agent_type::Fundamentalist;
                }
            }
            else if (agent->vector_type[asset_id] == Agent::Agent_type::Fundamentalist) {
                
                if (u < probs.fund_to_plus) {
                    agent->vector_type[asset_id] = Agent::Agent_type::Optimist;
                } 
                else if (u < probs.fund_to_plus + probs.fund_to_minus) {
                    agent->vector_type[asset_id] = Agent::Agent_type::Pessimist;
                }
            }

            if (agents[i]->vector_type[asset_id] == Agent::Agent_type::Optimist) {
                thread_counter_optimist[asset_id]++;
            } else if (agents[i]->vector_type[asset_id] == Agent::Agent_type::Pessimist) {
                thread_counter_pessimist[asset_id]++;
            } else {
                thread_counter_fundamentalist[asset_id]++;
            }
        }
    }  

    counter_optimist=thread_counter_optimist;
    counter_pessimist=thread_counter_pessimist;
    counter_fundamentalist=thread_counter_fundamentalist;
    return ;
}


std::vector<Counts> update_agents( Params& params, std::vector<Agent*>& agents, std::vector<Probs>& vector_probs, std::vector<std::mt19937>& vector_rng) {

    int nb_threads =params.nb_threads;
    int chunk_size = params.N/nb_threads;
    std::vector<std::thread> vector_threads(nb_threads);
    
    std::vector<Counts> vector_counts_after(params.nb_markets);
    for(int i=0; i<params.nb_markets;i++){
        vector_counts_after[i]=Counts(i);
    }
    std::vector<std::vector<int>> counter_optimist_per_market(params.nb_threads, std::vector<int> (params.nb_markets,0));
    std::vector<std::vector<int>> counter_pessimist_per_market(params.nb_threads, std::vector<int> (params.nb_markets,0));
    std::vector<std::vector<int>> counter_fundamentalist_per_market(params.nb_threads, std::vector<int> (params.nb_markets,0));

    for(int i=0; i<nb_threads;i++){
        int index_start= i*chunk_size;
        int index_end = index_start+chunk_size;

        if(i==nb_threads-1){
            index_end=params.N;
        }        
        vector_threads[i]= std::thread(update_agents_thread,std::ref(params), index_start, index_end,std::ref(agents), std::ref(counter_optimist_per_market[i]),std::ref(counter_pessimist_per_market[i]),std::ref(counter_fundamentalist_per_market[i]),std::ref(vector_probs), std::ref(vector_rng[i]));
    }

     for(int i=0; i<nb_threads;i++){
        vector_threads[i].join();

        for(int asset_id=0; asset_id<params.nb_markets; asset_id++){
            vector_counts_after[asset_id].nb_optimists+=counter_optimist_per_market[i][asset_id];
            vector_counts_after[asset_id].nb_pessimists+=counter_pessimist_per_market[i][asset_id];
            vector_counts_after[asset_id].nb_fundamentalists+=counter_fundamentalist_per_market[i][asset_id];
        }
     }
    return vector_counts_after;
}
    



//6 New order generation: 


void add_all_new_orders_per_market_batch(int index_start, int index_end,  Market& market,  Params& params, std::vector<Agent*>& agents, Order_book& local_order_book) {
    
    for (int i = index_start; i < index_end; i++) {
        Order new_order = agents[i]->new_order(market, params);
        if (new_order.quantity != 0.0) {
            local_order_book.add_order(agents,new_order); 
        }
    }
}

void add_all_new_orders(std::vector<Market>& vector_markets, Params& params, std::vector<Agent*>& agents, std::vector<Order_book>& vector_order_books){

    int nb_threads =params.nb_threads;
    int nb_markets=params.nb_markets;
    std::vector<std::thread> vector_threads(nb_threads);

    std::vector<Order_book> vector_order_book_batch(nb_threads);


    int count_thread=0;
    for(int asset_id=0; asset_id<nb_markets;asset_id++){
        int nb_thread_market=params.vector_nb_threads_per_market[asset_id];
        int chunk_size_market=params.N/nb_thread_market;

        for(int i=0; i<nb_thread_market;i++){
            int index_start=i*chunk_size_market;
            int index_end=index_start+chunk_size_market;

            if(i==nb_thread_market-1){
                index_end=params.N;
            }
            vector_order_book_batch[count_thread]= Order_book(params,asset_id);
            vector_threads[count_thread]= std::thread(add_all_new_orders_per_market_batch,index_start, index_end, std::ref(vector_markets[asset_id]),std::ref(params), std::ref(agents), std::ref(vector_order_book_batch[count_thread]));

            count_thread++;
        }   
    }

    count_thread=0;
    for(int asset_id=0; asset_id<params.nb_markets;asset_id++){
        int nb_thread_market=params.vector_nb_threads_per_market[asset_id];

        for(int i=0;i<nb_thread_market;i++){
            vector_threads[count_thread].join();
            for( Order& new_order : vector_order_book_batch[count_thread].order_storage){
                vector_order_books[asset_id].add_order(agents, new_order);
            }
            vector_order_books[asset_id].volume+= vector_order_book_batch[count_thread].volume;
            vector_order_books[asset_id].volume_weighted_sum+= vector_order_book_batch[count_thread].volume_weighted_sum;

            count_thread++;
        }
    }
    
}
   



// #7 Price update

std::vector<double> update_prices(std::vector<Order_book>& vector_order_books, Params& params, std::vector<Market>& vector_markets) {
    std::vector<double> vector_new_prices(params.nb_markets);

    for(int i=0; i<params.nb_markets;i++){
        vector_new_prices[i]= vector_order_books[i].volume_weighted_sum/vector_order_books[i].volume;
        vector_markets[i].p_prev=vector_markets[i].p;
        vector_markets[i].p=vector_new_prices[i];
    }
    return vector_new_prices;
}


// Simulation loop
SimTimes run_simulation( Params& params) {
    
    std::mt19937 init_rng(params.seed);

    std::vector<std::mt19937> vector_rng = initialize_vector_rng(params);

    std::vector<std::normal_distribution<double>> vector_normal_dist= initialize_vector_normal_dist(params);

    std::vector<Agent*> agents = initialize_agents(params, init_rng, vector_rng);
    std::vector<Market> vector_markets =initialize_vector_markets(params);

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

    std::vector<Order_book> vector_order_books= initialize_vector_order_books(params);

    for (int t = 0; t < params.T; ++t) {
        std::cerr<<"Period t= "<<t<<"\n";

        update_fundamental_value(params,vector_markets,vector_normal_dist,vector_rng);

        auto t0 = std::chrono::high_resolution_clock::now();
        std::vector<Counts> vector_counts_before = count_agents(agents,params);
        double dt_counting=std::chrono::duration<double, std::milli>(
            std::chrono::high_resolution_clock::now() - t0).count();
        time_counting += dt_counting;

        std::vector<Probs> vector_probs = compute_probabilities(params, vector_markets, vector_counts_before);

        auto t1 = std::chrono::high_resolution_clock::now();
        std::vector<Counts> vector_counts_after = update_agents(params,agents,vector_probs,vector_rng);
        double dt_updating=std::chrono::duration<double, std::milli>(
            std::chrono::high_resolution_clock::now() - t1).count();

        time_updating += dt_updating;

        auto t3 = std::chrono::high_resolution_clock::now();    
        add_all_new_orders(vector_markets, params, agents, vector_order_books);
        double dt_adding=std::chrono::duration<double, std::milli>(
            std::chrono::high_resolution_clock::now() - t3).count();
        time_adding += dt_adding;

        std::vector<double> vector_new_prices = update_prices(vector_order_books,params,vector_markets);

        //write_row(out, t, market, new_price, epsilon, counts_after);
        std::cerr << "t=" << t << " counting=" << dt_counting << "ms updating=" << dt_updating << "ms adding=" << dt_adding << "ms\n";
        int active_orders = 0;
for (auto& ob : vector_order_books)
    active_orders += ob.order_storage.size();
std::cerr << "t=" << t << " active_orders=" << active_orders << "\n";
        reset_vector_order_books(params,vector_order_books);

          for(int i=0;i<params.nb_markets;i++){
    std::cerr << "t=" << t << " market=" << i 
              << " p=" << vector_markets[i].p 
              << " pf=" << vector_markets[i].pf << "\n";
}
        
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
        
        

        
     
        

       


        


  

 