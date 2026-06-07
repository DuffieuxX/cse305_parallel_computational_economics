#include "market.hpp"

// #0 aux for random number generation
__device__
int permuted_rank(int i, int N, int seed) {
    int a = 7919;          // works well for N = 10000
    int b = seed % N;

    return (static_cast<long long>(a) * i + b) % N;
}

__device__
unsigned int rng_hash(unsigned int x) {
    x ^= x >> 16;
    x *= 0x7feb352d;
    x ^= x >> 15;
    x *= 0x846ca68b;
    x ^= x >> 16;
    return x;
}

__device__
double random_uniform01(int seed, int i, int t) {
    unsigned int h = rng_hash(
        static_cast<unsigned int>(seed)
        ^ (static_cast<unsigned int>(i) * 747796405u)
        ^ (static_cast<unsigned int>(t) * 2891336453u)
    );

    return static_cast<double>(h) / static_cast<double>(UINT_MAX);
}

__device__
double random_uniform_range(int seed, int i, int t, double min, double max) {
    double u = random_uniform01(seed, i, t);
    return min + (max - min) * u;
}

__device__
double random_normal_approx(int seed, int i, int t, double mean, double sigma) {
    
    double u1 = random_uniform01(seed, i, t);
    double u2 = random_uniform01(seed + 12345, i, t);

    u1 = fmax(u1, 1e-12);

    double z = sqrt(-2.0 * log(u1)) * cos(2.0 * 3.14159265358979323846 * u2);

    return mean + sigma * z;
}


// #1a initialization of agents (GPU)
__global__
void initialize_agents_kernel(Agent* agents, GpuParams params, int seed) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i >= params.N) {
        return;
    }

    int n_plus = params.N / 3;
    int n_minus = params.N / 3;

    int rank = permuted_rank(i, params.N, seed);

    if (rank < n_plus) {
        agents[i].type = Agent::Agent_type::Optimist;
    } else if (rank < n_plus + n_minus) {
        agents[i].type = Agent::Agent_type::Pessimist;
    } else {
        agents[i].type = Agent::Agent_type::Fundamentalist;
    }

    agents[i].agent_id = i;
    agents[i].cash = params.initial_cash;
    agents[i].asset_inventory = params.initial_inventory;

    agents[i].chartist_order_size = random_uniform_range(seed, i, 0, 0.2 * params.order_size, 2.0 * params.order_size);    
    agents[i].aggressiveness = fabs(random_normal_approx(seed, i, 1, 0.0, params.sigma_aggressiveness));
}

// #1b GPU counting of agent types
__global__
void count_agents_kernel(Agent* agents, int* partial_counts, int N) {
    
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int thread_index = threadIdx.x;

    extern __shared__ int sdata[];

    int* s_plus = sdata;
    int* s_minus = sdata + blockDim.x;
    int* s_fund = sdata + 2 * blockDim.x;

    s_plus[thread_index] = 0;
    s_minus[thread_index] = 0;
    s_fund[thread_index] = 0;

    if (i < N) {
        if (agents[i].type == Agent::Agent_type::Optimist) {
            s_plus[thread_index] = 1;
        } else if (agents[i].type == Agent::Agent_type::Pessimist) {
            s_minus[thread_index] = 1;
        } else {
            s_fund[thread_index] = 1;
        }
    }

    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
        if (thread_index < stride) {
            s_plus[thread_index] += s_plus[thread_index + stride];
            s_minus[thread_index] += s_minus[thread_index + stride];
            s_fund[thread_index] += s_fund[thread_index + stride];
        }
        __syncthreads();
    }

    if (thread_index == 0) {
        partial_counts[3 * blockIdx.x + 0] = s_plus[0];
        partial_counts[3 * blockIdx.x + 1] = s_minus[0];
        partial_counts[3 * blockIdx.x + 2] = s_fund[0];
    }
}


// #1c CPU wrapper returning Counts
Counts count_agents_gpu(Agent* agents, Params& params) {
    
    const int threads = 256;
    const int blocks = (params.N + threads - 1) / threads;
    
    int* partial_counts;
    cudaMallocManaged(&partial_counts, 3 * blocks * sizeof(int));

    size_t size_shared_mem = 3 * threads * sizeof(int);

    count_agents_kernel<<<blocks, threads, size_shared_mem>>>(
        agents,
        partial_counts,
        params.N
    );

    cudaDeviceSynchronize();

    Counts counts;

    for (int b = 0; b < blocks; ++b) {
        counts.nb_optimists += partial_counts[3 * b + 0];
        counts.nb_pessimists += partial_counts[3 * b + 1];
        counts.nb_fundamentalists += partial_counts[3 * b + 2];
    }

    cudaFree(partial_counts);

    return counts;
}


// #1d CPU scalar sentiment x_t
double sentiment( Counts& counts) {
    int noise_traders = counts.nb_optimists + counts.nb_pessimists;
    if (noise_traders == 0) {
        return 0.0;
    }
    return static_cast<double>(counts.nb_optimists- counts.nb_pessimists) / static_cast<double>(noise_traders);
}



// #2.A Computing Transition probabilities (CPU)
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


// #2.B updating of agent types (GPU)
__global__
void update_agents_kernel(Agent* agents, GpuParams params, Probs probs, int t) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i >= params.N) {
        return;
    }

    double u = random_uniform01(params.seed, i, t);

    Agent& agent = agents[i];

    if (agent.type == Agent::Agent_type::Optimist) {
        if (u < probs.plus_to_minus) {
            agent.type = Agent::Agent_type::Pessimist;
        } else if (u < probs.plus_to_minus + probs.plus_to_fund) {
            agent.type = Agent::Agent_type::Fundamentalist;
        }
    }

    else if (agent.type == Agent::Agent_type::Pessimist) {
        if (u < probs.minus_to_plus) {
            agent.type = Agent::Agent_type::Optimist;
        } else if (u < probs.minus_to_plus + probs.minus_to_fund) {
            agent.type = Agent::Agent_type::Fundamentalist;
        }
    }

    else {
        if (u < probs.fund_to_plus) {
            agent.type = Agent::Agent_type::Optimist;
        } else if (u < probs.fund_to_plus + probs.fund_to_minus) {
            agent.type = Agent::Agent_type::Pessimist;
        }
    }
}

// #2.C wrapper
void update_agents_gpu(Params& params, GpuParams& gpu_params, Agent* agents, Probs& probs, int t) {
    
    const int threads = 256;
    const int blocks = (params.N + threads - 1) / threads;

    update_agents_kernel<<<blocks, threads>>>(agents, gpu_params, probs, t);
    
    cudaDeviceSynchronize();
}


// #3a GPU generation of candidate orders
__global__
void generate_orders_kernel(Agent* agents, Order* orders, GpuParams params, Market market){
    
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i >= params.N) {
        return;
    }
    orders[i] = agents[i].make_order(market, params);
}


// #3b GPU order generation + CPU batched local clearing + sequential residual clearing
void add_all_new_orders_gpu(Market& market, Params& params, GpuParams& gpu_params, Agent* agents, Order* orders, Order_book& final_order_book) {
    const int threads = 256;
    const int blocks = (params.N + threads - 1) / threads;

    // 1. Generate all candidate orders on GPU
    generate_orders_kernel<<<blocks, threads>>>(agents, orders, gpu_params, market);
    cudaDeviceSynchronize();

    // 2. Use nb_threads as number of local batches
    int nb_batches = params.nb_threads;

    if (nb_batches <= 0) {
        nb_batches = 1;
    }

    if (nb_batches > params.N) {
        nb_batches = params.N;
    }

    int batch_size = (params.N + nb_batches - 1) / nb_batches;

    // 3. One local order book per batch
    std::vector<Order_book> local_books;
    local_books.reserve(nb_batches);

    for (int b = 0; b < nb_batches; ++b) {
        local_books.emplace_back(params);
    }

    // 4. Clear each batch locally in parallel on CPU
    std::vector<std::thread> batch_threads(nb_batches);

    for (int b = 0; b < nb_batches; ++b) {
        batch_threads[b] = std::thread([&, b]() {
            int start = b * batch_size;
            int end = std::min(start + batch_size, params.N);

            for (int i = start; i < end; ++i) {
                if (orders[i].quantity > 0.0) {
                    local_books[b].add_order(agents, orders[i]);
                }
            }
        });
    }

    for (int b = 0; b < nb_batches; ++b) {
        batch_threads[b].join();
    }

    // 5. Add local trade statistics to the final book statistics
    for (int b = 0; b < nb_batches; ++b) {
        final_order_book.volume += local_books[b].volume;
        final_order_book.volume_weighted_sum += local_books[b].volume_weighted_sum;
    }

    // 6. Collect only active residual orders from local books
    std::vector<Order> residual_orders;
    residual_orders.reserve(params.N);

    for (int b = 0; b < nb_batches; ++b) {

        for (std::size_t j = local_books[b].bid_head;
             j < local_books[b].bids.size();
             ++j) {

            Order* order = local_books[b].bids[j];

            if (order->quantity > 0.0) {
                residual_orders.push_back(*order);
            }
        }

        for (std::size_t j = local_books[b].ask_head;
             j < local_books[b].asks.size();
             ++j) {

            Order* order = local_books[b].asks[j];

            if (order->quantity > 0.0) {
                residual_orders.push_back(*order);
            }
        }
    }

    // 7. Sequentially reconcile remaining orders globally
    for (const Order& order : residual_orders) {
        if (order.quantity > 0.0) {
            final_order_book.add_order(agents, order);
        }
    }
}

// #7 Price update
double update_price(Order_book& order_book) {
    if (order_book.volume == 0) {
        return 100;
    }

    return order_book.volume_weighted_sum / order_book.volume;
}


// Simulation loop
//Parallel
SimTimes run_simulation(Params& params) {
    std::mt19937 rng(params.seed);
    std::normal_distribution<double> normal_dist(0.0, params.sigma_pf);

    GpuParams gpu_params = make_gpu_params(params);

    const int threads = 256;
    const int blocks = (params.N + threads - 1) / threads;

    Agent* agents;
    Order* orders;

    cudaMallocManaged(&agents, params.N * sizeof(Agent));
    cudaMallocManaged(&orders, params.N * sizeof(Order));

    initialize_agents_kernel<<<blocks, threads>>>(agents, gpu_params, params.seed);
    cudaDeviceSynchronize();

    Market market;
    market.p = params.p0;
    market.p_prev = params.p0;
    market.pf = params.pf0;

    std::ofstream out(params.output);

    if (!out.is_open()) {
        std::cerr << "Error: could not open output file " << params.output << "\n";
        cudaFree(orders);
        cudaFree(agents);
        throw std::runtime_error("Could not open output file");
    }

    double time_counting = 0.0;
    double time_updating = 0.0;
    double time_adding = 0.0;

    write_header(out);

    for (int t = 0; t < params.T; ++t) {
        std::cerr << "Period t= " << t << "\n";

        Order_book order_book(params);

        double epsilon = normal_dist(rng);
        market.pf *= std::exp(epsilon);

        auto t0 = std::chrono::high_resolution_clock::now();

        Counts counts_before = count_agents_gpu(agents, params);

        time_counting += std::chrono::duration<double, std::milli>(
            std::chrono::high_resolution_clock::now() - t0
        ).count();

        Probs probs = compute_probabilities(params, market, counts_before);

        auto t1 = std::chrono::high_resolution_clock::now();

        update_agents_gpu(params, gpu_params, agents, probs, t);

        time_updating += std::chrono::duration<double, std::milli>(
            std::chrono::high_resolution_clock::now() - t1
        ).count();

        auto t2 = std::chrono::high_resolution_clock::now();

        Counts counts_after = count_agents_gpu(agents, params);

        time_counting += std::chrono::duration<double, std::milli>(
            std::chrono::high_resolution_clock::now() - t2
        ).count();

        auto t3 = std::chrono::high_resolution_clock::now();

        add_all_new_orders_gpu(
            market,
            params,
            gpu_params,
            agents,
            orders,
            order_book
        );

        double new_price = update_price(order_book);

        time_adding += std::chrono::duration<double, std::milli>(
            std::chrono::high_resolution_clock::now() - t3
        ).count();

        if (!std::isfinite(new_price) || new_price <= 0.0) {
            std::cerr << "Numerical error at time step " << t << "\n";
            std::cerr << new_price << "\n";
            break;
        }

        write_row(out, t, market, new_price, epsilon, counts_after);

        market.p_prev = market.p;
        market.p = new_price;
    }

    out.close();

    cudaFree(orders);
    cudaFree(agents);

    std::cout << "Simulation completed.\n";

    SimTimes simtimes;
    simtimes.adding = time_adding;
    simtimes.counting = time_counting;
    simtimes.updating = time_updating;

    return simtimes;
}