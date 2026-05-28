#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <random>
#include <string>
#include <vector>

enum class AgentType {
    Optimist,
    Pessimist,
    Fundamentalist
};

struct Parameters {
    int agents = 10000;
    int steps = 1000;
    unsigned int seed = 42;

    double dt = 1.0;

    double initial_price = 100.0;
    double initial_fundamental_value = 100.0;

    // Fundamental value process: log(pf_t) - log(pf_{t-1}) = epsilon_t
    double sigma_epsilon = 0.001;

    // Price impact rule: p_{t+1} = p_t * exp(beta * ED_t / N)
    double beta = 0.01;

    // Excess demand parameters
    double q_noise = 1.0;
    double q_fundamentalist = 5.0;

    // Opinion-switching parameters
    double nu_opinion = 0.02;
    double a1_herding = 1.0;
    double a2_trend = 100.0;

    // Strategy-switching parameters
    double nu_strategy = 0.01;
    double r_alt = 0.0;
    double discount = 0.5;
    double profit_sensitivity = 100.0;

    // Numerical safety: prevents transition probabilities from becoming too large
    double max_total_transition_probability = 0.50;

    std::string output_file = "results/simulation.csv";
};

struct Counts {
    int optimists = 0;
    int pessimists = 0;
    int fundamentalists = 0;
};

struct Probabilities {
    double opt_to_pess = 0.0;
    double opt_to_fund = 0.0;

    double pess_to_opt = 0.0;
    double pess_to_fund = 0.0;

    double fund_to_opt = 0.0;
    double fund_to_pess = 0.0;
};

double clamp(double x, double low, double high) {
    if (x < low) return low;
    if (x > high) return high;
    return x;
}

double transition_probability(double nu, double U, double dt) {
    // Avoid numerical overflow in exp(U)
    U = clamp(U, -20.0, 20.0);
    return nu * std::exp(U) * dt;
}

void normalize_pair(double& p1, double& p2, double max_total_probability) {
    if (p1 < 0.0) p1 = 0.0;
    if (p2 < 0.0) p2 = 0.0;

    double total = p1 + p2;

    if (total > max_total_probability) {
        double scale = max_total_probability / total;
        p1 *= scale;
        p2 *= scale;
    }
}

double uniform_random(std::mt19937& rng) {
    static thread_local std::uniform_real_distribution<double> dist(0.0, 1.0);
    return dist(rng);
}

std::vector<AgentType> initialize_agents(const Parameters& params, std::mt19937& rng) {
    std::vector<AgentType> agents;
    agents.reserve(params.agents);

    std::uniform_real_distribution<double> dist(0.0, 1.0);

    for (int i = 0; i < params.agents; ++i) {
        double u = dist(rng);

        if (u < 1.0 / 3.0) {
            agents.push_back(AgentType::Optimist);
        } else if (u < 2.0 / 3.0) {
            agents.push_back(AgentType::Pessimist);
        } else {
            agents.push_back(AgentType::Fundamentalist);
        }
    }

    return agents;
}

Counts count_agents(const std::vector<AgentType>& agents) {
    Counts counts;

    for (AgentType agent : agents) {
        if (agent == AgentType::Optimist) {
            counts.optimists++;
        } else if (agent == AgentType::Pessimist) {
            counts.pessimists++;
        } else {
            counts.fundamentalists++;
        }
    }

    return counts;
}

double compute_sentiment(const Counts& counts) {
    int n_noise = counts.optimists + counts.pessimists;

    if (n_noise == 0) {
        return 0.0;
    }

    return static_cast<double>(counts.optimists - counts.pessimists)
         / static_cast<double>(n_noise);
}

Probabilities compute_transition_probabilities(
    const Parameters& params,
    double price,
    double previous_price,
    double fundamental_value,
    const Counts& counts
) {
    Probabilities probabilities;

    double r_t = std::log(price / previous_price);
    double sentiment = compute_sentiment(counts);
    double log_mispricing = std::log(fundamental_value / price);

    // 1. Opinion switching among noise traders
    double U_pess_to_opt = params.a1_herding * sentiment
                         + params.a2_trend * r_t;

    double U_opt_to_pess = -U_pess_to_opt;

    probabilities.pess_to_opt =
        transition_probability(params.nu_opinion, U_pess_to_opt, params.dt);

    probabilities.opt_to_pess =
        transition_probability(params.nu_opinion, U_opt_to_pess, params.dt);

    // 2. Strategy switching between noise traders and fundamentalists
    double profit_optimist = r_t;
    double profit_pessimist = params.r_alt - r_t;
    double profit_fundamentalist = params.discount * std::abs(log_mispricing);

    double U_fund_to_opt =
        params.profit_sensitivity * (profit_optimist - profit_fundamentalist);

    double U_opt_to_fund =
        params.profit_sensitivity * (profit_fundamentalist - profit_optimist);

    double U_fund_to_pess =
        params.profit_sensitivity * (profit_pessimist - profit_fundamentalist);

    double U_pess_to_fund =
        params.profit_sensitivity * (profit_fundamentalist - profit_pessimist);

    probabilities.fund_to_opt =
        transition_probability(params.nu_strategy, U_fund_to_opt, params.dt);

    probabilities.opt_to_fund =
        transition_probability(params.nu_strategy, U_opt_to_fund, params.dt);

    probabilities.fund_to_pess =
        transition_probability(params.nu_strategy, U_fund_to_pess, params.dt);

    probabilities.pess_to_fund =
        transition_probability(params.nu_strategy, U_pess_to_fund, params.dt);

    // Each agent can have two possible outgoing transitions.
    // We normalize each pair so that probabilities remain valid.
    normalize_pair(
        probabilities.opt_to_pess,
        probabilities.opt_to_fund,
        params.max_total_transition_probability
    );

    normalize_pair(
        probabilities.pess_to_opt,
        probabilities.pess_to_fund,
        params.max_total_transition_probability
    );

    normalize_pair(
        probabilities.fund_to_opt,
        probabilities.fund_to_pess,
        params.max_total_transition_probability
    );

    return probabilities;
}

void update_agent_states(
    std::vector<AgentType>& agents,
    const Probabilities& probabilities,
    std::mt19937& rng
) {
    for (AgentType& agent : agents) {
        double u = uniform_random(rng);

        if (agent == AgentType::Optimist) {
            if (u < probabilities.opt_to_pess) {
                agent = AgentType::Pessimist;
            } else if (u < probabilities.opt_to_pess + probabilities.opt_to_fund) {
                agent = AgentType::Fundamentalist;
            }
        }

        else if (agent == AgentType::Pessimist) {
            if (u < probabilities.pess_to_opt) {
                agent = AgentType::Optimist;
            } else if (u < probabilities.pess_to_opt + probabilities.pess_to_fund) {
                agent = AgentType::Fundamentalist;
            }
        }

        else if (agent == AgentType::Fundamentalist) {
            if (u < probabilities.fund_to_opt) {
                agent = AgentType::Optimist;
            } else if (u < probabilities.fund_to_opt + probabilities.fund_to_pess) {
                agent = AgentType::Pessimist;
            }
        }
    }
}

double compute_excess_demand(
    const Parameters& params,
    const Counts& counts,
    double price,
    double fundamental_value
) {
    double noise_demand =
        params.q_noise * static_cast<double>(counts.optimists - counts.pessimists);

    double fundamentalist_demand =
        params.q_fundamentalist
        * static_cast<double>(counts.fundamentalists)
        * std::log(fundamental_value / price);

    return noise_demand + fundamentalist_demand;
}

void write_header(std::ofstream& file) {
    file << "time,"
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

void write_state(
    std::ofstream& file,
    int time,
    double price,
    double fundamental_value,
    double log_return,
    double fundamental_shock,
    double log_mispricing,
    double sentiment,
    const Counts& counts,
    double excess_demand
) {
    file << time << ","
         << price << ","
         << fundamental_value << ","
         << log_return << ","
         << fundamental_shock << ","
         << log_mispricing << ","
         << sentiment << ","
         << counts.optimists << ","
         << counts.pessimists << ","
         << counts.fundamentalists << ","
         << excess_demand << "\n";
}

void run_sequential_simulation(const Parameters& params) {
    if (params.agents <= 0) {
        std::cerr << "Error: number of agents must be positive.\n";
        return;
    }

    if (params.steps <= 0) {
        std::cerr << "Error: number of steps must be positive.\n";
        return;
    }

    std::filesystem::path output_path(params.output_file);

    if (output_path.has_parent_path()) {
        std::filesystem::create_directories(output_path.parent_path());
    }

    std::ofstream output(params.output_file);

    if (!output.is_open()) {
        std::cerr << "Error: could not open output file " << params.output_file << "\n";
        return;
    }

    std::mt19937 rng(params.seed);
    std::normal_distribution<double> normal_dist(0.0, params.sigma_epsilon);

    std::vector<AgentType> agents = initialize_agents(params, rng);

    double price = params.initial_price;
    double previous_price = params.initial_price;
    double fundamental_value = params.initial_fundamental_value;

    write_header(output);

    for (int t = 0; t < params.steps; ++t) {
        // 1. Exogenous update of the fundamental value
        double epsilon = normal_dist(rng);
        fundamental_value *= std::exp(epsilon);

        // 2. Compute indicators from the current state
        Counts counts_before = count_agents(agents);
        double sentiment_before = compute_sentiment(counts_before);

        Probabilities probabilities = compute_transition_probabilities(
            params,
            price,
            previous_price,
            fundamental_value,
            counts_before
        );

        // 3. Update agent states using probabilities frozen at the beginning of the step
        update_agent_states(agents, probabilities, rng);

        // 4. Compute excess demand after the state update
        Counts counts_after = count_agents(agents);

        int total_agents =
            counts_after.optimists
            + counts_after.pessimists
            + counts_after.fundamentalists;

        if (total_agents != params.agents) {
            std::cerr << "Agent count error at time step " << t << "\n";
            break;
        }

        double log_mispricing = std::log(fundamental_value / price);
        double sentiment_after = compute_sentiment(counts_after);

        double excess_demand = compute_excess_demand(
            params,
            counts_after,
            price,
            fundamental_value
        );

        // 5. Endogenous market price update
        double new_price = price * std::exp(params.beta * excess_demand / params.agents);

        if (!std::isfinite(new_price) || new_price <= 0.0) {
            std::cerr << "Numerical error: invalid price at time step " << t << "\n";
            break;
        }

        double log_return = std::log(new_price / price);

        // 6. Record output
        write_state(
            output,
            t,
            new_price,
            fundamental_value,
            log_return,
            epsilon,
            log_mispricing,
            sentiment_after,
            counts_after,
            excess_demand
        );

        previous_price = price;
        price = new_price;
    }

    output.close();

    std::cout << "Sequential simulation completed.\n";
    std::cout << "Output written to: " << params.output_file << "\n";
}

void print_usage(const char* program_name) {
    std::cout
        << "Usage:\n"
        << "  " << program_name << " [options]\n\n"
        << "Options:\n"
        << "  --agents N              Number of agents\n"
        << "  --steps T               Number of time steps\n"
        << "  --seed S                Random seed\n"
        << "  --output FILE           Output CSV file\n"
        << "  --sigma-epsilon X       Standard deviation of fundamental shocks\n"
        << "  --beta X                Price-impact parameter\n"
        << "  --help                  Show this message\n";
}

int main(int argc, char* argv[]) {
    Parameters params;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];

        if (arg == "--agents" && i + 1 < argc) {
            params.agents = std::stoi(argv[++i]);
        }

        else if (arg == "--steps" && i + 1 < argc) {
            params.steps = std::stoi(argv[++i]);
        }

        else if (arg == "--seed" && i + 1 < argc) {
            params.seed = static_cast<unsigned int>(std::stoul(argv[++i]));
        }

        else if (arg == "--output" && i + 1 < argc) {
            params.output_file = argv[++i];
        }

        else if (arg == "--sigma-epsilon" && i + 1 < argc) {
            params.sigma_epsilon = std::stod(argv[++i]);
        }

        else if (arg == "--beta" && i + 1 < argc) {
            params.beta = std::stod(argv[++i]);
        }

        else if (arg == "--help") {
            print_usage(argv[0]);
            return 0;
        }

        else {
            std::cerr << "Unknown or incomplete argument: " << arg << "\n";
            print_usage(argv[0]);
            return 1;
        }
    }

    run_sequential_simulation(params);

    return 0;
}