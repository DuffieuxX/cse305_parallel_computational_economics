# cse305_parallel_computational_economics

Within this repository, you will find our project on parallel computational economics as part of the Concurrent and Distributed Computing (CSE305) course.  
Overall, we implement and then parallelize an agent-based financial market model inspired by Lux and Marchesi, *Scaling and criticality in a stochastic multi-agent model of a financial market*.

## Objective

Our goal is to simulate a stochastic multi-agent financial market and study how the computation can be parallelized on CPU.

Throughout the project, we propose:

- a sequential implementation of the model;
- a CPU-parallel implementation;
- benchmarks (runtime, speedup...) for different numbers of agents, time steps, and threads;

## Model

The simulated market contains heterogeneous agents with different trading rules:

- **fundamentalists**: trade according to the deviation between market price and fundamental value;
- **chartists**: follow market trends and are split into optimistic and pessimistic traders.

At each time step:

1. agents may switch between trading states according to stochastic transition rules;
2. aggregate excess demand is computed;
3. the market price is updated;
4. market statistics are recorded.

The main outputs are:

- price time series;
- log returns;
- number of agents in each state;
- excess demand;
- benchmark runtimes.

## Parallelization strategy

The sequential implementation is used as a reference.

The CPU-parallel version focuses on parallelizing operations over agents, especially:

- stochastic state updates;
- local computation of agent counts;
- local computation of excess demand;
- reduction of thread-local quantities into global market variables.

## Repository structure

```text
.
├── README.md
├── src/              # C++ source files
├── include/          # Header files
├── scripts/          # Benchmark and plotting scripts
└── results/         
```
