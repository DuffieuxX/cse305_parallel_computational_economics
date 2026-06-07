# Project README - (CSE305) Concurrent and Distributed Computing

Within this repository, you will find our project on parallel computational economics as part of the Concurrent and Distributed Computing (CSE305) course. The project is inspired by the agent-based financial market models of Lux and Marchesi, especially *Scaling and criticality in a stochastic multi-agent model of a financial market* (1999) and *Volatility clustering in financial markets: a microsimulation of interacting agents* (2000).

We use these papers as the starting point for the economic structure of the model: heterogeneous agents, optimists, pessimists, fundamentalists, and stochastic transitions driven by sentiment, trends, and mispricing. We then fix one computational model by adding limit-order generation and order-book market clearing. Our goal is then to study different parallelization strategies on this same model.

The computational complexity is that most agent-level operations are independent, while the market clearing is order-dependent. Agent initialization, counting, type updating, and order generation can be parallelized; order-book clearing is complex to parallelize since it relies on price-priority matching and sequential order arrival. We will propose ways to optimize this process.

## Repository structure

```text
.
├── CUDA/
│   ├── GPU_parallel_agent_level/
│   ├── GPU_parallel_batching/
│   └── GPU_parallel_batching_optimized/
│
├── src/
│   ├── CPU_parallel_agent_level/
│   ├── CPU_parallel_batching/
│   ├── Parrallel_multiple_markets/
│   ├── Parrallel_multiple_markets_batching/
│   ├── Parrallel_multiple_markets_batching_2/
│   ├── Sequential_array/
│   ├── Sequential_linked_list/
│   ├── Sequential_multiple_markets/
│   ├── analysis.py
│   └── simulation
│
├── README.md
└── .gitignore
```

The `src/` directory contains the CPU implementations. The `CUDA/` directory contains the CUDA implementations. Folder names are written exactly as they appear in the GitHub repository.

## Implemented versions

| Version                                      | Description                                                                                                                                    |
| -------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------- |
| `src/Sequential_linked_list/`                | Sequential baseline with linked-list-style order-book storage.                                                                                 |
| `src/Sequential_array/`                      | Sequential baseline using array/vector order-book storage.                                                                                     |
| `src/Sequential_multiple_markets/`           | Sequential extension with several assets and one order book per asset.                                                                         |
| `src/CPU_parallel_agent_level/`              | CPU-parallel version where agent-level operations are parallelized with C++ threads while market clearing remains sequential.                  |
| `src/CPU_parallel_batching/`                 | CPU batching version where agents are split into batches, local order books are cleared in parallel, and residual orders are merged afterward. |
| `src/Parrallel_multiple_markets/`            | Exact multiple-market parallel model where different markets are processed in parallel, with locks on shared agent cash.                       |
| `src/Parrallel_multiple_markets_batching/`   | Multiple-market batching version where each market uses batch-level parallel clearing.                                                         |
| `src/Parrallel_multiple_markets_batching_2/` | Alternative multiple-market batching implementation used for experimentation.                                                                  |
| `CUDA/GPU_parallel_agent_level/`             | CUDA version of the agent-level parallel architecture.                                                                                         |
| `CUDA/GPU_parallel_batching/`                | CUDA version of the batching architecture.                                                                                                     |
| `CUDA/GPU_parallel_batching_optimized/`      | Optimized CUDA batching implementation.                                                                                                        |

## Parallelization structure

| Step                               | Agent-level CPU/GPU models | Batching models                       | Multiple-market models                  |
| ---------------------------------- | -------------------------- | ------------------------------------- | --------------------------------------- |
| Agent initialization               | Parallelized               | Parallelized                          | Parallelized                            |
| Fundamental value update           | CPU                        | CPU                                   | CPU                                     |
| Agent counting                     | Parallelized               | Parallelized                          | Parallelized                            |
| Transition probability computation | CPU                        | CPU                                   | CPU                                     |
| Agent type update                  | Parallelized               | Parallelized                          | Parallelized                            |
| Candidate order generation         | Parallelized               | Parallelized inside batches           | Parallelized by market or batch         |
| Market clearing                    | Sequential                 | Local batch clearing + residual merge | Parallel by market or batched by market |
| Price update                       | CPU                        | CPU                                   | CPU                                     |
| Output writing                     | CPU                        | CPU                                   | CPU                                     |

Market clearing is the main computational bottleneck because the order book is stateful and order-dependent. Inserting one order modifies the book, changes agent inventories and cash, and affects subsequent matches.

## CPU configuration

In the CPU-parallel versions, the number of CPU threads is controlled by `Params::nb_threads` in `market.hpp`.

Agent-level parallelism uses a static block decomposition over the agent vector. Each thread processes a contiguous interval of agents, and the last thread receives the remaining agents when `N` is not exactly divisible by the number of threads.

In the batching versions, agents are split into batches. Each batch builds and clears a local order book. The remaining orders are then merged into a common order book. This relaxes strict global first-come-first-served clearing but reduces the cost of repeated sorted insertions.

In the multiple-market versions, each asset has its own market and order book. The exact multiple-market model parallelizes across markets. The batching multiple-market model applies the batch-clearing logic within markets.

## CUDA configuration

In the CUDA versions, GPU kernels map threads to agents for agent-wise operations such as counting, updating, and order generation. Market clearing remains partly or fully CPU-side depending on the version because the order book is sequential and stateful.

The default CUDA configuration follows the standard pattern:

```cpp
threads = 256;
blocks = (N + threads - 1) / threads;
```

This creates enough GPU threads to cover all agents, while excess threads return immediately.

## Build instructions

### CPU versions

Sequential linked-list baseline:

```bash
cd src/Sequential_linked_list
g++ -O3 -DNDEBUG -std=c++20 -pthread main.cpp model.cpp market.cpp output.cpp -o simulation
./simulation --agents 10000 --steps 100 --time
```

Sequential array baseline:

```bash
cd src/Sequential_array
g++ -O3 -DNDEBUG -std=c++20 -pthread main.cpp model.cpp market.cpp output.cpp -o simulation
./simulation --agents 10000 --steps 100 --time
```

Sequential multiple-markets model:

```bash
cd src/Sequential_multiple_markets
g++ -O3 -DNDEBUG -std=c++20 -pthread main.cpp model.cpp market.cpp output.cpp -o simulation
./simulation --agents 10000 --steps 100 --time
```

CPU parallel agent-level model:

```bash
cd src/CPU_parallel_agent_level
g++ -O3 -DNDEBUG -std=c++20 -pthread main.cpp model.cpp market.cpp output.cpp -o simulation
./simulation --agents 10000 --steps 100 --time
```

CPU batching model:

```bash
cd src/CPU_parallel_batching
g++ -O3 -DNDEBUG -std=c++20 -pthread main.cpp model_v4.cpp market.cpp output.cpp -o simulation
./simulation --agents 10000 --steps 100 --time
```

Exact parallel multiple-markets model:

```bash
cd src/Parrallel_multiple_markets
g++ -O3 -DNDEBUG -std=c++20 -pthread main.cpp model.cpp market.cpp output.cpp -o simulation
./simulation --agents 10000 --steps 100 --time
```

Multiple-market batching model:

```bash
cd src/Parrallel_multiple_markets_batching
g++ -O3 -DNDEBUG -std=c++20 -pthread main.cpp model.cpp market.cpp output.cpp -o simulation
./simulation --agents 10000 --steps 100 --time
```

Alternative multiple-market batching model:

```bash
cd src/Parrallel_multiple_markets_batching_2
g++ -O3 -DNDEBUG -std=c++20 -pthread main.cpp model.cpp market.cpp output.cpp -o simulation
./simulation --agents 10000 --steps 100 --time
```

### CUDA versions

On the university computers, compile CUDA versions with `nvcc`.

GPU parallel agent-level model:

```bash
cd CUDA/GPU_parallel_agent_level

/usr/local/cuda/bin/nvcc -O3 -std=c++17 \
  main.cpp model.cu market.cpp output.cpp \
  -o simulation

./simulation --agents 10000 --steps 100 --time
```

GPU parallel batching model:

```bash
cd CUDA/GPU_parallel_batching

/usr/local/cuda/bin/nvcc -O3 -std=c++17 \
  main.cpp model.cu market.cpp output.cpp \
  -o simulation

./simulation --agents 10000 --steps 100 --time
```

GPU parallel batching optimized model:

```bash
cd CUDA/GPU_parallel_batching_optimized

/usr/local/cuda/bin/nvcc -O3 -std=c++17 \
  main.cpp model.cu market.cpp output.cpp \
  -o simulation

./simulation --agents 10000 --steps 100 --time
```

Large-run example:

```bash
./simulation --agents 500000 --steps 10 --time
```

## Runtime options

| Option          | Meaning                                            |
| --------------- | -------------------------------------------------- |
| `--agents N`    | Number of agents                                   |
| `--steps T`     | Number of simulation time steps                    |
| `--seed S`      | Random seed                                        |
| `--output FILE` | Output CSV path                                    |
| `--sigma-pf X`  | Volatility of the fundamental value                |
| `--time`        | Print timing information                           |
| `--runs K`      | Repeat the simulation `K` times and print averages |

Example:

```bash
./simulation --agents 50000 --steps 100 --time --runs 5
```

## Benchmarking

The report benchmark tables were produced by running each model several times with fixed values of `N`, `T`, thread counts, and seeds. The GitHub repository focuses on the simulation implementations and their standard `main.cpp` entry points. Therefore, to reproduce a model run from the repository, compile the corresponding folder and execute the generated `simulation` program with the desired command-line parameters.

Example comparison between the agent-level CPU and GPU versions:

```bash
cd src/CPU_parallel_agent_level
g++ -O3 -DNDEBUG -std=c++20 -pthread main.cpp model.cpp market.cpp output.cpp -o simulation
./simulation --agents 500000 --steps 10 --time

cd ../../CUDA/GPU_parallel_agent_level
/usr/local/cuda/bin/nvcc -O3 -std=c++17 \
  main.cpp model.cu market.cpp output.cpp \
  -o simulation
./simulation --agents 500000 --steps 10 --time
```


## Terminal timing output

With `--time`, the program prints a runtime summary such as:

```text
Simulation completed.
Run 1: 2097.15 ms  (counting=56.2052 ms  updating=42.5583 ms  adding=1980.3 ms)
```

The timing fields have the following interpretation:

| Field               | Meaning                                                                       |
| ------------------- | ----------------------------------------------------------------------------- |
| `counting`          | Time spent counting optimists, pessimists, and fundamentalists.               |
| `updating`          | Time spent updating agent types.                                              |
| `adding`            | Time spent generating orders and processing market clearing.                  |
| `market_processing` | In multiple-market models, order generation plus per-market clearing.         |
| `order_generation`  | In batching models, order creation plus local batch clearing when applicable. |
| `clearing`          | In batching models, residual/global clearing after local batch clearing.      |









## Model

The market contains three types of agents:

* `Optimist`: chartist submitting buy orders;
* `Pessimist`: chartist submitting sell orders;
* `Fundamentalist`: trader reacting to mispricing between market price and fundamental value.

At each time step, the simulation executes the following sequence:

### 1. Fundamental value update

The fundamental value follows a multiplicative Gaussian shock:

$$
p_f(t+1)=p_f(t)\exp(\epsilon_t),
\qquad
\epsilon_t \sim \mathcal{N}(0,\sigma_{pf}^{2}).
$$

### 2. Agent counting and sentiment

The simulation counts

$$
n_+(t), \qquad n_-(t), \qquad n_f(t),
$$

where (n_+(t)) is the number of optimists, (n_-(t)) the number of pessimists, and (n_f(t)) the number of fundamentalists.

Chartist sentiment is

$$
x(t)=
\frac{n_+(t)-n_-(t)}
{n_+(t)+n_-(t)}.
$$

If there are no chartists, the implementation sets (x(t)=0).

### 3. Transition probability computation

The log return is

$$
r(t)=\log\left(\frac{p(t)}{p(t-1)}\right).
$$

The log mispricing is

$$
m_{\log}(t)=\log\left(\frac{p_f(t)}{p(t)}\right).
$$

Opinion switching between optimists and pessimists uses

$$
U_{-\to +}(t)=a_{\text{herding}}x(t)+a_{\text{trend}}r(t),
\qquad
U_{+\to -}(t)=-U_{-\to +}(t).
$$

The implemented switching probability has the form

$$
P(i\to j)=\nu \exp(U_{i\to j})\Delta t.
$$

For strategy switching, the implemented payoff proxies are

$$
\pi_+(t)=r(t),
\qquad
\pi_-(t)=-r(t),
\qquad
\pi_f(t)=|m_{\log}(t)|.
$$

The resulting probabilities are computed in `compute_probabilities`.

### 4. Agent type update

Once transition probabilities are fixed, each agent draws one uniform random number and updates its type independently. This step is naturally parallel.

### 5. Candidate order generation

Each agent generates at most one limit order.

An optimist submits a buy order:

$$
p_i^{limit}=p(t)(1+a_i),
\qquad
q_i=\min\left(q_i^c,\frac{cash_i}{p_i^{limit}}\right).
$$

A pessimist submits a sell order:

$$
p_i^{limit}=p(t)(1-a_i),
\qquad
q_i=\min(q_i^c,inventory_i).
$$

A fundamentalist uses the relative mispricing

$$
m(t)=\frac{p_f(t)-p(t)}{p(t)}.
$$

If (m(t)\geq m_{\min}), the fundamentalist submits a buy order.
If (m(t)\leq -m_{\min}), the fundamentalist submits a sell order.
Otherwise, no order is submitted.

For active fundamentalist orders, the desired order size is

$$
q_i^f=q_i^c(1+\gamma_f|m(t)|).
$$

No borrowing and no short-selling are enforced by taking the minimum with available cash or inventory.

### 6. Market clearing

The order book stores bids and asks sorted by limit price. A buy order matches against the best ask if

$$
p_{\text{ask}}\leq p_{\text{buy}}^{limit}.
$$

A sell order matches against the best bid if

$$
p_{\text{bid}}\geq p_{\text{sell}}^{limit}.
$$

Whenever a trade occurs, the implementation updates both agents' cash and inventory. The traded price is the price of the resting order.

### 7. Price update

The new market price is the volume-weighted average traded price:

$$
p(t+1)=
\frac{\sum_k q_k p_k}{\sum_k q_k}.
$$

If no trade occurs, the implementation uses the fallback price defined in the code.
