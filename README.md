# CSE305 Parallel Computational Economics

Within this repository, you will find our project on parallel computational economics as part of the Concurrent and Distributed Computing (CSE305) course. The project is inspired by the agent-based financial market models of Lux and Marchesi defined in their paper *Scaling and criticality in a stochastic multi-agent model of a financial market* (1999) and *Volatility clustering in financial markets: a microsimulation of interacting agents* (2000).

These papers are used only as initial source for the model. We copied from their models agent state transitions, .???, and we added limit-order generation and order-book market clearing. This model will be used a benchmark for parallelization. The objective of the project is to study how different parallelization strategies perform on this model.

The computational feature of the model is that most agent-level operations are independent, while market clearing is order-dependent. This creates a natural mixed parallel/sequential workload. For instance agent initialization, counting, type updating, and order generation can be parallelized, whereas order-book clearing, being a first come first served model, remains a sequential bottleneck.

## Repository structure

```text
.
├── CUDA/
│   └── GPU_V3/
│       ├── main.cpp
│       ├── market.cpp
│       ├── market.hpp
│       ├── model.cu
│       └── output.cpp
│
├── src/
│   ├── Sequential_linked_list/
│   ├── Sequential_array/
│   ├── CPU_parallel_V3/
│   ├── CPU_parallel_V4/
│   ├── analysis.py
│   └── simulation
│
├── README.md
└── .gitignore
```

The `src/` directory contains the CPU implementations.
The `CUDA/` directory contains the CUDA implementation. Currently, `CUDA/GPU_V3` is the GPU version corresponding to `src/CPU_parallel_V3`.

## Implemented versions

| Version                       | Description                                                                                                                                             |
| ----------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `src/Sequential_linked_list/` | Sequential baseline with linked-list-style order-book storage.                                                                                          |
| `src/Sequential_array/`       | Sequential baseline using array/vector order-book storage.                                                                                              |
| `src/CPU_parallel_V3/`        | CPU-parallel version. Agent-wise operations are parallelized with C++ threads; market clearing remains sequential.                                      |
| `src/CPU_parallel_V4/`        | CPU-parallel version with batched market-clearing improvements. Orders are grouped before clearing to reduce part of the sequential insertion overhead. |
| `CUDA/GPU_V3/`                | CUDA hybrid version. Agent-wise operations run on GPU; market clearing remains sequential on CPU.                                                       |

## Model implemented

The market contains three types of agents:

* `Optimist`: chartist submitting buy orders;
* `Pessimist`: chartist submitting sell orders;
* `Fundamentalist`: trader reacting to mispricing between market price and fundamental value.

At each time step, the simulation executes the following sequence.

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

## Parallelization structure

| Step                               | `CPU_parallel_V3` | `CUDA/GPU_V3`                         |
| ---------------------------------- | ----------------- | ------------------------------------- |
| Agent initialization               | CPU threads       | GPU                                   |
| Fundamental value update           | CPU               | CPU                                   |
| Agent counting                     | CPU threads       | GPU reduction + CPU final aggregation |
| Transition probability computation | CPU               | CPU                                   |
| Agent type update                  | CPU threads       | GPU                                   |
| Candidate order generation         | CPU threads       | GPU                                   |
| Market clearing                    | CPU sequential    | CPU sequential                        |
| Price update                       | CPU               | CPU                                   |
| Output writing                     | CPU               | CPU                                   |

Market clearing is kept sequential as order book is stateful and order-dependent. Inserting one order modify the book, change agent inventories and cash, and affect subsequent matches.

## CPU configuration

In `CPU_parallel_V3`, the number of CPU threads is controlled by `Params::nb_threads` in `market.hpp`.

We use a static block decomposition:

```cpp
chunk_size = N / nb_threads;
```

Each thread processes a contiguous interval of agents. The last thread receives the remaining agents when (N) is not exactly divisible by the number of threads.

In `CPU_parallel_V3`, threads are used for:

* agent initialization;
* agent counting;
* agent type update;
* candidate order generation.

After candidate orders are generated, the main thread joins the worker threads and inserts orders into the order book sequentially.

To benchmark different CPU thread counts, edit `nb_threads` in `market.hpp`.

## CUDA configuration

In `CUDA/GPU_V3`, each CUDA kernel maps one GPU thread to one agent.

The default configuration is

```cpp
threads = 256;
blocks = (N + threads - 1) / threads;
```

So for (N) agents, the grid contains enough blocks to cover all agents, and excess threads return immediately. The choice of 256 threads per block corresponds to 8 warps per block and is also convenient for the block-level reduction used in agent counting.

In this CUDA version, we use managed memory for the main `Agent*` and `Order*` arrays to allow both GPU kernels and CPU market clearing to access the same data.


## Build instructions

### CPU versions

Example for `CPU_parallel_V3`:

```bash
cd src/CPU_parallel_V3
g++ -O3 -std=c++17 main.cpp model.cpp market.cpp output.cpp -o simulation
```

Run:

```bash
./simulation --agents 10000 --steps 100 --time
```

Example for `CPU_parallel_V4`:

```bash
cd src/CPU_parallel_V4
g++ -O3 -std=c++17 main.cpp model.cpp market.cpp output.cpp -o simulation
./simulation --agents 10000 --steps 100 --time
```

### CUDA version

On the school machines, compile with:

```bash
cd CUDA/GPU_V3

/usr/local/cuda/bin/nvcc -arch=sm_60 -std=c++17 \
  -I/usr/local/cuda/include \
  main.cpp model.cu market.cpp output.cpp \
  -o simulation
```

Run:

```bash
./simulation --agents 10000 --steps 100 --time
```

Large benchmark example:

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

For fair comparison, CPU and GPU versions should be run on the same machine.

Example:

```bash
cd src/CPU_parallel_V3
./simulation --agents 500000 --steps 10 --time

cd ../../CUDA/GPU_V3
./simulation --agents 500000 --steps 10 --time
```

Preliminary benchmark on the school SSH machine:

|      N |  T | Version           | Total ms | Counting ms | Updating ms | Adding / clearing ms |
| -----: | -: | ----------------- | -------: | ----------: | ----------: | -------------------: |
| 500000 | 10 | `CPU_parallel_V3` |  32601.6 |       167.8 |       156.0 |              32178.0 |
| 500000 | 10 | `CUDA/GPU_V3`     |  28571.2 |        42.6 |         1.3 |              28266.8 |

The CUDA version strongly accelerates the agent-wise operations. The overall speedup is smaller because market clearing remains sequential and dominates runtime for large (N).

## Terminal timing output

With `--time`, the program prints a runtime summary:

```text
Simulation completed.
Run 1: 2097.15 ms  (counting=56.2052 ms  updating=42.5583 ms  adding=1980.3 ms)

And we define:
counting: time spent counting optimists, pessimists, and fundamentalists.
updating: time spent updating agents' types.
adding: time spent generating orders, clearing the market, and updating the price.