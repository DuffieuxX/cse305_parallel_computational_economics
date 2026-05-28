# cse305_parallel_computational_economics

Within this repository, you will find our project on parallel computational economics as part of the Concurrent and Distributed Computing (CSE305) course. We implement and then parallelize an agent-based financial market model inspired by Lux and Marchesi *Scaling and criticality in a stochastic multi-agent model of a financial market* 1999 paper and then their "Volatility clustering in financial markets: a microsimulation of interacting agents". 

## Model overview

The model consists in a financial market simulation with heterogeneous agents each behaving with different rules:

- **fundamentalists** trade according to the deviation between the market price and a fundamental value;
- **noise traders**, or chartists, react to price trends and market sentiment. They are divided into optimistic and pessimistic traders.

The model then combines three mechanisms:

1. stochastic changes in the fundamental value of the asset;
2. endogenous switching of agents between trading states;
3. endogenous price changes caused by aggregate excess demand.


## Repository structure

```text
.
├── README.md
├── src/              # C++ source files
├── include/          # Header files
├── scripts/          # Benchmark and plotting scripts
└── results/         
```


## Detailed model implementation


The implementation is based on the Lux--Marchesi microsimulation model of interacting chartists and fundamentalists. We use the detailed transition rules described in the 2000 paper, and we optionally add Gaussian shocks to the fundamental value as in the 1999 Nature version of the model.

The market contains three agent states:

- `Optimist`: chartist expecting a rising market;
- `Pessimist`: chartist expecting a declining market;
- `Fundamentalist`: trader expecting the market price to revert toward the fundamental value.

We denote the chartist population as

```math
n_c(t)=n_+(t)+n_-(t)
```

where `n_+` is the number of optimistic chartists and `n_-` is the number of pessimistic chartists. The total population is

```math
N=n_c(t)+n_f(t)
```

where `n_f` is the number of fundamentalists.


Following the paper, we will simulate the model with small discrete time increments:

```math
\Delta t = 0.01.
```

### Main state variables

| Variable                    | Meaning                                                 |
| --------------------------- | ------------------------------------------------------- |
| `p`                         | current market price                                    |
| `pf`                        | current fundamental value                               |
| `dt` / `Delta t` | micro time step used to discretize transition rates |
| `epsilon`                   | exogenous shock to the fundamental value                |
| `n_optimists` / `n_+`       | number of optimistic chartists                          |
| `n_pessimists` / `n_-`      | number of pessimistic chartists                         |
| `n_chartists` / `n_c`       | total number of chartists                               |
| `n_fundamentalists` / `n_f` | number of fundamentalists                               |
| `x`                         | opinion index among chartists                           |
| `z`                         | fraction of chartists in the population                 |
| `ED`                        | aggregate excess demand                                 |
| `mu`                        | noise in the market maker's perception of excess demand |

### Step 1. The Fundamental value

First, we allow for a stochastic fundamental value price update:

```math
p_{f,t}=p_{f,t-1}\exp(\epsilon_t),
```

with

```math
\epsilon_t \sim \mathcal{N}(0,\sigma_{\epsilon}^{2}).
```

So the exogenous input to the market is Gaussian.

### Step 2. The Opinion index and price trend

Then, we define the opinion index among chartists as

```math
x(t)
=
\frac{n_+(t)-n_-(t)}{n_c(t)},
\qquad
x(t)\in[-1,1].
```

And the fraction of chartists in the population being

```math
z(t)=\frac{n_c(t)}{N}.
```

The price trend `dp/dt` is estimated from the recent price path rather than from a single micro-step. Following Lux and Marchesi, we use the average price change over `[t - 0.2, t)`, which corresponds to 20 micro-steps when `dt = 0.01`.


### Step 3. Opinion switch between optimistic and pessimistic chartists

Opinion switching is driven by two forces:

* herding, measured by the opinion index `x`;
* trend-following, measured by the recent price trend.

Here `nu_1` is the (fixed) revision frequency for opinion changes. A larger `nu_1` means chartists reconsider their opinion more often, while the exponential term adjusts this baseline frequency according to herding and trend-following forces.


The transition intensities are

```math
\lambda_{-\to +}(t)
=
\nu_1
\frac{n_c(t)}{N}
\exp(U_1(t)),
```

and

```math
\lambda_{+\to -}(t)
=
\nu_1
\frac{n_c(t)}{N}
\exp(-U_1(t)).
```
With the forcing term being

```math
U_1(t)
=
\alpha_1 x(t)
+
\alpha_2
\frac{\dot p(t)}{\nu_1}.
```

At each micro-step, the corresponding transition probabilities are then approximately

```math
P(-\to +)
\approx
\lambda_{-\to +}(t)\Delta t,
\qquad
P(+\to -)
\approx
\lambda_{+\to -}(t)\Delta t.
```

Thus, if the price trend is positive and optimistic chartists are already the majority, pessimists are more likely to become optimists. Conversely, if the trend is negative and pessimists dominate, optimists are more likely to become pessimists.

### Step 4. Switch between chartists and fundamentalists

At each micro-step, agents may also switch between chartist and fundamentalist strategies. These switches are driven by relative profitability.

We use the following variables:

* `r`: nominal dividend of the asset, fixed in the 2000 baseline;
* `R`: average return from alternative investments, fixed;
* `s`: discount factor applied to fundamentalist arbitrage profits, fixed with `s < 1`;
* `nu_2`: revision frequency for switching between chartist and fundamentalist strategies, fixed;
* `alpha_3`: sensitivity to profit differences, fixed.

First, we update the dividend `r` with the current fundamental value by setting `r_t = R * pf_t`. Then, for optimistic chartists versus fundamentalists, the forcing term is

```math
U_{f,+}(t)
=
\alpha_3
\left(
\frac{r_t+\dot p(t)/\nu_2}{p(t)}
-
R
-
s
\left|
\frac{p_f(t)-p(t)}{p(t)}
\right|
\right)
```
The transition intensities are

```math
\lambda_{f\to +}(t)
=
\nu_2
\frac{n_+(t)}{N}
\exp(U_{f,+}(t))
```

and

```math
\lambda_{+\to f}(t)
=
\nu_2
\frac{n_f(t)}{N}
\exp(-U_{f,+}(t))
```

For pessimistic chartists versus fundamentalists, the forcing term is

```math
U_{f,-}(t)
=
\alpha_3
\left(
R
-
\frac{r_t+\dot p(t)/\nu_2}{p(t)}
-
s
\left|
\frac{p_f(t)-p(t)}{p(t)}
\right|
\right)
```

The transition intensities are

```math
\lambda_{f\to -}(t)
=
\nu_2
\frac{n_-(t)}{N}
\exp(U_{f,-}(t))
```

and

```math
\lambda_{-\to f}(t)
=
\nu_2
\frac{n_f(t)}{N}
\exp(-U_{f,-}(t))
```

Over one micro-step, each transition probability is approximated by

```math
P(i\to j)
\approx
\lambda_{i\to j}(t)\Delta t
``` 
Using these transition probabilities, each agent is updated stochastically. This is useful for parallelization since once the transition probabilities are fixed, agents can be processed independently.

### Step 5. Update Excess demand

At each micro-step, we aggregate excess demand as

```math
ED(t)=ED_c(t)+ED_f(t).
```

Chartists buy or sell a fixed number of units `t_c`. Optimists buy and pessimists sell, so chartist excess demand is

```math
ED_c(t)
=
(n_+(t)-n_-(t))t_c.
```

Fundamentalists trade against mispricing. Their excess demand is

```math
ED_f(t)
=
n_f(t)\gamma(p_f(t)-p(t)),
```

where `gamma` is the strength of the fundamentalists' reaction to the deviation between fundamental value and market price.

Hence,

```math
ED(t)
=
(n_+(t)-n_-(t))t_c
+
n_f(t)\gamma(p_f(t)-p(t)).
```

If `p < pf`, fundamentalists buy. If `p > pf`, fundamentalists sell.

### Step 6. Market price update

Since we combine the detailed transition mechanism of Lux--Marchesi (2000) with the Gaussian update of the fundamental value from Lux--Marchesi (1999), we use a log-price impact rule instead of the fixed-tick price rule of the 2000 microsimulation.

```math
p_{t+\Delta t}
=
p_t
\exp\left(
\beta \frac{ED_t}{N}
\right) 
```
Equivalently, excess demand changes the log-price by `beta * ED_t / N`, so positive excess demand increases the price and negative excess demand decreases it.

### Step 7. Recorded outputs

At each simulation step, the model records the aggregate market state. The main recorded return is

```math
ret_t
=
\log(p_t)-\log(p_{t-1})
```

The output file stores:

* market price;
* fundamental value;
* log return;
* fundamental-value shock;
* opinion index;
* number of optimistic chartists;
* number of pessimistic chartists;
* number of fundamentalists;
* excess demand.