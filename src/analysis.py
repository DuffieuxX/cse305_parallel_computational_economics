import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import sys
import os

def load_simulation(path: str) -> pd.DataFrame:
    with open(path) as f:
        content = f.read()

    lines = [l.strip() for l in content.strip().split('\n') if l.strip()]

    if len(lines) <= 2:
        # Header on line 1, all data crammed on line 2 (or single line)
        header_line = lines[0]
        data_line   = lines[1] if len(lines) == 2 else lines[0]
        cols   = [c.strip() for c in header_line.rstrip(',').split(',')]
        values = [v.strip() for v in data_line.rstrip(',').split(',') if v.strip()]
    else:
        cols   = [c.strip() for c in lines[0].rstrip(',').split(',')]
        values = [v.strip() for v in ','.join(lines[1:]).rstrip(',').split(',') if v.strip()]

    n = len(cols)
    values = [v for v in values if v]
    n_rows = len(values) // n
    arr = np.array(values[:n_rows * n], dtype=float).reshape(n_rows, n)
    return pd.DataFrame(arr, columns=cols)


def price_stats(df):
    r = df['log_return'].dropna()
    p = df['price']
    return {
        'mean_return':      r.mean(),
        'volatility':       r.std(),
        'skewness':         r.skew(),
        'excess_kurtosis':  r.kurtosis(),
        'min_price':        p.min(),
        'max_price':        p.max(),
        'price_range_pct':  (p.max()-p.min())/p.mean()*100,
        'crashes_gt3pct':   (r < -0.03).sum(),
        'rallies_gt3pct':   (r >  0.03).sum(),
        'max_drawdown_pct': ((p-p.cummax())/p.cummax()).min()*100,
    }

def agent_stats(df):
    N = df[['optimists','pessimists','fundamentalists']].sum(axis=1)
    return {
        'avg_optimists':       df['optimists'].mean(),
        'avg_pessimists':      df['pessimists'].mean(),
        'avg_fundamentalists': df['fundamentalists'].mean(),
        'avg_opt_pct':         (df['optimists']/N).mean()*100,
        'avg_pess_pct':        (df['pessimists']/N).mean()*100,
        'avg_fund_pct':        (df['fundamentalists']/N).mean()*100,
        'sentiment_std':       df['sentiment'].std(),
        'herding_events':      (df['sentiment'].abs() > 0.8).sum(),
    }

def mispricing_stats(df):
    m = df['log_mispricing'].dropna()
    return {
        'mean_abs':        m.abs().mean(),
        'max_abs':         m.abs().max(),
        'std':             m.std(),
        'pct_overvalued':  (m < 0).mean()*100,
        'pct_undervalued': (m > 0).mean()*100,
    }

def acf_abs_returns(df, lags=30):
    ar = df['log_return'].abs().dropna()
    return [ar.autocorr(lag=k) for k in range(1, lags+1)]

def print_summary(df):
    ps = price_stats(df)
    ag = agent_stats(df)
    ms = mispricing_stats(df)
    print('\n' + '='*52)
    print('  SIMULATION SUMMARY')
    print('='*52)
    print('\n-- Price dynamics --')
    print(f"  Mean log return      : {ps['mean_return']:.5f}")
    print(f"  Volatility (std)     : {ps['volatility']:.5f}")
    print(f"  Skewness             : {ps['skewness']:.3f}")
    print(f"  Excess kurtosis      : {ps['excess_kurtosis']:.3f}")
    print(f"  Price range          : {ps['min_price']:.2f} - {ps['max_price']:.2f}  ({ps['price_range_pct']:.1f}% of mean)")
    print(f"  Max drawdown         : {ps['max_drawdown_pct']:.1f}%")
    print(f"  Crashes  >3%         : {ps['crashes_gt3pct']}")
    print(f"  Rallies  >3%         : {ps['rallies_gt3pct']}")
    print('\n-- Agent dynamics --')
    print(f"  Avg optimists        : {ag['avg_optimists']:.0f}  ({ag['avg_opt_pct']:.1f}%)")
    print(f"  Avg pessimists       : {ag['avg_pessimists']:.0f}  ({ag['avg_pess_pct']:.1f}%)")
    print(f"  Avg fundamentalists  : {ag['avg_fundamentalists']:.0f}  ({ag['avg_fund_pct']:.1f}%)")
    print(f"  Sentiment std        : {ag['sentiment_std']:.3f}")
    print(f"  Herding events       : {ag['herding_events']}")
    print('\n-- Mispricing --')
    print(f"  Mean |log mispricing|: {ms['mean_abs']:.4f}")
    print(f"  Max  |log mispricing|: {ms['max_abs']:.4f}")
    print(f"  % time overvalued    : {ms['pct_overvalued']:.1f}%")
    print(f"  % time undervalued   : {ms['pct_undervalued']:.1f}%")
    print('\n-- Volatility clustering (ACF of |returns|) --')
    acf = acf_abs_returns(df, lags=5)
    for i, a in enumerate(acf, 1):
        print(f"  Lag {i}: {a:.4f}")
    print('='*52 + '\n')

def make_plots(df, out_dir='.'):
    os.makedirs(out_dir, exist_ok=True)
    T  = df['time']
    p  = df['price']
    pf = df['fundamental_value']
    r  = df['log_return']
    N  = df[['optimists','pessimists','fundamentalists']].sum(axis=1)

    # Figure 1: Price dynamics
    fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True)
    fig.suptitle('Price dynamics', fontsize=13, fontweight='bold')

    axes[0].plot(T, p,  color='#2171b5', lw=1,   label='Price')
    axes[0].plot(T, pf, color='#238b45', lw=1, ls='--', label='Fundamental')
    axes[0].set_ylabel('Price')
    axes[0].legend(fontsize=9)
    axes[0].grid(alpha=0.25)

    colors = np.where(r >= 0, '#2171b5', '#cb181d')
    axes[1].bar(T, r, color=colors, width=1, alpha=0.8)
    axes[1].axhline(0, color='black', lw=0.5)
    axes[1].set_ylabel('Log return')
    axes[1].grid(alpha=0.25)

    axes[2].plot(T, df['log_mispricing'], color='#6a3d9a', lw=0.8)
    axes[2].axhline(0, color='black', lw=0.5)
    axes[2].fill_between(T, df['log_mispricing'], 0, alpha=0.2, color='#6a3d9a')
    axes[2].set_ylabel('Log mispricing')
    axes[2].set_xlabel('Time step')
    axes[2].grid(alpha=0.25)

    plt.tight_layout()
    fig.savefig(f'{out_dir}/fig1_price_dynamics.png', dpi=150)
    plt.close()
    print(f'  Saved fig1_price_dynamics.png')

    # Figure 2: Agent composition
    fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
    fig.suptitle('Agent composition & sentiment', fontsize=13, fontweight='bold')

    axes[0].stackplot(T,
        df['fundamentalists']/N*100,
        df['pessimists']/N*100,
        df['optimists']/N*100,
        labels=['Fundamentalists','Pessimists','Optimists'],
        colors=['#969696','#cb181d','#2171b5'], alpha=0.8)
    axes[0].set_ylabel('Share (%)')
    axes[0].set_ylim(0,100)
    axes[0].legend(loc='upper right', fontsize=9)
    axes[0].grid(alpha=0.25)

    axes[1].plot(T, df['sentiment'], color='#6a3d9a', lw=0.8)
    axes[1].axhline(0, color='black', lw=0.5)
    axes[1].fill_between(T, df['sentiment'], 0,
        where=df['sentiment']>0, color='#2171b5', alpha=0.3, label='Bullish')
    axes[1].fill_between(T, df['sentiment'], 0,
        where=df['sentiment']<0, color='#cb181d', alpha=0.3, label='Bearish')
    axes[1].set_ylabel('Sentiment')
    axes[1].set_ylim(-1.1,1.1)
    axes[1].set_xlabel('Time step')
    axes[1].legend(fontsize=9)
    axes[1].grid(alpha=0.25)

    plt.tight_layout()
    fig.savefig(f'{out_dir}/fig2_agents.png', dpi=150)
    plt.close()
    print(f'  Saved fig2_agents.png')

    # Figure 3: Return distribution
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    fig.suptitle('Return distribution', fontsize=13, fontweight='bold')

    rd = r.dropna()
    axes[0].hist(rd, bins=60, color='#2171b5', alpha=0.7, density=True, edgecolor='white', lw=0.3)
    xg = np.linspace(rd.min(), rd.max(), 300)
    axes[0].plot(xg, stats.norm.pdf(xg, rd.mean(), rd.std()),
                 color='#cb181d', lw=1.5, label='Normal fit')
    axes[0].set_xlabel('Log return')
    axes[0].set_ylabel('Density')
    axes[0].legend(fontsize=9)
    axes[0].grid(alpha=0.25)
    axes[0].set_title(f'Kurtosis = {rd.kurtosis():.2f}')

    (osm, osr), (slope, intercept, r2) = stats.probplot(rd, dist='norm')
    axes[1].scatter(osm, osr, color='#2171b5', s=4, alpha=0.5)
    axes[1].plot(osm, slope*np.array(osm)+intercept, color='#cb181d', lw=1.5)
    axes[1].set_xlabel('Theoretical quantiles')
    axes[1].set_ylabel('Sample quantiles')
    axes[1].set_title('Q-Q plot vs normal')
    axes[1].grid(alpha=0.25)

    plt.tight_layout()
    fig.savefig(f'{out_dir}/fig3_return_distribution.png', dpi=150)
    plt.close()
    print(f'  Saved fig3_return_distribution.png')

    # Figure 4: Volatility clustering
    fig, ax = plt.subplots(figsize=(10, 4))
    fig.suptitle('Volatility clustering — ACF of |log returns|', fontsize=13, fontweight='bold')

    acf = acf_abs_returns(df, lags=30)
    lags = np.arange(1, len(acf)+1)
    ax.bar(lags, acf, color='#2171b5', alpha=0.7, width=0.7)
    ci = 1.96 / np.sqrt(len(rd))
    ax.axhline( ci, color='#cb181d', ls='--', lw=1, label='95% CI')
    ax.axhline(-ci, color='#cb181d', ls='--', lw=1)
    ax.axhline(0, color='black', lw=0.5)
    ax.set_xlabel('Lag')
    ax.set_ylabel('Autocorrelation')
    ax.legend(fontsize=9)
    ax.grid(alpha=0.25)

    plt.tight_layout()
    fig.savefig(f'{out_dir}/fig4_volatility_clustering.png', dpi=150)
    plt.close()
    print(f'  Saved fig4_volatility_clustering.png')

if __name__ == '__main__':
    path = sys.argv[1] if len(sys.argv) > 1 else 'results/simulation.csv'
    out  = sys.argv[2] if len(sys.argv) > 2 else 'results/figures'

    print(f'Loading: {path}')
    df = load_simulation(path)
    print(f'Loaded {len(df)} rows x {len(df.columns)} columns')
    print(f'Columns: {list(df.columns)}')

    print_summary(df)
    print(f'Saving figures to: {out}/')
    make_plots(df, out_dir=out)
    print('Done.')