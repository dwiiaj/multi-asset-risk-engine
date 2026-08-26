# ============================================================
# MULTI-ASSET RISK & PORTFOLIO ANALYTICS ENGINE
# ============================================================

from pathlib import Path
import warnings

import duckdb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
import yfinance as yf

from scipy.optimize import minimize
from scipy.stats import norm


warnings.filterwarnings("ignore")


# ============================================================
# 1. PROJECT CONFIGURATION
# ============================================================

print("=" * 70)
print("MULTI-ASSET RISK & PORTFOLIO ANALYTICS ENGINE")
print("=" * 70)


START_DATE = "2018-01-01"

# yfinance treats the end date as exclusive.
# 2026-01-01 therefore gives us data through 2025-12-31.
END_DATE = "2026-01-01"

TRADING_DAYS = 252

RISK_FREE_RATE = 0.02

MAX_ASSET_WEIGHT = 0.35

MONTE_CARLO_SIMULATIONS = 100_000

RANDOM_SEED = 42


ASSETS = {

    "US Equities":
        "SPY",

    "US Technology":
        "QQQ",

    "European Equities":
        "VGK",

    "Emerging Markets":
        "EEM",

    "US Treasuries":
        "TLT",

    "Gold":
        "GLD",

    "Oil":
        "USO",

    "Bitcoin":
        "BTC-USD"

}


ROOT = Path(".")

RAW_DIR = ROOT / "data" / "raw"

PROCESSED_DIR = ROOT / "data" / "processed"

FIGURE_DIR = ROOT / "reports" / "figures"

TABLE_DIR = ROOT / "reports" / "tables"


for directory in [

    RAW_DIR,
    PROCESSED_DIR,
    FIGURE_DIR,
    TABLE_DIR

]:

    directory.mkdir(
        parents=True,
        exist_ok=True
    )


print("\nHistorical sample:")
print(
    START_DATE,
    "to",
    "2025-12-31"
)

print("\nAsset universe:")

for asset, ticker in ASSETS.items():

    print(
        f"- {asset}: {ticker}"
    )


# ============================================================
# 2. DOWNLOAD MARKET DATA
# ============================================================

print("\n" + "=" * 70)
print("STEP 1 — DOWNLOADING MARKET DATA")
print("=" * 70)


def download_adjusted_price(
    asset_name,
    ticker
):

    print(
        f"Downloading {asset_name} ({ticker})..."
    )

    data = yf.download(
        ticker,
        start=START_DATE,
        end=END_DATE,
        auto_adjust=True,
        progress=False
    )

    if data.empty:

        raise ValueError(
            f"No data downloaded for {ticker}"
        )


    # yfinance may return MultiIndex columns.
    if isinstance(
        data.columns,
        pd.MultiIndex
    ):

        close = data["Close"]

        if isinstance(
            close,
            pd.DataFrame
        ):

            close = close.iloc[:, 0]

    else:

        close = data["Close"]


    close = pd.Series(
        close,
        index=data.index,
        name=asset_name
    )


    close.index = pd.to_datetime(
        close.index
    )


    if close.index.tz is not None:

        close.index = (
            close.index
            .tz_localize(None)
        )


    raw_output = pd.DataFrame({

        "date":
            close.index,

        "adjusted_close":
            close.values

    })


    raw_output.to_csv(
        RAW_DIR
        / f"{ticker.replace('-', '_')}.csv",
        index=False
    )


    return close


price_series = []


for asset_name, ticker in ASSETS.items():

    series = download_adjusted_price(
        asset_name,
        ticker
    )

    price_series.append(
        series
    )


prices = pd.concat(
    price_series,
    axis=1
)


print("\nCombined raw observations:")
print(len(prices))


print("\nMissing values before common-date filter:")
print(
    prices
    .isna()
    .sum()
)


# Use dates available across the complete asset universe.
prices = (
    prices
    .dropna()
    .sort_index()
)


print("\nCommon trading observations:")
print(len(prices))


print("\nDate range:")

print(
    prices.index.min(),
    "→",
    prices.index.max()
)


print("\nMissing values after filtering:")
print(
    int(
        prices
        .isna()
        .sum()
        .sum()
    )
)


prices.to_csv(
    PROCESSED_DIR
    / "multi_asset_prices.csv"
)


# ============================================================
# 3. DAILY RETURNS
# ============================================================

print("\n" + "=" * 70)
print("STEP 2 — CALCULATING RETURNS")
print("=" * 70)


returns = (
    prices
    .pct_change(
        fill_method=None
    )
    .dropna()
)


returns.to_csv(
    PROCESSED_DIR
    / "multi_asset_returns.csv"
)


print("\nReturn observations:")
print(len(returns))


print("\nReturn dataset preview:")

print(
    returns
    .head()
    .round(5)
)


# ============================================================
# 4. DUCKDB + SQL
# ============================================================

print("\n" + "=" * 70)
print("STEP 3 — DUCKDB + SQL ANALYTICS")
print("=" * 70)


prices_long = (
    prices
    .reset_index()
    .rename(
        columns={
            "Date":
                "date",
            "index":
                "date"
        }
    )
    .melt(
        id_vars="date",
        var_name="asset",
        value_name="adjusted_close"
    )
)


returns_long = (
    returns
    .reset_index()
    .rename(
        columns={
            "Date":
                "date",
            "index":
                "date"
        }
    )
    .melt(
        id_vars="date",
        var_name="asset",
        value_name="daily_return"
    )
)


con = duckdb.connect(
    "risk_engine.duckdb"
)


con.execute(
    """
    CREATE OR REPLACE TABLE asset_prices
    AS
    SELECT *
    FROM prices_long
    """
)


con.execute(
    """
    CREATE OR REPLACE TABLE asset_returns
    AS
    SELECT *
    FROM returns_long
    """
)


sql_summary = con.execute(

    """
    SELECT
        asset,

        COUNT(*) AS observations,

        AVG(daily_return) * 252
            AS annualised_mean_return,

        STDDEV_SAMP(daily_return)
            * SQRT(252)
            AS annualised_volatility,

        MIN(daily_return)
            AS worst_daily_return,

        MAX(daily_return)
            AS best_daily_return

    FROM asset_returns

    GROUP BY asset

    ORDER BY
        annualised_volatility DESC
    """

).df()


sql_summary.to_csv(
    TABLE_DIR
    / "sql_asset_summary.csv",
    index=False
)


print("\nSQL asset summary:")

print(
    sql_summary
    .round(4)
    .to_string(index=False)
)


# ============================================================
# 5. PERFORMANCE & RISK FUNCTIONS
# ============================================================

print("\n" + "=" * 70)
print("STEP 4 — ASSET PERFORMANCE & RISK")
print("=" * 70)


def annualised_return(
    return_series
):

    n = len(
        return_series
    )

    if n == 0:

        return np.nan


    cumulative_growth = (
        1
        + return_series
    ).prod()


    return (
        cumulative_growth
        ** (
            TRADING_DAYS / n
        )
        - 1
    )


def annualised_volatility(
    return_series
):

    return (
        return_series.std()
        * np.sqrt(
            TRADING_DAYS
        )
    )


def max_drawdown(
    return_series
):

    wealth = (
        1
        + return_series
    ).cumprod()


    running_peak = (
        wealth
        .cummax()
    )


    drawdown = (
        wealth
        / running_peak
        - 1
    )


    return drawdown.min()


def historical_var(
    return_series,
    confidence=0.95
):

    alpha = (
        1
        - confidence
    )


    quantile = (
        return_series
        .quantile(alpha)
    )


    return -quantile


def historical_es(
    return_series,
    confidence=0.95
):

    alpha = (
        1
        - confidence
    )


    threshold = (
        return_series
        .quantile(alpha)
    )


    tail = (
        return_series[
            return_series
            <= threshold
        ]
    )


    return -tail.mean()


def gaussian_var(
    return_series,
    confidence=0.95
):

    alpha = (
        1
        - confidence
    )


    mean = (
        return_series
        .mean()
    )


    sigma = (
        return_series
        .std()
    )


    quantile = (
        mean
        + sigma
        * norm.ppf(alpha)
    )


    return -quantile


asset_metrics = []


asset_metrics = []


for asset in returns.columns:

    series = (
        returns[asset]
        .dropna()
    )

    ann_return = (
        annualised_return(
            series
        )
    )

    ann_vol = (
        annualised_volatility(
            series
        )
    )

    annual_mean_return = (
        series.mean()
        * TRADING_DAYS
    )

    sharpe = (
        (
            annual_mean_return
            - RISK_FREE_RATE
        )
        / ann_vol
        if ann_vol > 0
        else np.nan
    )

    asset_metrics.append({

        "asset":
            asset,

        "annualised_return":
            ann_return,

        "annualised_volatility":
            ann_vol,

        "sharpe_ratio":
            sharpe,

        "max_drawdown":
            max_drawdown(
                series
            ),

        "historical_var_95":
            historical_var(
                series,
                0.95
            ),

        "historical_es_95":
            historical_es(
                series,
                0.95
            ),

        "historical_var_99":
            historical_var(
                series,
                0.99
            )

    })

asset_risk_summary = pd.DataFrame(
    asset_metrics
)

asset_risk_summary = (
    asset_risk_summary
    .sort_values(
        "annualised_volatility",
        ascending=False
    )
)


asset_risk_summary.to_csv(
    TABLE_DIR
    / "asset_risk_summary.csv",
    index=False
)


print("\nAsset risk summary:")

print(
    asset_risk_summary
    .round(4)
    .to_string(index=False)
)


# ============================================================
# 6. CORRELATION ANALYSIS
# ============================================================

print("\n" + "=" * 70)
print("STEP 5 — CORRELATION ANALYSIS")
print("=" * 70)


correlation_matrix = (
    returns
    .corr()
)


correlation_matrix.to_csv(
    TABLE_DIR
    / "correlation_matrix.csv"
)


print("\nReturn correlation matrix:")

print(
    correlation_matrix
    .round(3)
)


fig, ax = plt.subplots(
    figsize=(10, 8)
)


image = ax.imshow(
    correlation_matrix.values,
    aspect="auto"
)


ax.set_xticks(
    range(
        len(
            correlation_matrix.columns
        )
    )
)


ax.set_xticklabels(
    correlation_matrix.columns,
    rotation=45,
    ha="right"
)


ax.set_yticks(
    range(
        len(
            correlation_matrix.index
        )
    )
)


ax.set_yticklabels(
    correlation_matrix.index
)


for i in range(
    len(
        correlation_matrix.index
    )
):

    for j in range(
        len(
            correlation_matrix.columns
        )
    ):

        ax.text(
            j,
            i,
            f"{correlation_matrix.iloc[i, j]:.2f}",
            ha="center",
            va="center"
        )


ax.set_title(
    "Multi-Asset Return Correlation"
)


fig.colorbar(
    image,
    ax=ax
)


fig.tight_layout()


fig.savefig(
    FIGURE_DIR
    / "asset_correlation_matrix.png",
    dpi=200
)


plt.close(
    fig
)


# ============================================================
# 7. NORMALISED MARKET PERFORMANCE
# ============================================================

normalised_prices = (
    prices
    / prices.iloc[0]
    * 100
)


fig, ax = plt.subplots(
    figsize=(13, 7)
)


for asset in normalised_prices.columns:

    ax.plot(
        normalised_prices.index,
        normalised_prices[asset],
        label=asset
    )


ax.set_title(
    "Normalised Multi-Asset Performance"
)


ax.set_ylabel(
    "Growth of 100"
)


ax.legend(
    fontsize=8,
    ncol=2
)


ax.grid(
    alpha=0.25
)


fig.tight_layout()


fig.savefig(
    FIGURE_DIR
    / "normalised_asset_performance.png",
    dpi=200
)


plt.close(
    fig
)


# ============================================================
# 8. EQUAL-WEIGHT PORTFOLIO
# ============================================================

print("\n" + "=" * 70)
print("STEP 6 — EQUAL-WEIGHT PORTFOLIO")
print("=" * 70)


number_of_assets = (
    len(
        returns.columns
    )
)


equal_weights = np.repeat(
    1 / number_of_assets,
    number_of_assets
)


equal_weight_returns = pd.Series(

    returns.values
    @ equal_weights,

    index=returns.index,

    name="Equal Weight Portfolio"

)


equal_ann_return = (
    annualised_return(
        equal_weight_returns
    )
)


equal_ann_vol = (
    annualised_volatility(
        equal_weight_returns
    )
)

equal_annual_mean_return = (
    equal_weight_returns.mean()
    * TRADING_DAYS
)

equal_sharpe = (
    equal_annual_mean_return
    - RISK_FREE_RATE
) / equal_ann_vol


equal_max_dd = (
    max_drawdown(
        equal_weight_returns
    )
)


print(
    "\nEqual-weight annualised return:",
    round(
        equal_ann_return * 100,
        2
    ),
    "%"
)


print(
    "Equal-weight annualised volatility:",
    round(
        equal_ann_vol * 100,
        2
    ),
    "%"
)


print(
    "Equal-weight Sharpe ratio:",
    round(
        equal_sharpe,
        3
    )
)


print(
    "Equal-weight maximum drawdown:",
    round(
        equal_max_dd * 100,
        2
    ),
    "%"
)


# ============================================================
# 9. PORTFOLIO VAR / EXPECTED SHORTFALL
# ============================================================

print("\n" + "=" * 70)
print("STEP 7 — VALUE-AT-RISK & EXPECTED SHORTFALL")
print("=" * 70)


risk_results = pd.DataFrame({

    "metric": [

        "Historical VaR 95%",
        "Historical ES 95%",
        "Historical VaR 99%",
        "Parametric Gaussian VaR 95%"

    ],

    "daily_loss_fraction": [

        historical_var(
            equal_weight_returns,
            0.95
        ),

        historical_es(
            equal_weight_returns,
            0.95
        ),

        historical_var(
            equal_weight_returns,
            0.99
        ),

        gaussian_var(
            equal_weight_returns,
            0.95
        )

    ]

})


risk_results[
    "daily_loss_percent"
] = (
    risk_results[
        "daily_loss_fraction"
    ]
    * 100
)


# ============================================================
# 10. MONTE CARLO VAR
# ============================================================

np.random.seed(
    RANDOM_SEED
)


simulated_asset_returns = (
    np.random.multivariate_normal(

        mean=returns.mean().values,

        cov=returns.cov().values,

        size=MONTE_CARLO_SIMULATIONS

    )
)


simulated_portfolio_returns = (
    simulated_asset_returns
    @ equal_weights
)


monte_carlo_var_95 = -np.quantile(
    simulated_portfolio_returns,
    0.05
)


monte_carlo_var_99 = -np.quantile(
    simulated_portfolio_returns,
    0.01
)


monte_carlo_rows = pd.DataFrame({

    "metric": [

        "Monte Carlo VaR 95%",
        "Monte Carlo VaR 99%"

    ],

    "daily_loss_fraction": [

        monte_carlo_var_95,
        monte_carlo_var_99

    ]

})


monte_carlo_rows[
    "daily_loss_percent"
] = (

    monte_carlo_rows[
        "daily_loss_fraction"
    ]
    * 100
)


risk_results = pd.concat(

    [
        risk_results,
        monte_carlo_rows
    ],

    ignore_index=True

)


risk_results.to_csv(
    TABLE_DIR
    / "portfolio_var_es_summary.csv",
    index=False
)


print("\nEqual-weight portfolio risk:")

print(
    risk_results[
        [
            "metric",
            "daily_loss_percent"
        ]
    ]
    .round(3)
    .to_string(index=False)
)


# ============================================================
# 11. HISTORICAL STRESS TESTS
# ============================================================

print("\n" + "=" * 70)
print("STEP 8 — HISTORICAL STRESS TESTING")
print("=" * 70)


STRESS_SCENARIOS = {

    "COVID Crash":
        (
            "2020-02-19",
            "2020-03-23"
        ),

    "2022 Rate & Inflation Shock":
        (
            "2022-01-03",
            "2022-10-14"
        )

}


stress_rows = []


for scenario, (
    start,
    end
) in STRESS_SCENARIOS.items():

    price_window = prices.loc[
        start:end
    ]


    if len(
        price_window
    ) < 2:

        continue


    scenario_returns = (

        price_window.iloc[-1]
        / price_window.iloc[0]
        - 1

    )


    equal_portfolio_return = (

        scenario_returns.values
        @ equal_weights

    )


    for asset in prices.columns:

        stress_rows.append({

            "scenario":
                scenario,

            "asset":
                asset,

            "scenario_return":
                scenario_returns[
                    asset
                ]

        })


    stress_rows.append({

        "scenario":
            scenario,

        "asset":
            "Equal Weight Portfolio",

        "scenario_return":
            equal_portfolio_return

    })


stress_results = pd.DataFrame(
    stress_rows
)


stress_results[
    "scenario_return_percent"
] = (
    stress_results[
        "scenario_return"
    ]
    * 100
)


stress_results.to_csv(
    TABLE_DIR
    / "historical_stress_tests.csv",
    index=False
)


print("\nHistorical stress-test results:")

print(
    stress_results[
        [
            "scenario",
            "asset",
            "scenario_return_percent"
        ]
    ]
    .round(2)
    .to_string(index=False)
)


# ============================================================
# 12. CAPM-STYLE REGRESSION
# ============================================================

print("\n" + "=" * 70)
print("STEP 9 — ASSET-PRICING REGRESSION")
print("=" * 70)


market_return = (
    returns[
        "US Equities"
    ]
)


regression_rows = []


for asset in returns.columns:

    y = (
        returns[
            asset
        ]
    )

    X = sm.add_constant(
        market_return
    )

    regression = sm.OLS(
        y,
        X
    ).fit()

    alpha_daily = (
        regression.params[
            "const"
        ]
    )

    beta = (
        regression.params[
            "US Equities"
        ]
    )

    regression_rows.append({

        "asset":
            asset,

        "annualised_alpha":
            alpha_daily
            * TRADING_DAYS,

        "beta_to_spy":
            beta,

        "r_squared":
            regression.rsquared

    })


regression_results = pd.DataFrame(
    regression_rows
)


regression_results.to_csv(
    TABLE_DIR
    / "asset_pricing_regression.csv",
    index=False
)


print("\nCAPM-style regression results:")

print(
    regression_results
    .round(4)
    .to_string(index=False)
)

# ============================================================
# 13. PORTFOLIO OPTIMISATION
# ============================================================

print("\n" + "=" * 70)
print("STEP 10 — PORTFOLIO OPTIMISATION")
print("=" * 70)


expected_returns = (
    returns.mean()
    * TRADING_DAYS
)


covariance_matrix = (
    returns.cov()
    * TRADING_DAYS
)


def portfolio_return(
    weights
):

    return float(
        weights
        @ expected_returns.values
    )


def portfolio_volatility(
    weights
):

    return float(
        np.sqrt(
            weights
            @ covariance_matrix.values
            @ weights
        )
    )


def negative_sharpe(
    weights
):

    volatility = (
        portfolio_volatility(
            weights
        )
    )

    if volatility == 0:
        return 1e6

    sharpe = (
        portfolio_return(
            weights
        )
        - RISK_FREE_RATE
    ) / volatility

    return -sharpe


constraints = (
    {
        "type": "eq",
        "fun": lambda weights: np.sum(weights) - 1
    },
)


bounds = tuple(
    (
        0.0,
        MAX_ASSET_WEIGHT
    )
    for _ in range(
        number_of_assets
    )
)

initial_weights = (
    equal_weights.copy()
)

minimum_variance_result = minimize(

    portfolio_volatility,

    initial_weights,

    method="SLSQP",

    bounds=bounds,

    constraints=constraints,

    options={
        "maxiter":
            2000
    }

)


maximum_sharpe_result = minimize(

    negative_sharpe,

    initial_weights,

    method="SLSQP",

    bounds=bounds,

    constraints=constraints,

    options={
        "maxiter":
            2000
    }

)


if not minimum_variance_result.success:

    raise RuntimeError(
        "Minimum-variance optimisation failed: "
        + minimum_variance_result.message
    )


if not maximum_sharpe_result.success:

    raise RuntimeError(
        "Maximum-Sharpe optimisation failed: "
        + maximum_sharpe_result.message
    )


minimum_variance_weights = (
    minimum_variance_result.x
)


maximum_sharpe_weights = (
    maximum_sharpe_result.x
)


portfolio_weights = pd.DataFrame({

    "asset":
        returns.columns,

    "equal_weight":
        equal_weights,

    "minimum_variance":
        minimum_variance_weights,

    "maximum_sharpe":
        maximum_sharpe_weights

})


portfolio_weights.to_csv(
    TABLE_DIR
    / "optimised_portfolio_weights.csv",
    index=False
)


print("\nPortfolio weights:")

display_weights = (
    portfolio_weights.copy()
)


for column in [

    "equal_weight",
    "minimum_variance",
    "maximum_sharpe"

]:

    display_weights[
        column
    ] = (

        display_weights[
            column
        ]
        * 100

    )


print(
    display_weights
    .round(2)
    .to_string(index=False)
)


# ============================================================
# 14. PORTFOLIO COMPARISON
# ============================================================

def portfolio_summary(
    name,
    weights
):

    port_return = (
        portfolio_return(
            weights
        )
    )


    port_vol = (
        portfolio_volatility(
            weights
        )
    )


    sharpe = (

        (
            port_return
            - RISK_FREE_RATE
        )

        / port_vol

    )


    realised_returns = pd.Series(

        returns.values
        @ weights,

        index=returns.index

    )


    return {

        "portfolio":
            name,

        "expected_annual_return":
            port_return,

        "annualised_volatility":
            port_vol,

        "sharpe_ratio":
            sharpe,

        "historical_max_drawdown":
            max_drawdown(
                realised_returns
            ),

        "historical_var_95":
            historical_var(
                realised_returns,
                0.95
            ),

        "historical_es_95":
            historical_es(
                realised_returns,
                0.95
            )

    }


portfolio_comparison = pd.DataFrame([

    portfolio_summary(
        "Equal Weight",
        equal_weights
    ),

    portfolio_summary(
        "Minimum Variance",
        minimum_variance_weights
    ),

    portfolio_summary(
        "Maximum Sharpe",
        maximum_sharpe_weights
    )

])


portfolio_comparison.to_csv(
    TABLE_DIR
    / "portfolio_strategy_comparison.csv",
    index=False
)


print("\nPortfolio strategy comparison:")

print(
    portfolio_comparison
    .round(4)
    .to_string(index=False)
)


# ============================================================
# 15. RISK CONTRIBUTION
# ============================================================

print("\n" + "=" * 70)
print("STEP 11 — RISK CONTRIBUTION")
print("=" * 70)


def risk_contribution(
    weights
):

    portfolio_variance = (

        weights
        @ covariance_matrix.values
        @ weights

    )


    portfolio_sigma = (
        np.sqrt(
            portfolio_variance
        )
    )


    marginal_contribution = (

        covariance_matrix.values
        @ weights

    ) / portfolio_sigma


    component_contribution = (

        weights
        * marginal_contribution

    )


    percent_contribution = (

        component_contribution
        / portfolio_sigma

    )


    return percent_contribution


risk_contributions = pd.DataFrame({

    "asset":
        returns.columns,

    "equal_weight_risk_contribution":
        risk_contribution(
            equal_weights
        ),

    "minimum_variance_risk_contribution":
        risk_contribution(
            minimum_variance_weights
        ),

    "maximum_sharpe_risk_contribution":
        risk_contribution(
            maximum_sharpe_weights
        )

})


risk_contributions.to_csv(
    TABLE_DIR
    / "portfolio_risk_contributions.csv",
    index=False
)


print("\nRisk contributions:")

display_risk = (
    risk_contributions.copy()
)


for column in [

    "equal_weight_risk_contribution",
    "minimum_variance_risk_contribution",
    "maximum_sharpe_risk_contribution"

]:

    display_risk[
        column
    ] = (

        display_risk[
            column
        ]
        * 100

    )


print(
    display_risk
    .round(2)
    .to_string(index=False)
)


# ============================================================
# 16. EFFICIENT FRONTIER SIMULATION
# ============================================================

print("\n" + "=" * 70)
print("STEP 12 — EFFICIENT FRONTIER SIMULATION")
print("=" * 70)


np.random.seed(
    RANDOM_SEED
)

number_of_portfolios = 5000

frontier_rows = []

generated_portfolios = 0

while generated_portfolios < number_of_portfolios:

    weights = (
        np.random.dirichlet(
            np.ones(
                number_of_assets
            )
        )
    )

    port_return = (
        portfolio_return(
            weights
        )
    )

    port_vol = (
        portfolio_volatility(
            weights
        )
    )

    sharpe = (

        (
            port_return
            - RISK_FREE_RATE
        )
        / port_vol

    )

    frontier_rows.append({

        "annual_return":
            port_return,

        "annual_volatility":
            port_vol,

        "sharpe_ratio":
            sharpe

    })

    generated_portfolios += 1

efficient_frontier = pd.DataFrame(
    frontier_rows
)

efficient_frontier.to_csv(
    TABLE_DIR
    / "efficient_frontier_simulation.csv",
    index=False
)

fig, ax = plt.subplots(
    figsize=(10, 7)
)

scatter = ax.scatter(

    efficient_frontier[
        "annual_volatility"
    ]
    * 100,

    efficient_frontier[
        "annual_return"
    ]
    * 100,

    c=efficient_frontier[
        "sharpe_ratio"
    ],

    s=10,

    alpha=0.5

)

ax.scatter(

    portfolio_volatility(
        minimum_variance_weights
    )
    * 100,

    portfolio_return(
        minimum_variance_weights
    )
    * 100,

    marker="*",

    s=250,

    label="Minimum Variance"

)


ax.scatter(

    portfolio_volatility(
        maximum_sharpe_weights
    )
    * 100,

    portfolio_return(
        maximum_sharpe_weights
    )
    * 100,

    marker="*",

    s=250,

    label="Maximum Sharpe"

)


ax.set_xlabel(
    "Annualised Volatility (%)"
)


ax.set_ylabel(
    "Expected Annual Return (%)"
)


ax.set_title(
    "Simulated Portfolio Opportunity Set"
)


ax.legend()


fig.colorbar(
    scatter,
    ax=ax,
    label="Sharpe Ratio"
)


fig.tight_layout()


fig.savefig(
    FIGURE_DIR
    / "efficient_frontier.png",
    dpi=200
)


plt.close(
    fig
)


# ============================================================
# 17. ROLLING PORTFOLIO VOLATILITY
# ============================================================

rolling_volatility = (

    equal_weight_returns
    .rolling(
        63
    )
    .std()
    * np.sqrt(
        TRADING_DAYS
    )
)


fig, ax = plt.subplots(
    figsize=(12, 6)
)


ax.plot(
    rolling_volatility.index,
    rolling_volatility
    * 100
)


ax.set_title(
    "Equal-Weight Portfolio Rolling 63-Day Volatility"
)


ax.set_ylabel(
    "Annualised Volatility (%)"
)


ax.grid(
    alpha=0.25
)


fig.tight_layout()


fig.savefig(
    FIGURE_DIR
    / "rolling_portfolio_volatility.png",
    dpi=200
)


plt.close(
    fig
)


# ============================================================
# 18. PORTFOLIO DRAWDOWN
# ============================================================

portfolio_wealth = (

    1
    + equal_weight_returns

).cumprod()


portfolio_drawdown = (

    portfolio_wealth
    / portfolio_wealth.cummax()
    - 1

)


fig, ax = plt.subplots(
    figsize=(12, 6)
)


ax.plot(
    portfolio_drawdown.index,
    portfolio_drawdown
    * 100
)


ax.set_title(
    "Equal-Weight Portfolio Drawdown"
)


ax.set_ylabel(
    "Drawdown (%)"
)


ax.grid(
    alpha=0.25
)


fig.tight_layout()


fig.savefig(
    FIGURE_DIR
    / "portfolio_drawdown.png",
    dpi=200
)


plt.close(
    fig
)


# ============================================================
# 19. FINAL PROJECT SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("FINAL PROJECT SUMMARY")
print("=" * 70)


best_sharpe_row = (

    portfolio_comparison
    .sort_values(
        "sharpe_ratio",
        ascending=False
    )
    .iloc[0]

)


lowest_vol_row = (

    portfolio_comparison
    .sort_values(
        "annualised_volatility"
    )
    .iloc[0]

)


highest_risk_asset = (

    asset_risk_summary
    .sort_values(
        "annualised_volatility",
        ascending=False
    )
    .iloc[0]

)


print("\nHistorical observations:")
print(len(returns))


print("\nAssets analysed:")
print(number_of_assets)


print("\nHighest-volatility asset:")
print(
    highest_risk_asset[
        "asset"
    ]
)


print(
    "Annualised volatility:",
    round(
        highest_risk_asset[
            "annualised_volatility"
        ]
        * 100,
        2
    ),
    "%"
)


print("\nPortfolio with highest estimated Sharpe ratio:")
print(
    best_sharpe_row[
        "portfolio"
    ]
)


print(
    "Sharpe ratio:",
    round(
        best_sharpe_row[
            "sharpe_ratio"
        ],
        3
    )
)


print("\nLowest-volatility portfolio:")
print(
    lowest_vol_row[
        "portfolio"
    ]
)


print(
    "Annualised volatility:",
    round(
        lowest_vol_row[
            "annualised_volatility"
        ]
        * 100,
        2
    ),
    "%"
)


print("\nKey files created:")

print(
    "- reports/tables/asset_risk_summary.csv"
)

print(
    "- reports/tables/portfolio_var_es_summary.csv"
)

print(
    "- reports/tables/historical_stress_tests.csv"
)

print(
    "- reports/tables/asset_pricing_regression.csv"
)

print(
    "- reports/tables/optimised_portfolio_weights.csv"
)

print(
    "- reports/tables/portfolio_strategy_comparison.csv"
)

print(
    "- reports/tables/portfolio_risk_contributions.csv"
)

print(
    "- reports/figures/asset_correlation_matrix.png"
)

print(
    "- reports/figures/efficient_frontier.png"
)

print(
    "- reports/figures/rolling_portfolio_volatility.png"
)

print(
    "- reports/figures/portfolio_drawdown.png"
)


con.close()


print("\n" + "=" * 70)
print("RISK ANALYTICS PIPELINE COMPLETED SUCCESSFULLY")
print("=" * 70)