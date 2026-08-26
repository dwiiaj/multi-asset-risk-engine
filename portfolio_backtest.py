# ============================================================
# PORTFOLIO OUT-OF-SAMPLE BACKTEST
# ============================================================

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from scipy.optimize import minimize


# ============================================================
# 1. CONFIGURATION
# ============================================================

print("=" * 70)
print("PORTFOLIO OUT-OF-SAMPLE BACKTEST")
print("=" * 70)


TRADING_DAYS = 252
RISK_FREE_RATE = 0.02

TRAIN_END = "2022-12-31"
TEST_START = "2023-01-01"

ROOT = Path(".")

PROCESSED_DIR = ROOT / "data" / "processed"
TABLE_DIR = ROOT / "reports" / "tables"
FIGURE_DIR = ROOT / "reports" / "figures"


TABLE_DIR.mkdir(
    parents=True,
    exist_ok=True
)

FIGURE_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# 2. LOAD PROCESSED RETURNS
# ============================================================

returns_file = (
    PROCESSED_DIR
    / "multi_asset_returns.csv"
)


returns = pd.read_csv(
    returns_file,
    index_col=0,
    parse_dates=True
)


returns = (
    returns
    .sort_index()
)


print("\nLoaded observations:")
print(len(returns))


print("\nDate range:")
print(
    returns.index.min(),
    "→",
    returns.index.max()
)


# ============================================================
# 3. TRAIN / TEST SPLIT
# ============================================================

train_returns = (
    returns.loc[
        :TRAIN_END
    ]
    .copy()
)


test_returns = (
    returns.loc[
        TEST_START:
    ]
    .copy()
)


print("\nTraining period:")
print(
    train_returns.index.min(),
    "→",
    train_returns.index.max()
)


print("\nTraining observations:")
print(len(train_returns))


print("\nUnseen holdout period:")
print(
    test_returns.index.min(),
    "→",
    test_returns.index.max()
)


print("\nHoldout observations:")
print(len(test_returns))


# ============================================================
# 4. TRAINING-SAMPLE ESTIMATES
# ============================================================

expected_returns_train = (
    train_returns.mean()
    * TRADING_DAYS
)


covariance_train = (
    train_returns.cov()
    * TRADING_DAYS
)


number_of_assets = (
    len(
        train_returns.columns
    )
)


equal_weights = np.repeat(
    1 / number_of_assets,
    number_of_assets
)


# ============================================================
# 5. PORTFOLIO FUNCTIONS
# ============================================================

def expected_portfolio_return(
    weights
):

    return float(
        weights
        @ expected_returns_train.values
    )


def expected_portfolio_volatility(
    weights
):

    return float(
        np.sqrt(
            weights
            @ covariance_train.values
            @ weights
        )
    )


def negative_sharpe(
    weights
):

    volatility = (
        expected_portfolio_volatility(
            weights
        )
    )

    if volatility == 0:

        return 1e6


    sharpe = (
        expected_portfolio_return(
            weights
        )
        - RISK_FREE_RATE
    ) / volatility


    return -sharpe


constraints = (

    {
        "type":
            "eq",

        "fun":
            lambda weights:
            np.sum(weights) - 1
    },

)

TRADING_DAYS = 252
RISK_FREE_RATE = 0.02
MAX_ASSET_WEIGHT = 0.35

bounds = tuple(
    (0.0, MAX_ASSET_WEIGHT)
    for _ in range
    (number_of_assets)
)


# ============================================================
# 6. ESTIMATE WEIGHTS USING TRAINING DATA ONLY
# ============================================================

print("\n" + "=" * 70)
print("TRAINING-SAMPLE PORTFOLIO OPTIMISATION")
print("=" * 70)


minimum_variance_result = minimize(

    expected_portfolio_volatility,

    equal_weights.copy(),

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

    equal_weights.copy(),

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


weights_table = pd.DataFrame({

    "asset":
        train_returns.columns,

    "equal_weight":
        equal_weights,

    "minimum_variance":
        minimum_variance_weights,

    "maximum_sharpe":
        maximum_sharpe_weights

})


weights_table.to_csv(
    TABLE_DIR
    / "holdout_portfolio_weights.csv",
    index=False
)


display_weights = (
    weights_table.copy()
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


print("\nWeights estimated using 2018-2022 only:")

print(
    display_weights
    .round(2)
    .to_string(index=False)
)


# ============================================================
# 7. REALISED HOLDOUT PORTFOLIO RETURNS
# ============================================================

equal_test_returns = pd.Series(

    test_returns.values
    @ equal_weights,

    index=test_returns.index,

    name="Equal Weight"

)


min_var_test_returns = pd.Series(

    test_returns.values
    @ minimum_variance_weights,

    index=test_returns.index,

    name="Minimum Variance"

)


max_sharpe_test_returns = pd.Series(

    test_returns.values
    @ maximum_sharpe_weights,

    index=test_returns.index,

    name="Maximum Sharpe"

)


# ============================================================
# 8. PERFORMANCE FUNCTIONS
# ============================================================

def annualised_return(
    return_series
):

    n = len(
        return_series
    )

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


def sharpe_ratio(
    return_series
):

    annual_mean_return = (
        return_series.mean()
        * TRADING_DAYS
    )

    annual_volatility = (
        return_series.std()
        * np.sqrt(
            TRADING_DAYS
        )
    )

    if annual_volatility == 0:

        return np.nan

    return (
        annual_mean_return
        - RISK_FREE_RATE
    ) / annual_volatility


def maximum_drawdown(
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


    return -return_series.quantile(
        alpha
    )


def historical_es(
    return_series,
    confidence=0.95
):

    alpha = (
        1
        - confidence
    )


    threshold = (
        return_series.quantile(
            alpha
        )
    )


    tail = (
        return_series[
            return_series
            <= threshold
        ]
    )


    return -tail.mean()


# ============================================================
# 9. HOLDOUT PERFORMANCE SUMMARY
# ============================================================

portfolio_series = {

    "Equal Weight":
        equal_test_returns,

    "Minimum Variance":
        min_var_test_returns,

    "Maximum Sharpe":
        max_sharpe_test_returns

}


summary_rows = []


for name, series in portfolio_series.items():

    summary_rows.append({

        "portfolio":
            name,

        "annualised_return":
            annualised_return(
                series
            ),

        "annualised_volatility":
            annualised_volatility(
                series
            ),

        "sharpe_ratio":
            sharpe_ratio(
                series
            ),

        "maximum_drawdown":
            maximum_drawdown(
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
            )

    })


holdout_summary = pd.DataFrame(
    summary_rows
)


holdout_summary.to_csv(
    TABLE_DIR
    / "portfolio_holdout_summary.csv",
    index=False
)


print("\n" + "=" * 70)
print("UNSEEN HOLDOUT RESULTS")
print("=" * 70)


print(
    holdout_summary
    .round(4)
    .to_string(index=False)
)


# ============================================================
# 10. RELATIVE IMPROVEMENTS
# ============================================================

equal_row = (
    holdout_summary[
        holdout_summary[
            "portfolio"
        ]
        == "Equal Weight"
    ]
    .iloc[0]
)


minimum_variance_row = (
    holdout_summary[
        holdout_summary[
            "portfolio"
        ]
        == "Minimum Variance"
    ]
    .iloc[0]
)


maximum_sharpe_row = (
    holdout_summary[
        holdout_summary[
            "portfolio"
        ]
        == "Maximum Sharpe"
    ]
    .iloc[0]
)


volatility_reduction = (

    (
        equal_row[
            "annualised_volatility"
        ]
        - minimum_variance_row[
            "annualised_volatility"
        ]
    )

    / equal_row[
        "annualised_volatility"
    ]

    * 100
)


drawdown_improvement = (

    (
        abs(
            equal_row[
                "maximum_drawdown"
            ]
        )
        - abs(
            minimum_variance_row[
                "maximum_drawdown"
            ]
        )
    )

    / abs(
        equal_row[
            "maximum_drawdown"
        ]
    )

    * 100
)


sharpe_change = (

    maximum_sharpe_row[
        "sharpe_ratio"
    ]

    - equal_row[
        "sharpe_ratio"
    ]

)


print("\nMinimum-variance volatility reduction vs equal weight:")

print(
    round(
        volatility_reduction,
        2
    ),
    "%"
)


print("\nMinimum-variance drawdown reduction vs equal weight:")

print(
    round(
        drawdown_improvement,
        2
    ),
    "%"
)


print("\nMaximum-Sharpe holdout Sharpe change vs equal weight:")

print(
    round(
        sharpe_change,
        3
    )
)


# ============================================================
# 11. CUMULATIVE HOLDOUT PERFORMANCE
# ============================================================

cumulative_returns = pd.DataFrame({

    "Equal Weight":
        (
            1
            + equal_test_returns
        ).cumprod(),

    "Minimum Variance":
        (
            1
            + min_var_test_returns
        ).cumprod(),

    "Maximum Sharpe":
        (
            1
            + max_sharpe_test_returns
        ).cumprod()

})


fig, ax = plt.subplots(
    figsize=(12, 7)
)


for column in cumulative_returns.columns:

    ax.plot(
        cumulative_returns.index,
        cumulative_returns[column],
        label=column
    )


ax.set_title(
    "Unseen Holdout Portfolio Performance"
)


ax.set_ylabel(
    "Growth of 1"
)


ax.legend()


ax.grid(
    alpha=0.25
)


fig.tight_layout()


fig.savefig(
    FIGURE_DIR
    / "portfolio_holdout_performance.png",
    dpi=200
)


plt.close(
    fig
)


# ============================================================
# 12. HOLDOUT DRAWDOWNS
# ============================================================

fig, ax = plt.subplots(
    figsize=(12, 7)
)


for name, series in portfolio_series.items():

    wealth = (
        1
        + series
    ).cumprod()


    drawdown = (
        wealth
        / wealth.cummax()
        - 1
    )


    ax.plot(
        drawdown.index,
        drawdown * 100,
        label=name
    )


ax.set_title(
    "Unseen Holdout Portfolio Drawdowns"
)


ax.set_ylabel(
    "Drawdown (%)"
)


ax.legend()


ax.grid(
    alpha=0.25
)


fig.tight_layout()


fig.savefig(
    FIGURE_DIR
    / "portfolio_holdout_drawdowns.png",
    dpi=200
)


plt.close(
    fig
)


# ============================================================
# 13. FINAL SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("PORTFOLIO BACKTEST COMPLETED SUCCESSFULLY")
print("=" * 70)


print("\nTraining window:")
print("2018-2022")


print("\nUnseen holdout:")
print("2023-2025")


print("\nFiles created:")

print(
    "reports/tables/holdout_portfolio_weights.csv"
)

print(
    "reports/tables/portfolio_holdout_summary.csv"
)

print(
    "reports/figures/portfolio_holdout_performance.png"
)

print(
    "reports/figures/portfolio_holdout_drawdowns.png"
)