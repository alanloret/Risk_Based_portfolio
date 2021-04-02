from risk_based_portfolio import RiskBasedPortfolio
from tools import create_data
from datetime import date


if __name__ == "__main__":
    df = create_data()  # use your module to load prices

    # Create a Risk-Weighted portfolio with a monthly rebalacing frequency
    portfolio = RiskBasedPortfolio(
        df,
        "monthly",
        method="RW",
        start=date(2002, 1, 19)
    )

    # Compute weights and returns
    portfolio.compute_weights(window=252, verbose=0)
    portfolio.compute_returns(aum_start=100)

    # Show prices
    portfolio.visualize_prices(path=None)

    # Show weights
    portfolio.visualize_weights(path=None)

    # Show NAV
    portfolio.visualize_returns(aum_start=100)
