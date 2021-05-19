import logging
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from datetime import date
from risk_based_methods import PORTFOLIO_FUNCS
from tools import *


class RiskBasedPortfolio:
    def __init__(
            self,
            prices: pd.DataFrame,
            rebalancing_frequency: str,
            start: date = None,
            end: date = None,
            method: str = "EW"
    ):
        self.prices = prices
        self.method = method
        self.rebalancing_frequency = rebalancing_frequency
        self.weights = None
        self.returns = None
        self.nav = None

        self.start_date = start
        if start is None:  # Take first date of self.prices ?
            self.start_date = prices.index[0].date()

        self.end_date = end
        if end is None:  # Take last date of self.prices ?
            self.end_date = self.prices.index[-1].date()

        self.check_inputs()
        self.rebalancing_dates = self.compute_rebalancing_dates()

    def check_inputs(self):
        """
        This method checks whether the attributes are correctly initialized.
        :raise ValueError: if the start_date is greater than the end_date.
        :raise ValueError: if the price dataset contains NaN values.
        :raise ValueError: if the index is not a date.
        :raise NotImplementedError: if the rebalancing_frequency is not
            monthly, quarterly nor yearly.
        :raise NotImplementedError: if the risk-based method is not
            EW, RW, MinVar, MaxDiv nor ERC.
        """

        if self.start_date is not None and self.end_date is not None:
            if self.start_date >= self.end_date:
                raise (ValueError("End_date must be lower than start_date."))

        if self.method not in ['EW', 'RW', 'MinVar', 'MaxDiv', 'ERC']:
            raise NotImplementedError(
                f"This Risk-Based method({self.method}) is not implemented.\n"
                f"Choose between these strategies ['EW', 'RW', 'MinVar', 'MaxDiv', 'ERC']."
            )

        if self.rebalancing_frequency not in ["monthly", "quarterly", "yearly"]:
            raise NotImplementedError(
                f"{self.rebalancing_frequency} must belong to ['monthly', 'quarterly', 'yearly']."
            )

        if self.prices.isnull().values.any():
            raise ValueError("Price data must not contain any NaN value.")

        if (
            self.prices.index.inferred_type != "datetime64"
            or self.prices.index.isna().any()
        ):
            raise ValueError("Indexes must all be dates.")

    def compute_rebalancing_dates(self):
        """
        This method computes the rebalacing dates between the
        start and the end date accordingly to the rebalancing_frequency.
        :returns: list of the rebalacing dates.
        :rtype: list of pd.Timestamp.
        """

        freq = {"monthly": "M", "quarterly": "3M", "yearly": "Y"}.get(self.rebalancing_frequency)

        # We need to take the first training date after the start_date
        # and the last trading date before the end_date
        start = self.prices.loc[self.start_date:].index[0]
        end = self.prices.loc[: self.end_date].index[-1]

        dates = pd.Series(
            self.prices.groupby([pd.Grouper(freq=freq)]).head(1).index
        )
        return [start] + dates.loc[(start < dates) & (dates < end)].to_list() + [end]

    def compute_weights_once(self, date, window=252, verbose=0):
        # Compute returns and covariance
        returns = self.prices.pct_change()
        covariance = returns.rolling(window=window).cov()

        # Get balancing dates and apply to covariance
        covariance_matrices = covariance.loc[self.rebalancing_dates]

        return PORTFOLIO_FUNCS[self.method](
            np.array(covariance_matrices.loc[date]),
            verbose=verbose
        )

    def compute_weights(self, window=252, verbose=0):
        """
        This method computes the weights of the portfolio.
        :param int window: number of days to take into account
            for the covariance matrix.
        :param int verbose:
            - 0: prints nothing
            - 1: prints balancing dates
            - 2: prints balancing dates and messages of the optimization.
        :raise ValueError: if the covariance matrix contains
            NaN values.
        """

        # Compute returns and covariance
        returns = self.prices.pct_change()
        covariance = returns.rolling(window=window).cov()

        # Get balancing dates and apply to covariance
        covariance_matrices = covariance.loc[self.rebalancing_dates]

        # Check NaN in covariance
        if covariance_matrices.isnull().values.any():
            raise ValueError(
                "The covariance matrix has NaN values.\n"
                "Unable to compute weights : check start date, end date and window."
            )

        # Compute weights
        weights = []
        for balancing_date in self.rebalancing_dates:
            if verbose:
                print(f"Computing weights on {balancing_date}.")
            weights.append(
                PORTFOLIO_FUNCS[self.method](
                    np.array(covariance_matrices.loc[balancing_date]),
                    verbose=verbose
                )
            )

        self.weights = pd.DataFrame(weights, index=self.rebalancing_dates, columns=self.prices.columns)

    def compute_returns(self, aum_start=100):
        """
        This method computes the returns of the portfolio.
        :param int/float aum_start: initial value of the portfolio at
            the start date.
        :raise ValueError: if the value of method is not naive
            or realistic.
        """

        if self.weights is None:
            logging.warning("Weights do not exist: computing them...")
            self.compute_weights()

        assets_returns = self.prices.loc[
            self.weights.index[0]: self.weights.index[-1]
        ].pct_change()

        weights = self.weights.reindex(index=assets_returns.index).values
        nav, assets_change = [aum_start], assets_returns.values + 1.0

        for index in range(1, len(weights)):
            if np.isnan(weights[index, 0]):
                weights[index] = weights[index - 1] * assets_change[index]
                weights[index] /= weights[index].sum()
            nav.append(nav[index - 1] * (weights[index - 1] * assets_change[index]).sum())

        self.nav = pd.Series(
            nav, index=assets_returns.index, name="NAV"
        )
        self.returns = pd.Series(
            self.nav.pct_change(), name="Return"
        )

    def describe(self, method="naive", aum_start=100):
        """
        :param method: method for computing the portfolio returns (naive or realistic)
        :param aum_start: reference value at the beginning of the portfolio
        :return: descriptive statistics on the portfolio performance
        """

        if self.returns is None:
            self.compute_returns(aum_start=aum_start)

        portfolio_vol = self.returns.std()
        portfolio_ann_returns = annualized_return(self.returns, freq="daily")
        portfolio_sharpe = sharpe(self.returns, freq="daily")
        portfolio_sortino = sortino(self.returns, freq="daily")
        portfolio_calmar = calmar(self.returns, freq="daily")
        portfolio_value = (self.weights * self.prices.loc[self.rebalancing_dates]).sum(axis=1)
        max_dd = max_drawdown(portfolio_value)

        stats = self.describe(method=method)
        print(
            f"----- Statistics for {self} portfolio -----\n"
            f"Annualized Volatility : {portfolio_vol} \n"
            f"Annualized Returns : {portfolio_ann_returns} \n"
            f"Sharpe Ratio : {portfolio_sharpe} \n"
            f"Maximum Drawdown : {max_dd} \n"
            f"Sortino Ratio : {portfolio_sortino} \n"
            f"Calmar Ratio : {portfolio_calmar} \n")

        return {"volatility": portfolio_vol,
                "ann_returns": portfolio_ann_returns,
                "sharpe_ratio": portfolio_sharpe,
                "max_drawdown": max_dd,
                "sortino": portfolio_sortino,
                "calmar": portfolio_calmar}

    def visualize_prices(self, path=None):
        """
        This method plots the prices.
        :param str/None path: the path to save the graphic.
            - str: the graphic is saved.
            - None: the graphic is plotted.
        """

        plt.gcf().clear()
        fig = plt.figure(figsize=(10, 6))
        ax = fig.add_axes([0.065, 0.1, 0.45, 0.8])
        ax.plot(self.prices.index, self.prices.to_numpy())
        ax.set_title("Prices")
        ax.legend(
            self.prices.columns.to_list(),
            bbox_to_anchor=(1.05, 0.6),
            loc=2,
            borderaxespad=0.0,
            prop={'size': 9}
        )
        if path is None:
            plt.show()
        elif type(path) == str:
            fig.savefig(
                path + "prices.png",
                dpi=fig.dpi,
                bbox_inches='tight',
                pad_inches=0.5
            )
            print(f"File saved: 'prices.png'\nPath of the file: {path}")

    def truncated_nav(self, start_date=None, aum_start=100):
        """
        This method computes the returns of the portfolio.
        :param datetime start_date: start date.
        :param int/float aum_start: initial value of the portfolio at
            the start date.
        :return: the NAV (value of asset or portfolio) of our portfolio
            and of assets.
        :rtype: DataFrame
        """

        if start_date is None:
            start_date = pd.to_datetime(self.start_date)

        if self.nav is None:
            logging.warning("Returns do not exist: computing them...")
            self.compute_returns(aum_start=aum_start)

        truncated_nav = pd.Series(
            self.nav.loc[start_date:],
            name="truncated_nav"
        )
        return truncated_nav / truncated_nav.iloc[0] * aum_start

    def visualize_returns(self, start_date=None, aum_start=100, path=None):
        """
        This method plots the returns.
        :param datetime start_date: start date of the plot.
        :param int/float aum_start: initial value of the portfolio.
        :param str/None path: the path to save the graphic.
            - str: the graphic is saved.
            - None: the graphic is plotted.
        """

        plt.gcf().clear()
        fig = plt.figure(figsize=(10, 5))
        ax = fig.add_subplot(1, 1, 1)
        ax.plot(
            self.truncated_nav(start_date=start_date, aum_start=aum_start),
            label="NAV"
        )
        ax.set_title(f"Simulated NAV - {self.method} ")
        ax.legend()

        if path is None:
            plt.show()
        elif type(path) == str:
            fig.savefig(
                path + f"NAV_{self.method}.png",
                dpi=fig.dpi,
                bbox_inches='tight',
                pad_inches=0.5
            )
            print(f"File saved: 'NAV_{self.method}.png'\nPath of the file: {path}")

    def visualize_returns_all(self, start_date=None, aum_start=100, path=None):
        """
        This method plots the returns of all methods.
        :param datetime start_date: start date of the plot.
        :param int/float aum_start: initial value of the portfolio.
        :param str/None path: the path to save the graphic.
            - str: the graphic is saved.
            - None: the graphic is plotted.
        """

        save_method = self.method

        plt.gcf().clear()
        fig = plt.figure(figsize=(10, 5))
        ax = fig.add_subplot(1, 1, 1)

        for method in ['EW', 'RW', 'MinVar', 'MaxDiv', 'ERC']:
            # We need to update all the weights/returns
            self.method = method
            self.compute_weights()
            self.compute_returns(aum_start=aum_start)
            ax.plot(
                self.truncated_nav(start_date=start_date, aum_start=aum_start),
                label=method
            )

        self.method = save_method
        self.compute_weights()
        self.compute_returns(aum_start=aum_start)

        ax.set_title(f"Simulated NAV")
        ax.legend()

        if path is None:
            plt.show()
        elif type(path) == str:
            fig.savefig(
                path + f"NAV.png",
                dpi=fig.dpi,
                bbox_inches='tight',
                pad_inches=0.5
            )
            print(f"File saved: 'NAV.png'\nPath of the file: {path}")

    def visualize_weights(self, path=None):
        """
        This method plots the weights.
        :param str/None path: the path to save the graphic.
            - str: the graphic is saved.
            - None: the graphic is plotted.
        """

        if self.weights is None:
            logging.warning("Weights do not exist: computing them...")
            self.compute_weights()

        plt.gcf().clear()
        fig = plt.figure(figsize=(10, 6))
        ax = fig.add_axes([0.065, 0.1, 0.45, 0.8])
        ax.plot(self.weights.index, self.weights.to_numpy())
        ax.set_title(f"Weights - {self.method}")
        ax.legend(
            self.weights.columns.to_list(),
            bbox_to_anchor=(1.05, 0.6),
            loc=2,
            borderaxespad=0.0,
            prop={'size': 9}
        )
        if path is None:
            plt.show()
        elif type(path) == str:
            fig.savefig(
                path + f"weights_{self.method}.png",
                dpi=fig.dpi,
                bbox_inches='tight',
                pad_inches=0.5
            )
            print(
                f"File saved : 'weights_{self.method}.png'\nPath of the files: {path}"
            )

        plt.gcf().clear()
        fig_ = plt.figure(figsize=(10, 6))
        ax_ = fig_.add_axes([0.065, 0.1, 0.45, 0.8])
        ax_.stackplot(self.weights.index, self.weights.T)
        ax_.set_title(f"Stacked weights - {self.method}")
        ax_.legend(
            self.weights.columns.to_list(),
            bbox_to_anchor=(1.05, 0.6),
            loc=2,
            borderaxespad=0.0,
            prop={'size': 9}
        )
        if path is None:
            plt.show()

        elif type(path) == str:
            fig_.savefig(
                path + f"weights_stacked_{self.weights}.png",
                dpi=fig_.dpi,
                bbox_inches='tight',
                pad_inches=0.5
            )
            print(
                f"File saved : 'weights_stacked_{self.weights}.png'\nPath of the files: {path}"
            )

    def visualize_weights_all(self, path=None):
        """
        This method plots the weights of all methods.
        :param str/None path: the path to save the graphic.
            - str: the graphic is saved.
            - None: the graphic is plotted.
        """
        save_method = self.method

        for method in ['EW', 'RW', 'MinVar', 'MaxDiv', 'ERC']:
            # We need to update all the weights
            self.method = method
            self.compute_weights()

            plt.gcf().clear()
            fig = plt.figure(figsize=(10, 6))
            ax = fig.add_axes([0.065, 0.1, 0.45, 0.8])
            ax.plot(self.weights.index, self.weights.to_numpy())
            ax.set_title(f"Weights - {self.method}")
            ax.legend(
                self.weights.columns.to_list(),
                bbox_to_anchor=(1.05, 0.6),
                loc=2,
                borderaxespad=0.0,
                prop={'size': 9}
            )
            if path is None:
                plt.show()
            elif type(path) == str:
                fig.savefig(
                    path + f"weights_{self.method}.png",
                    dpi=fig.dpi,
                    bbox_inches='tight',
                    pad_inches=0.5
                )
                print(
                    f"File saved : 'weights_{self.method}.png'\nPath of the files: {path}"
                )

            plt.gcf().clear()
            fig_ = plt.figure(figsize=(10, 6))
            ax_ = fig_.add_axes([0.065, 0.1, 0.45, 0.8])
            ax_.stackplot(self.weights.index, self.weights.T)
            ax_.set_title(f"Stacked weights - {self.method}")
            ax_.legend(
                self.weights.columns.to_list(),
                bbox_to_anchor=(1.05, 0.6),
                loc=2,
                borderaxespad=0.0,
                prop={'size': 9}
            )
            if path is None:
                plt.show()

            elif type(path) == str:
                fig_.savefig(
                    path + f"weights_stacked_{self.weights}.png",
                    dpi=fig_.dpi,
                    bbox_inches='tight',
                    pad_inches=0.5
                )
                print(
                    f"File saved : 'weights_stacked_{self.weights}.png'\nPath of the files: {path}"
                )

        self.method = save_method
        self.compute_weights()
