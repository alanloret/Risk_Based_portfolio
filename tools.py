import pandas as pd
import numpy as np


def create_data():
    """
    This function loads price data from an Excel document.
    WARNING : Pay attention to the Excel document's directory path and name.
    :return: a DataFrame of the prices of the assets.
    :rtype: DataFrame
    """

    data = pd.read_excel(
        "data/StatApp_Data.xlsx", sheet_name='Data', parse_dates=['Dates']
    )
    data = data[data.Dates != 'None']
    data["Dates"] = pd.to_datetime(data.Dates, format="%Y-%m-%d %H:%M:%S")

    return data.set_index("Dates")


def annualized_return(return_data, freq="daily"):
    """annualized return: inputs are frequency of data and return data"""
    xd = {"daily": 252, "monthly": 12, "weekly": 52}.get(freq, "daily")
    px_data = return_to_price_base100(return_data)
    return (px_data.iloc[-1] / px_data.iloc[0]) ** (xd / len(px_data - 1)) - 1


def volatility(return_data, freq="daily"):
    """ Inputs are frequency of data and return data """

    vol = return_data.std()
    if freq == "monthly":
        return vol * np.sqrt(12)
    elif freq == "weekly":
        return vol * np.sqrt(52)
    elif freq == "daily":
        return vol * np.sqrt(252)
    return vol


def downside_vol(return_data, freq="daily"):
    return_data = return_data.loc[return_data < 0]
    vol_ = return_data.loc[return_data < 0].std()
    if freq == "monthly":
        return vol_ * np.sqrt(12)
    elif freq == "weekly":
        return vol_ * np.sqrt(52)
    elif freq == "daily":
        return vol_ * np.sqrt(252)
    return vol_


def sharpe(return_data, freq="daily"):
    return annualized_return(return_data, freq) / volatility(return_data, freq)


def sharpe_corrected(return_data, freq="daily"):
    """
    The correction for the sharpe and calmar ratio gives the same if return is
    positive than sharpe and calmar uncorrected; when return is negative it
    corrects the ratio.
    """

    ann_ret = annualized_return(return_data, freq)
    sigma = volatility(return_data, freq)
    sharpe = ann_ret / (sigma ** np.sign(ann_ret))
    return sharpe


def sortino(return_data, freq="daily"):
    return annualized_return(return_data, freq=freq) / downside_vol(return_data, freq=freq)


def calmar(return_data, freq="daily"):
    return annualized_return(return_data, freq=freq) / max_drawdown(return_data)


def max_drawdown(portfolio_value):
    max_dd = 0
    dates = portfolio_value.index.tolist()
    for i in range(len(dates) - 1):
        if i < len(dates) - 1:
            if portfolio_value.iloc[i+1] <= portfolio_value.iloc[i]:
                original = portfolio_value.iloc[i]
                j = i+1
                dd = portfolio_value.iloc[i] - portfolio_value.iloc[j]
                while j+1 < len(dates) and portfolio_value.iloc[j+1] <= portfolio_value.iloc[j]:
                    dd = portfolio_value.iloc[i] - portfolio_value.iloc[j]
                    j += 1
                dd = dd / portfolio_value.iloc[i]
                if dd > max_dd:
                    max_dd = dd
    return max_dd


def return_to_price_base100(return_data):
    """
    Works only if here is a pandas.nan in the first row, to fill it with with
    0 and start with 100. Otherwise starts with 100*first
    """

    return_data = return_data.copy()
    return_data = return_data.fillna(0) + 1.0
    return_data.iloc[0] *= 100
    return return_data.cumprod()


def concentration_function(correlations):
    """
    Function CF from ARB article where correlations is a numpy matrix of pairwise correlations
    """
    n = len(correlations)
    average_rho = ((correlations.sum() - 1) / (n - 1)).mean()  # Exclude auto-correlation
    return np.sqrt((1 + (n - 1) * average_rho) / n)


def implied_returns(weights, cov_matrix, risk_aversion=1):
    """
    :param weights: weights of the portfolio for which
                    we want to compute implied returns
    :type weights: numpy k*1 vector
    :param cov_matrix: covariance of returns of the assets considered
    :type cov_matrix: numpy k*k matrix
    :param risk_aversion: risk aversion to use in MVO, defaults to 1
    :return: k*1 numpy vector of implied returns
    """
    return risk_aversion * cov_matrix @ weights


def implied_returns_df(weights_df, cov_matrices, risk_aversion=1):
    """
    :param weights_df: weights of the portfolio over different dates
    :type weights_df: data frame with dates as index,
                        and assets names as column names
    :param cov_matrices: covariance of returns of the assets considered
                            over different dates
    :type cov_matrices: data frame with dates as index, with a cov matrix at each date
    :param risk_aversion: same as above
    :return: a data frame consisting of the implied returns of the argument portfolio
                for each date in the initial weights_df data frame
    """

    returns = weights_df.copy(deep=True)
    indices = weights_df.index.to_list()
    for index in indices:
        weights = weights_df.loc[index].to_numpy()
        cov_matrix = cov_matrices.loc[index].to_numpy()
        implied_return = implied_returns(weights, cov_matrix, risk_aversion)
        returns.loc[index] = implied_return
    return returns
