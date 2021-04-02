import pandas as pd


def create_data():
    """
    This function loads price data from an Excel document.
    WARNING : Pay attention to the Excel document's directory path and name.
    :return: a DataFrame of the prices of the assets.
    :rtype: DataFrame
    """

    data = pd.read_excel(
        "data/Data.xlsx", sheet_name='Data', parse_dates=['Dates']
    )
    data = data[data.Dates != 'None']
    data["Dates"] = pd.to_datetime(data.Dates, format="%Y-%m-%d %H:%M:%S")

    return data.set_index("Dates")

