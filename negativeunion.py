import glob
import pandas as pd
import numpy as np
import pickle

files = glob.glob(
    "E:\Documents\PythonProject\StockProject\StockData\TestTempData\*data.pkl"
)


Fil = files[0]
Fil1 = files[1]


def load_data(Fil):

    with open(Fil, "rb") as f:
        data = pickle.load(f)
    return data


def select_data_negative(data1, rolling_window=120, icband=0.01):

    icrolling = data1["IC"].rolling(rolling_window).mean()
    TradingDates = icrolling.index.get_level_values("TradingDates")
    negative_holdings = {}
    for date in TradingDates:
        ic = icrolling.loc[date]

        if (ic < -icband).any():
            max_group = (
                data1["grouped_StockCodes"]
                .loc[date]
                .index.get_level_values("Group")
                .max()
            )
            negative_holdings[date] = (
                data1["grouped_StockCodes"].loc[date].loc[max_group]
            )
        elif (ic > icband).any():
            min_group = (
                data1["grouped_StockCodes"]
                .loc[date]
                .index.get_level_values("Group")
                .min()
            )
            negative_holdings[date] = (
                data1["grouped_StockCodes"].loc[date].loc[min_group]
            )
        elif ic.isna().any():
            negative_holdings[date] = []
        else:
            negative_holdings[date] = []
    return pd.Series(negative_holdings)


def select_data_passitive(data1, rolling_window=120, icband=0.01):

    icrolling = data1["IC"].rolling(rolling_window).mean()
    TradingDates = icrolling.index.get_level_values("TradingDates")
    negative_holdings = {}
    for date in TradingDates:
        ic = icrolling.loc[date]

        if (ic < -icband).any():
            max_group = (
                data1["grouped_StockCodes"]
                .loc[date]
                .index.get_level_values("Group")
                .min()
            )
            negative_holdings[date] = (
                data1["grouped_StockCodes"].loc[date].loc[max_group]
            )
        elif (ic > icband).any():
            min_group = (
                data1["grouped_StockCodes"]
                .loc[date]
                .index.get_level_values("Group")
                .max()
            )
            negative_holdings[date] = (
                data1["grouped_StockCodes"].loc[date].loc[min_group]
            )
        elif ic.isna().any():
            negative_holdings[date] = []
        else:
            negative_holdings[date] = []
    return pd.Series(negative_holdings)


def data_group_to_quantile(data1, rolling_window=120, icband=0.01):
    icrolling = data1["IC"].rolling(rolling_window).mean()
    TradingDates = icrolling.index.get_level_values("TradingDates")
    quatiledata = {}
    data = data1["grouped_StockCodes"].reset_index(level=1)

    for date in TradingDates:
        ic = icrolling.loc[date]
        max_group = data.loc[date]["Group"].max()
        if ic.isna().any():
            data.loc[(data.index == date), "Group"] = 0.5
        elif (ic > icband).any():
            data.loc[date]["Group"] = data.loc[date]["Group"] / max_group
        elif (ic < -icband).any():
            data.loc[date]["Group"] = 1 - data.loc[date]["Group"] / max_group
        else:
            data.loc[(data.index == date), "Group"] = 0.5

    return data


# 定义一个函数来获取两个列表的并集
def union_lists(list1, list2):
    # 如果输入是数组，将其转换为列表
    if isinstance(list1, np.ndarray):
        list1 = list1.tolist()
    if isinstance(list2, np.ndarray):
        list2 = list2.tolist()
    # 如果输入是np.nan，将其视为一个空列表
    if np.all(pd.isnull(list1)):
        list1 = []
    if np.all(pd.isnull(list2)):
        list2 = []
    return list(set(list1) | set(list2))


def intersect_lists(list1, list2):
    # 如果输入是数组，将其转换为列表
    if isinstance(list1, np.ndarray):
        list1 = list1.tolist()
    if isinstance(list2, np.ndarray):
        list2 = list2.tolist()
    # 如果输入是np.nan，将其视为一个空列表
    if np.all(pd.isnull(list1)):
        list1 = []
    if np.all(pd.isnull(list2)):
        list2 = []
    return list(set(list1) & set(list2))


def loadingLogreturnData():
    Fil = r"E:\Documents\PythonProject\StockProject\StockData\LogReturn_daily_o2o.pkl"
    with open(Fil, "rb") as f:
        data = pickle.load(f)
    return data


def non_negative_return(LogReturn, negative_holdings):
    result = []

    # 遍历LogReturn的日期
    for date in negative_holdings.index.get_level_values(0).unique():
        # 获取当天的股票
        stocks_today = LogReturn.loc[date].index

        # 获取当天的negative_holdings
        negative_holdings_today = set(negative_holdings.loc[date])

        # 找出在LogReturn中但不在negative_holdings中的股票
        selected_stocks = [
            stock for stock in stocks_today if stock not in negative_holdings_today
        ]

        # 获取这些股票的LogReturn值
        df = LogReturn.loc[date].loc[selected_stocks]
        df["date"] = date  # 添加日期列
        result.append(df)

    # 将结果转换为DataFrame
    result = pd.concat(result)
    # 将原有的第一级索引"StockCodes"转换为列
    result = result.reset_index(level=0)

    # 将'date'和'StockCodes'设置为新的多级索引，且'date'为第一级索引
    result = result.set_index(["date", "StockCodes"])

    # 重命名索引
    result.index.names = ["TradingDates", "StockCodes"]
    return result


count = 0
for file in files:

    print(file)
    data1 = load_data(file)
    negative_holdings1 = select_data_negative(data1, rolling_window=120, icband=0.01)
    if count == 0:
        negative_holdings = negative_holdings1

    else:
        negative_holdings = negative_holdings.combine(
            negative_holdings1, func=union_lists
        )
    # if count == 3:
    #     break
    count += 1
    print(count)

count = 0
for file in files:

    print(file)
    data1 = load_data(file)
    passitive_holdings1 = select_data_passitive(data1, rolling_window=120, icband=0.01)
    if count == 0:
        passitive_holdings = passitive_holdings1

    else:
        passitive_holdings = passitive_holdings.combine(
            passitive_holdings1, func=intersect_lists
        )
    # if count == 3:
    #     break
    count += 1
    print(count)
result = non_negative_return(LogReturn, negative_holdings)

result1 = non_negative_return(LogReturn, passitive_holdings)

diffdata = (
    result1.groupby("TradingDates").mean() - LogReturn.groupby("TradingDates").mean()
)
selected_diffdata = diffdata[diffdata.index > "2020-01-01"]
selected_diffdata.cumsum().plot()


non_negative_return(LogReturn, negative_holdings)

# 使用apply方法获取negative_holdings和negative_holdings0的并集
union_holdings = negative_holdings.combine(negative_holdings1, func=union_lists)
negative_holdings = select_data_negative(load_data(Fil))
