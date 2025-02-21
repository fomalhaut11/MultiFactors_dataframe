import pandas as pd
import numpy as np
from SingleFactorTest import Remove_Outlier, quick_remove_outlier_np, Normlization
import matplotlib.pyplot as plt
from risk_model import sequential_orthog_df, dataloader


def double_factors_test_2triple(factordata, logreturn):
    factordataname = factordata.columns
    factordata = factordata.join(logreturn, how="left")
    factordata = factordata.dropna(subset=["LogReturn"])
    def _indayfunc_proportion(dayslice):
        dayslice['group1'] = pd.qcut(dayslice['data1'], q=3, labels=False, duplicates='drop')
        dayslice['group2'] = pd.qcut(dayslice['data2'], q=3, labels=False, duplicates='drop')
        dayslice['group_logreturn'] = pd.qcut(dayslice['LogReturn'], q=3, labels=False, duplicates='drop')
        group1_0 = dayslice[dayslice['group1'] == 0]
        group1_1 = dayslice[dayslice['group1'] == 1]
        group1_2 = dayslice[dayslice['group1'] == 2]
        # 计算 group_logreturn 为 1 的比例
        proportion = (group2_0['group_logreturn'] == 0).mean()
    return


if __name__ == "main":
    print("main")
    datapath = r"E:\Documents\PythonProject\StockProject\StockData"
    PriceDf = pd.read_pickle(datapath + "\\" + "Price.pkl")
    StockTradableDF = pd.read_pickle(datapath + "\\" + "TradableDF.pkl")
    PriceDf = PriceDf[~(PriceDf["exchange_id"] == "BJ")]  # 踢出北交所
    StockTradableDF = StockTradableDF[
        ~(StockTradableDF["exchange_id"] == "BJ")
    ]  # 踢出北交所

    PriceDf = PriceDf.join(
        StockTradableDF, how="left", lsuffix="_left", rsuffix="_right"
    )
    PriceDf = PriceDf[~(PriceDf["trade_status"] == "退市")]

    basenames = ['LogMarketCap', 'PEG', 'freeturnoverrate_ma120']
    factornames = [ 'zz2000_126_30_beta', 'DEDUCTEDPROFIT_yoy_zscores_4']
    basedata = dataloader(basenames)
    factordata = dataloader(factornames)
    factordata.columns=['data1', 'data2']  
    data1 = basedata.join(factordata, how="left")
    factordata1 = sequential_orthog_df(data1)

    logreturn = pd.read_pickle(datapath + "\\" + "LogReturn_daily_o2o.pkl")
