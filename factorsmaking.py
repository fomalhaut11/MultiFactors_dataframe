import numpy as np
import pandas as pd
import os
import sys

file_path = r"E:\Documents\PythonProject\StockProject"
sys.path.append(file_path)
file_path1 = r"E:\Documents\PythonProject\StockProject\MultiFactors"
sys.path.append(file_path1)

import StockDataPrepairing as SDP
import TechnicFactors as TF
import alpha191backtest_factor as alpha191
import 股票数据接口_zx as FinData
from datetime import date
from datetime import datetime
import matplotlib.pyplot as plt
import statsmodels.api as sm
from time import time

""" def factor_ROE(x1,cla):
    # x1=SDP.FactorMatrix_Report(StockData3dMatrix,cla)
    # x2=SDP.FactorMatrix_Report(StockData3dMatrix,cla)
    # x1.Title('lrb','DEDUCTEDPROFIT') 
    # x2.Title('fzb','EQY_BELONGTO_PARCOMSH')
    # f1=x1.InitFactor()
    # f2=x2.InitFactor()
    # x1.initfactor=f1/f2#roe：扣非利润/归母权益

    # templrb=x1.ReportData.lrb[x1.Report_idx_Stock3d,:,:]
    # index=[0,1,-3]
    # x1.initfactor_ReportDates= templrb[:,:,index]
    # f1expanding=x1.FactorExpanding() 
    # factorDF=x1.ExpandedFactor2DataFrame(f1expanding)
    return factorDF """


def factor_ROE_YoY(StockData3dMatrix, cla):
    x1 = SDP.FactorMatrix_Report(StockData3dMatrix, cla)
    x2 = SDP.FactorMatrix_Report(StockData3dMatrix, cla)
    x1.Title("lrb", "DEDUCTEDPROFIT")
    x2.Title("fzb", "EQY_BELONGTO_PARCOMSH")
    f1 = x1.InitFactor()
    f2 = x2.InitFactor()
    x1.initfactor = f1 / f2  # roe：扣非利润/归母权益

    dd = x1.apply_initial_data()
    x1.initfactor = dd
    f1expanding = x1.FactorExpanding()
    factorDF = x1.ExpandedFactor2DataFrame(f1expanding)
    return factorDF


def liq_amt(x2, para=20, show=False):
    CloseReturnM = x2.CloseReturn()
    FactorMatrix = np.ones([len(x2.TradingDates), len(x2.StockCodes)]) * np.nan
    s = np.shape(CloseReturnM)
    for i in range(len(x2.StockCodes)):
        if show == True:
            print(i)
        for j in range(len(x2.TradingDates)):
            if j <= para:
                continue
            temp = CloseReturnM[j - para : j, i]
            temp1 = x2.pricematrix[j - para : j, i, 5]  # 左开右闭，不用移动
            std = np.std(np.abs(temp))
            std1 = np.std(np.abs(temp1)) / 100000000
            if std1 > 0:
                FactorMatrix[j, i] = std / std1

    df = pd.DataFrame(FactorMatrix, index=x2.TradingDates, columns=x2.StockCodes)
    df = df.stack()
    df.index.names = ["TradingDates", "StockCodes"]
    return df


def liq_amihud_std(x2, para=20, show=False):
    FactorMatrix = np.ones([len(x2.TradingDates), len(x2.StockCodes)]) * np.nan
    CloseReturnM = x2.CloseReturn()
    data = CloseReturnM / (x2.pricematrix[:, :, 5] / 1000000000)
    for i in range(len(x2.StockCodes)):
        temp = data[:, i]
        for j in range(len(x2.TradingDates)):
            if j <= para:
                continue
            temp1 = temp[j - para : j]  # 左开右闭，不用移动
            FactorMatrix[j, i] = np.nanstd(temp1)
    df = pd.DataFrame(FactorMatrix, index=x2.TradingDates, columns=x2.StockCodes)
    df = df.stack()
    df.index.names = ["TradingDates", "StockCodes"]
    return df


def RoE(x1, r_method="ttm", e_method="avg"):
    x1.Title("lrb", "DEDUCTEDPROFIT")
    x1.Title_2("fzb", "EQY_BELONGTO_PARCOMSH")
    x1.initfactor = x1.InitFactor()
    DEDUCTEDPROFIT = x1.apply_initial_data(r_method)
    x1.initfactor = x1.InitFactor_2()
    EQY_BELONGTO_PARCOMSH = x1.apply_initial_data(e_method)
    x1.initfactor = DEDUCTEDPROFIT / EQY_BELONGTO_PARCOMSH
    f1expanding = x1.FactorExpanding()
    factorDF = x1.ExpandedFactor2DataFrame(f1expanding)
    return factorDF


def RoE_zscores(x1, r_method="ttm", e_method="avg", len1=8):
    x1.Title("lrb", "DEDUCTEDPROFIT")
    x1.Title_2("fzb", "EQY_BELONGTO_PARCOMSH")
    x1.initfactor = x1.InitFactor()
    DEDUCTEDPROFIT = x1.apply_initial_data(r_method)
    x1.initfactor = x1.InitFactor_2()
    EQY_BELONGTO_PARCOMSH = x1.apply_initial_data(e_method)
    x1.initfactor = DEDUCTEDPROFIT / EQY_BELONGTO_PARCOMSH
    x1.initfactor = x1.FinDataRollingZScores(len1)
    f1expanding = x1.FactorExpanding()
    factorDF = x1.ExpandedFactor2DataFrame(f1expanding)
    return factorDF


def RoE_ratio(x1, r_method="ttm", e_method="avg", len1=4):
    x1.Title("lrb", "DEDUCTEDPROFIT")
    x1.Title_2("fzb", "EQY_BELONGTO_PARCOMSH")
    x1.initfactor = x1.InitFactor()
    DEDUCTEDPROFIT = x1.apply_initial_data(r_method)
    x1.initfactor = x1.InitFactor_2()
    EQY_BELONGTO_PARCOMSH = x1.apply_initial_data(e_method)
    x1.initfactor = DEDUCTEDPROFIT / EQY_BELONGTO_PARCOMSH
    RoE_yoy_ratio = x1.apply_initial_data("ss")
    x1.initfactor = RoE_yoy_ratio
    f1expanding = x1.FactorExpanding()
    factorDF = x1.ExpandedFactor2DataFrame(f1expanding)
    x1.initfactor = x1.FinDataRollingZScores(len1)
    f1expanding = x1.FactorExpanding()
    factorDF_zscores = x1.ExpandedFactor2DataFrame(f1expanding)
    return factorDF, factorDF_zscores


def EP_ttm(x1, MarketCap):
    x1.Title("lrb", "DEDUCTEDPROFIT")
    x1.initfactor = x1.InitFactor()
    DEDUCTEDPROFIT = x1.apply_initial_data("ttm")
    x1.initfactor = DEDUCTEDPROFIT
    f1expanding = x1.FactorExpanding()
    factorDF = x1.ExpandedFactor2DataFrame(f1expanding)
    MarketCap1 = MarketCap.groupby(level=1).shift(1)
    factorDF1 = factorDF / MarketCap1
    return factorDF1


def EP_ss(x1, MarketCap):
    x1.Title("lrb", "DEDUCTEDPROFIT")
    x1.initfactor = x1.InitFactor()
    DEDUCTEDPROFIT = x1.apply_initial_data("ss")
    x1.initfactor = DEDUCTEDPROFIT
    f1expanding = x1.FactorExpanding()
    factorDF = x1.ExpandedFactor2DataFrame(f1expanding)
    MarketCap1 = MarketCap.groupby(level=1).shift(1)
    factorDF1 = factorDF / MarketCap1
    return factorDF1


def BP(x1, MarketCap):
    x1.Title("fzb", "EQY_BELONGTO_PARCOMSH")
    x1.initfactor = x1.InitFactor()
    EQY_BELONGTO_PARCOMSH = x1.apply_initial_data("avg")
    x1.initfactor = EQY_BELONGTO_PARCOMSH
    f1expanding = x1.FactorExpanding()
    factorDF = x1.ExpandedFactor2DataFrame(f1expanding)
    MarketCap1 = MarketCap.groupby(level=1).shift(1)
    factorDF1 = factorDF / MarketCap1
    return factorDF1


def SP_ttm(x1, MarketCap):
    x1.Title("xjlb", "CASH_RECP_SG_AND_RS")
    x1.initfactor = x1.InitFactor()
    DEDUCTEDPROFIT = x1.apply_initial_data("ttm")
    x1.initfactor = DEDUCTEDPROFIT
    f1expanding = x1.FactorExpanding()
    factorDF = x1.ExpandedFactor2DataFrame(f1expanding)
    MarketCap1 = MarketCap.groupby(level=1).shift(1)
    factorDF1 = factorDF / MarketCap1
    return factorDF1


def SP_ss(x1, MarketCap):
    x1.Title("xjlb", "CASH_RECP_SG_AND_RS")
    x1.initfactor = x1.InitFactor()
    DEDUCTEDPROFIT = x1.apply_initial_data("ss")
    x1.initfactor = DEDUCTEDPROFIT
    f1expanding = x1.FactorExpanding()
    factorDF = x1.ExpandedFactor2DataFrame(f1expanding)
    MarketCap1 = MarketCap.groupby(level=1).shift(1)
    factorDF1 = factorDF / MarketCap1
    return factorDF1


def SUE_ss(x1, len1=4):
    x1.Title("lrb", "DEDUCTEDPROFIT")
    x1.initfactor = x1.InitFactor()
    x1.initfactor = x1.apply_initial_data("ss")
    x1.initfactor = x1.FinDataRollingZScores(len1)
    f1expanding = x1.FactorExpanding()
    factorDF = x1.ExpandedFactor2DataFrame(f1expanding)
    return factorDF


def SUE_ttm(x1, len1=4):
    x1.Title("lrb", "DEDUCTEDPROFIT")
    x1.initfactor = x1.InitFactor()
    x1.initfactor = x1.apply_initial_data("ttm")
    x1.initfactor = x1.FinDataRollingZScores(len1)
    f1expanding = x1.FactorExpanding()
    factorDF = x1.ExpandedFactor2DataFrame(f1expanding)
    return factorDF


def SUE_yoy(x1, len1=4):
    x1.Title("lrb", "DEDUCTEDPROFIT")
    x1.initfactor = x1.InitFactor()
    x1.initfactor = x1.apply_initial_data("yoy")
    x1.initfactor = x1.FinDataRollingZScores(len1)
    f1expanding = x1.FactorExpanding()
    factorDF = x1.ExpandedFactor2DataFrame(f1expanding)
    return factorDF


def SUE_qoq(x1, len1=4):
    x1.Title("lrb", "DEDUCTEDPROFIT")
    x1.initfactor = x1.InitFactor()
    x1.initfactor = x1.apply_initial_data("qoq")
    x1.initfactor = x1.FinDataRollingZScores(len1)
    f1expanding = x1.FactorExpanding()
    factorDF = x1.ExpandedFactor2DataFrame(f1expanding)
    return factorDF


def DEDUCTEDPROFIT_qoq(x1, len1=4):
    x1.Title("lrb", "DEDUCTEDPROFIT")
    x1.initfactor = x1.InitFactor()
    x1.initfactor = x1.apply_initial_data("qoq")
    x1.initfactor = x1.FinDataRollingZScores(len1)
    f1expanding = x1.FactorExpanding()
    factorDF = x1.ExpandedFactor2DataFrame(f1expanding)
    x1.initfactor = x1.FinDataRollingZScores(len1)
    f1expanding = x1.FactorExpanding()
    factorDF_zscores = x1.ExpandedFactor2DataFrame(f1expanding)
    return factorDF, factorDF_zscores


def DEDUCTEDPROFIT_yoy(x1, len1=4):
    x1.Title("lrb", "DEDUCTEDPROFIT")
    x1.initfactor = x1.InitFactor()
    x1.initfactor = x1.apply_initial_data("yoy")
    x1.initfactor = x1.FinDataRollingZScores(len1)
    f1expanding = x1.FactorExpanding()
    factorDF = x1.ExpandedFactor2DataFrame(f1expanding)
    x1.initfactor = x1.FinDataRollingZScores(len1)
    f1expanding = x1.FactorExpanding()
    factorDF_zscores = x1.ExpandedFactor2DataFrame(f1expanding)
    return factorDF, factorDF_zscores


def OPER_REV_yoy(x1, len1=4):
    # 营业收入同比增长率
    x1.Title("lrb", "OPER_REV")
    x1.initfactor = x1.InitFactor()
    x1.initfactor = x1.apply_initial_data("yoy")
    f1expanding = x1.FactorExpanding()
    factorDF = x1.ExpandedFactor2DataFrame(f1expanding)
    x1.initfactor = x1.FinDataRollingZScores(len1)
    f1expanding = x1.FactorExpanding()
    factorDF_zscores = x1.ExpandedFactor2DataFrame(f1expanding)
    return factorDF, factorDF_zscores


def STOT_CASH_INFLOWS_OPER_ACT_yoy(x1, len1=4):
    # 经营活动现金流入小计同比增长率
    x1.Title("xjlb", "STOT_CASH_INFLOWS_OPER_ACT")
    x1.initfactor = x1.InitFactor()
    x1.initfactor = x1.apply_initial_data("yoy")
    f1expanding = x1.FactorExpanding()
    factorDF = x1.ExpandedFactor2DataFrame(f1expanding)
    x1.initfactor = x1.FinDataRollingZScores(len1)
    f1expanding = x1.FactorExpanding()
    factorDF_zscores = x1.ExpandedFactor2DataFrame(f1expanding)
    return factorDF, factorDF_zscores


def STOT_CASH_INFLOWS_OPER_ACT_divedeby_TOT_CUR_LIAB(x1, len1=4):
    # 经营活动现金流入小计/流动负债
    x1.Title("xjlb", "STOT_CASH_INFLOWS_OPER_ACT")
    STOT_CASH_INFLOWS_OPER_ACT = x1.InitFactor()
    x1.Title("fzb", "TOT_CUR_LIAB")
    TOT_CUR_LIAB = x1.InitFactor()
    x1.initfactor = STOT_CASH_INFLOWS_OPER_ACT / TOT_CUR_LIAB
    f1expanding = x1.FactorExpanding()
    factorDF = x1.ExpandedFactor2DataFrame(f1expanding)
    x1.initfactor = x1.FinDataRollingZScores(len1)
    f1expanding = x1.FactorExpanding()
    factorDF_zscores = x1.ExpandedFactor2DataFrame(f1expanding)
    return factorDF, factorDF_zscores


def FreeCashflow(x1, len1=4):
    x1.Title("xjlb", "NET_CASH_FLOWS_OPER_ACT")  # 经营活动产生的现金流量净额
    NET_CASH_FLOWS_OPER_ACT = x1.InitFactor()
    x1.Title_2(
        "xjlb", "CASH_PAY_ACQ_CONST_FIOLTA"
    )  # 构建固定资产、无形资产和其他长期资产支付的现金
    CASH_PAY_ACQ_CONST_FIOLTA = x1.InitFactor_2()
    x1.initfactor = NET_CASH_FLOWS_OPER_ACT - CASH_PAY_ACQ_CONST_FIOLTA
    f1expanding = x1.FactorExpanding()
    factorDF = x1.ExpandedFactor2DataFrame(f1expanding)
    return factorDF


def FreeCashflow_ratio(x1, len1=4):
    x1.Title("xjlb", "NET_CASH_FLOWS_OPER_ACT")  # 经营活动产生的现金流量净额
    NET_CASH_FLOWS_OPER_ACT = x1.InitFactor()
    x1.Title_2(
        "xjlb", "CASH_PAY_ACQ_CONST_FIOLTA"
    )  # 构建固定资产、无形资产和其他长期资产支付的现金
    CASH_PAY_ACQ_CONST_FIOLTA = x1.InitFactor_2()
    x1.initfactor = NET_CASH_FLOWS_OPER_ACT - CASH_PAY_ACQ_CONST_FIOLTA
    x1.initfactor = x1.apply_initial_data("yoy")
    f1expanding = x1.FactorExpanding()
    factorDF = x1.ExpandedFactor2DataFrame(f1expanding)
    x1.initfactor = x1.FinDataRollingZScores(len1)
    f1expanding = x1.FactorExpanding()
    factorDF_zscores = x1.ExpandedFactor2DataFrame(f1expanding)
    return factorDF, factorDF_zscores


def NET_CASH_FLOWS_OPER_ACT_yoy(x1, len1=4):
    # 经营活动产生的现金流量净额同比增长率
    x1.Title("xjlb", "NET_CASH_FLOWS_OPER_ACT")
    x1.initfactor = x1.InitFactor()
    x1.initfactor = x1.apply_initial_data("yoy")
    f1expanding = x1.FactorExpanding()
    factorDF = x1.ExpandedFactor2DataFrame(f1expanding)
    x1.initfactor = x1.FinDataRollingZScores(len1)
    f1expanding = x1.FactorExpanding()
    factorDF_zscores = x1.ExpandedFactor2DataFrame(f1expanding)
    return factorDF, factorDF_zscores


def CurrentRatio(x1, len1=8):
    # 流动比率
    x1.Title("fzb", "TOT_CUR_ASSETS")
    x1.initfactor = x1.InitFactor()
    tca = x1.apply_initial_data("avg")  # 流动资产
    x1.Title("fzb", "TOT_CUR_LIAB")
    x1.initfactor = x1.InitFactor()
    tcl = x1.apply_initial_data("avg")  # 流动负债
    x1.initfactor = tca / tcl
    f1expanding = x1.FactorExpanding()
    factorDF = x1.ExpandedFactor2DataFrame(f1expanding)
    x1.initfactor = x1.FinDataRollingZScores(len1)
    f1expanding = x1.FactorExpanding()
    factorDF_zscores = x1.ExpandedFactor2DataFrame(f1expanding)
    return factorDF, factorDF_zscores


def QuickRatio(x1, len1=8):
    # 速动比率
    x1.Title("fzb", "TOT_CUR_ASSETS")
    x1.initfactor = x1.InitFactor()
    tca = x1.apply_initial_data("avg")  # 流动资产

    x1.Title("fzb", "INVENTORIES")
    x1.initfactor = x1.InitFactor()
    INVENTORIES = x1.apply_initial_data("avg")  # 存货

    x1.Title("fzb", "PREPAY")
    x1.initfactor = x1.InitFactor()
    PREPAY = x1.apply_initial_data("avg")  # 预付款项

    x1.Title("fzb", "TOT_CUR_LIAB")
    x1.initfactor = x1.InitFactor()
    tcl = x1.apply_initial_data("avg")  # 流动负债
    x1.initfactor = (tca - INVENTORIES - PREPAY) / tcl
    f1expanding = x1.FactorExpanding()
    factorDF = x1.ExpandedFactor2DataFrame(f1expanding)
    x1.initfactor = x1.FinDataRollingZScores(len1)
    f1expanding = x1.FactorExpanding()
    factorDF_zscores = x1.ExpandedFactor2DataFrame(f1expanding)
    return factorDF, factorDF_zscores


def GOODWILLRatio(x1, len1=8):
    x1.Title("fzb", "GOODWILL")
    x1.initfactor = x1.InitFactor()
    GOODWILL = x1.apply_initial_data("avg")  # 商誉
    x1.Title("fzb", "TOT_ASSETS")
    x1.initfactor = x1.InitFactor()
    TOT_ASSETS = x1.apply_initial_data("avg")  # 总资产
    x1.initfactor = GOODWILL / TOT_ASSETS
    f1expanding = x1.FactorExpanding()
    factorDF = x1.ExpandedFactor2DataFrame(f1expanding)
    x1.initfactor = x1.FinDataRollingZScores(len1)
    f1expanding = x1.FactorExpanding()
    factorDF_zscores = x1.ExpandedFactor2DataFrame(f1expanding)
    return factorDF, factorDF_zscores


def ACCT_RCV__NET_CASH_FLOWS_OPER_ACT(x1, len1=8):
    x1.Title("fzb", "ACCT_RCV")
    x1.initfactor = x1.InitFactor()
    ACCT_RCV = x1.apply_initial_data("ttm")  # 应收账款
    x1.Title("xjlb", "NET_CASH_FLOWS_OPER_ACT")
    x1.initfactor = x1.InitFactor()
    NET_CASH_FLOWS_OPER_ACT = x1.apply_initial_data("ttm")  # 经营活动产生的现金流量净额
    x1.initfactor = ACCT_RCV / NET_CASH_FLOWS_OPER_ACT
    f1expanding = x1.FactorExpanding()
    factorDF = x1.ExpandedFactor2DataFrame(f1expanding)
    x1.initfactor = x1.FinDataRollingZScores(len1)
    f1expanding = x1.FactorExpanding()
    factorDF_zscores = x1.ExpandedFactor2DataFrame(f1expanding)
    return factorDF, factorDF_zscores


def wet_profit_ratio(x1, len1=8):
    x1.Title("xjlb", "STOT_CASH_INFLOWS_OPER_ACT")
    x1.initfactor = x1.InitFactor()
    CASH_RECP_SG_AND_RS = x1.apply_initial_data("ttm")  # 经营活动现金流入小计
    x1.Title("xjlb", "STOT_CASH_OUTFLOWS_OPER_ACT")
    x1.initfactor = x1.InitFactor()
    NET_INCR_LENDING_FUND = x1.apply_initial_data("ttm")  # 经营活动现金流出小计
    x1.initfactor = CASH_RECP_SG_AND_RS / NET_INCR_LENDING_FUND
    f1expanding = x1.FactorExpanding()
    factorDF = x1.ExpandedFactor2DataFrame(f1expanding)
    x1.initfactor = x1.FinDataRollingZScores(len1)
    f1expanding = x1.FactorExpanding()
    factorDF_zscores = x1.ExpandedFactor2DataFrame(f1expanding)
    return factorDF, factorDF_zscores


def net_profit_FIN_EXP_CS_ratio(x1, len1=8):
    x1.Title("xjlb", "FIN_EXP_CS")
    x1.initfactor = x1.InitFactor()
    FIN_EXP_CS = x1.apply_initial_data("ttm")  # 财务费用,但是只有年报和半年报
    x1.Title("xjlb", "NET_PROFIT_CS")
    x1.initfactor = x1.InitFactor()
    NETPROFIT = x1.apply_initial_data("ttm")  # 净利润,但是只有年报和半年报
    x1.initfactor = FIN_EXP_CS / NETPROFIT
    f1expanding = x1.FactorExpanding()
    factorDF = x1.ExpandedFactor2DataFrame(f1expanding)
    x1.initfactor = x1.FinDataRollingZScores(len1)
    f1expanding = x1.FactorExpanding()
    factorDF_zscores = x1.ExpandedFactor2DataFrame(f1expanding)
    return factorDF, factorDF_zscores


def none_linear_marketcap(MarketCap):  # 非线性市值因子
    logmc = np.log(MarketCap)
    logmc3 = logmc**3
    data = pd.DataFrame({"logmc": logmc, "logmc3": logmc3})
    logmcustack = data["logmc"].unstack()
    logmc3ustack = data["logmc3"].unstack()
    TradingDates = logmcustack.index
    StockCodes = logmcustack.columns
    none_linear_np = np.ones((len(TradingDates), len(StockCodes))) * np.nan
    np1 = logmcustack.values
    np2 = logmc3ustack.values
    for i in range(len(TradingDates)):
        temp1 = np1[i, :]
        temp2 = np2[i, :]
        nanmask = np.isnan(temp1)
        x = temp1[~nanmask]
        y = temp2[~nanmask]
        if len(x) == 0:
            continue
        model = sm.OLS(y, sm.add_constant(x), missing="drop")
        results = model.fit()
        none_linear_np[i, ~nanmask] = results.resid
    none_linear_marketcap_pd = pd.DataFrame(
        none_linear_np, index=TradingDates, columns=StockCodes
    )
    none_linear_marketcap_pd = none_linear_marketcap_pd.shift(1).stack()
    return none_linear_marketcap_pd


def log3marketcap(MarketCap):
    logmc = np.log(MarketCap)
    logmc3 = logmc**3
    return logmc3


def IndexComponentWeight(IndexComponent, FreeMarketCap):
    weightdata = FreeMarketCap.loc[IndexComponent.index]
    weightdata1 = (
        weightdata.groupby(level=0, group_keys=False)
        .apply(lambda x: x / x.sum())
        .to_frame(name="weight")
    )

    return weightdata1


def BetaFactorbyIndex(PriceDf, indexname, window=[20, 60, 120, 240]):
    benchmark = SDP.IndexPoint(indexname)
    benchmark["tradingday"] = pd.to_datetime(benchmark["tradingday"], format="%Y%m%d")
    benchmark = benchmark.rename(columns={"tradingday": "TradingDates"})
    benchmark.set_index("TradingDates", inplace=True)
    benchmarkreturn = np.log(benchmark["c"] / benchmark["c"].shift(1)).to_frame(
        name="benchmarkreturn"
    )
    stockreturn = (
        PriceDf.groupby(level=1, group_keys=False)
        .apply(
            lambda x: np.log(
                (x["c"] * x["adjfactor"]) / (x["c"].shift(1) * x["adjfactor"].shift(1))
            )
        )
        .sort_index(level=0)
    )
    stockreturn = stockreturn.unstack()
    stockreturn = stockreturn.fillna(0)
    data = pd.concat([stockreturn, benchmarkreturn], axis=1)
    data = data.loc[benchmarkreturn.index]
    data = data.loc[stockreturn.index]
    nan_rows = stockreturn[stockreturn.isna().all(axis=1)]
    nan_index = nan_rows.index
    data = data.drop(nan_index)
    columns_without_benchmarkreturn = [
        col for col in data.columns if col != "benchmarkreturn"
    ]
    betas0 = np.zeros((len(data.index), len(columns_without_benchmarkreturn)))
    betas1 = np.zeros((len(data.index), len(columns_without_benchmarkreturn)))
    betas2 = np.zeros((len(data.index), len(columns_without_benchmarkreturn)))
    betas3 = np.zeros((len(data.index), len(columns_without_benchmarkreturn)))

    resid0 = np.zeros((len(data.index), len(columns_without_benchmarkreturn)))
    resid1 = np.zeros((len(data.index), len(columns_without_benchmarkreturn)))
    resid2 = np.zeros((len(data.index), len(columns_without_benchmarkreturn)))
    resid3 = np.zeros((len(data.index), len(columns_without_benchmarkreturn)))

    datanp = data.to_numpy()

    for i in range(len(data.index)):
        print(data.index[i])
        if data.index[i] <= pd.Timestamp("2018-01-01"):
            continue
        if i >= window[0]:
            Y0 = datanp[i - window[0] : i, -1]  # 左开右闭不用移动
            for j, col in enumerate(columns_without_benchmarkreturn):
                X = datanp[i - window[0] : i, j]
                X = sm.add_constant(X)
                model = sm.OLS(Y0, X, missing="drop")
                results = model.fit()
                betas0[i, j] = results.params[1]
                resid0[i, j] = results.resid[-1]
        if i >= window[1]:
            Y1 = datanp[i - window[1] : i, -1]
            for j, col in enumerate(columns_without_benchmarkreturn):
                X = datanp[i - window[1] : i, j]
                X = sm.add_constant(X)
                model = sm.OLS(Y1, X, missing="drop")
                results = model.fit()
                betas1[i, j] = results.params[1]
                resid1[i, j] = results.resid[-1]
        if i >= window[2]:
            Y2 = datanp[i - window[2] : i, -1]
            for j, col in enumerate(columns_without_benchmarkreturn):
                X = datanp[i - window[2] : i, j]
                X = sm.add_constant(X)
                model = sm.OLS(Y2, X, missing="drop")
                results = model.fit()
                betas2[i, j] = results.params[1]
                resid2[i, j] = results.resid[-1]
        if i >= window[3]:
            Y3 = datanp[i - window[3] : i, -1]
            for j, col in enumerate(columns_without_benchmarkreturn):
                X = datanp[i - window[3] : i, j]
                X = sm.add_constant(X)
                model = sm.OLS(Y3, X, missing="drop")
                results = model.fit()
                betas3[i, j] = results.params[1]
                resid3[i, j] = results.resid[-1]

    BetaFactor0 = (
        pd.DataFrame(betas0, index=data.index, columns=columns_without_benchmarkreturn)
        .stack()
        .to_frame(name="Beta_" + str(window[0]) + "days")
    )
    BetaFactor0.index.names = ["TradingDates", "StockCodes"]
    BetaFactor1 = (
        pd.DataFrame(betas1, index=data.index, columns=columns_without_benchmarkreturn)
        .stack()
        .to_frame(name="Beta_" + str(window[1]) + "days")
    )
    BetaFactor1.index.names = ["TradingDates", "StockCodes"]
    BetaFactor2 = (
        pd.DataFrame(betas2, index=data.index, columns=columns_without_benchmarkreturn)
        .stack()
        .to_frame(name="Beta_" + str(window[2]) + "days")
    )
    BetaFactor2.index.names = ["TradingDates", "StockCodes"]
    BetaFactor3 = (
        pd.DataFrame(betas3, index=data.index, columns=columns_without_benchmarkreturn)
        .stack()
        .to_frame(name="Beta_" + str(window[3]) + "days")
    )
    BetaFactor3.index.names = ["TradingDates", "StockCodes"]

    ResidFactor0 = (
        pd.DataFrame(resid0, index=data.index, columns=columns_without_benchmarkreturn)
        .stack()
        .to_frame(name="Resid_" + str(window[0]) + "days")
    )
    ResidFactor0.index.names = ["TradingDates", "StockCodes"]
    ResidFactor1 = (
        pd.DataFrame(resid1, index=data.index, columns=columns_without_benchmarkreturn)
        .stack()
        .to_frame(name="Resid_" + str(window[1]) + "days")
    )
    ResidFactor1.index.names = ["TradingDates", "StockCodes"]
    ResidFactor2 = (
        pd.DataFrame(resid2, index=data.index, columns=columns_without_benchmarkreturn)
        .stack()
        .to_frame(name="Resid_" + str(window[2]) + "days")
    )
    ResidFactor2.index.names = ["TradingDates", "StockCodes"]
    ResidFactor3 = (
        pd.DataFrame(resid3, index=data.index, columns=columns_without_benchmarkreturn)
        .stack()
        .to_frame(name="Resid_" + str(window[3]) + "days")
    )
    ResidFactor3.index.names = ["TradingDates", "StockCodes"]
    return (
        BetaFactor0,
        BetaFactor1,
        BetaFactor2,
        BetaFactor3,
        ResidFactor0,
        ResidFactor1,
        ResidFactor2,
        ResidFactor3,
    )  # 20,60,120,240


def BetafactorbyIndexWLS(PriceDf, indexname, halfdecay=60, window=252):
    assert halfdecay <= 0.8 * window, "halfdecay should be larger than 0.5*window"
    t = np.arange(window)[::-1]
    # 生成权重序列
    weights = (0.5) ** (t / halfdecay)
    weights = weights / np.sum(weights)
    benchmark = SDP.IndexPoint(indexname)
    benchmark["tradingday"] = pd.to_datetime(benchmark["tradingday"], format="%Y%m%d")
    benchmark = benchmark.rename(columns={"tradingday": "TradingDates"})
    benchmark.set_index("TradingDates", inplace=True)
    benchmarkreturn = np.log(benchmark["c"] / benchmark["c"].shift(1)).to_frame(
        name="benchmarkreturn"
    )
    begindate = pd.Timestamp("2015-01-01")
    stockreturn = (
        PriceDf.groupby(level=1, group_keys=False)
        .apply(
            lambda x: np.log(
                (x["c"] * x["adjfactor"]) / (x["c"].shift(1) * x["adjfactor"].shift(1))
            )
        )
        .sort_index(level=0)
    ).to_frame(name="stockreturn")

    stockreturn = stockreturn.join(PriceDf["v"]).unstack()
    stockreturn = stockreturn.loc[begindate:]
    benchmarkreturn = benchmarkreturn.loc[begindate:]
    r = stockreturn["stockreturn"]
    v = stockreturn["v"]
    dateindex = r.index
    bench_np = benchmarkreturn.loc[dateindex].values.flatten()
    stockcodes = r.columns
    r_np = r.to_numpy()
    v_np = v.to_numpy()

    temps = np.nan * np.ones((window, len(stockcodes)))
    idx0 = np.zeros(len(stockcodes)).astype(int)
    tempinputlag = np.zeros(len(stockcodes)).astype(int)
    tempinputlagmatrix = -1 * np.ones((window, len(stockcodes))).astype(int)
    linspace1 = np.linspace(0, window - 1, window).astype(int)[::-1]

    alfa_matrix = np.ones((len(dateindex), len(stockcodes))) * np.nan  # 存储alpha
    beta_matrix = np.ones((len(dateindex), len(stockcodes))) * np.nan  # 存储beta
    resid_matrix = np.ones((len(dateindex), len(stockcodes))) * np.nan  # 存储残差
    resid_std_matrix = (
        np.ones((len(dateindex), len(stockcodes))) * np.nan
    )  # 存储残差标准差
    for i in range(len(dateindex)):
        linspace2 = i - linspace1
        print(dateindex[i])
        for j in range(len(stockcodes)):

            if v_np[i, j] > 0:
                tempinputlag[j] = 0  # 如果本日有交易量 ，则将tempinputlag置为0
                if idx0[j] < window:  # 填充位置小于窗口长度，说明动态窗口数据未填充满
                    temps[idx0[j], j] = r_np[i, j]
                    tempinputlagmatrix[idx0[j], j] = tempinputlag[j]
                    idx0[j] += 1
                else:  # 动态窗口数据已经填充满，将数据往前移动一位，最后一位填充新数据
                    temps[:-1, j] = temps[1:, j]
                    temps[-1, j] = r_np[i, j]
                    tempinputlagmatrix[:-1, j] = tempinputlagmatrix[1:, j]
                    tempinputlagmatrix[-1, j] = tempinputlag[j]
            else:  # 如果本日无交易量
                tempinputlag[j] = (
                    tempinputlag[j] + 1
                )  # 当前lag记录+1， 说明当前股票已经连续多少天没有交易
                if idx0[j] < window:
                    tempinputlagmatrix[idx0[j], j] = tempinputlag[j]
                else:
                    tempinputlagmatrix[:-1, j] = tempinputlagmatrix[1:, j]
                    tempinputlagmatrix[-1, j] = tempinputlag[j]
            if ~np.any(np.isnan(temps[:, j])):  # 如果动态窗口数据已经填充满
                benchmarkidx = linspace2 - tempinputlagmatrix[:, j][::-1].cumsum()[::-1]
                if benchmarkidx[0] >= 0:
                    tempbench = bench_np[benchmarkidx]
                    X = sm.add_constant(temps[:, j])
                    model = sm.WLS(tempbench, X, weights=weights)
                    results = model.fit()
                    alfa_matrix[i, j] = results.params[0]
                    beta_matrix[i, j] = results.params[1]
                    resid_matrix[i, j] = results.resid[-1]
                    resid_std_matrix[i, j] = results.resid.std()

    Alfa_Df = (
        pd.DataFrame(alfa_matrix, index=dateindex, columns=stockcodes)
        .stack()
        .to_frame(name="Alfa")
    )
    Alfa_Df.index.names = ["TradingDates", "StockCodes"]
    Beta_Df = (
        pd.DataFrame(beta_matrix, index=dateindex, columns=stockcodes)
        .stack()
        .to_frame(name="Beta")
    )
    Beta_Df.index.names = ["TradingDates", "StockCodes"]
    Resid_Df = (
        pd.DataFrame(resid_matrix, index=dateindex, columns=stockcodes)
        .stack()
        .to_frame(name="Resid")
    )
    Resid_Df.index.names = ["TradingDates", "StockCodes"]
    ResidStd_Df = (
        pd.DataFrame(resid_std_matrix, index=dateindex, columns=stockcodes)
        .stack()
        .to_frame(name="ResidStd")
    )
    ResidStd_Df.index.names = ["TradingDates", "StockCodes"]
    return Alfa_Df, Beta_Df, Resid_Df, ResidStd_Df


def run_Index_regression(
    PriceDf,
    datasavepath=r"E:\Documents\PythonProject\StockProject\StockData\RawFactors",
) -> None:
    (
        zz2000_252_60_alfa,
        zz2000_252_60_beta,
        zz2000_252_60_resid,
        zz2000_252_60_residstd,
    ) = BetafactorbyIndexWLS(PriceDf, "中证2000", halfdecay=60, window=252)
    pd.to_pickle(zz2000_252_60_alfa, datasavepath + r"\zz2000_252_60_alfa.pkl")
    pd.to_pickle(zz2000_252_60_beta, datasavepath + r"\zz2000_252_60_beta.pkl")
    pd.to_pickle(zz2000_252_60_resid, datasavepath + r"\zz2000_252_60_resid.pkl")
    pd.to_pickle(zz2000_252_60_residstd, datasavepath + r"\zz2000_252_60_residstd.pkl")
    (
        zz2000_126_30_alfa,
        zz2000_126_30_beta,
        zz2000_126_30_resid,
        zz2000_126_30_residstd,
    ) = BetafactorbyIndexWLS(PriceDf, "中证2000", halfdecay=30, window=126)
    pd.to_pickle(zz2000_252_60_alfa, datasavepath + r"\zz2000_126_30_alfa.pkl")
    pd.to_pickle(zz2000_252_60_beta, datasavepath + r"\zz2000_126_30_beta.pkl")
    pd.to_pickle(zz2000_252_60_resid, datasavepath + r"\zz2000_126_30_resid.pkl")
    pd.to_pickle(zz2000_252_60_residstd, datasavepath + r"\zz2000_126_30_residstd.pkl")
    return


def run_IndexBeta(
    PriceDf,
    datasavepath=r"E:\Documents\PythonProject\StockProject\StockData\RawFactors",
) -> None:
    (
        zz2000_20day_beta,
        zz2000_60day_beta,
        zz2000_120day_beta,
        zz2000_240day_beta,
        zz2000_20day_resid,
        zz2000_60day_resid,
        zz2000_120day_resid,
        zz2000_240day_resid,
    ) = BetaFactorbyIndex(PriceDf, "中证2000")
    pd.to_pickle(zz2000_20day_beta, datasavepath + r"\zz2000_20day_beta.pkl")
    pd.to_pickle(zz2000_60day_beta, datasavepath + r"\zz2000_60day_beta.pkl")
    pd.to_pickle(zz2000_120day_beta, datasavepath + r"\zz2000_120day_beta.pkl")
    pd.to_pickle(zz2000_240day_beta, datasavepath + r"\zz2000_240day_beta.pkl")

    pd.to_pickle(zz2000_20day_resid, datasavepath + r"\zz2000_20day_resid.pkl")
    pd.to_pickle(zz2000_60day_resid, datasavepath + r"\zz2000_60day_resid.pkl")
    pd.to_pickle(zz2000_120day_resid, datasavepath + r"\zz2000_120day_resid.pkl")
    pd.to_pickle(zz2000_240day_resid, datasavepath + r"\zz2000_240day_resid.pkl")

    (
        zz500_20day_beta,
        zz500_60day_beta,
        zz500_120day_beta,
        zz500_240day_beta,
        zz500_20day_resid,
        zz500_60day_resid,
        zz500_120day_resid,
        zz500_240day_resid,
    ) = BetaFactorbyIndex(PriceDf, "中证500")
    pd.to_pickle(zz500_20day_beta, datasavepath + r"\zz500_20day_beta.pkl")
    pd.to_pickle(zz500_60day_beta, datasavepath + r"\zz500_60day_beta.pkl")
    pd.to_pickle(zz500_120day_beta, datasavepath + r"\zz500_120day_beta.pkl")
    pd.to_pickle(zz500_240day_beta, datasavepath + r"\zz500_240day_beta.pkl")

    pd.to_pickle(zz500_20day_resid, datasavepath + r"\zz500_20day_resid.pkl")
    pd.to_pickle(zz500_60day_resid, datasavepath + r"\zz500_60day_resid.pkl")
    pd.to_pickle(zz500_120day_resid, datasavepath + r"\zz500_120day_resid.pkl")
    pd.to_pickle(zz500_240day_resid, datasavepath + r"\zz500_240day_resid.pkl")
    return


def run_finacialfactors(
    x1, datasavepath, datapath=r"E:\Documents\PythonProject\StockProject\StockData"
) -> None:
    MarketCap = pd.read_pickle(datasavepath + "\\" + "MarketCap.pkl")
    none_linear_marketcap_data = none_linear_marketcap(MarketCap)
    pd.to_pickle(
        none_linear_marketcap_data, datasavepath + r"\none_linear_marketcap.pkl"
    )  # 非线性市值因子
    log3marketcap_data = log3marketcap(MarketCap)
    pd.to_pickle(
        log3marketcap_data, datasavepath + r"\log3marketcap.pkl"
    )  # log3市值因子

    factorDF = EP_ttm(x1, MarketCap)
    pd.to_pickle(factorDF, datasavepath + r"\EP_ttm.pkl")
    factorDF = EP_ss(x1, MarketCap)
    pd.to_pickle(factorDF, datasavepath + r"\EP_ss.pkl")
    factorDF = BP(x1, MarketCap)
    pd.to_pickle(factorDF, datasavepath + r"\BP.pkl")
    factorDF = SP_ttm(x1, MarketCap)
    pd.to_pickle(factorDF, datasavepath + r"\SP_ttm.pkl")
    factorDF = SP_ss(x1, MarketCap)
    pd.to_pickle(factorDF, datasavepath + r"\SP_ss.pkl")
    factorDF, factorDF_zscores = DEDUCTEDPROFIT_yoy(x1)
    pd.to_pickle(factorDF, datasavepath + r"\DEDUCTEDPROFIT_yoy.pkl")
    pd.to_pickle(factorDF_zscores, datasavepath + r"\DEDUCTEDPROFIT_yoy_zscores_4.pkl")
    factorDF = RoE(x1)
    pd.to_pickle(factorDF, datasavepath + r"\ROE.pkl")
    factorDF = RoE_zscores(x1, r_method="ttm", e_method="avg", len1=8)
    pd.to_pickle(factorDF, datasavepath + r"\ROEzscores_8.pkl")
    factorDF = RoE_zscores(x1, r_method="ttm", e_method="avg", len1=4)
    pd.to_pickle(factorDF, datasavepath + r"\ROEzscores_4.pkl")
    factorDF, factorDF_zscores = RoE_ratio(x1)
    pd.to_pickle(factorDF, datasavepath + r"\ROE_ratio.pkl")
    pd.to_pickle(factorDF_zscores, datasavepath + r"\ROE_ratio_zscores_4.pkl")
    factorDF, factorDF_zscores = NET_CASH_FLOWS_OPER_ACT_yoy(
        x1, 4
    )  # 经营活动产生的现金流量净额同比增长率
    pd.to_pickle(factorDF, datasavepath + r"\NET_CASH_FLOWS_OPER_ACT_yoy_4.pkl")
    pd.to_pickle(
        factorDF_zscores, datasavepath + r"\NET_CASH_FLOWS_OPER_ACT_yoy_zscores_4.pkl"
    )
    factorDF, factorDF_zscores = OPER_REV_yoy(x1, 4)  # 营业收入同比增长率
    pd.to_pickle(factorDF, datasavepath + r"\OPER_REV_yoy_4.pkl")
    pd.to_pickle(factorDF_zscores, datasavepath + r"\OPER_REV_yoy_zscores_4.pkl")
    factorDF, factorDF_zscores = STOT_CASH_INFLOWS_OPER_ACT_yoy(
        x1, 4
    )  # 经营活动现金流入小计同比增长率
    pd.to_pickle(factorDF, datasavepath + r"\STOT_CASH_INFLOWS_OPER_ACT_yoy_4.pkl")
    pd.to_pickle(
        factorDF_zscores,
        datasavepath + r"\STOT_CASH_INFLOWS_OPER_ACT_yoy_zscores_4.pkl",
    )
    factorDF, factorDF_zscores = FreeCashflow_ratio(x1, 4)  # 自由现金流入小计同比增长率
    pd.to_pickle(factorDF, datasavepath + r"\FreeCashflow_ratio.pkl")
    pd.to_pickle(factorDF_zscores, datasavepath + r"\FreeCashflow_ratio_zscores_4.pkl")

    factorDF = SUE_ss(x1, 4)
    pd.to_pickle(factorDF, datasavepath + r"\SUE_ss_4.pkl")
    factorDF = SUE_ttm(x1, 4)
    pd.to_pickle(factorDF, datasavepath + r"\SUE_ttm_4.pkl")
    factorDF = SUE_yoy(x1, 4)
    pd.to_pickle(factorDF, datasavepath + r"\SUE_yoy_4.pkl")
    factorDF = SUE_qoq(x1, 4)
    pd.to_pickle(factorDF, datasavepath + r"\SUE_qoq_4.pkl")
    factorDF, factorDF_zscores = CurrentRatio(x1, 8)
    pd.to_pickle(factorDF, datasavepath + r"\CurrentRatio.pkl")
    pd.to_pickle(factorDF_zscores, datasavepath + r"\CurrentRatio_zscores_8.pkl")
    factorDF, factorDF_zscores = QuickRatio(x1, 8)
    pd.to_pickle(factorDF, datasavepath + r"\QuickRatio.pkl")
    pd.to_pickle(factorDF_zscores, datasavepath + r"\QuickRatio_zscores_8.pkl")
    factorDF, factorDF_zscores = GOODWILLRatio(x1, 8)
    pd.to_pickle(factorDF, datasavepath + r"\GOODWILLRatio.pkl")
    pd.to_pickle(factorDF_zscores, datasavepath + r"\GOODWILLRatio_zscores_8.pkl")
    factorDF, factorDF_zscores = ACCT_RCV__NET_CASH_FLOWS_OPER_ACT(x1, 8)
    pd.to_pickle(factorDF, datasavepath + r"\ACCT_RCV__NET_CASH_FLOWS_OPER_ACT.pkl")
    pd.to_pickle(
        factorDF_zscores,
        datasavepath + r"\ACCT_RCV__NET_CASH_FLOWS_OPER_ACT_zscores_8.pkl",
    )
    factorDF, factorDF_zscores = wet_profit_ratio(x1, 8)
    pd.to_pickle(factorDF, datasavepath + r"\wet_profit_ratio.pkl")
    pd.to_pickle(factorDF_zscores, datasavepath + r"\wet_profit_ratio_zscores_8.pkl")
    realead_dates_count_df = pd.read_pickle(
        datapath + "\\" + "realesed_dates_count_df.pkl"
    )
    SUE_ss_4 = pd.read_pickle(datasavepath + r"\SUE_ss_4.pkl")
    SUE_ss_4_hd5 = half_decay_factor(
        SUE_ss_4, realead_dates_count_df, para=5, show=False
    )
    pd.to_pickle(SUE_ss_4_hd5, datasavepath + r"\SUE_ss_4_hd5.pkl")
    testdata1 = pd.read_pickle(datasavepath + r"\ROE.pkl")
    testdata2 = pd.read_pickle(datasavepath + r"\CurrentRatio.pkl")
    ROE_d_Currentratio = testdata1 / testdata2
    pd.to_pickle(ROE_d_Currentratio, datasavepath + r"\ROE_d_Currentratio.pkl")

    gw = pd.read_pickle(datasavepath + r"\GOODWILLRatio.pkl")
    qr = pd.read_pickle(datasavepath + r"\QuickRatio.pkl")
    gw_divide_qr = gw / qr
    pd.to_pickle(gw_divide_qr, datasavepath + r"\GOODWILLRatio_divide_QuickRatio.pkl")

    ep = pd.read_pickle(datasavepath + r"\EP_ttm.pkl")
    sp = pd.read_pickle(datasavepath + r"\SP_ttm.pkl")
    g = pd.read_pickle(datasavepath + r"\DEDUCTEDPROFIT_yoy.pkl")
    PEG = ep * g
    pd.to_pickle(PEG, datasavepath + r"\PEG.pkl")
    spg = sp * (g + 1)
    pd.to_pickle(spg, datasavepath + r"\spg.pkl")
    roe = pd.read_pickle(datasavepath + r"\ROEzscores_4.pkl")
    roe_d_g = roe * g
    pd.to_pickle(roe_d_g, datasavepath + r"\ROEzscores_4_t_profitgrowth.pkl")
    bp = pd.read_pickle(datasavepath + r"\BP.pkl")
    df = bp * (g + 1)
    pd.to_pickle(df, datasavepath + r"\BP_t_profitgrowth.pkl")

    none_linear_marketcap_data = pd.read_pickle(
        datasavepath + r"\none_linear_marketcap.pkl"
    )
    df = roe_d_g * 100 / none_linear_marketcap_data
    pd.to_pickle(
        df,
        datasavepath + r"\ROEzscores_4_t_profitgrowth_divide_none_linear_marketcap.pkl",
    )

    return


def Techfactors():
    pass


def run_main():
    datasavepath = (
        r"E:\Documents\PythonProject\StockProject\StockData\RawFactors"  # 原始因子存放
    )
    datapath = r"E:\Documents\PythonProject\StockProject\StockData"
    today = datetime.today()
    int_today = int(today.strftime("%Y%m%d"))
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
    StockDataDF = PriceDf.copy()
    StockDataDF.index = PriceDf.index.set_names(["tradingday", "code"])
    StockDataDF = StockDataDF.reset_index()
    StockDataDF["tradingday"] = StockDataDF["tradingday"].dt.strftime("%Y%m%d")
    columns = [
        "tradingday",
        "code",
        "o",
        "h",
        "l",
        "c",
        "v",
        "amt",
        "adjfactor",
        "total_shares",
        "free_float_shares",
        "MC",
        "FMC",
        "turnoverrate",
        "vwap",
        "freeturnoverrate",
    ]
    StockDataDF = StockDataDF[columns]
    StockData3dMatrix = SDP.StockDataDF2Matrix(StockDataDF)
    StockDataDF = None

    cla = FinData.财报数据()
    cla.复制三张表()
    begindate = 20140101
    cla.读取财报数据(begindate, int_today)
    x1 = SDP.FactorMatrix_Report(StockData3dMatrix, cla)
    a = time()
    run_Index_regression(PriceDf)
    b = time()
    print(b - a)
    # run_IndexBeta(PriceDf)
    PriceDf = None
    run_finacialfactors(x1, datasavepath)

    return


def factor_releaseddates_(factor, releaseddata):
    if isinstance(factor, pd.DataFrame):
        factor.columns = ["factor"]
    else:
        factor = factor.to_frame(name="factor")
    releasedmin = releaseddata.min(axis=1).to_frame(name="releasedmin")
    factor1 = factor.join(releasedmin, how="left")
    newfactor = (factor1["factor"] / factor1["releasedmin"]).to_frame(name="factor")
    return newfactor


def half_decay_factor(factor, releaseddata, para=20, show=False):
    if isinstance(factor, pd.DataFrame):
        factor.columns = ["factor"]
    else:
        factor = factor.to_frame(name="factor")
    logbase = np.exp(-np.log(2) / para)
    minvalues = releaseddata.min(axis=1).to_frame(name="releasedmin")
    minvalues[minvalues < 0] = np.nan
    factor1 = factor.join(minvalues, how="left")
    factor1["indice"] = logbase ** factor1["releasedmin"]
    df = factor1["factor"] * factor1["indice"]
    return df.to_frame(name="factor")


if __name__ == "__main__":
    print("制作因子 start")
    SDP.Main_Data_Renew()  # StockDataPrepairing中的数据更新
    run_main()
    alpha191.Main_Data_Renew()
    # datasavepath = (
    #     r"E:\Documents\PythonProject\StockProject\StockData\RawFactors"  # 原始因子存放
    # )
    # datapath = r"E:\Documents\PythonProject\StockProject\StockData"
    # today = datetime.today()
    # int_today = int(today.strftime("%Y%m%d"))
    # PriceDf = pd.read_pickle(datapath + "\\" + "Price.pkl")
    # StockTradableDF = pd.read_pickle(datapath + "\\" + "TradableDF.pkl")
    # PriceDf = PriceDf[~(PriceDf["exchange_id"] == "BJ")]  # 踢出北交所
    # StockTradableDF = StockTradableDF[
    #     ~(StockTradableDF["exchange_id"] == "BJ")
    # ]  # 踢出北交所

    # PriceDf = PriceDf.join(
    #     StockTradableDF, how="left", lsuffix="_left", rsuffix="_right"
    # )
    # PriceDf = PriceDf[~(PriceDf["trade_status"] == "退市")]
    # BetafactorbyIndexWLS(PriceDf, '中证2000', halfdecay=60, window=252)

    # zz500 = SDP.GetWideBaseByDateSerries(
    #     alldates, 指数类型="沪深交易所核心指数", 指数code="000905"
    # )
    # zz500Weight = IndexComponentWeight(zz500, FreeMarketCap)
    # pd.to_pickle(zz500Weight, datasavepath + r"\zz500Weight.pkl")
    # hs300 = SDP.GetWideBaseByDateSerries(
    #     alldates, 指数类型="沪深交易所核心指数", 指数code="000300"
    # )
    # hs300Weight = IndexComponentWeight(hs300, FreeMarketCap)
    # pd.to_pickle(hs300Weight, datasavepath + r"\hs300Weight.pkl")

    # hs300dayk = SDP.IndexPoint(indexname="沪深300")
    # zz500dayk = SDP.IndexPoint(indexname="中证500")
