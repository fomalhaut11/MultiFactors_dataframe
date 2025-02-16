import psutil
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import configparser
import os
from scipy.stats import spearmanr, kendalltau, t, shapiro, norm
import matplotlib.pyplot as plt
import pickle

from multiprocessing import Pool
from types import SimpleNamespace
import sys

file_path = r"E:\Documents\PythonProject\StockProject"
sys.path.append(file_path)
import StockDataPrepairing as SDP
import statsmodels.api as sm

file_path1 = r"E:\Documents\PythonProject\StockProject\MultiFactors"
sys.path.append(file_path1)
import factorsmaking as fm

# import alpha191backtest_factor as alpha191


def print_memory_usage():
    # 当前内存占用
    process = psutil.Process(os.getpid())
    print("Memory Usage: %.2f MB" % (process.memory_info().rss / 1024 / 1024))


def Remove_Outlier(input_x, method="mean", para=3):

    x = input_x.astype(
        float
    )  # 使用.astype(float)将数据转换为浮点型，则是已经创建了一个新的对象。
    if isinstance(x, np.ndarray):
        xmax = np.nanmax(x)
        xmin = np.nanmin(x)
        xmedian = np.nanmedian(x)
        x[np.isposinf(x)] = xmax
        x[np.isneginf(x)] = xmin
        x[np.isnan(x)] = xmedian
        if method == "IQR":
            medianvalue = np.nanmedian(x)
            Q1 = np.percentile(x, 5)
            Q3 = np.percentile(x, 95)
            IQR = Q3 - Q1
            x[x > Q3 + para * IQR] = Q3 + para * IQR
            x[x < Q1 - para * IQR] = Q1 - para * IQR
        if method == "median":
            medianvalue = np.nanmedian(x)
            mad = np.nanmedian(np.abs(x - medianvalue))
            x[x - medianvalue > para * mad] = para * mad
            x[x - medianvalue < -para * mad] = -para * mad
        elif method == "mean":
            meanvalue = np.nanmean(x)
            std = np.std(x)
            x[x - meanvalue > para * std] = para * std
            x[x - meanvalue < -para * std] = -para * std
    else:
        x = x.copy()
        if method == "IQR":
            medianvalue = x[np.isfinite(x)].median()
            Q1 = np.percentile(x[np.isfinite(x)], 5)
            Q3 = np.percentile(x[np.isfinite(x)], 95)
            x.fillna(medianvalue, inplace=True)
            IQR = Q3 - Q1
            x[x > Q3 + para * IQR] = Q3 + para * IQR
            x[x < Q1 - para * IQR] = Q1 - para * IQR
        if method == "median":
            medianvalue = x[np.isfinite(x)].median()
            mad = np.nanmedian(np.abs(x - medianvalue))
            x.fillna(medianvalue, inplace=True)
            x[x - medianvalue > para * mad] = para * mad
            x[x - medianvalue < -para * mad] = -para * mad
        elif method == "mean":
            meanvalue = x[np.isfinite(x)].mean(axis=0)
            std = np.std(x[np.isfinite(x)], axis=0)
            x.fillna(meanvalue, inplace=True)
            x[x - meanvalue > para * std] = para * std
            x[x - meanvalue < -para * std] = -para * std
    return x


def quick_remove_outlier_np(input_x, method="mean", para=3):
    assert isinstance(input_x, np.ndarray), "input_x is not np.ndarray"
    input_x = input_x.astype(float)
    meanvalue = np.nanmean(input_x)
    input_x[np.isnan(input_x) | np.isinf(input_x)] = meanvalue

    if method == "IQR":
        medianvalue = np.median(input_x)
        Q1 = np.percentile(input_x, 5)
        Q3 = np.percentile(input_x, 95)
        IQR = Q3 - Q1
        input_x[input_x > Q3 + para * IQR] = Q3 + para * IQR
        input_x[input_x < Q1 - para * IQR] = Q1 - para * IQR
    elif method == "median":
        medianvalue = np.median(input_x)
        mad = np.median(np.abs(input_x - medianvalue))
        input_x[input_x - medianvalue > para * mad] = para * mad
        input_x[input_x - medianvalue < -para * mad] = -para * mad
    elif method == "mean":
        meanvalue = np.mean(input_x)
        std = np.std(input_x)
        input_x[input_x - meanvalue > para * std] = para * std
        input_x[input_x - meanvalue < -para * std] = -para * std
    return input_x


def Normlization(x_input, method="zscore"):
    # 标准化
    x = x_input.copy()
    if isinstance(x, pd.core.frame.DataFrame):
        if x.std().item() <= 0.000001:
            x = x * 0
        else:
            if method == "zscore":
                x = (x - x.mean()) / x.std()
            if method == "ppf":
                cdf_values = (np.argsort(np.argsort(x)) + 0.5) / len(x)
                x = norm.ppf(cdf_values)
    if isinstance(x, pd.core.series.Series):
        if x.std() <= 0.000001:
            x = x * 0
        else:
            if method == "zscore":
                x = (x - x.mean()) / x.std()
            if method == "ppf":
                cdf_values = (np.argsort(np.argsort(x)) + 0.5) / len(x)
                x = norm.ppf(cdf_values)
    if isinstance(x, np.ndarray):
        if np.std(x) <= 0.000001:
            x = x * 0
        else:
            if method == "zscore":
                x = (x - np.mean(x)) / np.std(x)
            if method == "ppf":
                cdf_values = (np.argsort(np.argsort(x)) + 0.5) / len(x)
                x = norm.ppf(cdf_values)
    return x


def cal_IC_P_GroupReturn_inday(newfactor, next_log_return, groupnum=10):
    sort_indice = np.argsort(newfactor)
    sorted_factor_group = np.array_split(newfactor[sort_indice], groupnum)
    sorted_return_group = np.array_split(next_log_return[sort_indice], groupnum)
    sorted_indice_group = np.array_split(sort_indice, groupnum)
    mean_return_group = [np.mean(x) for x in sorted_return_group]
    std_return_group = [np.std(x) for x in sorted_return_group]
    mean_factor_group = [np.mean(x) for x in sorted_factor_group]
    IC, P = spearmanr(mean_factor_group, mean_return_group)
    return (
        IC,
        P,
        mean_return_group,
        std_return_group,
        mean_factor_group,
        sorted_indice_group,
    )


def PickupStocksByAmount(
    PriceDF: pd.DataFrame, windows=20, para=0, mc_limit_min=None, mc_limit_max=None
) -> pd.DataFrame:
    """过去5天平均成交额大于para,市值>mclimit,amt>0,l<h,的股票"""
    average_amount = (
        PriceDF.groupby("StockCodes")["amt"]
        .rolling(window=windows)
        .mean()
        .reset_index(level=0, drop=True)
        .shift(1)
    )
    filtered_stocks = average_amount[average_amount > para]
    if mc_limit_min is None:
        pass
    else:
        filtered_stocks = filtered_stocks[PriceDF["MC"] > mc_limit_min]
    if mc_limit_max is None:
        pass
    else:
        filtered_stocks = filtered_stocks[PriceDF["MC"] < mc_limit_max]

    filtered_stocks = filtered_stocks[PriceDF["amt"] > 0]  # 剔除成交额为0的股票
    # filtered_stocks = filtered_stocks[PriceDF['l'] < PriceDF['h']]
    filtered_stocks = filtered_stocks[
        PriceDF["o"] < PriceDF["high_limit"] - 0.02
    ]  # 剔除开盘涨停股票
    filtered_stocks = filtered_stocks[
        PriceDF["o"] > PriceDF["low_limit"] + 0.02
    ]  # 剔除开盘跌停股票
    # 找出所有股票代码的第一个数字小于7的所有数据, 剔除北交所
    filtered_stocks = filtered_stocks[
        filtered_stocks.index.get_level_values("StockCodes").str[0].astype(int) < 7
    ]
    filtered_stocks = filtered_stocks.to_frame().sort_index(level=0)
    return filtered_stocks


def PickupStocksByMarketCap(
    PriceDF: pd.DataFrame, groupnums=10, targetgroup=0
) -> pd.DataFrame:
    """按市值分组，选取市值最小的股票"""
    filtered_stocks = PriceDF
    # 找出所有股票代码的第一个数字小于7的所有数据, 剔除北交所
    filtered_stocks = filtered_stocks[
        filtered_stocks.index.get_level_values("StockCodes").str[0].astype(int) < 7
    ]
    filtered_stocks = filtered_stocks.sort_index(level=0)
    filtered_stocks = filtered_stocks[filtered_stocks["amt"] > 0]  # 剔除成交额为0的股票

    mc = filtered_stocks["MC"]
    mc = mc.groupby("TradingDates").transform(
        lambda x: pd.qcut(x, groupnums, labels=False, duplicates="drop")
    )
    return mc[mc == targetgroup]


def Stock_Filter(PriceDf):
    # 股票池过滤
    pass


def backtest_merged_data(merged_data, groupnums, IndusNetralBase=True, extradates=0):
    """merged_data:具有列名['Factor','LogReturn','BaseFactor1','BaseFactor2',...]
    双重行索引['TraodingDates','StockCodes']的DataFrame
    """
    tradingdates = merged_data.index.get_level_values(0).unique().sort_values()
    base_factor_columns = merged_data.filter(like="BaseFactor").columns
    columns_to_exclude = base_factor_columns.tolist() + ["Factor"] + ["LogReturn"]
    industry_columns = merged_data.drop(columns=columns_to_exclude).columns

    New_Factor = pd.DataFrame(index=merged_data.index, columns=["New_Factor"])
    Resid_Return = pd.DataFrame(
        index=merged_data.index, columns=["Resid_Return"]
    )  # 残差收益率
    """输出数据初始化"""
    analysis_data = pd.DataFrame(
        index=tradingdates,
        columns=[
            "Factor_Return_LongShort",
            "Factor_Return_Regress",
            "Factor_Return_Last",
            "Factor_Return_First",
            "IC_Regress",
            "IC_Grouped",
            "P_Regress",
            "P_Grouped",
            "T_Regress",
        ],
    )  # 因子收益率回测结果
    groups = np.arange(groupnums, dtype=np.int64)
    Grouped_index = pd.MultiIndex.from_product(
        [tradingdates, groups], names=["TradingDates", "Group"]
    )
    Grouped_StockCodes = pd.DataFrame(
        index=Grouped_index, columns=["StockCodes", "ReturnMean", "ReturnStd"]
    )
    paramsdata_list = []
    pvalues_list = []
    tvalues_list = []
    """输出数据初始化"""

    for date in tradingdates:
        print(date)
        if date < pd.to_datetime("2018-06-01", format="%Y-%m-%d"):
            continue
        if extradates > 0:
            days_ago = date - pd.Timedelta(days=extradates)
            m1slice = merged_data.loc[days_ago:date, :].copy()
        else:
            m1slice = merged_data.loc[(date,)].copy()

        factor = m1slice["Factor"].dropna()
        y0 = Remove_Outlier(factor, method="IQR", para=5)
        y0 = Normlization(y0, method="zscore").to_frame()
        y0 = y0.fillna(0)  # 无效值填充为0
        if (
            y0.std().values[0] <= 0.00001
        ):  # 如果因子是布尔值(0,1)离散型，则不做去极值，只做标准化
            y0 = Normlization(factor, method="zscore").to_frame()  # 标准化
            continue
        if (y0 == 0).all().all():  # 当日因子值全为0，不需要进行回归分析
            continue

        if len(base_factor_columns) > 0:
            base_factors = m1slice[base_factor_columns].copy().dropna()
            basesize = len(base_factors.columns)
            common_index = y0.index.intersection(base_factors.index)
            base_factors = base_factors.loc[common_index]
            y0 = y0.loc[common_index]
            if len(industry_columns) > 0:
                one_hot = m1slice[industry_columns].dropna(axis=1, how="all").fillna(0)
                if IndusNetralBase:
                    for j in range(basesize):
                        result_base = sm.OLS(
                            base_factors.iloc[:, j], one_hot.loc[base_factors.index]
                        ).fit()
                        temp_base = result_base.resid
                        temp_base = Remove_Outlier(temp_base, method="IQR", para=3)
                        temp_base = Normlization(temp_base, method="zscore")
                        base_factors.iloc[:, j] = temp_base
                else:
                    for j in range(basesize):
                        base_factors.iloc[:, j] = Remove_Outlier(
                            base_factors.iloc[:, j], method="IQR", para=5
                        )
                        base_factors.iloc[:, j] = Normlization(
                            base_factors.iloc[:, j], method="zscore"
                        )

                X = base_factors.join(
                    one_hot.loc[common_index]
                )  # 基准因子+行业独热编码
                X1 = sm.add_constant(X)  # 添加常数项
            else:
                X1 = sm.add_constant(base_factors)
            result_y = sm.OLS(y0, X1).fit()  # 因子暴露回归
            newfactor = result_y.resid
        else:
            if len(industry_columns) > 0:
                one_hot = m1slice[industry_columns].dropna(axis=1, how="all").fillna(0)
                X1 = sm.add_constant(one_hot)
                result_y = sm.OLS(y0, X1.loc[y0.index]).fit()  # 因子暴露回归
                newfactor = result_y.resid
            else:
                newfactor = y0
                X1 = pd.DataFrame(index=newfactor.index)
        newfactor.name = "newfactor"
        """因子 收益率 回归测试"""
        X2 = sm.add_constant(X1.join(newfactor))
        y2 = m1slice["LogReturn"].loc[X2.index]
        result_logreturn = sm.OLS(y2, X2).fit()
        resid_return = result_logreturn.resid  # 残差收益率
        analysis_data.loc[date, "Factor_Return_Regress"] = result_logreturn.params[
            "newfactor"
        ]
        if np.abs(result_logreturn.params["newfactor"]) > 1:
            print(date)
        analysis_data.loc[date, "T_Regress"] = result_logreturn.tvalues["newfactor"]
        analysis_data.loc[date, "P_Regress"] = result_logreturn.pvalues["newfactor"]
        analysis_data.loc[date, "IC_Regress"] = spearmanr(newfactor, y2)[0]
        """因子 收益率 分组测试 """
        m1slice["Group"] = pd.qcut(
            newfactor, groupnums, labels=False, duplicates="drop"
        )
        grouped = m1slice.groupby("Group")
        groupedmean = grouped["LogReturn"].mean()
        analysis_data.loc[date, "Factor_Return_LongShort"] = (
            groupedmean.loc[max(groupedmean.index)]
            - groupedmean.loc[min(groupedmean.index)]
        )
        m1slice["newfactor"] = newfactor
        groupedfactormean = m1slice.groupby("Group")["newfactor"].mean()
        correlation, p_value = spearmanr(groupedfactormean, groupedmean)
        analysis_data.loc[date, "IC_Grouped"] = correlation
        analysis_data.loc[date, "P_Grouped"] = p_value
        analysis_data.loc[date, "Factor_Return_Last"] = groupedmean.loc[
            max(groupedmean.index)
        ]
        analysis_data.loc[date, "Factor_Return_First"] = groupedmean.loc[
            min(groupedmean.index)
        ]
        groupedmean.index = pd.MultiIndex.from_product(
            [[date], groups], names=["TradingDates", "Group"]
        )
        Grouped_StockCodes.loc[groupedmean.index, "ReturnMean"] = groupedmean
        groupedstd = grouped["LogReturn"].std()
        groupedstd.index = pd.MultiIndex.from_product(
            [[date], groups], names=["TradingDates", "Group"]
        )
        Grouped_StockCodes.loc[groupedstd.index, "ReturnStd"] = groupedstd
        groupedcodes = m1slice.groupby("Group").apply(
            lambda x: x.index.get_level_values("StockCodes").tolist()
        )

        groupedcodes.index = pd.MultiIndex.from_product(
            [[date], groups], names=["TradingDates", "Group"]
        )
        Grouped_StockCodes.loc[groupedcodes.index, "StockCodes"] = groupedcodes
        newfactor.index = pd.MultiIndex.from_product(
            [[date], newfactor.index], names=["TradingDates", "StockCodes"]
        )
        New_Factor.loc[newfactor.index, "New_Factor"] = newfactor
        resid_return.index = pd.MultiIndex.from_product(
            [[date], resid_return.index], names=["TradingDates", "StockCodes"]
        )
        Resid_Return.loc[resid_return.index, "Resid_Return"] = resid_return.values
        paramsdata_daily = {
            "TradingDates": date,
            "params": result_logreturn.params.copy(),
        }
        paramsdata_list.append(paramsdata_daily)
        pvalues_daily = {
            "TradingDates": date,
            "pvalues": result_logreturn.pvalues.copy(),
        }
        pvalues_list.append(pvalues_daily)
        tvalues_daily = {
            "TradingDates": date,
            "tvalues": result_logreturn.tvalues.copy(),
        }
        tvalues_list.append(tvalues_daily)

    return (
        analysis_data,
        New_Factor,
        Resid_Return,
        Grouped_StockCodes,
        paramsdata_list,
        pvalues_list,
        tvalues_list,
    )


def check_listdata(listdata, keys, keys2) -> pd.DataFrame:
    newfactor_df = pd.DataFrame()
    for entry in listdata:
        # 检查'newfactor'键是否存在于字典中
        if keys2 in entry[keys]:
            # 将'newfactor'的值添加到newfactor_list中
            newfactor_df.loc[entry["TradingDates"], keys2] = entry[keys][keys2]
    return newfactor_df


def backtest_nearestmatrix_np(inputdata, NetralBase=True):
    """回测最近邻矩阵"""
    """inputdata{'NearestIndic':NearestIndic,
                 'merged_data':merged_data,
                 'groupnums':groupnums,
                 'merged_data_tradingdates':merged_data_tradingdates,
                 'BeginDate':BeginDate,
                 'EndDate':EndDate,
                 'StockCodes_unstacked':StockCodes_unstacked,
                 'Factor_2dMatrix':Factor_2dMatrix,
                 'BaseFactor_2dMatrix':BaseFactor_2dMatrix
                 'base_data':base_data,
                 'NextLogReturn_2d':NextLogReturn_2d,
                 'NearestNums':NearestNums,
                 'NearestTradingDates':NearestTradingDates,
                 'Intersect_StockCodes':Intersect_StockCodes,
                 'Intersect_StockCodes_indice_StockCodesUnstacked':Intersect_StockCodes_indice_StockCodesUnstacked,
                 'Intersect_StockCodes_indice_NearestStockCodes':Intersect_StockCodes_indice_NearestStockCodes,
                 }"""
    inputdata_ns = SimpleNamespace(**inputdata)
    groupnums = inputdata_ns.groupnums
    backtestindex = np.where(
        (
            inputdata_ns.merged_data_tradingdates
            >= pd.to_datetime(inputdata_ns.BeginDate, format="%Y-%m-%d")
        )
        & (
            inputdata_ns.merged_data_tradingdates
            <= pd.to_datetime(inputdata_ns.EndDate, format="%Y-%m-%d")
        )
    )[0]

    """输出数据初始化"""
    tradingdates = inputdata_ns.merged_data_tradingdates[backtestindex]
    New_Factor_numpy = (
        np.zeros((len(tradingdates), len(inputdata_ns.StockCodes_unstacked))) * np.nan
    )
    standared_basefactor_numpy = (
        np.zeros(
            (
                len(tradingdates),
                len(inputdata_ns.StockCodes_unstacked),
                len(inputdata_ns.base_data.columns),
            )
        )
        * np.nan
    )

    Resid_Return_numpy = (
        np.zeros((len(tradingdates), len(inputdata_ns.StockCodes_unstacked))) * np.nan
    )
    paramsdata_numpy = (
        np.zeros((len(tradingdates), len(inputdata_ns.base_data.columns) + 1)) * np.nan
    )
    pvalues_numpy = (
        np.zeros((len(tradingdates), len(inputdata_ns.base_data.columns) + 1)) * np.nan
    )
    tvalues_numpy = (
        np.zeros((len(tradingdates), len(inputdata_ns.base_data.columns) + 1)) * np.nan
    )
    analysis_data_columns = [
        "Factor_Return_LongShort",
        "Factor_Return_Regress",
        "Factor_Return_Last",
        "Factor_Return_First",
        "IC_Regress",
        "IC_Grouped",
        "P_Regress",
        "P_Grouped",
        "T_Regress",
    ]

    analysis_data_numpy = (
        np.zeros((len(tradingdates), len(analysis_data_columns))) * np.nan
    )
    groups = np.arange(groupnums, dtype=np.int64)
    Grouped_index = pd.MultiIndex.from_product(
        [tradingdates, groups], names=["TradingDates", "Group"]
    )
    Grouped_columns = ["StockCodes", "ReturnMean", "ReturnStd"]

    Grouped_StockCodes_numpy = np.empty(
        (len(tradingdates), len(groups), 3), dtype=object
    )

    """输出数据初始化"""
    for i in range(0, len(backtestindex)):
        today = inputdata_ns.merged_data_tradingdates[backtestindex[i]]
        print(today)
        date_idx_nearestmatrix = np.where(
            inputdata_ns.NearestTradingDates
            <= inputdata_ns.merged_data_tradingdates[backtestindex[i]]
        )[0]
        # 相似度矩阵日期坐标。因为相似度矩阵计算是左闭右开，当天可用
        if len(date_idx_nearestmatrix) == 0:
            print("相似度矩阵数据不足")
            print(inputdata_ns.merged_data_tradingdates[backtestindex[i]])
            continue
        else:
            date_idx_nearestmatrix = date_idx_nearestmatrix[-1]
        # 相似度矩阵日期坐标

        factor = inputdata_ns.Factor_2dMatrix[
            backtestindex[i],
            inputdata_ns.Intersect_StockCodes_indice_StockCodesUnstacked,
        ]  # (合并时已经取了前一日factor和base数据)取出前一日, 所有股票池与相似度矩阵股票交集 因子数据
        factor_nonan_indice = np.argwhere(~np.isnan(factor)).flatten()
        # 今天有self.Intersect_StockCodes_indice_StockCodesUnstacked[factor_nonan_indice]
        factor = factor[factor_nonan_indice]
        # 取出前一日所有股票池因子

        base = []
        for k in range(len(inputdata_ns.base_data.columns)):
            basefactor = inputdata_ns.BaseFactor_2dMatrix[
                k,
                backtestindex[i],
                inputdata_ns.Intersect_StockCodes_indice_StockCodesUnstacked[
                    factor_nonan_indice
                ],
            ]
            # 取出前一日 所有股票池 与相似度矩阵股票交集 基准因子数据
            base.append(basefactor)
        base = np.array(base)  # 取出前一日 所有股票池

        nearest = inputdata_ns.NearestIndice[
            date_idx_nearestmatrix,
            inputdata_ns.Intersect_StockCodes_indice_NearestStockCodes[
                factor_nonan_indice
            ],
            :,
        ]
        # 取出前一日 所有股票池 与相似度矩阵股票交集 最相近的股票索引
        today_stock_codes = inputdata_ns.Intersect_StockCodes[
            inputdata_ns.Intersect_StockCodes_indice_StockCodesUnstacked[
                factor_nonan_indice
            ]
        ]
        # 当前日相似度矩阵
        today_newfactor0 = np.zeros((len(factor_nonan_indice)))
        # 只计算StockUnivers_indice的股票
        today_newBaseFactor = np.zeros(np.shape(base))
        today_newBaseFactor1 = np.zeros(np.shape(base))

        for j in range(len(factor_nonan_indice)):
            Nindice = nearest[j, :]
            Nindice = Nindice[Nindice >= 0].astype(int)
            # 最相近的股票索引(原始索引，在self.NearestStockCodes中)
            if len(Nindice) == 0:
                continue

            indice1 = inputdata_ns.StockCodes_Unstacked_indice_of_Nearest[Nindice]
            indice1 = indice1[indice1 >= 0]
            # 无效值为-1 该组中的股票 所对应的 StockCodes_unstacked 的索引
            I1, i1, i2 = np.intersect1d(
                inputdata_ns.Intersect_StockCodes_indice_StockCodesUnstacked[
                    factor_nonan_indice
                ],
                indice1,
                return_indices=True,
            )

            Factor_j_Nearest = quick_remove_outlier_np(factor[i1])
            # 取出第j个股票的最相近的股票的factor数据并去极值

            if len(np.where(~np.isnan(Factor_j_Nearest))[0]) <= 0.4 * len(indice1):
                print("该组中有效值数量过少")
                break
            else:
                today_newfactor0[j] = (factor[j] - np.mean(Factor_j_Nearest)) / (
                    np.std(Factor_j_Nearest) + 0.00000001
                )
            new_base = 0
            if NetralBase:
                for k in range(len(inputdata_ns.base_data.columns)):
                    base_j = base[k, j]
                    # 取出第j个股票的第k个basefactor值
                    base_j_nearest = quick_remove_outlier_np(base[k, i1])
                    # 取出第j个股票的最相近的股票的第k个basefactor数据并去极值
                    nan_mask = np.isnan(base_j_nearest)
                    if nan_mask.sum() / len(nan_mask) < 0.6:
                        new_base = (base_j - np.mean(base_j_nearest)) / (
                            np.std(base_j_nearest) + 0.00000001
                        )
                        today_newBaseFactor[k, j] = new_base
                    else:  # 如果组内nan值太多，就默认为0
                        pass

        for k in range(len(inputdata_ns.base_data.columns)):
            if k == 0:
                today_newBaseFactor1[k, :] = today_newBaseFactor[k, :]
            else:  # 依次将基准因子对之前的基准因子进行回归
                tempx = today_newBaseFactor1[k - 1 : k, :]
                tempy = today_newBaseFactor[k, :]
                tempy = np.nan_to_num(tempy)
                tempx = np.nan_to_num(tempx)
                X = sm.add_constant(tempx.T)
                results = sm.OLS(tempy, X).fit()
                today_newBaseFactor1[k, :] = results.resid
        X = sm.add_constant(today_newBaseFactor1.T)
        results = sm.OLS(today_newfactor0, X).fit()
        today_newfactor = results.resid
        standared_basefactor_numpy[
            i,
            inputdata_ns.Intersect_StockCodes_indice_StockCodesUnstacked[
                factor_nonan_indice
            ],
            :,
        ] = today_newBaseFactor1.T
        New_Factor_numpy[
            i,
            inputdata_ns.Intersect_StockCodes_indice_StockCodesUnstacked[
                factor_nonan_indice
            ],
        ] = today_newfactor
        """因子收益率回归测试"""
        X2 = sm.add_constant(
            np.concatenate(
                [today_newBaseFactor1.T, today_newfactor.reshape(-1, 1)], axis=1
            )
        )
        nextlogreturn = inputdata_ns.NextLogReturn_2d[
            backtestindex[i],
            inputdata_ns.Intersect_StockCodes_indice_StockCodesUnstacked[
                factor_nonan_indice
            ],
        ]
        results_logreturn = sm.OLS(nextlogreturn, X2).fit()
        reisd_return = results_logreturn.resid  # 残差收益率
        Resid_Return_numpy[
            i,
            inputdata_ns.Intersect_StockCodes_indice_StockCodesUnstacked[
                factor_nonan_indice
            ],
        ] = reisd_return
        analysis_data_numpy[i, 1] = results_logreturn.params[-1]
        analysis_data_numpy[i, 8] = results_logreturn.tvalues[-1]
        analysis_data_numpy[i, 6] = results_logreturn.pvalues[-1]
        analysis_data_numpy[i, 4] = spearmanr(today_newfactor, nextlogreturn)[0]
        """收益率分组测试 (np.digitize 函数无法等数量地分割数组)"""
        sorted_newfactor_indice = np.argsort(today_newfactor)
        sorted_newfactor = today_newfactor[sorted_newfactor_indice]
        sorted_nextlogreturn = nextlogreturn[sorted_newfactor_indice]
        sorted_stock_codes = today_stock_codes[sorted_newfactor_indice]
        target_size = len(today_newfactor) // groupnums
        grouped_means = np.zeros(groupnums)
        grouped_std = np.zeros(groupnums)
        grouped_factor_mean = np.zeros(groupnums)
        grouped_codes = []
        for g in range(groupnums):
            start_idx = g * target_size
            end_idx = (
                (g + 1) * target_size if g < groupnums - 1 else len(today_newfactor)
            )
            group_logreturn = sorted_nextlogreturn[start_idx:end_idx]
            grouped_means[g] = group_logreturn.mean()
            grouped_std[g] = group_logreturn.std()
            grouped_factor_mean[g] = sorted_newfactor[start_idx:end_idx].mean()
            grouped_codes.append(sorted_stock_codes[start_idx:end_idx])
        analysis_data_numpy[i, 0] = grouped_means[-1] - grouped_means[0]
        analysis_data_numpy[i, 2] = grouped_means[-1]
        analysis_data_numpy[i, 3] = grouped_means[0]
        analysis_data_numpy[i, 5] = spearmanr(grouped_factor_mean, grouped_means)[0]
        analysis_data_numpy[i, 7] = spearmanr(grouped_factor_mean, grouped_means)[1]
        Grouped_StockCodes_numpy[i, :, 0] = grouped_codes
        Grouped_StockCodes_numpy[i, :, 1] = grouped_means
        Grouped_StockCodes_numpy[i, :, 2] = grouped_std
        paramsdata_numpy[i, :] = results_logreturn.params[:-1]
        pvalues_numpy[i, :] = results_logreturn.pvalues[:-1]
        tvalues_numpy[i, :] = results_logreturn.tvalues[:-1]
    analysis_data = pd.DataFrame(
        analysis_data_numpy, index=tradingdates, columns=analysis_data_columns
    )
    New_Factor = pd.DataFrame(
        New_Factor_numpy, index=tradingdates, columns=inputdata_ns.StockCodes_unstacked
    )
    Resid_Return = pd.DataFrame(
        Resid_Return_numpy,
        index=tradingdates,
        columns=inputdata_ns.StockCodes_unstacked,
    )
    Grouped_StockCodes = pd.DataFrame(
        Grouped_StockCodes_numpy.reshape(len(tradingdates) * len(groups), 3),
        index=Grouped_index,
        columns=Grouped_columns,
    )
    paramsdata_pd = pd.DataFrame(
        paramsdata_numpy,
        index=tradingdates,
        columns=["constant", inputdata_ns.base_data.columns],
    )
    pvalues_pd = pd.DataFrame(
        pvalues_numpy,
        index=tradingdates,
        columns=["constant", inputdata_ns.base_data.columns],
    )
    tvalues_pd = pd.DataFrame(
        tvalues_numpy,
        index=tradingdates,
        columns=["constant", inputdata_ns.base_data.columns],
    )
    return (
        analysis_data,
        New_Factor,
        Resid_Return,
        Grouped_StockCodes,
        paramsdata_pd,
        pvalues_pd,
        tvalues_pd,
    )


def data_collinearity_removal(inputdata, threshold=0.8):
    pass


class Single_Factor_Test:
    def __init__(self, config_file):
        config = configparser.ConfigParser()
        config.read(config_file)
        self.BeginDate = config.get("SingleFactorTest", "BeginDate")  # 回测开始日期
        self.EndDate = config.get("SingleFactorTest", "EndDate")  # 回测结束日期
        self.EndDate_tag = (
            "default"  # 如果使用default，则默认测试结束日期为因子的最新日期
        )
        self.backtesttype = config.get(
            "SingleFactorTest", "backtesttype"
        )  # 回测交易频率 daily weekly monthly
        self.basic_data_path = config.get(
            "SingleFactorTest", "basic_data_path"
        )  # 基础数据存储路径
        self.rawfactorsavepath = config.get(
            "SingleFactorTest", "raw_factor_path"
        )  # 原始因子数据存储路径
        self.basedatasavepath = config.get(
            "SingleFactorTest", "base_data_path"
        )  # 基准数据存储路径
        self.tempdatapath = config.get(
            "SingleFactorTest", "temp_data_path"
        )  # 临时数据存储路径
        self.base_name = config.get("SingleFactorTest", "base_name").split(
            ","
        )  # 基准名称
        self.matrixname = config.get("SingleFactorTest", "matrixname")  # 相似度矩阵名称
        self.nearestnums = config.getint(
            "SingleFactorTest", "nearestnums"
        )  # 最近邻个数
        self.c_name = config.get("SingleFactorTest", "c_name")  # 独热分类器名称
        self.groupnums = config.getint("SingleFactorTest", "groupnums")  # 回测分组数量
        self.NetralBase = config.getboolean(
            "SingleFactorTest", "NetralBase"
        )  # 是否中性化基准
        self.backtesttradingprice = config.get(
            "SingleFactorTest", "back_test_trading_price"
        )  # 回测交易价格#vwap o2o

    def stock_filter(self, PriceDF) -> None:
        """股票池过滤"""
        self.filtered_stocks = PickupStocksByAmount(PriceDF)

    def data_merging(self) -> None:
        assert hasattr(self, "filtered_stocks"), "filtered_stocks not loaded"
        if len(self.base_name) > 0:
            assert hasattr(self, "base_data"), "base_data not loaded"
        assert hasattr(self, "factor_data"), "factor_data not loaded"
        assert hasattr(self, "next_log_return"), "next_log_return not loaded"
        factor_data = (
            self.factor_data.reindex(self.filtered_stocks.index)
            .groupby(level=1)
            .fillna(0)
        )
        next_log_return = (
            self.next_log_return.reindex(self.filtered_stocks.index)
            .groupby(level=1)
            .fillna(0)
        )
        merged_data = pd.concat([factor_data, next_log_return], axis=1)

        if len(self.base_name) > 0:
            if hasattr(self, "base_data"):
                base_data = (
                    self.base_data.reindex(self.filtered_stocks.index)
                    .groupby(level=1)
                    .fillna(method="ffill")
                )
                merged_data = pd.concat([merged_data, base_data], axis=1)
            else:
                print("base_data not loaded")
        else:
            pass

        if self.c_name == "classification_one_hot":
            if hasattr(self, "classification_data"):
                classification_data = (
                    self.classification_data.reindex(self.filtered_stocks.index)
                    .groupby(level=1)
                    .fillna(method="ffill")
                )
                merged_data = pd.concat([merged_data, classification_data], axis=1)
            else:
                print("One_hot classification data not loaded")
        else:
            pass
        self.merged_data = merged_data

    def data_switching_factordata_inmergeddata(self, c_type) -> None:
        assert hasattr(self, "merged_data"), "merged_data not loaded"

    def data_swithcing_basefactor_inmergeddata(self, c_type) -> None:
        assert hasattr(self, "merged_data"), "merged_data not loaded"

    def data_loading_1st_time(self, c_type=None) -> None:
        """第一次加载数据，不加载测试因子"""
        """ 读取下期收益数据 """
        if c_type is None:
            c_type = self.c_name

        if self.backtesttradingprice == "o2o":
            if self.backtesttype == "daily":
                self.next_log_return = pd.read_pickle(
                    os.path.join(self.basic_data_path, "LogReturn_daily_o2o.pkl")
                )
            if self.backtesttype == "weekly":
                self.next_log_return = pd.read_pickle(
                    os.path.join(self.basic_data_path, "LogReturn_weekly_o2o.pkl")
                )
            if self.backtesttype == "monthly":
                self.next_log_return = pd.read_pickle(
                    os.path.join(self.basic_data_path, "LogReturn_monthly_o2o.pkl")
                )
        """读取下期收益数据"""
        if self.backtesttradingprice == "vwap":
            if self.backtesttype == "daily":
                self.next_log_return = pd.read_pickle(
                    os.path.join(self.basic_data_path, "LogReturn_daily_vwap.pkl")
                )
            if self.backtesttype == "weekly":
                self.next_log_return = pd.read_pickle(
                    os.path.join(self.basic_data_path, "LogReturn_weekly_vwap.pkl")
                )
            if self.backtesttype == "monthly":
                self.next_log_return = pd.read_pickle(
                    os.path.join(self.basic_data_path, "LogReturn_monthly_vwap.pkl")
                )
        self.next_log_return = self.next_log_return.sort_index(level=0)
        # 读取下期收益数据  本日时间戳买进，下个时间戳卖出的收益
        """读取回测日期"""
        self.backtestdates = (
            self.next_log_return.index.get_level_values(0).unique().sort_values()
        )
        self.backtestdates = self.backtestdates[
            self.backtestdates >= pd.to_datetime(self.BeginDate, format="%Y-%m-%d")
        ]
        self.backtestdates = self.backtestdates[
            self.backtestdates <= pd.to_datetime(self.EndDate, format="%Y-%m-%d")
        ]
        """读取基准因子数据"""
        if isinstance(self.base_name, list):  # 考虑基准因子为多个的情况
            base_data = pd.DataFrame()
            for i in range(0, len(self.base_name)):
                temp = pd.read_pickle(
                    os.path.join(self.rawfactorsavepath, self.base_name[i] + ".pkl")
                )
                if i == 0:
                    base_data = pd.DataFrame()
                    base_data["BaseFactor" + str(i + 1)] = temp
                else:
                    base_data["BaseFactor" + str(i + 1)] = temp
        elif isinstance(self.base_name, str):  # basedata也存放在rawfactorsavepath中
            base_data = pd.read_pickle(
                os.path.join(self.rawfactorsavepath, self.base_name + ".pkl")
            ).to_frame(name="BaseFactor")
        self.base_data = base_data.sort_index(level=0)  # 读取基准数据
        """读取分类数据"""
        if (c_type == "one_hot") | (c_type == "classification_one_hot"):
            classification_path = os.path.join(
                self.basic_data_path + r"\Classificationdata"
            )
            self.classification_data = pd.read_pickle(
                os.path.join(classification_path, self.c_name + ".pkl")
            )
            # 将索引转换为列
            self.classification_data = self.classification_data.reset_index()
            # 去除重复的行
            self.classification_data = self.classification_data.drop_duplicates(
                subset=["TradingDates", "StockCodes"]
            )
            # 将索引转换回来
            self.classification_data = self.classification_data.set_index(
                ["TradingDates", "StockCodes"]
            )
            if hasattr(self, "filtered_stocks"):  # 以filtered_stocks为所有数据索引
                base_data = (
                    self.base_data.reindex(self.filtered_stocks.index)
                    .groupby(level=1)
                    .fillna(method="ffill")
                )
                next_log_return = (
                    self.next_log_return.reindex(self.filtered_stocks.index)
                    .groupby(level=1)
                    .fillna(0)
                )
                classification_data = (
                    self.classification_data.reindex(self.filtered_stocks.index)
                    .groupby(level=1)
                    .fillna(method="ffill")
                )
                self.merged_data = pd.concat(
                    [base_data, next_log_return, classification_data], axis=1
                )
                duplicated_indices = self.merged_data.index.duplicated()
                self.merged_data = self.merged_data[~duplicated_indices]
            else:
                print("股票池未过滤")
        else:
            if hasattr(self, "filtered_stocks"):
                base_data = (
                    self.base_data.reindex(self.filtered_stocks.index)
                    .groupby(level=1)
                    .fillna(method="ffill")
                )
                next_log_return = (
                    self.next_log_return.reindex(self.filtered_stocks.index)
                    .groupby(level=1)
                    .fillna(0)
                )
                self.merged_data = pd.concat([base_data, next_log_return], axis=1)
                duplicated_indices = self.merged_data.index.duplicated()
                self.merged_data = self.merged_data[~duplicated_indices]
            if self.c_name == "nearestmatrix":
                if hasattr(self, "matrixname"):
                    if hasattr(self, "merged_data"):
                        if hasattr(self.merged_data, "Factor"):
                            self.data_preprocessing_numpy()
                else:
                    print("请指定相似度矩阵名称")
            else:
                print("股票池未过滤")

    def data_loading_factor(self, factor_name, factorspath=None) -> None:
        """加载测试因子"""
        if factorspath is None:
            factorspath = self.rawfactorsavepath
        self.factor_name = factor_name
        self.factor_data = pd.read_pickle(
            os.path.join(factorspath, self.factor_name + ".pkl")
        )
        if isinstance(self.factor_data, pd.Series):
            self.factor_data = (
                self.factor_data.sort_index(level=0)
                .to_frame()
                .rename(columns={0: "Factor"})
            )
        if isinstance(self.factor_data, pd.DataFrame):
            self.factor_data = self.factor_data.sort_index(level=0).rename(
                columns={0: "Factor"}
            )
        factor_data = (
            self.factor_data.reindex(self.merged_data.index)
            .groupby(level=1)
            .fillna(method="ffill")
        )
        self.merged_data["Factor"] = factor_data

    def data_reloading_base(self, newbasename):
        """重新加载基准数据"""
        # 删除原有basefactor
        columns_to_drop = [col for col in self.base_data.columns if "BaseFactor" in col]
        self.merged_data.drop(columns=columns_to_drop, inplace=True)
        # 重新加载基准数据
        if isinstance(newbasename, list):
            for i in range(0, len(newbasename)):
                temp = pd.read_pickle(
                    os.path.join(self.rawfactorsavepath, newbasename[i] + ".pkl")
                ).to_frame(name="BaseFactor" + str(i + 1))
                if i == 0:
                    base_data = temp
                else:
                    base_data = pd.merge(
                        base_data, temp, left_index=True, right_index=True
                    )
        elif isinstance(newbasename, str):
            base_data = pd.read_pickle(
                os.path.join(self.rawfactorsavepath, newbasename + ".pkl")
            ).to_frame(name="BaseFactor")
        self.base_data = base_data
        base_data = (
            base_data.reindex(self.merged_data.index)
            .groupby(level=1)
            .fillna(method="ffill")
        )
        self.merged_data = pd.concat([self.merged_data, base_data], axis=1)

    def data_mergedata(self):
        if hasattr(self, "merged_data"):
            print("renew merged_data")
        if ~hasattr(self, "factor_data"):
            print("factor_data not loaded")
            return
        if ~hasattr(self, "base_data"):
            pass

    def backtest_pandas(
        self, factor_name=None, groupnums=None, NetralBase=False, factorspath=None
    ):
        """默认无分类的回测"""
        if factor_name is None:
            factor_name = self.factor_name
        if groupnums is None:
            groupnums = self.groupnums
        if factorspath is None:
            factorspath = self.rawfactorsavepath
        if ~hasattr(self, "factor_data"):
            self.data_loading_factor(factor_name, factorspath)
        if self.factor_name != factor_name:
            self.data_loading_factor(factor_name, factorspath)
        if len(self.factor_data) < 100:
            print("因子有效数据不足")
            return

        if groupnums is None:
            groupnums = self.groupnums
        if hasattr(self, "merged_data"):
            pass
        else:
            print("ReloadingData")
            self.data_loading_1st_time()
        factorbegindate = (
            self.factor_data.index.get_level_values(0).unique().sort_values()[0]
        )
        factorenddate = (
            self.factor_data.index.get_level_values(0).unique().sort_values()[-1]
        )
        begindate = max(
            pd.to_datetime(self.BeginDate, format="%Y-%m-%d"), factorbegindate
        )
        if self.EndDate_tag == "EndDate":
            enddate = min(
                pd.to_datetime(self.EndDate, format="%Y-%m-%d"), factorenddate
            )
        elif self.EndDate_tag == "default":
            enddate = factorenddate
        m1 = self.merged_data.copy()
        m1 = m1.loc[
            (m1.index.get_level_values(0) >= begindate)
            & ((m1.index.get_level_values(0) <= enddate))
        ]

        (analysis_data,
        New_Factor,
        Resid_Return,
        Grouped_StockCodes,
        paramsdata_list,
        pvalues_list,
        tvalues_list) = (
            backtest_merged_data(m1, self.groupnums, IndusNetralBase=True, extradates=0)
        )

        return analysis_data, New_Factor, Resid_Return, Grouped_StockCodes

    def backtest_one_hot_cat_pandas(
        self, factor_name, groupnums=None, NetralBase=False, factorspath=None
    ):

        if factorspath is None:
            factorspath = self.rawfactorsavepath
        if ~hasattr(self, "factor_data"):
            self.data_loading_factor(factor_name, factorspath)
        if self.factor_name != factor_name:
            self.data_loading_factor(factor_name, factorspath)

        if len(self.factor_data) < 100:
            return

        if groupnums is None:
            groupnums = self.groupnums
        else:
            self.groupnums = groupnums
        if hasattr(self, "merged_data"):
            pass
        else:
            print("ReloadingData")
            self.data_loading_1st_time()
        factorbegindate = (
            self.factor_data.index.get_level_values(0).unique().sort_values()[0]
        )
        factorenddate = (
            self.factor_data.index.get_level_values(0).unique().sort_values()[-1]
        )
        begindate = max(
            pd.to_datetime(self.BeginDate, format="%Y-%m-%d"), factorbegindate
        )
        if self.EndDate_tag == "EndDate":
            enddate = min(
                pd.to_datetime(self.EndDate, format="%Y-%m-%d"), factorenddate
            )
        elif self.EndDate_tag == "default":
            enddate = factorenddate
        m1 = self.merged_data.copy()
        m1 = m1.loc[
            (m1.index.get_level_values(0) >= begindate)
            & ((m1.index.get_level_values(0) <= enddate))
        ]
        tradingdates = m1.index.get_level_values(0).unique().sort_values()
        New_Factor = pd.DataFrame(index=m1.index, columns=["New_Factor"])
        Resid_Return = pd.DataFrame(index=m1.index, columns=["Resid_Return"])

        analysis_data = pd.DataFrame(
            index=tradingdates,
            columns=[
                "Factor_Return_LongShort",
                "Factor_Return_Regress",
                "Factor_Return_Last",
                "Factor_Return_First",
                "IC_Regress",
                "IC_Grouped",
                "P_Regress",
                "P_Grouped",
                "T_Regress",
            ],
        )
        groups = np.arange(groupnums, dtype=np.int64)
        Grouped_index = pd.MultiIndex.from_product(
            [tradingdates, groups], names=["TradingDates", "Group"]
        )
        Grouped_StockCodes = pd.DataFrame(
            index=Grouped_index, columns=["StockCodes", "ReturnMean", "ReturnStd"]
        )
        for date in tradingdates:
            print(date)
            m1slice = m1.loc[(date,)]

            y0 = Remove_Outlier(m1slice["Factor"], method="IQR", para=5)
            y1 = Normlization(y0, method="zscore").to_frame()  # 标准化
            y2 = y1.fillna(0)  # 无效值填充为0
            if (
                y2.std().values[0] <= 0.00001
            ):  # 如果因子是布尔值(0,1)离散型，则不做去极值，只做标准化
                y1 = m1slice["Factor"].copy()
                y2 = Normlization(y1, method="zscore").to_frame()  # 标准化
                continue
            if (y2 == 0).all().all():  # 当日因子值全为0，不需要进行回归分析
                continue
            else:
                base_factors = m1slice[m1slice.filter(like="BaseFactor").columns].copy()
                basesize = len(base_factors.columns)
                for j in range(basesize):
                    base_factors.iloc[:, j] = Remove_Outlier(
                        base_factors.iloc[:, j], method="IQR", para=5
                    )
                    base_factors.iloc[:, j] = Normlization(
                        base_factors.iloc[:, j], method="zscore"
                    )
                base_factors = base_factors.fillna(0)
                industry_columns = m1slice.drop(
                    columns=base_factors.columns.tolist() + ["Factor", "LogReturn"]
                ).dropna(
                    axis=1, how="all"
                )  # 行业独热编码,剔除在所有分类上都为nan的行
                x = base_factors.join(industry_columns)  # 基准因子+行业独热编码
                x = sm.add_constant(x)  # 添加常数项
                x = x.dropna(axis=0, how="any")
                y = y2.loc[x.index]
                model = sm.OLS(y, x)
                result = model.fit()  # 因子暴露回归
                newfactor = result.resid  # 新因子
                if newfactor.std() <= 0.00001:
                    continue
                newfactor.name = "newfactor"
                """收益率因子回归测试"""
                x1 = x.join(newfactor)
                y1 = m1slice["LogReturn"].loc[x1.index.values]
                model1 = sm.OLS(y1, x1)
                result1 = model1.fit()
                resid_return = result1.resid  # 残差收益率
                analysis_data.loc[date, "Factor_Return_Regress"] = result1.params[
                    "newfactor"
                ]
                analysis_data.loc[date, "T_Regress"] = result1.tvalues["newfactor"]
                analysis_data.loc[date, "P_Regress"] = result1.pvalues["newfactor"]
                analysis_data.loc[date, "IC_Regress"] = spearmanr(newfactor, y1)[0]
                """收益率分组测试 """
                m1slice["Group"] = pd.qcut(
                    newfactor, groupnums, labels=False, duplicates="drop"
                )
                m1slice["newfactor"] = newfactor
                grouped = m1slice.groupby("Group")
                groupedmean = grouped["LogReturn"].mean()
                analysis_data.loc[date, "Factor_Return_LongShort"] = (
                    groupedmean.loc[max(groupedmean.index)]
                    - groupedmean.loc[min(groupedmean.index)]
                )
                groupedfactormean = m1slice.groupby("Group")["newfactor"].mean()
                correlation, p_value = spearmanr(groupedfactormean, groupedmean)
                analysis_data.loc[date, "IC_Grouped"] = correlation
                analysis_data.loc[date, "P_Grouped"] = p_value

                analysis_data.loc[date, "Factor_Return_Last"] = groupedmean.loc[
                    max(groupedmean.index)
                ]
                analysis_data.loc[date, "Factor_Return_First"] = groupedmean.loc[
                    min(groupedmean.index)
                ]

                groupedmean.index = pd.MultiIndex.from_product(
                    [[date], groups], names=["TradingDates", "Group"]
                )
                Grouped_StockCodes.loc[groupedmean.index, "ReturnMean"] = groupedmean
                groupedstd = grouped["LogReturn"].std()
                groupedstd.index = pd.MultiIndex.from_product(
                    [[date], groups], names=["TradingDates", "Group"]
                )
                Grouped_StockCodes.loc[groupedstd.index, "ReturnStd"] = groupedstd
                groupedcodes = m1slice.groupby("Group").apply(
                    lambda x: x.index.get_level_values("StockCodes").tolist()
                )
                groupedcodes.index = pd.MultiIndex.from_product(
                    [[date], groups], names=["TradingDates", "Group"]
                )
                Grouped_StockCodes.loc[groupedcodes.index, "StockCodes"] = groupedcodes

                newfactor.index = pd.MultiIndex.from_product(
                    [[date], newfactor.index], names=["TradingDates", "StockCodes"]
                )
                New_Factor.loc[newfactor.index, "New_Factor"] = newfactor
                resid_return.index = pd.MultiIndex.from_product(
                    [[date], resid_return.index], names=["TradingDates", "StockCodes"]
                )
                Resid_Return.loc[resid_return.index, "Resid_Return"] = (
                    resid_return.values
                )
        self.analysis_data = analysis_data
        self.New_Factor = New_Factor
        self.Resid_Return = Resid_Return
        self.Grouped_StockCodes = Grouped_StockCodes

        return analysis_data, New_Factor, Resid_Return, Grouped_StockCodes

    def data_preprocessing_numpy(
        self, nearestdata_path=None, nearestMatrixname=None, nearestnums=None
    ) -> None:
        """使用最近邻矩阵的数据的预处理"""
        assert hasattr(self, "merged_data"), "merged_data not loaded"
        assert hasattr(self, "factor_data"), "factor_data not loaded"
        assert hasattr(self.merged_data, "Factor"), "Factor not loaded"
        if nearestdata_path is None:
            nearestdata_path = os.path.join(self.basic_data_path, "Classificationdata")
        if nearestMatrixname is None:
            nearestMatrixname = self.matrixname
        if nearestnums is None:
            nearestnums = self.nearestnums

        merge_data_unstacked = self.merged_data.unstack().sort_index()
        self.merged_data_tradingdates = merge_data_unstacked.index.to_numpy()
        self.StockCodes_unstacked = merge_data_unstacked["Factor"].columns.to_numpy()
        self.Factor_2dMatrix = merge_data_unstacked["Factor"].to_numpy()
        self.NextLogReturn_2d = merge_data_unstacked["LogReturn"].to_numpy()
        BaseFactor_2dMatrix = []
        for k in range(len(self.base_data.columns)):
            BaseFactor_2dMatrix.append(
                merge_data_unstacked[self.base_data.columns[k]].to_numpy()
            )
        self.BaseFactor_2dMatrix = np.array(BaseFactor_2dMatrix)
        if hasattr(self, "NearestIndice"):
            pass
        else:
            NearestIndice = pd.read_pickle(
                os.path.join(nearestdata_path, nearestMatrixname + ".pkl")
            )
            self.NearestIndice = NearestIndice["NearestIndiceMatrix"][
                :, :, -nearestnums:
            ]
            self.NearestTradingDates = NearestIndice["TradingDates"]
            self.NearestTradingDates = pd.to_datetime(
                self.NearestTradingDates, format="%Y%m%d"
            )
            self.NearestStockCodes = NearestIndice["StockCodes"]
            stock1, stockindice1, stockindice2 = np.intersect1d(
                self.StockCodes_unstacked, self.NearestStockCodes, return_indices=True
            )
            StockCode_in_Nearest_indice = []
            self.Intersect_StockCodes = stock1
            self.Intersect_StockCodes_indice_StockCodesUnstacked = stockindice1
            self.Intersect_StockCodes_indice_NearestStockCodes = stockindice2
            for a in self.NearestStockCodes:
                if a not in self.StockCodes_unstacked:
                    temp = -1
                else:
                    temp = np.where(self.StockCodes_unstacked == a)[0][0]
                StockCode_in_Nearest_indice.append(temp)
            self.StockCodes_Unstacked_indice_of_Nearest = np.array(
                StockCode_in_Nearest_indice
            )
            # 按照NearestStockCodes的顺序，在StockCodes_unstacked中的索引 """

    def backtest_nearest_numpy(self, NetralBase="True"):
        # 因为相似度矩阵是numpy形式的，所以回测主要基于numpy进行
        assert hasattr(self, "NearestIndice"), "NearestIndice not loaded"
        assert hasattr(self, "merged_data"), "merged_data not loaded"
        groupnums = self.groupnums
        backtestindex = np.where(
            (
                self.merged_data_tradingdates
                >= pd.to_datetime(self.BeginDate, format="%Y-%m-%d")
            )
            & (
                self.merged_data_tradingdates
                <= pd.to_datetime(self.EndDate, format="%Y-%m-%d")
            )
        )[0]

        """输出数据初始化"""
        tradingdates = self.merged_data_tradingdates[backtestindex]
        New_Factor_numpy = (
            np.zeros((len(tradingdates), len(self.StockCodes_unstacked))) * np.nan
        )
        standared_basefactor_numpy = (
            np.zeros(
                (
                    len(tradingdates),
                    len(self.StockCodes_unstacked),
                    len(self.base_data.columns),
                )
            )
            * np.nan
        )

        Resid_Return_numpy = (
            np.zeros((len(tradingdates), len(self.StockCodes_unstacked))) * np.nan
        )
        paramsdata_numpy = (
            np.zeros((len(tradingdates), len(self.base_data.columns) + 1)) * np.nan
        )
        pvalues_numpy = (
            np.zeros((len(tradingdates), len(self.base_data.columns) + 1)) * np.nan
        )
        tvalues_numpy = (
            np.zeros((len(tradingdates), len(self.base_data.columns) + 1)) * np.nan
        )
        analysis_data_columns = [
            "Factor_Return_LongShort",
            "Factor_Return_Regress",
            "Factor_Return_Last",
            "Factor_Return_First",
            "IC_Regress",
            "IC_Grouped",
            "P_Regress",
            "P_Grouped",
            "T_Regress",
        ]

        analysis_data_numpy = (
            np.zeros((len(tradingdates), len(analysis_data_columns))) * np.nan
        )
        groups = np.arange(groupnums, dtype=np.int64)
        Grouped_index = pd.MultiIndex.from_product(
            [tradingdates, groups], names=["TradingDates", "Group"]
        )
        Grouped_columns = ["StockCodes", "ReturnMean", "ReturnStd"]

        Grouped_StockCodes_numpy = np.empty(
            (len(tradingdates), len(groups), 3), dtype=object
        )

        """输出数据初始化"""
        for i in range(0, len(backtestindex)):
            today = self.merged_data_tradingdates[backtestindex[i]]
            print(today)
            date_idx_nearestmatrix = np.where(
                self.NearestTradingDates
                <= self.merged_data_tradingdates[backtestindex[i]]
            )[0]
            # 相似度矩阵日期坐标。因为相似度矩阵计算是左闭右开，当天可用
            if len(date_idx_nearestmatrix) == 0:
                print("相似度矩阵数据不足")
                print(self.merged_data_tradingdates[backtestindex[i]])
                continue
            else:
                date_idx_nearestmatrix = date_idx_nearestmatrix[-1]
            # 相似度矩阵日期坐标

            factor = self.Factor_2dMatrix[
                backtestindex[i], self.Intersect_StockCodes_indice_StockCodesUnstacked
            ]  # (合并时已经取了前一日factor和base数据)取出前一日, 所有股票池与相似度矩阵股票交集 因子数据
            factor_nonan_indice = np.argwhere(~np.isnan(factor)).flatten()
            # 今天有self.Intersect_StockCodes_indice_StockCodesUnstacked[factor_nonan_indice]
            factor = factor[factor_nonan_indice]
            # 取出前一日所有股票池因子

            base = []
            for k in range(len(self.base_data.columns)):
                basefactor = self.BaseFactor_2dMatrix[
                    k,
                    backtestindex[i],
                    self.Intersect_StockCodes_indice_StockCodesUnstacked[
                        factor_nonan_indice
                    ],
                ]
                # 取出前一日 所有股票池 与相似度矩阵股票交集 基准因子数据
                base.append(basefactor)
            base = np.array(base)  # 取出前一日 所有股票池

            nearest = self.NearestIndice[
                date_idx_nearestmatrix,
                self.Intersect_StockCodes_indice_NearestStockCodes[factor_nonan_indice],
                :,
            ]
            # 取出前一日 所有股票池 与相似度矩阵股票交集 最相近的股票索引
            today_stock_codes = self.Intersect_StockCodes[
                self.Intersect_StockCodes_indice_StockCodesUnstacked[
                    factor_nonan_indice
                ]
            ]
            # 当前日相似度矩阵
            today_newfactor0 = np.zeros((len(factor_nonan_indice)))
            # 只计算StockUnivers_indice的股票
            today_newBaseFactor = np.zeros(np.shape(base))
            today_newBaseFactor1 = np.zeros(np.shape(base))

            for j in range(len(factor_nonan_indice)):
                Nindice = nearest[j, :]
                Nindice = Nindice[Nindice >= 0].astype(int)
                # 最相近的股票索引(原始索引，在self.NearestStockCodes中)
                if len(Nindice) == 0:
                    continue

                indice1 = self.StockCodes_Unstacked_indice_of_Nearest[Nindice]
                indice1 = indice1[indice1 >= 0]
                # 无效值为-1 该组中的股票 所对应的 StockCodes_unstacked 的索引
                I1, i1, i2 = np.intersect1d(
                    self.Intersect_StockCodes_indice_StockCodesUnstacked[
                        factor_nonan_indice
                    ],
                    indice1,
                    return_indices=True,
                )

                Factor_j_Nearest = quick_remove_outlier_np(factor[i1])
                # 取出第j个股票的最相近的股票的factor数据并去极值

                if len(np.where(~np.isnan(Factor_j_Nearest))[0]) <= 0.4 * len(indice1):
                    print("该组中有效值数量过少")
                    break
                else:
                    today_newfactor0[j] = (factor[j] - np.mean(Factor_j_Nearest)) / (
                        np.std(Factor_j_Nearest) + 0.00000001
                    )
                if ~hasattr(self, "standared_basefactor_numpy"):
                    new_base = 0
                    if NetralBase:
                        for k in range(len(self.base_data.columns)):
                            base_j = base[k, j]
                            # 取出第j个股票的第k个basefactor值
                            base_j_nearest = quick_remove_outlier_np(base[k, i1])
                            # 取出第j个股票的最相近的股票的第k个basefactor数据并去极值
                            nan_mask = np.isnan(base_j_nearest)
                            if nan_mask.sum() / len(nan_mask) < 0.6:
                                new_base = (base_j - np.mean(base_j_nearest)) / (
                                    np.std(base_j_nearest) + 0.00000001
                                )
                                today_newBaseFactor[k, j] = new_base
                            else:  # 如果组内nan值太多，就默认为0
                                pass
                else:
                    pass

            if ~hasattr(self, "standared_basefactor_numpy"):
                for k in range(len(self.base_data.columns)):
                    if k == 0:
                        today_newBaseFactor1[k, :] = today_newBaseFactor[k, :]
                    else:  # 依次将基准因子对之前的基准因子进行回归
                        tempx = today_newBaseFactor1[k - 1 : k, :]
                        tempy = today_newBaseFactor[k, :]
                        tempy = np.nan_to_num(tempy)
                        tempx = np.nan_to_num(tempx)
                        X = sm.add_constant(tempx.T)
                        results = sm.OLS(tempy, X).fit()
                        today_newBaseFactor1[k, :] = results.resid
                X = sm.add_constant(today_newBaseFactor1.T)
                results = sm.OLS(today_newfactor0, X).fit()
                today_newfactor = results.resid
                standared_basefactor_numpy[
                    i,
                    self.Intersect_StockCodes_indice_StockCodesUnstacked[
                        factor_nonan_indice
                    ],
                    :,
                ] = today_newBaseFactor1.T
            else:
                today_newBaseFactor1 = self.standared_basefactor_numpy[
                    i,
                    self.Intersect_StockCodes_indice_StockCodesUnstacked[
                        factor_nonan_indice
                    ],
                    :,
                ].T

            New_Factor_numpy[
                i,
                self.Intersect_StockCodes_indice_StockCodesUnstacked[
                    factor_nonan_indice
                ],
            ] = today_newfactor
            """因子收益率回归测试"""
            X2 = sm.add_constant(
                np.concatenate(
                    [today_newBaseFactor1.T, today_newfactor.reshape(-1, 1)], axis=1
                )
            )
            nextlogreturn = self.NextLogReturn_2d[
                backtestindex[i],
                self.Intersect_StockCodes_indice_StockCodesUnstacked[
                    factor_nonan_indice
                ],
            ]
            results_logreturn = sm.OLS(nextlogreturn, X2).fit()
            reisd_return = results_logreturn.resid  # 残差收益率
            Resid_Return_numpy[
                i,
                self.Intersect_StockCodes_indice_StockCodesUnstacked[
                    factor_nonan_indice
                ],
            ] = reisd_return
            analysis_data_numpy[i, 1] = results_logreturn.params[-1]
            analysis_data_numpy[i, 8] = results_logreturn.tvalues[-1]
            analysis_data_numpy[i, 6] = results_logreturn.pvalues[-1]
            analysis_data_numpy[i, 4] = spearmanr(today_newfactor, nextlogreturn)[0]
            """收益率分组测试 (np.digitize 函数无法等数量地分割数组)"""
            sorted_newfactor_indice = np.argsort(today_newfactor)
            sorted_newfactor = today_newfactor[sorted_newfactor_indice]
            sorted_nextlogreturn = nextlogreturn[sorted_newfactor_indice]
            sorted_stock_codes = today_stock_codes[sorted_newfactor_indice]
            target_size = len(today_newfactor) // groupnums
            grouped_means = np.zeros(groupnums)
            grouped_std = np.zeros(groupnums)
            grouped_factor_mean = np.zeros(groupnums)
            grouped_codes = []
            for g in range(groupnums):
                start_idx = g * target_size
                end_idx = (
                    (g + 1) * target_size if g < groupnums - 1 else len(today_newfactor)
                )
                group_logreturn = sorted_nextlogreturn[start_idx:end_idx]
                grouped_means[g] = group_logreturn.mean()
                grouped_std[g] = group_logreturn.std()
                grouped_factor_mean[g] = sorted_newfactor[start_idx:end_idx].mean()
                grouped_codes.append(sorted_stock_codes[start_idx:end_idx])
            analysis_data_numpy[i, 0] = grouped_means[-1] - grouped_means[0]
            analysis_data_numpy[i, 2] = grouped_means[-1]
            analysis_data_numpy[i, 3] = grouped_means[0]
            analysis_data_numpy[i, 5] = spearmanr(grouped_factor_mean, grouped_means)[0]
            analysis_data_numpy[i, 7] = spearmanr(grouped_factor_mean, grouped_means)[1]
            Grouped_StockCodes_numpy[i, :, 0] = grouped_codes
            Grouped_StockCodes_numpy[i, :, 1] = grouped_means
            Grouped_StockCodes_numpy[i, :, 2] = grouped_std
            paramsdata_numpy[i, :] = results_logreturn.params[:-1]
            pvalues_numpy[i, :] = results_logreturn.pvalues[:-1]
            tvalues_numpy[i, :] = results_logreturn.tvalues[:-1]
        analysis_data = pd.DataFrame(
            analysis_data_numpy, index=tradingdates, columns=analysis_data_columns
        )
        New_Factor = pd.DataFrame(
            New_Factor_numpy, index=tradingdates, columns=self.StockCodes_unstacked
        )
        Resid_Return = pd.DataFrame(
            Resid_Return_numpy, index=tradingdates, columns=self.StockCodes_unstacked
        )
        Grouped_StockCodes = pd.DataFrame(
            Grouped_StockCodes_numpy.reshape(len(tradingdates) * len(groups), 3),
            index=Grouped_index,
            columns=Grouped_columns,
        )
        paramsdata_pd = pd.DataFrame(
            paramsdata_numpy,
            index=tradingdates,
            columns=["constant", self.base_data.columns],
        )
        pvalues_pd = pd.DataFrame(
            pvalues_numpy,
            index=tradingdates,
            columns=["constant", self.base_data.columns],
        )
        tvalues_pd = pd.DataFrame(
            tvalues_numpy,
            index=tradingdates,
            columns=["constant", self.base_data.columns],
        )
        self.standared_basefactor_numpy = standared_basefactor_numpy
        return (
            analysis_data,
            New_Factor,
            Resid_Return,
            Grouped_StockCodes,
            paramsdata_pd,
            pvalues_pd,
            tvalues_pd,
            standared_basefactor_numpy,
        )

    def data_save(
        self,
        newfactorsavepath=None,
        analysisdatasavepath=None,
        resid_return_savepath=None,
    ):
        if newfactorsavepath is None:
            newfactorsavepath = self.tempdatapath
        if analysisdatasavepath is None:
            analysisdatasavepath = self.tempdatapath
        if resid_return_savepath is None:
            resid_return_savepath = self.tempdatapath

        pd.to_pickle(
            self.New_Factor,
            os.path.join(newfactorsavepath, self.factor_name + "_newfactor.pkl"),
        )
        pd.to_pickle(
            self.analysis_data,
            os.path.join(analysisdatasavepath, self.factor_name + "_analysis_data.pkl"),
        )
        pd.to_pickle(
            self.Resid_Return,
            os.path.join(resid_return_savepath, self.factor_name + "_Resid_Return.pkl"),
        )

    def data_plot(self, meanlen=120, AbsReturn=False, savepath=None):
        if savepath is None:
            savepath = self.tempdatapath

        if AbsReturn is False:
            fig, axs = plt.subplots(2, gridspec_kw={"height_ratios": [4, 1]})
            for i in range(0, self.groupnums):
                axs[0].plot(
                    self.Grouped_StockCodes.xs(i, level=1)["ReturnMean"].cumsum(),
                    label="Group" + str(i),
                )
            axs[0].legend(loc="upper left")
            ax2 = axs[1].twinx()
            line1 = (
                self.analysis_data["T_Regress"]
                .dropna()
                .rolling(meanlen)
                .mean()
                .plot(ax=ax2, color="b", label=None)
            )
            # 使用第二个 y 轴绘制 self.T
            line2 = (
                self.analysis_data["IC_Grouped"]
                .dropna()
                .rolling(meanlen)
                .mean()
                .plot(ax=axs[1], color="r", label=None)
            )
            # 使用原始 y 轴绘制 self.IC
            ax2.yaxis.label.set_color("b")  # 设置第二个 y 轴的标签颜色
            axs[1].yaxis.label.set_color("r")  # 设置原始 y 轴的标签颜色
            # 如果需要，你还可以设置每个 y 轴的标签
            ax2.set_ylabel("T_Regress")
            axs[1].set_ylabel("IC_Group")
        else:
            meanreturn = self.Grouped_StockCodes.groupby(level=0)["ReturnMean"].mean()
            relative_meanreturn = self.Grouped_StockCodes["ReturnMean"] - meanreturn
            fig, axs = plt.subplots(2, gridspec_kw={"height_ratios": [4, 1]})
            for i in range(0, self.groupnums):
                axs[0].plot(
                    relative_meanreturn.xs(i, level=1).cumsum(), label="Group" + str(i)
                )
            axs[0].legend(loc="upper left")
            ax2 = axs[1].twinx()
            line1 = (
                self.analysis_data["T_Regress"]
                .dropna()
                .rolling(meanlen)
                .mean()
                .plot(ax=ax2, color="b", label=None)
            )  # 使用第二个 y 轴绘制 self.T
            line2 = (
                self.analysis_data["IC_Grouped"]
                .dropna()
                .rolling(meanlen)
                .mean()
                .plot(ax=axs[1], color="r", label=None)
            )  # 使用原始 y 轴绘制 self.IC
            ax2.yaxis.label.set_color("b")  # 设置第二个 y 轴的标签颜色
            axs[1].yaxis.label.set_color("r")  # 设置原始 y 轴的标签颜色
            # 如果需要，你还可以设置每个 y 轴的标签
            ax2.set_ylabel("T_Regress")
            axs[1].set_ylabel("IC_Group")
        plt.savefig(os.path.join(savepath, self.factor_name + "_figure.png"))
        plt.show()
        plt.close()


def NearestMatrixSingleFactorTest():
    test = Single_Factor_Test(
        r"E:\Documents\PythonProject\StockProject\StockData\SingleFactorTestData03\[SingleFactorTest].ini"
    )  # 读取配置文件
    PriceDf = pd.read_pickle(
        r"E:\Documents\PythonProject\StockProject\StockData\Price.pkl"
    )
    test.filtered_stocks = PickupStocksByAmount(
        PriceDf, windows=20, para=100000, mc_limit_min=None, mc_limit_max=None
    )  # 股票池过滤
    test.data_loading_1st_time(c_type=test.c_name)  # 第一次加载数据
    test.data_loading_factor(
        "PEG", r"E:\Documents\PythonProject\StockProject\StockData\RawFactors"
    )


def run_singletest():
    pass


def single_factor_test_data(factordata, normedbasedata, logreturndata):

    mergeddata = factordata.join(logreturndata, how="left").join(
        normedbasedata, how="left"
    )

    def dailyregress(m1):
        if m1.empty:
            return (
                0,
                0,
                0,
                0
                )        
        mask = m1["LogReturn"].notna()
        m1 = m1.loc[mask]
        y0 = Remove_Outlier(m1["factor"], method="IQR", para=5)
        y1 = Normlization(y0, method="zscore").to_frame()
        y3 = y1.fillna(0)
        if y3.std().values[0] <= 0.000001:
            y1 = m1["factor"].copy()
            y3 = Normlization(y1, method="zscore").to_frame()
        m2 = m1.drop(columns=["factor", "LogReturn"])
        x = sm.add_constant(m2)
        x = x.dropna(axis=0, how="any")
        y = y3.loc[x.index]
        model = sm.OLS(y, x)
        result = model.fit()
        newfactor = result.resid
        m2["newfactor"] = Normlization(newfactor, method="zscore")
        x1 = sm.add_constant(m2)
        x1 = x1.dropna(axis=0, how="any")
        y1 = m1["LogReturn"].loc[x1.index]
        model1 = sm.OLS(y1, x1)
        result1 = model1.fit()
        return (
            result1.params["newfactor"],
            result1.tvalues["newfactor"],
            result1.pvalues["newfactor"],
            spearmanr(newfactor, y1)[0],
        )

    regress_result = mergeddata.groupby(level=0).apply(dailyregress)
    regressreturn = regress_result.apply(lambda x: x[0])
    regressreturn.cumsum().plot()
    return regress_result


if __name__ == "__main__":
    print("main")
    factor = pd.read_pickle(
        r"E:\Documents\PythonProject\StockProject\StockData\RawFactors\PEG.pkl"
    )
    releaseddata = pd.read_pickle(
        r"E:\Documents\PythonProject\StockProject\StockData\realesed_dates_count_df.pkl"
    )
    newfactor = fm.factor_releaseddates_(factor, releaseddata)
    pd.to_pickle(
        newfactor,
        r"E:\Documents\PythonProject\StockProject\StockData\RawFactors\PEG_new_released.pkl",
    )

    test = Single_Factor_Test(
        r"E:\Documents\PythonProject\StockProject\StockData\SingleFactorTestData\[SingleFactorTest].ini"
    )  # 读取配置文件
    PriceDf = pd.read_pickle(
        r"E:\Documents\PythonProject\StockProject\StockData\Price.pkl"
    )
    test.filtered_stocks = PickupStocksByAmount(
        PriceDf, windows=20, para=10000000, mc_limit_min=None, mc_limit_max=None
    )  # 股票池过滤
    # test.filtered_stocks = PickupStocksByMarketCap(PriceDf, 3, 2)
    test.data_loading_1st_time(test.c_name)  # 第一次加载数据
    test.data_loading_factor(
        "PEG_new_released",
        r"E:\Documents\PythonProject\StockProject\StockData\RawFactors",
    )
    test.backtest_pandas("PEG_new_released")

    test.data_preprocessing_numpy()
    test.backtest_one_hot_cat_pandas("PEG_new_released")
    (
        analysis_data,
        New_Factor,
        Resid_Return,
        Grouped_StockCodes,
        paramsdata_list,
        pvalues_list,
        tvalues_list,
    ) = test.backtest_nearest_numpy(NetralBase=True)

    test.data_plot(
        meanlen=120,
        AbsReturn=False,
        savepath=r"E:\Documents\PythonProject\StockProject\StockData\SingleFactorTestData\graphic",
    )
    test.data_save(
        r"E:\Documents\PythonProject\StockProject\StockData\SingleFactorTestData\newfactors",
        r"E:\Documents\PythonProject\StockProject\StockData\SingleFactorTestData\analysisdata",
        r"E:\Documents\PythonProject\StockProject\StockData\SingleFactorTestData\residreturn",
    )
