import pandas as pd
import numpy as np
from scipy.stats import norm
from scipy.optimize import linprog
import os
import statsmodels.api as sm
import sys

sys.path.append(r"E:\Documents\PythonProject\StockProject\MultiFactors")
import SingleFactorTest as sft
import factorsmaking as fm


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
    namask = (np.isnan(input_x)) | (np.isinf(input_x))
    if method == "IQR":
        medianvalue = np.median(input_x[~namask])
        Q1 = np.percentile(input_x, 10)
        Q3 = np.percentile(input_x, 90)
        IQR = Q3 - Q1
        input_x[input_x > Q3 + para * IQR] = Q3 + para * IQR
        input_x[input_x < Q1 - para * IQR] = Q1 - para * IQR
        input_x[namask] = medianvalue
    elif method == "median":
        medianvalue = np.median(input_x[~namask])
        mad = np.median(np.abs(input_x[namask] - medianvalue))
        input_x[input_x - medianvalue > para * mad] = para * mad
        input_x[input_x - medianvalue < -para * mad] = -para * mad
        input_x[namask] = medianvalue
    elif method == "mean":
        meanvalue = np.mean(input_x[~namask])
        std = np.std(input_x[~namask])
        input_x[input_x - meanvalue > para * std] = para * std
        input_x[input_x - meanvalue < -para * std] = -para * std
        input_x[namask] = meanvalue
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


def test_dataloader():
    datapath1 = "E:\Documents\PythonProject\StockProject\StockData\RawFactors"
    factordata = pd.read_pickle(
        os.path.join(datapath1, "DEDUCTEDPROFIT_yoy_zscores_4.pkl")
    )
    if isinstance(factordata, pd.Series):
        factordata = factordata.to_frame(name="factor")
    if isinstance(factordata, pd.DataFrame):
        factordata.columns = ["factor"]

    if factordata.index.names[1] is None or factordata.index.names[1] != "StockCodes":
        factordata.index.set_names(["TradingDates", "StockCodes"], inplace=True)

    riskdata1 = pd.read_pickle(os.path.join(datapath1, "LogMarketCap.pkl")).to_frame(
        name="LogMarketCap"
    )
    riskdata2 = pd.read_pickle(
        os.path.join(datapath1, "none_linear_marketcap.pkl")
    ).to_frame(name="none_linear_marketcap")
    riskdata3 = pd.read_pickle(os.path.join(datapath1, "BP.pkl")).to_frame(name="BP")
    riskdata4 = pd.read_pickle(os.path.join(datapath1, "ROE.pkl")).to_frame(name="ROE")
    riskdata5 = pd.read_pickle(os.path.join(datapath1, "ma_20.pkl")).to_frame(
        name="ma_20"
    )
    riskdata6 = pd.read_pickle(os.path.join(datapath1, "Vol_20.pkl")).to_frame(
        name="Vol_20"
    )
    riskdata7 = pd.read_pickle(
        os.path.join(datapath1, "freeturnoverrate_ma20.pkl")
    ).to_frame(name="freeturnoverrate_ma20")
    riskdata8 = pd.read_pickle(os.path.join(datapath1, "EP_ttm.pkl")).to_frame(
        name="EP_ttm"
    )
    # riskdata7 = pd.read_pickle(
    #     os.path.join(datapath1, "zz2000_20day_beta.pkl")
    # ).to_frame(name="zz2000_20day_beta")
    # if (riskdata7.index.names[1] is None or riskdata7.index.names[1] != "StockCodes"):
    #     riskdata7.index.set_names(["TradingDates", "StockCodes"], inplace=True)

    merged_data = pd.concat(
        [
            riskdata1,
            riskdata2,
            riskdata3,
            riskdata4,
            riskdata5,
            riskdata6,
            riskdata7,
            riskdata8,
        ],
        axis=1,
    )
    merged_data.sort_index(level=0, inplace=True)
    merged_data = merged_data[
        (merged_data.index.get_level_values("TradingDates") > "2018-01-01")
        & (merged_data.index.get_level_values("TradingDates") < "2024-12-31")
    ]
    factordata = factordata[
        (factordata.index.get_level_values("TradingDates") > "2018-01-01")
        & (factordata.index.get_level_values("TradingDates") < "2024-12-31")
    ]

    return factordata, merged_data


def formatting_data(inputdata):
    pass
    return


def stocks_weights_pd(
    inputfactor: pd.DataFrame,
    riskfactors: pd.DataFrame,
    risk_lower: list,
    risk_upper: list,
    weightrange: list = [0.0, 0.1],
) -> pd.DataFrame:
    # 默认输入因子ic为负数
    m1 = inputfactor.join(riskfactors, how="left").sort_index(level=0).fillna(0)
    m1 = m1.dropna(how="any", axis=0)
    m1unstack = m1.unstack()
    TradingDates = m1unstack.index.to_list()
    StockCodes = m1unstack["factor"].columns.to_list()
    basenamelist = riskfactors.columns.tolist()
    factor_np = m1unstack["factor"].values
    risks_np = []
    for i in range(len(basenamelist)):
        risks_np.append(m1unstack[basenamelist[i]].values)
    risks_np = np.array(risks_np)
    weight_np = stock_weights_np(
        factor_np=factor_np,
        risks_np=risks_np,
        b_risk_lower=np.array(risk_lower),
        b_risk_upper=np.array(risk_upper),
        weightrange=weightrange,
    )
    weight_pd = pd.DataFrame(weight_np, index=TradingDates, columns=StockCodes)
    weight_pd = weight_pd.stack().to_frame(name="weight")
    weight_pd.index.names = ["TradingDates", "StockCodes"]
    return weight_pd


def sequential_orthog_df(input: pd.DataFrame) -> pd.DataFrame:
    # 顺序正交化
    input_unstack = input.unstack().sort_index(level=0)
    factorsname_list = input_unstack.columns.get_level_values(0).unique().tolist()
    TradingDates = input_unstack.index.get_level_values(0).tolist()
    StockCodes = input_unstack[factorsname_list[0]].columns.to_list()
    factors_np = []
    for i in range(len(factorsname_list)):
        factors_np.append(input_unstack[factorsname_list[i]].values)
    factors_np = np.array(factors_np)
    newdata_np = sequential_orthog_np(factors_np)
    multi_index = pd.MultiIndex.from_product(
        [TradingDates, StockCodes], names=["TradingDates", "StockCodes"]
    )
    empty_df = pd.DataFrame(index=multi_index, columns=factorsname_list)
    for i in range(len(factorsname_list)):
        df_i = pd.DataFrame(newdata_np[i, :, :], index=TradingDates, columns=StockCodes)
        df_i_stack = df_i.stack().to_frame(name=factorsname_list[i])
        df_i_stack.index.names = ["TradingDates", "StockCodes"]
        empty_df[factorsname_list[i]] = df_i_stack
    newdata_df = empty_df
    return newdata_df


def sequential_orthog_np(input_np: np.ndarray) -> np.ndarray:
    # 顺序正交化
    assert input_np.shape[0] < 500, "第一维为基准因子数量"
    assert input_np.shape[0] < input_np.shape[2], "第三维为股票"
    newdata_np = np.zeros(input_np.shape)
    for i in range(input_np.shape[1]):
        today_factor = input_np[:, i, :]
        for j in range(input_np.shape[0]):
            tempfactor = today_factor[j, :]
            outlierfactor = quick_remove_outlier_np(tempfactor, method="IQR")
            normfactor = Normlization(outlierfactor, method="zscore")
            if j == 0:
                newdata_np[j, i, :] = normfactor
            else:
                x = sm.add_constant(newdata_np[:j, i, :].T)
                y = normfactor
                try:
                    model = sm.OLS(y, x)
                    resid = model.fit().resid
                    newdata_np[j, i, :] = Normlization(
                        resid, method="zscore"
                    )  # 正交化重新缩放
                except:
                    continue
    return newdata_np


def sequential_orthog_np_daily(input: np.ndarray) -> np.ndarray:
    pass
    return newdata_np


def stock_weights_np(
    factor_np: np.ndarray,
    risks_np: np.ndarray,
    b_risk_lower: np.ndarray,
    b_risk_upper: np.ndarray,
    weightrange,
    risk_orthog: bool = True,
) -> np.ndarray:
    # b_risk_lower = -0.3 * np.ones(3)  # 风险因子暴露的下界
    # b_risk_upper = 0.3 * np.ones(3)
    min_weight = weightrange[0]
    max_weight = weightrange[1]
    assert (
        factor_np.shape[0] == risks_np.shape[1]
    ), "factor_np and risks_np[0] have different length"
    # 确保日期相同
    weight_np = np.zeros((factor_np.shape[0], factor_np.shape[1]))
    for i in range(factor_np.shape[0]):  # 按日期循环
        today_factor = factor_np[i, :]
        avilmask = ~np.isnan(today_factor)
        outlierfactor = quick_remove_outlier_np(today_factor[avilmask], method="IQR")
        normfactor = Normlization(outlierfactor, method="zscore")
        today_risks = np.nan * np.ones((normfactor.shape[0], risks_np.shape[0]))
        for j in range(risks_np.shape[0]):  # 风险因子个数
            if risk_orthog:
                today_risks[:, j] = risks_np[j, i, avilmask]
            else:
                outlierdata = quick_remove_outlier_np(
                    risks_np[j, i, avilmask], method="IQR"
                )
                normdata = Normlization(outlierdata, method="zscore")
                if j == 0:
                    today_risks[:, j] = normdata
                else:
                    today_risks[:, j] = normdata
                Q, R = np.linalg.qr(today_risks)
                # 验证Q的正交性
                is_orthogonal = np.allclose(np.eye(Q.shape[1]), np.dot(Q.T, Q))
                if is_orthogonal:
                    pass

        c = normfactor
        A_risk = today_risks.T
        # 合并不等式约束
        A_ub = np.vstack([-A_risk, A_risk])
        b_ub = np.hstack((-b_risk_lower, b_risk_upper))
        # 等式约束：所有股票的权重之和为 1
        A_eq = np.ones((1, normfactor.shape[0]))
        b_eq = np.array([1.0])
        # 变量界限：[0, 1]
        bounds = [(min_weight, max_weight) for _ in range(normfactor.shape[0])]
        res = linprog(
            c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method="highs"
        )
        weight_np[i, avilmask] = res.x
    return weight_np


def stock_weights_np_daily(
    today_factor: np.ndarray,
    today_risks: np.ndarray,
    b_risk_lower: np.ndarray,
    b_risk_upper: np.ndarray,
    max_weight: float,
) -> np.ndarray:
    assert (
        today_factor.shape[0] == today_risks.shape[1]
    ), "factor_np and risks_np[0] have different length"
    # 确保股票数量相同
    weight_np = np.zeros(today_factor.shape[0])
    outlierfactor = quick_remove_outlier_np(today_factor, method="IQR")
    normfactor = Normlization(outlierfactor, method="zscore")
    for j in range(today_risks.shape[0]):  # 风险因子个数
        outlierdata = quick_remove_outlier_np(today_risks[j, :], method="IQR")
        normdata = Normlization(outlierdata, method="zscore")
        if j == 0:
            today_risks[:, j] = normdata
        else:
            today_risks[:, j] = normdata
        # Q, R = np.linalg.qr(today_risks)
        # # 验证Q的正交性
        # is_orthogonal = np.allclose(np.eye(Q.shape[1]), np.dot(Q.T, Q))
        # if is_orthogonal:
        #     pass
    c = normfactor
    A_risk = today_risks.T
    # 合并不等式约束
    A_ub = np.vstack([-A_risk, A_risk])
    b_ub = np.hstack((-b_risk_lower, b_risk_upper))
    # 等式约束：所有股票的权重之和为 1
    A_eq = np.ones((1, normfactor.shape[0]))
    b_eq = np.array([1.0])
    # 变量界限：[0, max_weight]
    bounds = [(0, max_weight) for _ in range(normfactor.shape[0])]
    res = linprog(
        c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method="highs"
    )
    weight_np = res.x
    return weight_np


def risks_explosure_pd(
    inputweight: pd.DataFrame,
    riskfactors: pd.DataFrame,
) -> pd.DataFrame:
    factornames = riskfactors.columns.tolist()
    m1 = inputweight.join(riskfactors, how="left").sort_index(level=0)

    def _calc_weighted_risk(row):
        weighted_risk = row[factornames] * row["weight"]
        return weighted_risk

    weight_exlosure = m1.groupby(level=0).apply(_calc_weighted_risk, axis=1)

    return weight_exlosure


def risks_explosure(
    inputweight: pd.DataFrame,
    riskfactors: pd.DataFrame,
) -> pd.DataFrame:
    factornames = riskfactors.columns.tolist()
    m1 = inputweight.join(riskfactors, how="left").sort_index(level=0)
    m1unstack = m1.unstack()
    TradingDates = m1unstack.index.to_list()

    weight_np = m1unstack["weight"].values
    risks_np = []
    for j in range(len(factornames)):
        risks_np.append(m1unstack[factornames[j]].values)
    risks_np = np.array(risks_np)

    exploresure_np = np.zeros((len(TradingDates), len(factornames)))
    for i in range(len(TradingDates)):
        todayweight = weight_np[i, :]
        for j in range(len(factornames)):
            todayfactor = risks_np[j, i, :]
            exploresure_np[i, j] = np.dot(todayweight, todayfactor)
    explosure_df = pd.DataFrame(exploresure_np, index=TradingDates, columns=factornames)
    return explosure_df


def risks_explosure_np(
    inputweight: np.ndarray,
    riskfactors: np.ndarray,
) -> np.ndarray:
    assert (
        inputweight.shape[0] == riskfactors.shape[1]
    ), "inputweight and riskfactors have different length"
    weighted_risk = np.zeros((inputweight.shape[0], riskfactors.shape[0]))
    for i in range(inputweight.shape[0]):
        for j in range(riskfactors.shape[0]):
            weighted_risk[i, j] = inputweight[i, :] @ riskfactors[j, i, :]

    return weighted_risk


def dataloader(
    list, path=r"E:\Documents\PythonProject\StockProject\StockData\RawFactors"
):
    merged_data = pd.DataFrame()
    for i in list:
        data = pd.read_pickle(path + r"\\" + i + ".pkl")
        if isinstance(data, pd.Series):
            data = data.to_frame(name=i)
        if merged_data.empty:
            merged_data = data
        else:
            merged_data = merged_data.join(data, how="inner")
    return merged_data


if __name__ == "__main__":

    # factordata, merged_data = test_dataloader()
    factordata = pd.read_pickle(
        r"E:\Documents\PythonProject\StockProject\StockData\RawFactors\zz2000_60day_beta.pkl"
    )
    factordata.columns = ["factor"]
    merged_data = dataloader(["LogMarketCap", "none_linear_marketcap"])
    merged_data = merged_data.groupby("TradingDates").fillna(method="ffill")
    merged_data = merged_data.dropna(axis=0)
    riskdata = sequential_orthog_df(merged_data)

    testfactor = dataloader(["DEDUCTEDPROFIT_yoy_zscores_4", "PEG"])

    LogReturn = pd.read_pickle(
        r"E:\Documents\PythonProject\StockProject\StockData\LogReturn_daily_o2o.pkl"
    )
    LogReturn = LogReturn[LogReturn.index.get_level_values(0) > "2018-01-01"]
    testdata = sft.single_factor_test_data(factordata, riskdata, LogReturn)
    # ic = testdata.apply(lambda x: x[3])
    realesed_dataes = pd.read_pickle(
        r"E:\Documents\PythonProject\StockProject\StockData\realesed_dates_count_df.pkl"
    )
    factordecay = fm.half_decay_factor(factordata, realesed_dataes, para=20)
    # testdata1 = sft.single_factor_test_data(factordecay, riskdata, LogReturn)
    # ic1 = testdata1.apply(lambda x: x[3])

    weights = stocks_weights_pd(
        factordecay * -1,
        riskdata,
        risk_lower=[-0.1 for i in range(2)],
        risk_upper=[0.1 for i in range(2)],
        weightrange=[0.0, 0.05],
    )
    w1 = weights.join(LogReturn)
    w1x = w1["weight"] * w1["LogReturn"]
    # w1x.groupby('TradingDates').sum().cumsum().plot()
    t = w1x.groupby("TradingDates").sum().to_frame(name="sum")
    b = LogReturn.groupby("TradingDates").mean()
    (t["sum"] - b["LogReturn"]).cumsum().plot()

    risk1 = risks_explosure(weights, riskdata)

    realesed_dataes = pd.read_pickle(
        r"E:\Documents\PythonProject\StockProject\StockData\realesed_dates_count_df.pkl"
    )
    f1 = factordata.join(realesed_dataes, how="left").sort_index(level=0)
