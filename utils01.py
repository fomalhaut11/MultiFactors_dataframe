import pandas as pd
import numpy as np
from scipy.stats import norm


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


