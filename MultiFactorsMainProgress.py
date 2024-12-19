#MultiFactorsMainProgress.py

import numpy as np      
import pandas as pd
import os
import sys
sys.path.append(r'E:\Documents\PythonProject\StockProject\MultiFactors')
import SingleFactorTest as sft
import MultiFactorsTool as mft
import logging
import statsmodels.api as sm
from scipy.stats import skew, kurtosis, spearmanr, t, mode, norm
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

from copy import deepcopy


def find_pkl_files(directory):
    # 存储找到的.pkl文件路径
    pkl_files = []
    # 遍历指定目录及其子目录
    for root, dirs, files in os.walk(directory):
        for file in files:
            # 检查文件扩展名是否为.pkl
            if file.endswith('.pkl'):
                # 将文件的完整路径添加到列表中
                pkl_files.append(os.path.join(root, file))
    return pkl_files


def E_Multicollinearity(data, threshold=0.8):
    # 计算相关系数矩阵
    corr_matrix = data.corr().abs()
    # 选择上三角矩阵
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))
    # 找到相关系数大于阈值的特征
    to_drop = [column for column in upper.columns if any(upper[column] > threshold)]
    return to_drop


def Stepwise_Regression(data, threshold=0.05):
 
    # 初始化特征
    features = data.columns.tolist()
    # 初始化已选择的特征
    selected_features = []
    # 初始化最小p值
    min_p_value = 1
    # 进行逐步回归
    while len(features) > 0:
        # 初始化最小p值的特征
        feature_with_min_p_value = None
        # 初始化最小p值的模型
        model_with_min_p_value = None
        # 初始化最小p值的p值
        p_value_with_min_p_value = 1
        # 遍历特征
        for feature in features:
            # 选择已选择的特征和当前特征
            X = data[selected_features + [feature]]
            # 添加常数项
            X = sm.add_constant(X)
            # 训练模型
            model = sm.OLS(data, X).fit()
            # 获取当前特征的p值
            p_value = model.pvalues[feature]
            # 如果当前特征的p值小于最小p值
            if p_value < p_value_with_min_p_value:
                # 更新最小p值的特征
                feature_with_min_p_value = feature
                # 更新最小p值的模型
                model_with_min_p_value = model
                # 更新最小p值
                p_value_with_min_p_value = p_value
        # 如果最小p值小于阈值
        if p_value_with_min_p_value < threshold:
            # 更新已选择的特征
            selected_features.append(feature_with_min_p_value)
            # 更新最小p值
            min_p_value = p_value_with_min_p_value
        # 如果最小p值大于等于阈值
        else:
            # 结束循环
            break
    return selected_features


def Three_grouped_analysis(factordata, nextreturn):
    if 'Factor_grouped' in factordata.columns:
        pass
    else:
        factordata['Factor_grouped'] = \
            factordata.groupby('TradingDates')['New_Factor'].apply(lambda x: pd.qcut(x, 3, labels=False)) 
    if 'Return_grouped' in nextreturn.columns:
        pass
    else:  
        print('computing return grouped')
        nextreturn['Return_grouped'] = \
            nextreturn.groupby('TradingDates')['LogReturn'].apply(lambda x: pd.qcut(x, 3, labels=False, duplicates='drop'))
    grouped_data = pd.merge(factordata['Factor_grouped'], nextreturn['Return_grouped'], on=['TradingDates', 'StockCodes'])
    grouped_data['correctidex'] = grouped_data['Factor_grouped'] == grouped_data['Return_grouped']  
    total_accuracy = grouped_data.groupby('Factor_grouped')['correctidex'].mean()
    daily_accuracy = grouped_data.groupby(['TradingDates', 'Factor_grouped'])['correctidex'].mean()
    #daily_accuracy.groupby(level='TradingDates').sum().rolling(120).mean().plot()
    return total_accuracy, daily_accuracy


def re_analysis(datapath=r'E:\Documents\PythonProject\StockProject\StockData\SingleFactorTestData02'
               ):
    LogReturn_daily_o2o = pd.read_pickle(r'E:\Documents\PythonProject\StockProject\StockData\LogReturn_daily_o2o.pkl')
    LogReturn_daily_o2o['Return_grouped'] = \
        LogReturn_daily_o2o.groupby('TradingDates', group_keys=False)['LogReturn'].apply(lambda x: pd.qcut(x, 3, labels=False, duplicates='drop'))
    folder = os.path.join(datapath, 'newfactors')
    files = find_pkl_files(folder)
    all_daily_accuracy = pd.DataFrame()
    all_daily_accuracy_sum = pd.DataFrame()
    for file in files:
        data = pd.read_pickle(file)
        data = data.dropna()
        file_name = os.path.basename(file)[0:-5]
        print(file_name)
        data['New_Factor'] = pd.to_numeric(data['New_Factor'], errors='coerce')
        data['Factor_grouped'] = data.groupby('TradingDates', group_keys=False)['New_Factor'].apply(lambda x: pd.qcut(x, 3, labels=False)) 
        total_accuracy, daily_accuracy = Three_grouped_analysis(data, LogReturn_daily_o2o)
        dailysum = daily_accuracy.groupby('TradingDates', group_keys=False).sum()
        all_daily_accuracy = pd.concat([all_daily_accuracy,
                daily_accuracy.to_frame(name=file_name)], axis=1)
        all_daily_accuracy_sum = pd.concat([all_daily_accuracy_sum, dailysum.to_frame(name = file_name)], axis=1)


def factor_timing(data, window_size=20) -> pd.DataFrame:
    # 从单因子测试数据中制备时序指标
    result = pd.DataFrame()
    rolling_mean = data['IC_Regress'].rolling(window=window_size).mean()
    rolling_std = data['IC_Regress'].rolling(window=window_size).std()
    result['ICIR'] = rolling_mean/rolling_std
    result['IC_ema1'] = data['IC_Regress'].ewm(window_size, adjust=True).mean()
    result['PIR'] = data['P_Regress'].rolling(window=window_size).mean()/data['P_Regress'].rolling(window=window_size).std()
    result['P_ema1'] = data['P_Regress'].ewm(window_size, adjust=True).mean()
    result['P_ema2'] = data['P_Regress'].ewm(window_size*3, adjust=True).mean()
    return result


def loop_through_factor_timing(
    adjustwindow='M',
    datapath=r'E:\Documents\PythonProject\StockProject\StockData\SingleFactorTestData'
                                ) -> pd.DataFrame:
    folder = os.path.join(datapath, 'analysisdata')
    files = find_pkl_files(folder)
    all_results = pd.DataFrame()
    for file in files:
        data = pd.read_pickle(file)
        data = data.dropna()
        file_name = os.path.basename(file)[0:-18]
        print(file_name) 
        result = factor_timing(data, window_size=20)
        result['file_name'] = file_name
        result = result.set_index('file_name', append=True)
        x1 = result.unstack()
        all_results = pd.concat([all_results, x1], axis=1)
    # all_results.loc[:, ('ICIR','ACCT_RCV__NET_CASH_FLOWS_OPER_ACT')]
    # all_results.loc[:, ('ICIR',slice(None))]
    P_ema1 = all_results.loc[:, ('P_ema1', slice(None))]['P_ema1']
    P_ema2 = all_results.loc[:, ('P_ema2', slice(None))]['P_ema2']
    filter1 = (P_ema1 < P_ema2) & (P_ema2 < 0.2)  # 选取P值指数平滑线性下降的因子且 P值小于0.2
    f1 = all_results['ICIR'][filter1].shift(1)  # p值当天不可用，要移动一位
    pickupdata = []
    for tradingdate, row in f1.iterrows():
        no_nan = row[~np.isnan(row)].to_dict()
        pickupdata.append({'TradingDates':tradingdate, 'ICIR':no_nan})
    pick_data_df = pd.DataFrame(pickupdata).set_index('TradingDates')
    return pick_data_df


def factors_combination(
        pickupdata,
        backregresslength=20,
        adjustwindow='M',
        datapath=r'E:\Documents\PythonProject\StockProject\StockData\SingleFactorTestData'
                    ):
    pickupdata.index = pd.to_datetime(pickupdata.index)
    pickupdata['W'] = pickupdata.index.to_period('W')
    newfactorspath = os.path.join(datapath, 'newfactors')
    #  获取每个月的第一个数据点
    lastdaym = None
    begindates = []
    enddates = []
    
    for dates, data in pickupdata.iterrows():
         
        if lastdaym is None:
            lastdaym = dates
            begindates.append(dates)
            continue
        if dates.month != lastdaym.month:          
            begindates.append(dates)
            enddates.append(lastdaym)
        lastdaym = dates

    newfactor = pd.DataFrame() 
    for i, begindate in enumerate(begindates):
        print(begindate)
        try:
            enddate = enddates[i]
        except IndexError as e:
            break
        if begindate < pd.Timestamp('2018-06-01'):
            continue    

        lookback_date = begindate - pd.Timedelta(days=60)
        data = pickupdata.loc[begindate]['ICIR']
        if len(data) == 0:
            continue
        icirabssum = sum([abs(i) for i in data.values()])
        
        sorted_data_abs_desc = sorted(data.items(), key=lambda item: abs(item[1]), reverse=True)
        
        all_data = []
        regress_data = []
        pridict_data = []
        for factorname, ICIR in sorted_data_abs_desc:
            factorfilename = factorname + '_newfactor.pkl'
            factorfile = os.path.join(newfactorspath, factorfilename)
            factor = pd.read_pickle(factorfile)
            factor.columns = [factorname]
            all_data.append(factor)
            regress_data.append(factor.loc[lookback_date:begindate])
            pridict_data.append(factor.loc[begindate:enddate])
        regress_data_df = pd.concat(regress_data, axis=1)
        pridict_data_df = pd.concat(pridict_data, axis=1)

        regressed_data = pd.DataFrame()
        params0 = {}
        New_Factors = pd.DataFrame()
        for factorname in regress_data_df.columns:
            if len(regressed_data) == 0:
                regressed_data = pd.to_numeric(regress_data_df[factorname].dropna(), errors='coerce').to_frame(name=factorname)
                New_Factors = pd.to_numeric(pridict_data_df[factorname].dropna(), errors='coerce').to_frame(name=factorname)
                continue
            tempdata = pd.to_numeric(regress_data_df[factorname].loc[regressed_data.index].fillna(0), errors='coerce').to_frame(name=factorname)
            tempNew_Factor = pd.to_numeric(pridict_data_df[factorname].loc[New_Factors.index].fillna(0), errors='coerce').to_frame(name=factorname)
            result = sm.OLS(tempdata, sm.add_constant(regressed_data)).fit()
            resid = result.resid
            tempdata = sft.Normlization(resid, method='zscore').to_frame(name=factorname) 
            regressed_data = pd.concat([regressed_data, tempdata], axis=1)
            params0[factorname] = result.params 
            tempNew_Factor = tempNew_Factor - (result.params[0] + result.params[1] * tempNew_Factor)  # 观测值-预测值
            tempNew_Factor = sft.Normlization(tempNew_Factor, method='zscore') 
            New_Factors = pd.concat([New_Factors, tempNew_Factor], axis=1)
        weights_df = pd.DataFrame([data])
        weights_df = weights_df[New_Factors.columns]
        New_Factors_weighted_sum = (New_Factors.mul(weights_df.iloc[0])/icirabssum).sum(axis=1).to_frame(name='CombinedFactor')
 
        newfactor = pd.concat([newfactor, New_Factors_weighted_sum.copy()], axis=0)
    return newfactor


def quick_test(factor, logreturn, single_base_data=None, groupnums=10):
    if single_base_data is not None:
        df = pd.merge(factor, logreturn, on=['TradingDates', 'StockCodes'])
        df = pd.merge(df, single_base_data, on=['TradingDates', 'StockCodes'])
    else:
        df = pd.merge(factor, logreturn, on=['TradingDates', 'StockCodes'])
    
    tradingdates = df.index.get_level_values(0).unique().sort_values()
    New_Factor = pd.DataFrame(index=df.index, columns=['newfactor'])   # 新因子
    Resid_Return = pd.DataFrame(index=df.index, columns=['Resid_Return'])  # 残差收益率
    analysis_data = pd.DataFrame(
        index=tradingdates, columns=[
            'Factor_Return_LongShort',
            'Factor_Return_Regress',
            'Factor_Return_Last',
            'Factor_Return_First',
            'IC_Regress', 'IC_Grouped',
            'P_Regress', 'P_Grouped',
            'T_Regress'])
    groups = np.arange(groupnums, dtype=np.int64)
    Grouped_index = pd.MultiIndex.from_product([tradingdates, groups], names=['TradingDates', 'Group'])
    Grouped_StockCodes = pd.DataFrame(
        index=Grouped_index, 
        columns=['StockCodes', 'ReturnMean', 'ReturnStd'])    

    for date in tradingdates:
        print(date)
        m1slice = df.loc[(date,)].copy()

        #y0 = sft.Remove_Outlier(m1slice['Factor'], method='mean', para=5)  # 已经是处理过的数据则不需要去极值
        y0 = m1slice[factor.columns[0]]
        y1 = sft.Normlization(deepcopy(y0), method='zscore').to_frame()  # 标准化
        y2 = y1.fillna(0)   # 无效值填充为0 
        if y2.std().values[0] <= 0.00001:  # 如果因子是布尔值(0,1)离散型，则不做去极值，只做标准化
            y1 = m1slice['Factor'].copy()
            y2 = sft.Normlization(y1, method='zscore').to_frame()  # 标准化
            continue
        if (y2 == 0).all().all():  # 当日因子值全为0，不需要进行回归分析
            continue
        if single_base_data is not None:
            base_factors = m1slice[m1slice.filter(like='BaseFactor').columns].copy()
            base_factors = base_factors.fillna(0)
            X = sm.add_constant(base_factors)
            model = sm.OLS(y2, X).fit()
            resid = model.resid
            newfactor = sft.Normlization(resid, method='zscore').to_frame(name = 'Newfactor')
            x1 = X.join(newfactor)
        else:
            newfactor = y2
            newfactor.columns = ['newfactor']
            x1 = newfactor
        """收益率因子回归测试"""
        y1 = m1slice['LogReturn'].loc[x1.index.values]
        model1 = sm.OLS(y1, x1)
        result1 = model1.fit()
        resid_return = result1.resid  # 残差收益率
        analysis_data.loc[date, 'Factor_Return_Regress'] = \
            result1.params['newfactor']
        analysis_data.loc[date, 'T_Regress'] = \
            result1.tvalues['newfactor']
        analysis_data.loc[date, 'P_Regress'] = \
            result1.pvalues['newfactor'] 
        analysis_data.loc[date, 'IC_Regress'] = \
            spearmanr(newfactor, resid_return)[0]
        """收益率分组测试 """
        m1slice['Group'] = pd.qcut(
            newfactor['newfactor'], groupnums, labels=False, duplicates='drop') 
        m1slice['newfactor'] = newfactor
        grouped = m1slice.groupby('Group')
        groupedmean = grouped['LogReturn'].mean()
        analysis_data.loc[date, 'Factor_Return_LongShort'] = \
            groupedmean.loc[max(groupedmean.index)] -\
            groupedmean.loc[min(groupedmean.index)]
        groupedfactormean = m1slice.groupby('Group')['newfactor'].mean()
        correlation, p_value = spearmanr(
            groupedfactormean, groupedmean)
        analysis_data.loc[date, 'IC_Grouped'] = correlation
        analysis_data.loc[date, 'P_Grouped'] = p_value
    
        analysis_data.loc[date, 'Factor_Return_Last'] = \
            groupedmean.loc[max(groupedmean.index)]
        analysis_data.loc[date, 'Factor_Return_First'] = \
            groupedmean.loc[min(groupedmean.index)]

        groupedmean.index = pd.MultiIndex.from_product(
            [[date], groups], names=['TradingDates', 'Group']
            )
        Grouped_StockCodes.loc[groupedmean.index, 'ReturnMean'] = groupedmean                
        groupedstd = grouped['LogReturn'].std()
        groupedstd.index = pd.MultiIndex.from_product(
            [[date], groups], names=['TradingDates', 'Group']
            )
        Grouped_StockCodes.loc[groupedstd.index, 'ReturnStd'] = groupedstd
        groupedcodes = m1slice.groupby('Group').apply(
            lambda x: x.index.get_level_values('StockCodes').tolist())
        groupedcodes.index = pd.MultiIndex.from_product(
            [[date], groups], names=['TradingDates', 'Group']
            )
        Grouped_StockCodes.loc[groupedcodes.index, 'StockCodes'] = groupedcodes

        newfactor.index = pd.MultiIndex.from_product(
            [[date], newfactor.index], names=['TradingDates', 'StockCodes']
            )
        
        resid_return.index = pd.MultiIndex.from_product(
            [[date], resid_return.index], names=['TradingDates', 'StockCodes']
            )
        Resid_Return.loc[resid_return.index, 'Resid_Return'] = resid_return.values   
        New_Factor.loc[newfactor.index, 'newfactor'] = newfactor[newfactor.columns[0]]   
        Grouped_StockCodes['ReturnMean'] = pd.to_numeric(Grouped_StockCodes['ReturnMean'], errors='coerce')
        grouped_cumsum = Grouped_StockCodes['ReturnMean'].groupby(Grouped_StockCodes.index.get_level_values('Group')).cumsum() 

    return analysis_data, New_Factor, Resid_Return, Grouped_StockCodes

    
class LoopThroughSingleFactorTest():  
    '''循环测试单因子'''
    def __init__(self, single_test) -> None:
        self.RawFactorsPath = single_test.rawfactorsavepath
        self.tempdatapath = single_test.tempdatapath
        self.recordinfopath = r'E:\Documents\PythonProject\StockProject\StockData\multifactorrecord'
        self.single_test = single_test  
        
    def single_test_run(self, datasavepath):
        self.factorsfiles = find_pkl_files(single_test.rawfactorsavepath)
        for factorfile in self.factorsfiles:
            file_path = factorfile
            file_name_with_extension = os.path.basename(file_path)
            file_name = os.path.splitext(file_name_with_extension)[0]
            savename = file_name + '_newfactor.pkl'
            if os.path.exists(os.path.join(datasavepath, 'newfactors', savename)):
                continue
 
            print(file_name)
            self.single_test.backtest_one_hot_cat_pandas(file_name) 
            self.single_test.data_plot(meanlen=120, AbsReturn=False, savepath = os.path.join(datasavepath, 'graphic'))
            self.single_test.data_save(os.path.join(datasavepath, 'newfactors'),
                        os.path.join(datasavepath, 'analysisdata'),
                        os.path.join(datasavepath, 'residreturn'))
 
      
class MultiFactorCombination():
    '''多因子组合'''
    def __init__(self, data_loading_path) -> None:
        self.data_loading_path = data_loading_path
        self.analysisdata_path = os.path.join(data_loading_path, 'analysisdata')

    def loop_through_indicator(self, concerned_indicator_name):
        self.concerned_indicator_name = concerned_indicator_name
        files = find_pkl_files(self.analysisdata_path)
        concerned_indicator = []
        for file in files:
            data = pd.read_pickle(file)
            concerned_indicator.append(data[concerned_indicator_name])
        self.concerned_indicator = concerned_indicator
        return concerned_indicator
    

def run_multifactors_Main(input_data_path):
    '''多因子模型主程序'''

    picked_factor = loop_through_factor_timing(datapath=input_data_path) 
    # 在指定路径下，遍历所有因子，计算因子的时序指标，然后挑选出使用的因子
    pass



if __name__ == '__main__':

#     picked_factor = loop_through_factor_timing(datapath=r'E:\Documents\PythonProject\StockProject\StockData\SingleFactorTestData02') 
#     combined_factor = factors_combination(picked_factor, datapath=r'E:\Documents\PythonProject\StockProject\StockData\SingleFactorTestData02')
#     logreturn =  pd.read_pickle(r'E:\Documents\PythonProject\StockProject\StockData\LogReturn_daily_o2o.pkl')
#     analysis_data, New_Factor, Resid_Return, Grouped_StockCodes = \
#         quick_test(combined_factor, logreturn)
#     factor_name = 'combined_factor_191'
#     save_path = r'E:\Documents\PythonProject\StockProject\StockData\CombinedFactors'

#     pd.to_pickle(New_Factor, os.path.join(save_path, 'newfactors', factor_name+'_newfactor.pkl'))
#     pd.to_pickle(analysis_data, os.path.join(save_path, 'analysisdata',  factor_name+'_analysis_data.pkl'))
#     pd.to_pickle(Resid_Return, os.path.join(save_path, 'residreturn',  factor_name+'_Resid_Return.pkl'))
#     fig, ax = plt.subplots()

# # 遍历每个Group，绘制cumsum，并添加标签
#     for group in grouped_cumsum.index.get_level_values(1).unique():
#         group_data = grouped_cumsum.xs(group, level=1)
#         ax.plot(group_data.index.get_level_values('TradingDates'), group_data.values, label=f'Group {group}')

#     # 添加图例
#     ax.legend()

#     # 设置图表标题和标签
#     ax.set_title('Cumulative Sum of ReturnMean by Group')
#     ax.set_xlabel('Trading Dates')
#     ax.set_ylabel('Cumulative ReturnMean')

#     # 显示图表
#     plt.savefig(os.path.join(save_path, 'graphic', factor_name+'_cumsum.png'))    
#     plt.show()
#     plt.close()

    # single_test = sft.Single_Factor_Test(r'E:\Documents\PythonProject\StockProject\StockData\SingleFactorTestData\[SingleFactorTest].ini')
    # PriceDf = pd.read_pickle(r'E:\Documents\PythonProject\StockProject\StockData\Price.pkl')
    # single_test.stock_filter(PriceDf)
    # single_test.data_loading_1st_time()
    # datasavepath = r'E:\Documents\PythonProject\StockProject\StockData\SingleFactorTestData'
    # Model1 = LoopThroughSingleFactorTest(single_test)
    # Model1.single_test.rawfactorsavepath = r'E:\Documents\PythonProject\StockProject\StockData\RawFactors'
    # Model1.single_test_run(datasavepath)


    # mc=MultiFactorCombination(r'E:\Documents\PythonProject\StockProject\StockData\SingleFactorTestData02')
    # mc.loop_through_indicator('ResidualReturn')

    # newfactor = pd.read_pickle(r'E:\Documents\PythonProject\StockProject\StockData\SingleFactorTestData02\newfactors\alpha_001_newfactor.pkl')
    # newfactor = newfactor.dropna()
    # newfactor['New_Factor'] = pd.to_numeric(newfactor['New_Factor'], errors='coerce')
    # newfactor['Group'] = newfactor.groupby('TradingDates')['New_Factor'].apply(lambda x: pd.qcut(x, 10, labels=False)) 

    # LogReturn_daily_o2o = pd.read_pickle(r'E:\Documents\PythonProject\StockProject\StockData\LogReturn_daily_o2o.pkl')
    # daily_return_mean = LogReturn_daily_o2o.groupby('TradingDates').mean()
    # daily_return_std = LogReturn_daily_o2o.groupby('TradingDates').std()
    # fig, ax = plt.subplots()
    # for i in range(10):
    #     group0 = LogReturn_daily_o2o.loc[newfactor[newfactor['Group'] == i].index]
    #     normalized_data = (group0.sub(daily_return_mean, level='TradingDates').div(daily_return_std, level='TradingDates'))
    #     skewness = normalized_data.groupby('TradingDates').skew()
    #     kurtosis = normalized_data.groupby('TradingDates').apply(lambda x: x.kurtosis())

    #     ax.plot(kurtosis.rolling(480).mean())
    # ax.legend([str(i) for i in range(10)])
    # plt.show()