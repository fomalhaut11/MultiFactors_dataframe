import matplotlib.pyplot as plt 
import statsmodels.api as sm
from scipy import stats
from time import time 
import pandas as pd
import numpy as np
import importlib
import psutil
import pickle
import h5py
import pulp
import os
import gc
 
import sys
file_path =r'E:\Documents\PythonProject\StockProject'
sys.path.append(file_path)
import SingleFactorTest as SFT


def FactorCombining(pickfacorlist,datasavepath=r'E:\Documents\PythonProject\StockProject\StockData\TestTempData'):
    for i in range(len(pickfactor)):
        filestr = pickfactor[i] + 'data.pkl'
        data = pd.read_pickle(os.path.join(datasavepath,filestr))
        data1=data['newfactor'].to_frame(name=pickfactor[i]).reset_index(level=0,drop=True).sort_index(level=0)
        data2=data['T'].squeeze().sort_index(level=0).shift(1) #T是计算下期收益和当期因子值的 ，所以要shift一期
        data2rolling=data2.ewm(span=60,adjust=False).mean()
        data2sign=np.sign(data2rolling)
        if i == 0:
            factor=data1
        else:
            data1=data1.reindex(factor.index)
            data1.fillna(0,inplace=True)
            df=factor.merge(data1,left_index=True,right_index=True)
            df=df.rename(columns={pickfactor[i]:'data1'})
            residuals = df.groupby(level=0).apply(lambda x: sm.OLS(x['data1'], sm.add_constant(x.filter(like='_factor'))).fit().resid)
            residuals = residuals.reset_index(level=0, drop=True)
            factor[pickfactor[i]] = residuals.loc[factor.index]*data2sign
    return factor.sum(axis=1)     


def FactorCombiningAnalysis(pickfactor,stockreturndata,datasavepath=r'E:\Documents\PythonProject\StockProject\StockData\TestTempData'):
    for i in range(len(pickfactor)):
        filestr = pickfactor[i] + 'data.pkl'
        data = pd.read_pickle(os.path.join(datasavepath,filestr))
        data1=data['newfactor'].to_frame(name=pickfactor[i]).reset_index(level=0,drop=True).sort_index(level=0)
        data2=data['T'].squeeze().sort_index(level=0).shift(1) #T是计算下期收益和当期因子值的 ，所以要shift一期
        data2rolling=data2.ewm(span=60,adjust=False).mean()
        data2sign=np.sign(data2rolling)
        if i == 0:
            factor=data1
        else:
            data1=data1.reindex(factor.index)
            data1.fillna(0,inplace=True)
            df=factor.merge(data1,left_index=True,right_index=True)
            df=df.rename(columns={pickfactor[i]:'data1'})
            residuals = df.groupby(level=0).apply(lambda x: sm.OLS(x['data1'], sm.add_constant(x.filter(like='_factor'))).fit().resid)
            residuals = residuals.reset_index(level=0, drop=True)
            reiduals.groupby(level=0).apply(lambda x: sm.OLS(sm.add_constant(stockreturndata),x).fit().tvalues['newfactor'])
            factor[pickfactor[i]] = residuals.loc[factor.index]*data2sign  
    return factor.sum(axis=1)  

def weight_linear_program_numpy(
        factors_explosure,
        benchmark_explosure,
        constraints,
        logreturn    ):
    prob = pulp.LpProblem('WeightOptimization', pulp.LpMaximize)

if __name__ == '__main__':
    datasavepath=r'E:\Documents\PythonProject\StockProject\StockData'
    PriceDf=pd.read_pickle(r'E:\Documents\PythonProject\StockProject\StockData\Price.pkl')
    test=SFT.Single_Factor_Test(r'E:\Documents\PythonProject\StockProject\MultiFactors\[SingleFactorTest].ini')
    test.filtered_stocks=SFT.PickupStocksByAmount(PriceDf)#股票池过滤
    test.data_loading_1st_time()
    next_log_return=pd.read_pickle(os.path.join(datasavepath,'LogReturn_daily_o2o.pkl'))
    #pickfactor = ['SUE_ss_4_hd5','ROE_ratio_zscores_4','log3marketcap','PE_ttm','PS_ttm','PB','alpha_001','alpha_032']
    pickfactor=['BP_t_profitgrowth','zz2000_20day_beta','spg','log3marketcap','SUE_ss_4_hd5','alpha_001','alpha_032']
    tempcombinefactor=FactorCombining(pickfactor)
    pd.to_pickle(tempcombinefactor,os.path.join(datasavepath,'BP_t_profitgrowth_zz2000_20day_beta_spg_log3marketcap_SUE_ss_4_hd5_alpha001_alpha032.pkl'))
    test.data_backtest_one_hot('BP_t_profitgrowth_zz2000_20day_beta_spg_log3marketcap_SUE_ss_4_hd5_alpha001_alpha032')
    test.data_plot(meanlen=120)
    test.data_save()
