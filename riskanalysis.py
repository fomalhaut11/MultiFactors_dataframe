import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import configparser
import sys
file_path =r'E:\Documents\PythonProject\StockProject'
sys.path.append(file_path)
import os
import statsmodels.api as sm
import StockDataPrepairing as SDP
import alpha191backtest_factor as alpha191
from scipy.stats import spearmanr,kendalltau, t
import matplotlib.pyplot as plt 
import pickle
import SingleFactorTest as SFT

def FactorCombining(pickfacorlist):
    for i in range(len(pickfactor)):
        filestr = pickfactor[i] + 'data.pkl'
        data = pd.read_pickle(os.path.join(datasavepath1,filestr))
        data1=data['newfactor'].to_frame(name=pickfactor[i]).reset_index(level=0,drop=True).sort_index(level=0)
        data2=data['T'].squeeze().sort_index(level=0).shift(1) #T是计算下期收益和当期因子值的 ，所以要shift一期
        data2rolling=data2.ewm(span=5,adjust=False).mean()
        data3=data1.mul(data2rolling,axis=0)
        data3=data1 
        if i == 0:
            factor=data3
        else:

            factor = factor.merge(data3,left_index=True,right_index=True)


def RiskExposure(Factorname,Holdingweight,factortype='marketcapneutraled',weight=None):
    datasavepath = r'E:\Documents\PythonProject\StockProject\StockData'
    datasavepath1=r'E:\Documents\PythonProject\StockProject\StockData\TestTempData'
    if factortype=='marketcapneutraled':
        filestr = Factorname + 'data.pkl'
        data1 = pd.read_pickle(os.path.join(datasavepath1,filestr))
        factor=data1['newfactor'].reset_index(level=0,drop=True).sort_index(level=0)
        factor_reindex=factor.reindex(Holdingweight.index,fill_value=0)
        Holdingweight['factor']=factor_reindex
    else:
        if factortype=='raw':
            filestr=Factorname+'.pkl'
            data1 = pd.read_pickle(os.path.join(datasavepath,filestr))
            factor=data1.groupby('TradingDates').apply(lambda group:SFT.Normlization(SFT.Remove_Outlier(group,para=5)))
            factor_reindex=factor.reindex(Holdingweight.index,fill_value=0)
            Holdingweight['factor']=factor_reindex
            
           
     # 计算每个日期中 weight 和 factor 乘积的和   
    product_sum = Holdingweight.groupby('TradingDates').apply(lambda group: (group['weight'] * group['factor']).sum())

    return product_sum

def get_HoldingStocks(factorname,datasavepath1=r'E:\Documents\PythonProject\StockProject\StockData\TestTempData'):
    filestr = factorname + 'data.pkl'
    data = pd.read_pickle(os.path.join(datasavepath1,filestr))
    label1=np.sign(data['IC'].mean().values)
    Groupnum=data['grouped_StockCodes'].index.get_level_values(1).nunique()
    if label1>0:
        HoldingStocks=data['grouped_StockCodes'].xs(Groupnum-1,level=1)
        #data['GroupedMeanReturn'].xs(Groupnum-1,level=1).cumsum().plot()
    else:
        HoldingStocks=data['grouped_StockCodes'].xs(0,level=1)

    index = pd.MultiIndex.from_tuples([(date, stock) for date, stocks in HoldingStocks.items() for stock in stocks], names=['TradingDates', 'StockCodes'])
    df=pd.DataFrame(1,index=index,columns=['weight'])
    Holdingweight=df/df.groupby('TradingDates').sum()

    return Holdingweight
def get_IndexWeight():
    datasavepath = r'E:\Documents\PythonProject\StockProject\StockData'
    filestr='IndexWeight.pkl'
    data1 = pd.read_pickle(os.path.join(datasavepath,filestr))
    data1['weight']=data1['weight']/data1['weight'].sum()
    return data1
if __name__ == '__main__':
    Holdingweight=get_HoldingStocks('BP_t_profitgrowth_zz2000_20day_beta_spg_log3marketcap_SUE_ss_4_hd5_alpha001_alpha032')
    risk=RiskExposure('LogMarketCap',Holdingweight,'raw',weight=None)
    risk.plot()
    datasavepath = r'E:\Documents\PythonProject\StockProject\StockData'
    zz500Weight= pd.read_pickle(datasavepath+r'\zz500Weight.pkl') 
    risk=RiskExposure('spg',zz500Weight,'raw',weight=None)
    risk.plot()