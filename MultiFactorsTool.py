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
import os
import gc

import sys
file_path =r'E:\Documents\PythonProject\StockProject'
sys.path.append(file_path)
# import StockDataPrepairing as SDP
# import 股票数据接口_zx as FinData
def print_memory_usage():
    process = psutil.Process(os.getpid())
    print("Memory Usage: %.2f MB" % (process.memory_info().rss / 1024 / 1024))
def LogReturnDF(PriceDF1,returntype='c2c',inputtype='unadjusted'):
    '''输入复权数据，输出对数收益率数据'''
    if inputtype=='adjusted':
        if returntype=='c2c':
            #昨日收盘进，当日收盘价对数收益
            return1=PriceDF1.groupby('StockCodes').apply(lambda x: np.log(x['c']/x['c'].shift(1))).reset_index(level=0,drop=True)
        if returntype=='o2o':
            #昨日开盘进，当日开盘价对数收益
            return1=PriceDF1.groupby('StockCodes').apply(lambda x: np.log(x['o']/x['o'].shift(1))).reset_index(level=0,drop=True)
        if returntype=='o2c':
            #昨日开盘进，当日收盘价对数收益
            return1=PriceDF1.groupby('StockCodes').apply(lambda x: np.log(x['c']/x['o'].shift(1))).reset_index(level=0,drop=True)
        if returntype=='c2o':
            #昨日收盘进，当日开盘价对数收益
            return1=PriceDF1.groupby('StockCodes').apply(lambda x: np.log(x['o']/x['c'].shift(1))).reset_index(level=0,drop=True)
    if inputtype=='unadjusted':
        if returntype=='c2c':
            #昨日收盘进，当日收盘价对数收益
            return1=PriceDF1.groupby('StockCodes').apply(lambda x: np.log((x['c']*x['adjfactor'])/(x['c'].shift(1)*x['adjfactor'].shift(1)))).reset_index(level=0,drop=True)
        if returntype=='o2o':
            #昨日收盘进，当日收盘价对数收益
            return1=PriceDF1.groupby('StockCodes').apply(lambda x: np.log((x['o']*x['adjfactor'])/(x['o'].shift(1)*x['adjfactor'].shift(1)))).reset_index(level=0,drop=True)
        if returntype=='o2c':
            #昨日收盘进，当日收盘价对数收益
            return1=PriceDF1.groupby('StockCodes').apply(lambda x: np.log((x['c']*x['adjfactor'])/(x['o'].shift(1)*x['adjfactor'].shift(1)))).reset_index(level=0,drop=True)
        if returntype=='c2o':
            #昨日收盘进，当日收盘价对数收益
            return1=PriceDF1.groupby('StockCodes').apply(lambda x: np.log((x['o']*x['adjfactor'])/(x['c'].shift(1)*x['adjfactor'].shift(1)))).reset_index(level=0,drop=True)

    return1.sort_index(level=0,inplace=True)
    return1=return1.to_frame(name='LogReturn')
    return return1 

def PickupStocksByAmount(PriceDF,windows=5,para=10000000):
    #过去5天平均成交额大于1000万,amt>0,l<h,的股票
    average_amount=PriceDF.groupby('StockCodes')['amt'].rolling(window=windows).mean().reset_index(level=0,drop=True).shift(1)
    filtered_stocks=average_amount[average_amount>para]
    filtered_stocks=filtered_stocks[PriceDF['MC']>100000000]
    filtered_stocks=filtered_stocks[PriceDF['amt']>0]
    filtered_stocks=filtered_stocks[PriceDF['l']<PriceDF['h']]
    filtered_stocks=filtered_stocks[PriceDF['o']<PriceDF['high_limit']-0.02]
    filtered_stocks=filtered_stocks.to_frame().sort_index(level=0)
    return filtered_stocks

    
def Remove_Outlier_and_Normalize0(x_input,para=1.5,method='zscore'):
    x=x_input.copy()
    if type(x)==pd.core.frame.DataFrame or type(x)==pd.core.series.Series:
        medianvalue=x[np.isfinite(x)].median()
        x.fillna(medianvalue,inplace=True)
        Q1=np.percentile(x,25)
        Q3=np.percentile(x,75)
        IQR=Q3-Q1
        x[x>Q3+para*IQR]=Q3+para*IQR
        x[x<Q1-para*IQR]=Q1-para*IQR
        if method=='zscore':
            x=(x-x.mean())/x.std()
        if method=='ppf':
            cdf_values=(np.argsort(np.argsort(x))+0.5)/len(x)
            x=stats.norm.ppf(cdf_values)
    if type(x)==np.ndarray:
        medianvalue=np.median(x[np.isfinite(x)])
        x[np.isnan(x)]=medianvalue
        Q1=np.percentile(x,25)
        Q3=np.percentile(x,75)
        IQR=Q3-Q1
        x[x>Q3+para*IQR]=Q3+para*IQR
        x[x<Q1-para*IQR]=Q1-para*IQR
        if method=='zscore':
            x=(x-np.mean(x))/np.std(x)
        if method=='ppf':
            cdf_values=(np.argsort(np.argsort(x))+0.5)/len(x)
            x=stats.norm.ppf(cdf_values)

    return x


def Remove_Outlier_and_Normalize(x,method='median',para=3):
    x=x.astype(float)
     
    if method=='median':
        medianvalue=x[np.isfinite(x)].median()
        mad=np.nanmedian(np.abs(x-medianvalue))
        x.fillna(medianvalue,inplace=True)
        x[x-medianvalue>para*mad]['TestFactor']=para*mad
        x[x-medianvalue<-para*mad]['TestFactor']=-para*mad
        x=(x-medianvalue)/(mad)    
    elif method=='mean':
        meanvalue=x[np.isfinite(x)].mean(axis=0)
        std=np.std(x[np.isfinite(x)],axis=0)
        x.fillna(meanvalue,inplace=True)
        x[x-meanvalue>para*std]['TestFactor']=para*std 
        x[x-meanvalue<-para*std]['TestFactor']=-para*std 
        x=(x-meanvalue)/(std)
    return x

def Remove_Outlier(x,method='IQR',para=3):
    x=x.astype(float)
    
    if method=='IQR':
        medianvalue=x[np.isfinite(x)].median()
        x.fillna(medianvalue,inplace=True)
        Q1=np.percentile(x,25)
        Q3=np.percentile(x,75)
        IQR=Q3-Q1
        x[x>Q3+para*IQR]=Q3+para*IQR
        x[x<Q1-para*IQR]=Q1-para*IQR     
    if method=='median':
        medianvalue=x[np.isfinite(x)].median()
        mad=np.nanmedian(np.abs(x-medianvalue))
        x.fillna(medianvalue,inplace=True)
        x[x-medianvalue>para*mad]['TestFactor']=para*mad
        x[x-medianvalue<-para*mad]['TestFactor']=-para*mad
    elif method=='mean':
        meanvalue=x[np.isfinite(x)].mean(axis=0)
        std=np.std(x[np.isfinite(x)],axis=0)
        x.fillna(meanvalue,inplace=True)
        x[x-meanvalue>para*std]['TestFactor']=para*std 
        x[x-meanvalue<-para*std]['TestFactor']=-para*std
    return x
def Normlization(x_input,method='ppf'):
    x=x_input.copy()
    if type(x)==pd.core.frame.DataFrame or type(x)==pd.core.series.Series:
        if method=='zscore':
            x=(x-x.mean())/x.std()
        if method=='ppf':
            cdf_values=(np.argsort(np.argsort(x))+0.5)/len(x)
            x=stats.norm.ppf(cdf_values)
    if type(x)==np.ndarray:
        if method=='zscore':
            x=(x-np.mean(x))/np.std(x)
        if method=='ppf':
            cdf_values=(np.argsort(np.argsort(x))+0.5)/len(x)
            x=stats.norm.ppf(cdf_values)
    return x


def Normalized_UpandDown(x0,method='median'):
    x0=x0.astype(float)
    x=x0.copy()
     
    if method=='median':
        medianvalue=x[np.isfinite(x)].median()
        x.fillna(medianvalue,inplace=True)
        x1=x.copy()
        mad_plus=np.nanmedian(x[x>medianvalue]-medianvalue)
        mad_minus=np.nanmedian(medianvalue-x[x<medianvalue])
        x1[x>medianvalue]=(x[x>medianvalue]-medianvalue)/mad_plus
        x1[x<medianvalue]=(x[x<medianvalue]-medianvalue)/mad_minus
        x1[x==medianvalue]=0
    elif method=='mean':
        meanvalue=x[np.isfinite(x)].mean(axis=0)
        x.fillna(meanvalue,inplace=True)
        x1=x.copy()
        std_plus=np.std(x[x>meanvalue])
        std_minus=np.std(x[x<meanvalue])
        x.fillna(meanvalue,inplace=True)
        x1[x>meanvalue]=(x[x>meanvalue]-meanvalue)/std_plus 
        x1[x<meanvalue]=(x[x<meanvalue]-meanvalue)/std_minus
        x1[x==meanvalue]=0

    return x1   

 



def cal_IC_P_GroupReturn_inday(newfactor,next_log_return,groupnum=10):
    sort_indice=np.argsort(newfactor)
   
    sorted_factor_group=np.array_split(newfactor[sort_indice],groupnum)
    sorted_return_group=np.array_split(next_log_return[sort_indice],groupnum)
    sorted_indice_group=np.array_split(sort_indice,groupnum)

    mean_return_group=[np.mean(x) for x in sorted_return_group]
    std_return_group=[np.std(x) for x in sorted_return_group]
    mean_factor_group=[np.mean(x) for x in sorted_factor_group]
    IC,P=stats.pearsonr(newfactor,next_log_return)
    if P<0:
        print(P)
    return IC,P,mean_return_group,std_return_group,mean_factor_group,sorted_indice_group

class SingleFactorTest:
    def __init__(self,BeginDate,EndDate,datasavepath=r'E:\Documents\PythonProject\StockProject\StockData',tempdatapath=r'E:\Documents\PythonProject\StockProject\StockData\TestTempData'):
        #初始化单因子回测类，输入回测起始日期，回测结束日期，数据保存路径，临时数据保存路径   
        self.BeginDate = BeginDate
        self.EndDate = EndDate
        self.datasavepath = datasavepath    
        self.tempdatapath = tempdatapath
    def get_price_data(self):
        #获取股票价格数据
        PriceDF=pd.read_pickle(self.datasavepath+ '\\Price.pkl')
        self.PriceData=PriceDF
    def Calc_dailylogreturn(self,returntype='o2o',inputtype='unadjusted'):
        #计算日收益率
        if os.path.exists(tempdatapath+'\\dailylogreturn.pkl'):
            self.dailylogreturn=pd.read_pickle(tempdatapath+'\\dailylogreturn.pkl')
        else:    
            if hasattr(self,'PriceData'):
                self.dailylogreturn=LogReturnDF(self.PriceData,returntype=returntype,inputtype=inputtype)
            else:
                self.get_price_data()
                self.dailylogreturn=LogReturnDF(self.PriceData,returntype=returntype,inputtype=inputtype)   

    def get_filtered_stocks(self):
        #获取股票池
        filtered_stocks=PickupStocksByAmount(self.PriceData)
        self.filtered_stocks=filtered_stocks.loc[(filtered_stocks.index.get_level_values(0)>=self.BeginDate)&(filtered_stocks.index.get_level_values(0)<=self.EndDate)]

    def get_factor_data(self,factor_name):
        self.factor_name = factor_name
        self.factor_data=pd.read_pickle(self.datasavepath+ '\\'+self.factor_name+'.pkl')
        self.factor_data=self.factor_data.loc[(self.factor_data.index.get_level_values(0)>=self.BeginDate)&(self.factor_data.index.get_level_values(0)<=self.EndDate)]
        if type(self.factor_data)==pd.core.frame.DataFrame:
            self.factor_data.columns=['TestFactor']
        elif type(self.factor_data)==pd.core.series.Series:
            self.factor_data=self.factor_data.to_frame(name='TestFactor')
         

    def get_base_data(self,base_name):
        self.base_name=base_name
        if type(base_name)==list:
            for i in range(len(base_name)):
                temp=pd.read_pickle(self.datasavepath+ '\\'+self.base_name[i]+'.pkl').to_frame(name='BaseFactor'+str(i+1))
                if i==0:
                    self.base_data=temp
                else:
                    self.base_data=pd.merge(self.base_data,temp,left_index=True,right_index=True)
        elif type(base_name)==str:
            self.base_data=pd.read_pickle(self.datasavepath+ '\\'+self.base_name+'.pkl').to_frame(name='BaseFactor')
    def get_classification_one_hot(self,filename='classification_one_hot'):
        self.classification_one_hot=pd.read_pickle(self.datasavepath+ '\\'+filename+'.pkl')
        classification_one_hot_column_names=['classification' + str(i) for i in range(1,self.classification_one_hot.shape[1]+1)]
        self.classification_one_hot.columns=classification_one_hot_column_names
        datesindice=np.argwhere(self.classification_one_hot.index.get_level_values(0).unique()<=self.BeginDate)
        bdates=self.classification_one_hot.index.get_level_values(0).unique()[datesindice[-1]]
        self.classification_one_hot=self.classification_one_hot.loc[self.classification_one_hot.index.get_level_values(0)>=bdates[0]]
        self.classification_one_hot=self.classification_one_hot[~self.classification_one_hot.index.duplicated(keep='first')]
    def get_nearestmatrix(self,matrixname='NearestIndice250_100',nearestnums=50):
        #获取相似度矩阵数据
        NearestIndice=pd.read_pickle(self.datasavepath+ '\\'+matrixname+'.pkl')
        self.NearestIndice=NearestIndice['NearestIndiceMatrix'][:,:,-nearestnums:]
        self.NearestTradingDates=NearestIndice['TradingDates']
        self.NearestTradingDates=pd.to_datetime(self.NearestTradingDates,format='%Y%m%d')
        self.NearestStockCodes=NearestIndice['StockCodes'] 
        dateindice=np.argwhere(self.NearestTradingDates<=self.BeginDate)
        if len(dateindice)>0:
            bi=dateindice[-1]
            ei=np.argwhere(self.NearestTradingDates<=self.EndDate)
            self.NearestTradingDates=self.NearestTradingDates[dateindice[-1][0]:ei[-1][0]]
            self.NearestIndice=self.NearestIndice[dateindice[-1][0]:ei[-1][0],:,:]
        gc.collect()



    def back_freq(self,type='daily'):
        if hasattr(self,'factor_data'):
            pass
        else:
            print('请先读取因子值')
            return
        self.Dateserries=self.factor_data.index.get_level_values(0).unique()
        self.Dateserries=self.Dateserries[(self.Dateserries>=self.BeginDate)&(self.Dateserries<=self.EndDate)]
        if type=='daily':
            self.back_freq_type='daily'
            pass
        elif  type=='monthly':
            self.back_freq_type='monthly'
            monthly_mask=self.Dateserries.to_series().dt.to_period('M')!=self.Dateserries.to_series().shift(1).dt.to_period('M')
            self.Dateserries=self.Dateserries[monthly_mask]
        elif  type=='weekly': 
            self.back_freq_type='weekly'
            weekly_mask=self.Dateserries.to_series().dt.to_period('W')!=self.Dateserries.to_series().shift(1).dt.to_period('W')
            self.Dateserries=self.Dateserries[weekly_mask]
        elif  type=='weekly_end':
            self.back_freq_type='weekly_end'
            weekly_mask=self.Dateserries.to_series().dt.to_period('W')!=self.Dateserries.to_series().shift(-1).dt.to_period('W')
            self.Dateserries=self.Dateserries[weekly_mask]
    def apply_dateseries(self,dataname='factor_data'):
        if dataname=='factor_data':
            self.factor_data=self.factor_data.loc[self.Dateserries]
        elif dataname=='base_data':
            self.base_data=self.base_data.loc[self.Dateserries]

        elif dataname=='dailyreturn':
            if  os.path.exists(tempdatapath+'\\next_log_return_'+self.back_freq_type+'.pkl'):
                next_log_return=pd.read_pickle(tempdatapath+'\\next_log_return_'+self.back_freq_type+'.pkl')
                if len(self.Dateserries)==len(next_log_return):
                    self.next_log_return=next_log_return
                    self.next_log_return.columns=['next_log_return']
                    return
               
            else:
                if hasattr(self,'dailylogreturn'):
                    tempdates=self.dailylogreturn.index.get_level_values(0).unique()
                    if 0:#len(self.Dateserries)==len(tempdates[(tempdates>=self.Dateserries[0])&(tempdates<=self.Dateserries[-1])]):
                        
                        next_log_return=self.dailylogreturn.groupby(level=1).shift(-1)#仔细思考，是前移还是后移
                        self.next_log_return=next_log_return.loc[self.Dateserries]
                        next_log_return=None
                    else:
                        self.next_log_return=pd.DataFrame()
                        for i in range(0, len(self.Dateserries)-1):
                            slice0=self.dailylogreturn[(self.dailylogreturn.index.get_level_values(0)>self.Dateserries[i])&(self.dailylogreturn.index.get_level_values(0)<=self.Dateserries[i+1])]
                            slice_sum=slice0.groupby(level=1).sum()
                            slice_sum['TradingDates']=self.Dateserries[i]
                            slice_sum.set_index(['TradingDates',slice_sum.index],inplace=True)
                            self.next_log_return=pd.concat([self.next_log_return,slice_sum],axis=0)
                            #从本日期的下一个日期开始计算，直到序列中的下一个日期的日收益
                    self.next_log_return.columns=['next_log_return']
                else:
                    self.Calc_dailylogreturn()
                    self.next_log_return=pd.DataFrame()
                    tempdates=self.dailylogreturn.index.get_level_values(0).unique()
                    if 0:#len(self.Dateserries)==len(tempdates[(tempdates>=self.Dateserries[0])&(tempdates<=self.Dateserries[-1])]):
                        next_log_return=self.dailylogreturn.groupby(level=1).shift(1)
                        self.next_log_return=next_log_return.loc[self.Dateserries]
                        next_log_return=None

                    else:
                        self.next_log_return=pd.DataFrame()
                        for i in range(0, len(self.Dateserries)-1):
                            slice0=self.dailylogreturn[(self.dailylogreturn.index.get_level_index(0)>self.Dateserries[i])&(self.dailylogreturn.index.get_level_index(0)<=self.Dateserries[i+1])]
                            slice_sum=slice0.groupby(level=1).sum()
                            slice_sum['TradingDates']=self.Dateserries[i]
                            slice_sum.set_index(['TradingDates',slice_sum.index],inplace=True)
                            self.next_log_return=pd.concat([self.next_log_return,slice_sum],axis=0)
                            #从本日期的下一个日期开始计算，直到序列中的下一个日期的日收益
                    self.next_log_return.columns=['next_log_return']
        if dataname=='NearestIndice':
            tempi=[]
            for i in range(len(self.Dateserries)):
                idc=np.argwhere(self.NearestTradingDates<self.Dateserries[i])
                if len(idc)>0:
                    tempi.append(idc[-1][0])
            self.NearestTradingDates=self.NearestTradingDates[tempi]
            self.NearestIndice=self.NearestIndice[tempi,:,:]
            gc.collect()    



    def data_initialization(self):#输入数据融合和对齐,让日频的因子数据，股票池数据，基准因子数据，下期收益率数据拥有相同索引的多维度矩阵

        if hasattr(self,'filtered_stocks'):
            pass
        else:
            print('请生成股票池')
            return
        if hasattr(self,'factor_data'):
            pass
        else:
            print('请读取因子数据')
            return
        if hasattr(self,'next_log_return'):
            pass
        else:
            print('请生成日(序列)收益')
            return
        if hasattr(self,'base_data'):
            pass
        else:
            print('请读取基准因子数据')
            return
        factordata=pd.merge(self.factor_data,self.filtered_stocks,left_index=True,right_index=True,how='inner')
       #factordata=self.factor_data.loc[self.filtered_stocks.index]
        MergeData=pd.merge(factordata,self.base_data,left_index=True,right_index=True,how='left')
        MergeData=pd.merge(MergeData,self.next_log_return,left_index=True,right_index=True,how='left') 
        MergeData=MergeData[~MergeData.index.duplicated(keep='first')]
        MergeData=MergeData.sort_index(level=0)
        MergeData=MergeData.groupby(level=1).fillna(method='ffill')
        MergeData_unstack=MergeData.unstack()

        self.backtestdates=MergeData_unstack.index.get_level_values(0).unique()
        self.StockCodes_unstack=MergeData_unstack['TestFactor'].columns.to_numpy()#展开的股票代码全集
        self.Factor_2dMatrix=MergeData_unstack['TestFactor'].to_numpy()
        self.NextLogReturn_2dMatrix=MergeData_unstack['next_log_return'].to_numpy()
        BaseFactor_2dMatrix=[]
        for k in range(len(self.base_data.columns)):
            BaseFactor_2dMatrix.append(MergeData_unstack[self.base_data.columns[k]].to_numpy())
        self.BaseFactor_2dMatrix=np.array(BaseFactor_2dMatrix)
 
         
    
    def SingleFactorTest_classifcation_one_hot(self,GroupNum=10,NetralBase='True'):
        self.GroupNum=GroupNum
        holding_list_all=[]
        NewFactor=np.ones((len(self.backtestdates)-1,len(self.StockCodes_unstack)))*np.nan
        GroupedMeanReturn=np.ones((len(self.backtestdates)-1,self.GroupNum))*np.nan
        GroupedStdReturn=np.ones((len(self.backtestdates)-1,self.GroupNum))*np.nan
        IC_P_serreis=np.zeros((len(self.backtestdates)-1,2))
        if hasattr(self,'StockCodes_unstack'):
            pass
        else:
            print('先初始化输入数据 data_initialization(self)')
            return
        if hasattr(self,'classification_one_hot'):
            pass
        else:
            print('读取分类数据')
            return
        df=self.classification_one_hot.unstack(level=1)
        arr=df.values
        c01_matrix=arr.reshape((len(df),len(df.columns.get_level_values(0).unique()),-1))
        temp0= np.reshape(df.columns.get_level_values(0),(len(df.columns.get_level_values(0).unique()),-1))
        temp1= np.reshape(df.columns.get_level_values(1),(len(df.columns.get_level_values(0).unique()),-1))
        c_name=temp0[:,0]
        StockCodes_C=temp1[0,:]
        c_dates=df.index.get_level_values(0)
        StockCodes0,codei1,codei2=np.intersect1d(self.StockCodes_unstack,StockCodes_C,return_indices=True)

        for i in range(0,len(self.backtestdates)-1):
            print(self.backtestdates[i])
            idxinC=np.where(c_dates<=self.backtestdates[i])[0] 
            if len(idxinC)==0:
                continue
            else:
                idxinC=idxinC[-1]
            factor=self.Factor_2dMatrix[i,codei1]
            factor_nona_indice=np.argwhere(~np.isnan(factor)).reshape(-1)
            one_hot=c01_matrix[idxinC,:,codei2[factor_nona_indice]]
            one_hot[np.isnan(one_hot)]=0
            one_hot_nan_col=np.all(one_hot==0,axis=0)
            one_hot=one_hot[:,~one_hot_nan_col]
            fullsumindice=np.where(one_hot.sum(axis=1)>0)[0]#保证所有股票都有分类，不然回归会出错
            factor_nona_indice=factor_nona_indice[fullsumindice]
            one_hot=one_hot[fullsumindice,:]
            factor=Remove_Outlier_and_Normalize0(factor[factor_nona_indice],para=1.5)
            X=sm.add_constant(one_hot)
            model=sm.OLS(factor,X)
            results=model.fit()
            newfactor=results.resid
            base= self.BaseFactor_2dMatrix[:,i,codei1[factor_nona_indice]]
            for k in range(len(self.base_data.columns)):
                base[k,:]=Remove_Outlier_and_Normalize0(base[k,:],para=1.5)

            if NetralBase=='True':
                newbase=[]
                for k in range(len(self.base_data.columns)):
                    basefactor=base[k,:]
                    X=sm.add_constant(one_hot)
                    model=sm.OLS(basefactor,X)
                    results=model.fit()
                    newbasefactor=results.resid
                    newbase.append(newbasefactor)
                newbase=np.array(newbase)
            else :
                newbase=base
            X=sm.add_constant(np.concatenate((newbase.T,one_hot),axis=1))
            model=sm.OLS(newfactor,X)
            results=model.fit()
            newfactor=results.resid
            NewFactor[i,codei1[factor_nona_indice]]=newfactor
            nextlogreturn=self.NextLogReturn_2dMatrix[i,codei1[factor_nona_indice]]
            IC,P,mean_return_group,std_return_group,mean_factor_group,sorted_indice_group=cal_IC_P_GroupReturn_inday(newfactor,nextlogreturn,groupnum=self.GroupNum)
            list2=[]
            for list1 in sorted_indice_group:
                templist=codei1[factor_nona_indice[list1]]
                list2.append(templist)
            GroupedMeanReturn[i,:]=mean_return_group
            GroupedStdReturn[i,:]=std_return_group
            IC_P_serreis[i,0]=IC
            IC_P_serreis[i,1]=P
            if P < 0:
                print(P)

            holding_list_all.append(list2)

        self.NewFactor=NewFactor
        self.GroupedMeanReturn=GroupedMeanReturn
        self.GroupedStdReturn=GroupedStdReturn
        self.IC_P_serreis=IC_P_serreis
        self.holding_list_all=holding_list_all
        # for i in range(self.GroupNum):
        #     plt.plot(np.cumsum(self.GroupedMeanReturn[:,i]-np.mean(self.GroupedMeanReturn,axis=1)),label='Group'+str(i+1))
        #     plt.legend()
        # plt.show()



       

             

    
    def SingleFactorTest_NearestMatrix(self,GroupNum=10,NetralBase='True'):  
        self.GroupNum=GroupNum
        holding_list_all=[]
        NewFactor=np.ones((len(self.backtestdates)-1,len(self.StockCodes_unstack)))*np.nan
        GroupedMeanReturn=np.ones((len(self.backtestdates)-1,self.GroupNum))*np.nan
        GroupedStdReturn=np.ones((len(self.backtestdates)-1,self.GroupNum))*np.nan
        IC_P_serreis=np.zeros((len(self.backtestdates)-1,2))
        if hasattr(self,'StockCodes_unstack'):
            pass
        else:
            print('先初始化输入数据 data_initialization(self)')
            return
        if hasattr(self,'NearestIndice'):
            pass
        else:
            print('请读取相似度矩阵：get_nearestmatrix(self,matrixname=''NearestIndice250_100'',nearestnums=50)')
            return

        nearest_matrix=self.NearestIndice
         
        stock1,stockindice1,stockindice2=np.intersect1d(self.StockCodes_unstack,self.NearestStockCodes,return_indices=True)
        StockCode_in_Nearest_indice=[]
        for a in self.NearestStockCodes:
            if a not in self.StockCodes_unstack:
                temp=-1
            else:
                temp=np.where(self.StockCodes_unstack==a)[0][0]
            StockCode_in_Nearest_indice.append(temp)
        StockCode_in_Nearest_indice=np.array(StockCode_in_Nearest_indice)  

        
        
        for i in range(0,len(self.backtestdates)-1):
             
            idxinNearest=np.where(self.NearestTradingDates<=self.backtestdates[i])[0]#因为相似度矩阵是左闭右开，当天可用
            if len(idxinNearest)==0:
                print('short')
                print(self.backtestdates[i])
                continue
            else:
                idxinNearest=idxinNearest[-1]
            factor=self.Factor_2dMatrix[i,stockindice1]
            factor_nonan_indice=np.argwhere(~np.isnan(factor))
            factor=factor[factor_nonan_indice]
            factor=Remove_Outlier_and_Normalize0(factor,para=3)

            base=[]
            for k in range(len(self.base_data.columns)):
                basefactor=self.BaseFactor_2dMatrix[k,i,stockindice1[factor_nonan_indice]]
                basefactor=Remove_Outlier_and_Normalize0(basefactor,para=3,method='ppf')
                base.append(basefactor)
            base=np.array(base)
            base=np.squeeze(base,axis=2)
            nearest=self.NearestIndice[idxinNearest,stockindice2[factor_nonan_indice],:]
            today_newfactor=np.zeros((len(factor_nonan_indice)))#只计算StockUnivers_indice的股票
            #today_newBaseFactor=np.zeros((len(self.base_data.columns),len(factor_nonan_indice)))#只计算StockUnivers_indice的股票 
            today_newBaseFactor=base
            for j in range(len(factor_nonan_indice)):
                Nindice=nearest[j,:]
                Nindice=Nindice[Nindice>=0].astype(int)
                if len(Nindice)==0:
                    continue
                indice1=StockCode_in_Nearest_indice[Nindice]
                indice1=indice1[indice1>=0]#无效值为-1

                Factor_j_Nearest=self.Factor_2dMatrix[i,indice1]
                if len(np.where(~np.isnan(Factor_j_Nearest)))==0:
                    break
                today_newfactor[j]=(factor[j]-np.nanmean(Factor_j_Nearest))/(np.nanstd(Factor_j_Nearest)+0.00000001) 

                if NetralBase=='True':
                    newbase=[]
                    for k in range(len(self.base_data.columns)):
                        base_j=base[k,j]
                        base_j_nearest=self.BaseFactor_2dMatrix[k,i,indice1]
                        nan_mask=np.isnan(base_j_nearest)
                        if nan_mask.sum()/len(nan_mask)<0.4:
                            today_newBaseFactor[k,j]=(base_j-np.nanmean(base_j_nearest)/(np.nanstd(base_j_nearest)+0.00000001))
                        else:#如果组内nan值太多，就默认为0
                            today_newBaseFactor[k,j]=0
 
            X=sm.add_constant(today_newBaseFactor.T)
            model=sm.OLS(today_newfactor,X)
            results=model.fit()
            todayNewFactor=results.resid
            NewFactor[i,stockindice1[factor_nonan_indice]]=todayNewFactor.reshape((len(todayNewFactor),1))
            nextlogreturn=self.NextLogReturn_2dMatrix[i,stockindice1[factor_nonan_indice]]
            IC,P,mean_return_group,std_return_group,mean_factor_group,sorted_indice_group=cal_IC_P_GroupReturn_inday(todayNewFactor,nextlogreturn,groupnum=self.GroupNum)
            if np.isnan(mean_return_group).any(): 
                print(self.backtestdates[i])
                break
            list2=[]
            for list1 in sorted_indice_group:
                templist=stockindice1[factor_nonan_indice[list1]]
                list2.append(templist)
            GroupedMeanReturn[i,:]=mean_return_group
            GroupedStdReturn[i,:]=std_return_group
            IC_P_serreis[i,0]=IC
            IC_P_serreis[i,1]=P
            if P<0:
                print(P)

            holding_list_all.append(list2)


        self.NewFactor=NewFactor
        self.GroupedMeanReturn=GroupedMeanReturn
        self.GroupedStdReturn=GroupedStdReturn
        self.IC_P_serreis=IC_P_serreis
        self.holding_list_all=holding_list_all
 

        for i in range(self.GroupNum):
            plt.plot(np.cumsum(self.GroupedMeanReturn[:,i]-np.mean(self.GroupedMeanReturn,axis=1)),label='Group'+str(i+1))
            plt.legend()
        plt.show()        
    def cal_holding_turnoverrate(self):
        if hasattr(self,'holding_list_all'):
            turnoverrate=np.zeros((len(self.backtestdates)-1,self.GroupNum))
            holdingnums=np.zeros((len(self.backtestdates)-1,self.GroupNum))
            for i in range(len(self.backtestdates)-1):
                if i==0:
                    turnoverrate[i,:]=0
                else:
                    for j in range(self.GroupNum):
                        turnoverrate[i,j]=min(1,len(np.setdiff1d(self.holding_list_all[i][j],self.holding_list_all[i-1][j]))/len(self.holding_list_all[i-1][j]))#换手率最大为1
                        holdingnums[i,j]=len(self.holding_list_all[i][j])
            self.turnoverrate=turnoverrate
            self.holdingnums=  holdingnums      



            
    def preparation(self,factor_name,base_name,c_name,nearestnums,backtesttype,Groupnums,c_type='one_hot'):

        self.get_price_data()#读取行情数据（不复权）
        self.Calc_dailylogreturn(returntype='o2o',inputtype='unadjusted')#计算每日收益率
        pd.to_pickle(self.dailylogreturn,tempdatapath+'\\'+'dailylogreturn.pkl')
        self.get_filtered_stocks()#成交量过滤行情选取股票域
        self.get_factor_data(factor_name)#读取因子数据
        self.get_base_data(base_name)#读取基准因子数据
        self.base_data['BaseFactor1']=np.log(self.base_data['BaseFactor1'])#因为基准因子BaseFactor1是市值，所以对数化
        
        gc.collect()#释放内存
        self.back_freq(type=backtesttype)#'daily' 回测类型，'monthly' 按月回测 'daily' 按日回测 weekly每周一的信号 weekly_end 每周五的信号
        self.apply_dateseries(dataname='factor_data')#通过回测时间节点序列，对齐因子数据
        self.apply_dateseries(dataname='base_data')#通过回测时间节点序列，对齐因子数据
        if os.path.exists(tempdatapath+'\\next_log_return_'+backtesttype+'.pkl'):
            self.next_log_return=pd.read_pickle(tempdatapath+'\\next_log_return_'+backtesttype+'.pkl') 
        else:
            self.apply_dateseries(dataname='dailyreturn')#通过回测时间节点序列，对齐因子数据
            pd.to_pickle(self.next_log_return,tempdatapath+'\\'+'next_log_return_'+backtesttype+'.pkl')
        self.data_initialization()#通过回测时间节点序列，对齐因子数据
        
        if c_type=='one_hot':
            self.get_classification_one_hot(filename=c_name)#读取分类数据
            self.classification_type='one_hot'#记录分类数据类型
        elif c_type=='nearest':
            self.get_nearestmatrix(c_name,nearestnums=50)#读取分类数据
            self.apply_dateseries(dataname='NearestIndice')#记录分类数据类型
            self.classification_type='nearest'


    def quickstart(self,factor_name,base_name,c_name,nearestnums,Groupnums,c_type,NetralBase):
        self.preparation(factor_name,base_name,c_name,nearestnums,backtesttype,Groupnums,c_type='one_hot')
        if self.classification_type=='one_hot':
            self.SingleFactorTest_classifcation_one_hot(Groupnums,NetralBase)#单因子检验
        elif self.classification_type=='nearest':
            self.SingleFactorTest_NearestMatrix(Groupnums,NetralBase)#单因子检验
        self.cal_holding_turnoverrate()     
def MultiFactors(Factorlist):

    pass
 
 

if __name__ == '__main__':
    BeginDate='20180101'
    EndDate='20231125'
    backtesttype='daily'
    datasavepath=r'E:\Documents\PythonProject\StockProject\StockData'
    tempdatapath=r'E:\Documents\PythonProject\StockProject\StockData\TestTempData'
    factor_name='GTJA_191_alpha_001'
    f1=pd.read_pickle(datasavepath+ '\\'+factor_name+'.pkl')
    daily_count=f1.groupby(level=0).count()
    daily_count.plot()
    #factor_name='ma_5'
    #base_name=['MarketCap','ROE','ma_20','freeturnoverrate_ma20'] 
   # base_name=['LogMarketCap','CurrentRatio_zscores_8','QuickRatio_zscores_8','ma_20','freeturnoverrate_ma20','GapReturn_df']
    base_name=['MarketCap']
    matrixname='NearestIndice60_120_zscorefirst' 
    c_name='classification_one_hot'
    nearestnums=50
    groupnums=10
    NetralBase='True'
    test=SingleFactorTest(BeginDate,EndDate,datasavepath)
    '''配置 '''



    #test.quickstart(factor_name,base_name,matrixname,c_name,groupnums,c_type='one_hot',NetralBase=NetralBase)
    #test.quickstart(factor_name,base_name,matrixname,nearestnums,groupnums,c_type='nearest',NetralBase=NetralBase)#初次启动 读取数据

    test.preparation(factor_name,base_name,matrixname,nearestnums,backtesttype,groupnums,c_type='nearest')
    #test.SingleFactorTest_NearestMatrix(groupnums,NetralBase)#单因子检验
    test.SingleFactorTest_classifcation_one_hot(Groupnums,NetralBase)
    gc.collect()
  #  test.next_log_return.to_pickle(r'E:\Documents\PythonProject\StockProject\StockData\OLSedData\NextLogReturndaily.pkl')
    fig=plt.figure()
    plt.subplot(2,1,1)
    for i in range(groupnums):
        plt.plot(np.cumsum(test.GroupedMeanReturn[:,i]-np.mean(test.GroupedMeanReturn,axis=1)),label='Group'+str(i+1))

   
    plt.subplot(2,1,2)
    test.cal_holding_turnoverrate()
    plt.plot(test.turnoverrate[:,groupnums-1])
    print('meanturnover='+str(np.mean(test.turnoverrate[:,groupnums-1])))
    

    # matrixname='NearestIndice60_50'#更换相似度矩阵
    # test.get_nearestmatrix(matrixname,nearestnums=50)#读取分类数据
    # test.apply_dateseries(dataname='NearestIndice')# 分类数据按回测窗口序列选取
    # gc.collect()
    # test.classification_type='nearest'
    # test.SingleFactorTest_NearestMatrix(groupnums,NetralBase='False')

    factor_name1='SUE_ss_4_hd'#更换因子
    test.get_factor_data(factor_name1)#读取因子数据
    test.apply_dateseries(dataname='factor_data')#按回测时间窗口序列选取因子数据
    test.data_initialization() #数据对齐(因子，基准因子，股票池，下期收益率)
    test.SingleFactorTest_NearestMatrix(groupnums,NetralBase='False')


    # plt.plot(test.turnoverrate[:,0])#画出换手率
    # '''存储新因子'''
    # storagepath=r'E:\Documents\PythonProject\StockProject\StockData\OLSedData'
    # NewFactorDF=pd.DataFrame(test.NewFactor,index=test.backtestdates[:-1],columns=test.StockCodes_unstack)
    # NewFactorDF=NewFactorDF.stack()
    # NewFactorDF.index.names=['TradingDates','StockCodes']     
    # NewFactorDF.to_pickle(storagepath+'\\'+test.factor_name+'_OLSed.pkl')

    # '''存储回测结果'''
    # GroupedMeanReturnDF=pd.DataFrame(test.GroupedMeanReturn,index=test.backtestdates[:-1])
    # GroupedMeanReturnDF.columns=['Group'+str(i+1) for i in range(test.GroupNum)]
    # GroupedMeanReturnDF.to_pickle(storagepath+'\\'+test.factor_name+'_OLSed_GroupedMeanReturn.pkl')


    # text="""
    # 1.原因子名称：SUE_ss_4
    # 2.因子描述：SUE_ss_4
    # 3.股票池:过去5天平均成交额》1000w,开盘《涨停价-0.02
    # 4.因子计算方法：使用NearestIndice60_120_zscorefirst 的收益率相关性最近50个股票，对数市值为基准因子，且NetralBase='True'
    # 基准因子在相似股票中做标准化处理， 因子在相似股票中做标准化处理，然后做回归，取残差作为新因子
    # """
    # with open(storagepath+'\\'+test.factor_name+'_OLSed.txt','w') as f:
    #     f.write(text)
    

  