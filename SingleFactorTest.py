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

def print_memory_usage():
    #当前内存占用
    process = psutil.Process(os.getpid())
    print("Memory Usage: %.2f MB" % (process.memory_info().rss / 1024 / 1024))


def Remove_Outlier(input_x,method='IQR',para=3):
    #IQR剔除极值
    original_type=type(input_x)
    x=input_x.astype(float) #使用.astype(float)将数据转换为浮点型，则是已经创建了一个新的对象。
    
    if isinstance(x,np.ndarray):
        x=pd.Series(x)
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

    if original_type==np.ndarray:
        x=x.values
    return x

def Normlization(x_input,method='zscore'):
    #标准化
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
 

def cal_IC_P_GroupReturn_inday(newfactor,next_log_return,groupnum=10):
    sort_indice=np.argsort(newfactor)
    sorted_factor_group=np.array_split(newfactor[sort_indice],groupnum)
    sorted_return_group=np.array_split(next_log_return[sort_indice],groupnum)
    sorted_indice_group=np.array_split(sort_indice,groupnum)
    mean_return_group=[np.mean(x) for x in sorted_return_group]
    std_return_group=[np.std(x) for x in sorted_return_group]
    mean_factor_group=[np.mean(x) for x in sorted_factor_group]
    IC,P=spearmanr(mean_factor_group,mean_return_group)
    return IC,P,mean_return_group,std_return_group,mean_factor_group,sorted_indice_group

def PickupStocksByAmount(PriceDF,windows=5,para=10000000,mclimit=100000000):
    #过去5天平均成交额大于para,市值>mclimit,amt>0,l<h,的股票
    average_amount=PriceDF.groupby('StockCodes')['amt'].rolling(window=windows).mean().reset_index(level=0,drop=True).shift(1)
    filtered_stocks=average_amount[average_amount>para]
    filtered_stocks=filtered_stocks[PriceDF['MC']>mclimit]
    filtered_stocks=filtered_stocks[PriceDF['amt']>0]
    filtered_stocks=filtered_stocks[PriceDF['l']<PriceDF['h']]
    filtered_stocks=filtered_stocks[PriceDF['o']<PriceDF['high_limit']-0.02]
    # 找出所有股票代码的第一个数字小于7的所有数据
    filtered_stocks = filtered_stocks[filtered_stocks.index.get_level_values('StockCodes').str[0].astype(int) < 7]
    filtered_stocks=filtered_stocks.to_frame().sort_index(level=0)
    return filtered_stocks

def Stock_Filter(PriceDf):
    #股票池过滤
    pass

class Single_Factor_Test():
    def __init__(self, config_file):#读取配置文件
        config = configparser.ConfigParser()
        config.read(config_file)

        self.BeginDate = config.get('SingleFactorTest', 'BeginDate')#回测开始日期
        self.EndDate = config.get('SingleFactorTest', 'EndDate')#回测结束日期
        self.backtesttype = config.get('SingleFactorTest', 'backtesttype')#回测交易频率 daily weekly monthly
        self.datasavepath = config.get('SingleFactorTest', 'datasavepath')#数据存储路径
        self.tempdatapath = config.get('SingleFactorTest', 'tempdatapath')#临时数据存储路径
        self.base_name = config.get('SingleFactorTest', 'base_name')#基准名称
        self.matrixname = config.get('SingleFactorTest', 'matrixname')#相似度矩阵名称
        self.nearestnums = config.getint('SingleFactorTest', 'nearestnums')#最近邻个数
        self.c_name = config.get('SingleFactorTest', 'c_name')#独热分类器名称
        self.groupnums = config.getint('SingleFactorTest', 'groupnums')#回测分组数量
        self.NetralBase = config.getboolean('SingleFactorTest', 'NetralBase')#是否中性化基准
        self.backtesttradingprice=config.get('SingleFactorTest', 'backtesttradingprice')#回测交易价格#vwap o2o
        
    def stock_filter(self,PriceDF):
        #股票池过滤
        self.filtered_stocks=PickupStocksByAmount(PriceDF,windows=5,para=10000000)
            

    def data_loading_1st_time(self,c_type='one_hot'):
        #第一次加载数据，不加载测试因子
         
        if self.backtesttradingprice=='o2o':
            if self.backtesttype=='daily':
                self.next_log_return=pd.read_pickle(os.path.join(self.datasavepath,'LogReturn_daily_o2o.pkl'))
            if self.backtesttype=='weekly':
                self.next_log_return=pd.read_pickle(os.path.join(self.datasavepath,'LogReturn_weekly_o2o.pkl'))
            if self.backtesttype=='monthly':
                self.next_log_return=pd.read_pickle(os.path.join(self.datasavepath,'LogReturn_monthly_o2o.pkl'))
        if self.backtesttradingprice=='vwap':
            if self.backtesttype=='daily':
                self.next_log_return=pd.read_pickle(os.path.join(self.datasavepath,'LogReturn_daily_vwap.pkl'))
            if self.backtesttype=='weekly':
                self.next_log_return=pd.read_pickle(os.path.join(self.datasavepath,'LogReturn_weekly_vwap.pkl'))
            if self.backtesttype=='monthly':
                self.next_log_return=pd.read_pickle(os.path.join(self.datasavepath,'LogReturn_monthly_vwap.pkl'))
        self.next_log_return=self.next_log_return.sort_index(level=0)#读取下期收益数据  本日时间戳买进，下个时间戳卖出的收益    
        
        self.backtestdates=self.next_log_return.index.get_level_values(0).unique().sort_values()
        self.backtestdates=self.backtestdates[self.backtestdates>=pd.to_datetime(self.BeginDate,format='%Y-%m-%d')]
        self.backtestdates=self.backtestdates[self.backtestdates<=pd.to_datetime(self.EndDate,format='%Y-%m-%d')]#读取回测日期
 

        if type(self.base_name)==list:#考虑基准因子为多个的情况
            for i in range(0,len(base_name)):
                temp=pd.read_pickle(os.path.join(self.datasavepath,self.base_name[i]+'.pkl')).to_frame(name='BaseFactor'+str(i+1))
 
                if i==0:
                    base_data=temp
                else:
                    base_data=pd.merge(self.base_data,temp,left_index=True,right_index=True)
        elif type(self.base_name)==str:
            base_data=pd.read_pickle(os.path.join(self.datasavepath,self.base_name+'.pkl')).to_frame(name='BaseFactor')  
        self.base_data=base_data.sort_index(level=0)#读取基准数据

        if c_type=='one_hot':
            self.classification_data=pd.read_pickle(os.path.join(self.datasavepath,self.c_name+'.pkl'))
            # 将索引转换为列
            self.classification_data = self.classification_data.reset_index()
            # 去除重复的行
            self.classification_data = self.classification_data.drop_duplicates(subset=['TradingDates', 'StockCodes'])
            # 将索引转换回来
            self.classification_data = self.classification_data.set_index(['TradingDates', 'StockCodes'])
            if hasattr(self,'filtered_stocks'):#以filtered_stocks为所有数据索引
                base_data=self.base_data.reindex(self.filtered_stocks.index).groupby(level=1).fillna(method='ffill')
                next_log_return=self.next_log_return.reindex(self.filtered_stocks.index).groupby(level=1).fillna(0)
                classification_data=self.classification_data.reindex(self.filtered_stocks.index).groupby(level=1).fillna(method='ffill')
                self.merged_data= pd.concat([base_data, next_log_return, classification_data], axis=1)
                duplicated_indices = self.merged_data.index.duplicated()
                self.merged_data = self.merged_data[~duplicated_indices]
            else:
                print('股票池未过滤')
        if c_type=='nearest':
            raise ValueError('暂不支持最近邻分类器')

    def data_loading_factor(self,factor_name):
        self.factor_name=factor_name
        self.factor_data=pd.read_pickle(os.path.join(self.datasavepath,self.factor_name+'.pkl'))
        self.factor_data=self.factor_data.sort_index(level=0).to_frame().rename(columns={0:'Factor'})
        factor_data=self.factor_data.reindex(self.merged_data.index).groupby(level=1).fillna(method='ffill')
        self.merged_data['Factor']=factor_data
 
    def data_reloading_base(self,newbasename ):#替换基准因子值
        #删除原有basefactor
        columns_to_drop = [col for col in self.base_data.columns if 'BaseFactor' in col]
        self.merged_data.drop(columns=columns_to_drop, inplace=True)
        #重新加载基准数据
        if type(newbasename)==list:
            for i in range(0,len(newbasename)):
                temp=pd.read_pickle(os.path.join(self.datasavepath,newbasename[i]+'.pkl')).to_frame(name='BaseFactor'+str(i+1))
                if i==0:
                    base_data=temp
                else:
                    base_data=pd.merge(base_data,temp,left_index=True,right_index=True)
        elif type(newbasename)==str:
            base_data=pd.read_pickle(os.path.join(self.datasavepath,newbasename+'.pkl')).to_frame(name='BaseFactor')   
        self.base_data=base_data 
        base_data=base_data.reindex(self.merged_data.index).groupby(level=1).fillna(method='ffill')
        self.merged_data=pd.concat([self.merged_data,base_data],axis=1)


   


    def data_backtest_one_hot(self,factor_name,groupnums=None,NetralBase=False):
        self.data_loading_factor(factor_name)
        if groupnums==None:
            groupnums=self.groupnums
        else:
            self.groupnums=groupnums

        if hasattr(self,'merged_data'):
            pass
        else:
            print('ReloadingData')
            self.data_loading_1st_time()

        m1=self.merged_data.copy()
        m1=m1.loc[(m1.index.get_level_values(0)>=pd.to_datetime(self.BeginDate,format='%Y-%m-%d'))&((m1.index.get_level_values(0)<=pd.to_datetime(self.EndDate,format='%Y-%m-%d')))]
 
        def m1dailycount(m1slice,groupnums):
           # print(m1slice.index.get_level_values(0).unique())
            y=m1slice['Factor']
            y=Remove_Outlier(y,method='IQR',para=5)#去极值
            y=Normlization(y,method='zscore').to_frame()# 标准化
            y=y.fillna(0)   #无效值填充为0
            if y.std().values[0]<=0.00001:#如果因子是布尔值等 离散型，则不做去极值，只做标准化
                y=m1slice['Factor']
                y=Normlization(y,method='zscore').to_frame()# 标准化
            y=y.fillna(0)#无效值填充为0

            if (y==0).all().all():#当日因子值全为0，不需要进行回归分析
                print(m1slice.index.get_level_values(0).unique())
                resid=y
                factor_return=0
                resid_return=m1slice['LogReturn']
                groupedmean  = pd.DataFrame({'Group': [0]*10}, index=range(10))
                groupedmean.index=groupedmean.index.astype('float')
                groupedstd  = pd.DataFrame({'Group': [0]*10}, index=range(10))
                groupedstd.index=groupedmean.index.astype('float')          
                correlation=np.nan
                p_value=np.nan
                tvalues=np.nan
                params=[]
                grouped_StockCodes=[]
                return pd.Series({'newfactor':resid,'factor_return':factor_return,'resid_return':resid_return,'groupedmean':groupedmean,'groupedstd':groupedstd,'IC':correlation,'P':p_value,'T':tvalues,'params':params,'grouped_StockCodes':grouped_StockCodes})    

      
            else:
                base_factors =m1slice[m1slice.filter(like='BaseFactor').columns].copy()
                basesize=len(base_factors.columns)
                for i in range(basesize):
                    base_factors.iloc[:,i]=Remove_Outlier(base_factors.iloc[:,i],method='IQR',para=5)
                    base_factors.iloc[:,i]=Normlization(base_factors.iloc[:,i],method='zscore')
                base_factors=base_factors.fillna(0)
                industry_columns=m1slice.drop(columns=base_factors.columns.tolist()+['Factor','LogReturn']).dropna(axis=1,how='all')#行业独热编码,剔除在所有分类上都为nan的行
                x=base_factors.join(industry_columns)#基准因子+行业独热编码
                x=sm.add_constant(x)#添加常数项
                x=x.dropna(axis=0,how='any')
                y=y.loc[x.index.values]
                model=sm.OLS(y,x)
                result=model.fit()
                newfactor=result.resid#新因子
                newfactor.name='newfactor'
                ##收益率因子回归测试
                x1=x.join(newfactor)
                y1=m1slice['LogReturn'].loc[x1.index.values]
                model1=sm.OLS(y1,x1)
                result1=model1.fit()
                resid_return=result1.resid#残差收益率
                factor_return=result1.params['newfactor']
                tvalues=result1.tvalues
                ######分组测试#####
                
                params=result.params
                m1slice['Group']=pd.qcut(newfactor,groupnums,labels=False,duplicates='drop')
                grouped=m1slice.groupby('Group')
                groupedmean=grouped['LogReturn'].mean()
                groupedstd=grouped['LogReturn'].std()
                grouped_StockCodes=m1slice.groupby('Group').apply(lambda x: x.index.get_level_values('StockCodes').tolist())
                groupedfactormean=  m1slice.groupby('Group')['Factor'].mean()
                correlation, p_value = spearmanr(groupedfactormean, groupedmean)
                #####分组测试#####
 
                return pd.Series({'newfactor':newfactor,'factor_return':factor_return,'resid_return':resid_return,'groupedmean':groupedmean,'groupedstd':groupedstd,'IC':correlation,'P':p_value,'T':tvalues['newfactor'],'params':params,'grouped_StockCodes':grouped_StockCodes})    

      
        resultdata=m1.groupby('TradingDates').apply(m1dailycount,self.groupnums)
        self.BackTestResult=resultdata
        self.newfactor_df = pd.concat(self.BackTestResult['newfactor'].values, 
                         keys=self.BackTestResult['newfactor'].index)
        self.factor_return=pd.DataFrame(self.BackTestResult['factor_return'])
        self.T=pd.DataFrame(self.BackTestResult['T'])
        self.Params=pd.DataFrame(self.BackTestResult['params'])

        self.IC=pd.DataFrame(self.BackTestResult['IC'])
        self.P=pd.DataFrame(self.BackTestResult['P'])
        self.GroupedMeanReturn=pd.concat(self.BackTestResult['groupedmean'].values, 
                         keys=self.BackTestResult['groupedmean'].index)
        self.GroupedStdReturn=pd.concat(self.BackTestResult['groupedstd'].values,
                         keys=self.BackTestResult['groupedstd'].index)
 
        self.grouped_StockCodes=pd.concat(self.BackTestResult['grouped_StockCodes'].values,
                         keys=self.BackTestResult['grouped_StockCodes'].index)
        return resultdata
   

    def data_preprocessing_numpy(self):
        merge_data_unstacked=self.merged_data.unstack()
        self.merged_data_tradingdates=merge_data_unstacked.index.to_numpy()
        self.StockCodes_unstacked=merge_data_unstacked['Factor'].columns.to_numpy()
        self.Factor_2dMatrix=merge_data_unstacked['Factor'].to_numpy()
        self.NextLogReturn_2d=merge_data_unstacked['LogReturn'].to_numpy()
        BaseFactor_2dMatrix=[]
        for k in range(len(self.base_data.columns)):
            BaseFactor_2dMatrix.append(merge_data_unstacked[self.base_data.columns[k]].to_numpy())
        self.BaseFactor_2dMatrix=np.array(BaseFactor_2dMatrix)  
        if hasattr(self,'NearestIndice'):
            pass
        else:
            NearestIndice=pd.read_pickle(os.path.join(self.datasavepath,self.matrixname+'.pkl'))
            self.NearestIndice=NearestIndice['NearestIndiceMatrix'][:,:,self.nearestnums:]
            self.NearestTradingDates=NearestIndice['TradingDates']
            self.NearestTradingDates=pd.to_datetime(self.NearestTradingDates,format='%Y%m%d')
            self.NearestStockCodes=NearestIndice['StockCodes'] 
            stock1,stockindice1,stockindice2=np.intersect1d(self.StockCodes_unstacked,self.NearestStockCodes,return_indices=True)
            StockCode_in_Nearest_indice=[]
            self.Intersect_StockCodes=stock1
            self.Intersect_StockCodes_indice_StockCodesUnstacked=stockindice1
            self.Intersect_StockCodes_indice_NearestStockCodes=stockindice2
            for a in self.NearestStockCodes:
                if a not in self.StockCodes_unstacked:
                    temp=-1
                else:
                    temp=np.where(self.StockCodes_unstacked==a)[0][0]
                StockCode_in_Nearest_indice.append(temp)
            self.StockCodes_Unstacked_indice_of_Nearest=np.array(StockCode_in_Nearest_indice)
            #按照NearestStockCodes的顺序，在StockCodes_unstacked中的索引 """



    def data_backtest_nearest(self,NetralBase='True'):
        #因为相似度矩阵是numpy形式的，所以回测主要基于numpy进行

        if hasattr(self,'NearestIndice'):
            pass
        else:
            self.data_preprocessing_numpy()
        backtestindex=np.where((self.merged_data_tradingdates>=pd.to_datetime(self.BeginDate,format='%Y-%m-%d'))&(self.merged_data_tradingdates<=pd.to_datetime(self.EndDate,format='%Y-%m-%d')))[0]

        day_newfactor=[]
        IC=np.zeros(len(backtestindex))*np.nan
        P=np.zeros(len(backtestindex))*np.nan
        mean_return_group=np.zeros([len(backtestindex),self.groupnums])*np.nan
        std_return_group=np.zeros([len(backtestindex),self.groupnums])*np.nan
        mean_factor_group=np.zeros([len(backtestindex),self.groupnums])*np.nan

        for i in range(0,len(backtestindex)):
            print(self.merged_data_tradingdates[backtestindex[i]])
            idxinNearest=np.where(self.NearestTradingDates<=self.merged_data_tradingdates[backtestindex[i]])[0]#相似度矩阵日期坐标。因为相似度矩阵计算是左闭右开，当天可用
            if len(idxinNearest)==0:
                print('相似度矩阵数据不足')
                print(self.merged_data_tradingdates[backtestindex[i]])
                continue
            else:
                idxinNearest=idxinNearest[-1]#相似度矩阵日期坐标

            factor=self.Factor_2dMatrix[backtestindex[i],self.Intersect_StockCodes_indice_StockCodesUnstacked]#(合并时已经取了前一日factor和base数据)取出前一日, 所有股票池与相似度矩阵股票交集 因子数据 
            factor_nonan_indice=np.argwhere(~np.isnan(factor))#今天有self.Intersect_StockCodes_indice_StockCodesUnstacked[factor_nonan_indice]
            factor=factor[factor_nonan_indice].flatten()
            factor=Remove_Outlier(factor,para=3)
            factor=Normlization(factor,method='zscore')#取出前一日所有股票池因子数据 并去极值标准化

            base=[]
            for k in range(len(self.base_data.columns)):
                basefactor=self.BaseFactor_2dMatrix[k,backtestindex[i],self.Intersect_StockCodes_indice_StockCodesUnstacked[factor_nonan_indice]].flatten()#取出前一日 所有股票池 与相似度矩阵股票交集 基准因子数据
                basefactor=Remove_Outlier(basefactor,para=3)
                basefactor=Normlization(basefactor,method='zscore')
                base.append(basefactor)
            base=np.array(base)#取出前一日 所有股票池 基准因子数据 并去极值标准化

            nearest=self.NearestIndice[idxinNearest,self.Intersect_StockCodes_indice_NearestStockCodes[factor_nonan_indice],:]#当前日相似度矩阵
            today_newfactor=np.zeros((len(factor_nonan_indice)))#只计算StockUnivers_indice的股票
            today_newBaseFactor=base
            for j in range(len(factor_nonan_indice)):
                Nindice=nearest[j,:]
                Nindice=Nindice[Nindice>=0].astype(int)
                if len(Nindice)==0:
                    continue
                indice1=self.StockCodes_Unstacked_indice_of_Nearest[Nindice]
                indice1=indice1[indice1>=0]#无效值为-1 该组中的股票 所对应的 StockCodes_unstacked 的索引
                intersect1d,i1,i2=np.intersect1d(factor_nonan_indice,indice1,return_indices=True)
                Factor_j_Nearest=factor[i1]#取出第j个股票的最相近的股票的factor数据

                if len(np.where(~np.isnan(Factor_j_Nearest))[0])<=0.4*len(indice1):
                    break
                else:
                    today_newfactor[j]=(factor[j]-np.nanmean(Factor_j_Nearest))/(np.nanstd(Factor_j_Nearest)+0.00000001) 

                if NetralBase=='True':
                    newbase=[]
                    for k in range(len(self.base_data.columns)):
                        base_j=base[k,j]
                        base_j_nearest=base[k,i1]
                        nan_mask=np.isnan(base_j_nearest)
                        if nan_mask.sum()/len(nan_mask)<0.6:
                            today_newBaseFactor[k,j]=(base_j-np.nanmean(base_j_nearest)/(np.nanstd(base_j_nearest)+0.00000001))
                        else:#如果组内nan值太多，就默认为0
                            today_newBaseFactor[k,j]=0
 
            X=sm.add_constant(today_newBaseFactor.T)
            model=sm.OLS(today_newfactor,X)
            results=model.fit()
            today_newfactor=results.resid

            StockCodes=self.StockCodes_unstacked[self.Intersect_StockCodes_indice_StockCodesUnstacked[factor_nonan_indice]].flatten()
            newfactor = pd.DataFrame({'newfactor': today_newfactor}, index= StockCodes)
            newfactor['TradingDates']=self.merged_data_tradingdates[backtestindex[i]]
            newfactor.set_index('TradingDates', append=True, inplace=True)
            newfactor = newfactor.reorder_levels(['TradingDates', None])
            day_newfactor.append(newfactor)
            tvalues=results.tvalues
            nextlogreturn=self.NextLogReturn_2d[backtestindex[i],self.Intersect_StockCodes_indice_StockCodesUnstacked[factor_nonan_indice]]
            IC[i],P[i],mean_return_group[i,:],std_return_group[i,:],mean_factor_group[i,:],sorted_indice_group=cal_IC_P_GroupReturn_inday(today_newfactor,nextlogreturn,groupnum=self.groupnums)
        

        testdates=self.merged_data_tradingdates[backtestindex]
        self.newfactor_df=pd.concat(day_newfactor)
        self.IC=pd.DataFrame(IC,index=testdates)
        self.P=pd.DataFrame(P,index=testdates)
        index=pd.MultiIndex.from_product([testdates,range(self.groupnums)],names=['TradingDates','Group'])  
        self.GroupedMeanReturn=pd.DataFrame(mean_return_group.flatten(),index=index)
        self.GroupedStdReturn=pd.DataFrame(std_return_group.flatten(),index=index)
        self.GroupedMeanFactor=pd.DataFrame(mean_factor_group.flatten(),index=index)
       
        # self.GroupedMeanReturn= 
        # self.GroupedStdReturn= 
        # self.Tvalues=pd.DataFrame(resultdata['tvalues'])  

  


    def data_save(self,savepath=None):
        if savepath==None:
            savepath=self.tempdatapath

        with open('[SingleFactorTest].ini', 'r') as f:
            config = f.read()        
        data={
            'newfactor':self.newfactor_df,
            'IC':self.IC,
            'P':self.P,
            'T':self.T,
            'GroupedMeanReturn':self.GroupedMeanReturn,
            'GroupedStdReturn':self.GroupedStdReturn,
            'grouped_StockCodes':self.grouped_StockCodes,
            'config':config
        }
        with open(os.path.join(savepath,self.factor_name+'data.pkl'),'wb') as f:
            pickle.dump(data,f)
 
    def data_plot(self,savepath=None):
        if savepath==None:
            savepath=self.tempdatapath

        meanreturn=self.GroupedMeanReturn.groupby(level=0).mean()
        relative_meanreturn=self.GroupedMeanReturn-meanreturn
        cumsum_relative_meanreturn=relative_meanreturn.groupby(level=1).cumsum() 

        fig,axs=plt.subplots(2,gridspec_kw={'height_ratios': [4, 1]})
        for i in range(0,self.groupnums):
            axs[0].plot(cumsum_relative_meanreturn.xs(i,level=1),label='Group'+str(i))
        axs[0].legend(loc='upper left')   
        ax2 = axs[1].twinx()  # 创建第二个 y 轴
        line1 = self.T.rolling(120).mean().plot(ax=ax2, color='b', label='self.T')  # 使用第二个 y 轴绘制 self.T
        line2 = self.IC.rolling(120).mean().plot(ax=axs[1], color='r', label='self.IC')  # 使用原始 y 轴绘制 self.IC

        ax2.yaxis.label.set_color('b')  # 设置第二个 y 轴的标签颜色
        axs[1].yaxis.label.set_color('r')  # 设置原始 y 轴的标签颜色

        # 如果需要，你还可以设置每个 y 轴的标签
        ax2.set_ylabel('T')
        axs[1].set_ylabel('IC')


        plt.savefig(os.path.join(savepath,self.factor_name+'_figure.png'))
        plt.show()
        plt.close()


if __name__ == '__main__':
    PriceDf=pd.read_pickle(r'E:\Documents\PythonProject\StockProject\StockData\Price.pkl')
    test=Single_Factor_Test(r'E:\Documents\PythonProject\StockProject\MultiFactors\[SingleFactorTest].ini')
    test.filtered_stocks=PickupStocksByAmount(PriceDf)#股票池过滤
    test.data_loading_1st_time()
    test.data_backtest_one_hot('alpha_035')
    test.data_plot()
    test.data_save()

    for i in range(1,60):
        if i <=0:
            continue
        alpha = f'alpha_{i:03}'
        print(alpha)
        try:
            test.data_backtest_one_hot(alpha)
            test.data_plot()
            test.data_save()
        except:
            continue    
    # test.data_reloading_base('MarketCap')
    # test.data_preprocessing()
    # test.data_backtest_one_hot()
    # test.data_plot()
    # test.data_save()

    for i in range(1,190):
        if i <=120:
            continue
        alpha = f'alpha_{i:03}'
        print(alpha)
        try:
            test.data_reloading_factor(alpha)
            test.data_preprocessing()
            test.data_backtest_one_hot()
            test.data_plot()
            test.data_save()
        except:
            continue


    
 

 

