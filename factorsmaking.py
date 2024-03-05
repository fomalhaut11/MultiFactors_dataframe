import numpy as np
import pandas as pd
import sys
import GTJA_Alpha191 as a191
file_path =r'E:\Documents\PythonProject\StockProject'
sys.path.append(file_path)
import StockDataPrepairing  as SDP
import 股票数据接口_zx as FinData
from datetime import date
from datetime import datetime
import matplotlib.pyplot as plt
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
def factor_ROE_YoY(StockData3dMatrix,cla):
    x1=SDP.FactorMatrix_Report(StockData3dMatrix,cla)
    x2=SDP.FactorMatrix_Report(StockData3dMatrix,cla)
    x1.Title('lrb','DEDUCTEDPROFIT')
    x2.Title('fzb','EQY_BELONGTO_PARCOMSH')
    f1=x1.InitFactor()
    f2=x2.InitFactor()
    x1.initfactor=f1/f2#roe：扣非利润/归母权益

    dd=x1.apply_initial_data()
    x1.initfactor=dd
    f1expanding=x1.FactorExpanding()
    factorDF=x1.ExpandedFactor2DataFrame(f1expanding)
    return factorDF


def liq_amt(x2,para=20,show=False):
    CloseReturnM=x2.CloseReturn()
    FactorMatrix=np.ones([len(x2.TradingDates),len(x2.StockCodes)])*np.nan
    s=np.shape(CloseReturnM)
    for i in range(len(x2.StockCodes)):
        if show==True:
            print(i)
        for j in range(len(x2.TradingDates)):
            if j <=para:
                continue
            temp=CloseReturnM[j-para:j,i] 
            temp1=x2.pricematrix[j-para:j,i,5]
            std=np.std(np.abs(temp))
            std1=np.std(np.abs(temp1))/100000000
            if std1>0:
                FactorMatrix[j,i]=std/std1

    df=pd.DataFrame(FactorMatrix,index=x2.TradingDates,columns=x2.StockCodes)
    df=df.stack()
    df.index.names=['TradingDates','StockCodes']
    return df

def liq_amihud_std(x2,para=20,show=False):
    FactorMatrix=np.ones([len(x2.TradingDates),len(x2.StockCodes)])*np.nan
    CloseReturnM=x2.CloseReturn()
    data=CloseReturnM/(x2.pricematrix[:,:,5]/1000000000)
    for i in range(len(x2.StockCodes)):
        temp=data[:,i]
        for j in range(len(x2.TradingDates)):
            if j <=para:
                continue
            temp1=temp[j-para:j]
            FactorMatrix[j,i]=np.nanstd(temp1)
    df=pd.DataFrame(FactorMatrix,index=x2.TradingDates,columns=x2.StockCodes)
    df=df.stack()
    df.index.names=['TradingDates','StockCodes']
    return df     

def RoE(x1,r_method='ttm',e_method='avg'):
    x1.Title('lrb','DEDUCTEDPROFIT')
    x1.Title_2('fzb','EQY_BELONGTO_PARCOMSH')
    x1.initfactor=x1.InitFactor() 
    DEDUCTEDPROFIT=x1.apply_initial_data(r_method)
    x1.initfactor=x1.InitFactor_2()
    EQY_BELONGTO_PARCOMSH=x1.apply_initial_data(e_method)
    x1.initfactor=DEDUCTEDPROFIT/EQY_BELONGTO_PARCOMSH
    f1expanding=x1.FactorExpanding()
    factorDF=x1.ExpandedFactor2DataFrame(f1expanding)
    return factorDF
def RoE_zscores(x1,r_method='ttm',e_method='avg',len1=8):
    x1.Title('lrb','DEDUCTEDPROFIT')
    x1.Title_2('fzb','EQY_BELONGTO_PARCOMSH')
    x1.initfactor=x1.InitFactor() 
    DEDUCTEDPROFIT=x1.apply_initial_data(r_method)
    x1.initfactor=x1.InitFactor_2()
    EQY_BELONGTO_PARCOMSH=x1.apply_initial_data(e_method)
    x1.initfactor=DEDUCTEDPROFIT/EQY_BELONGTO_PARCOMSH
    x1.initfactor=x1.FinDataRollingZScores(len1)
    f1expanding=x1.FactorExpanding()
    factorDF=x1.ExpandedFactor2DataFrame(f1expanding)
    return factorDF

def SUE_ss(x1,len1=4):
    x1.Title('lrb','DEDUCTEDPROFIT')
    x1.initfactor=x1.InitFactor()
    x1.initfactor=x1.apply_initial_data('ss')
    x1.initfactor=x1.FinDataRollingZScores(len1)
    f1expanding=x1.FactorExpanding()
    factorDF=x1.ExpandedFactor2DataFrame(f1expanding)
    return factorDF

def SUE_ttm(x1,len1=4):
    x1.Title('lrb','DEDUCTEDPROFIT')
    x1.initfactor=x1.InitFactor()
    x1.initfactor=x1.apply_initial_data('ttm')
    x1.initfactor=x1.FinDataRollingZScores(len1)
    f1expanding=x1.FactorExpanding()
    factorDF=x1.ExpandedFactor2DataFrame(f1expanding)
    return factorDF
def SUE_yoy(x1,len1=4):
    x1.Title('lrb','DEDUCTEDPROFIT')
    x1.initfactor=x1.InitFactor()
    x1.initfactor=x1.apply_initial_data('yoy')
    x1.initfactor=x1.FinDataRollingZScores(len1)
    f1expanding=x1.FactorExpanding()
    factorDF=x1.ExpandedFactor2DataFrame(f1expanding)
    return factorDF

def CurrentRatio(x1,len1=8):
    x1.Title('fzb','TOT_CUR_ASSETS')
    x1.initfactor=x1.InitFactor()
    tca=x1.apply_initial_data('avg')#流动资产
    x1.Title('fzb','TOT_CUR_LIAB')
    x1.initfactor=x1.InitFactor()
    tcl=x1.apply_initial_data('avg')#流动负债
    x1.initfactor=tca/tcl
    f1expanding=x1.FactorExpanding()
    factorDF=x1.ExpandedFactor2DataFrame(f1expanding)
    x1.initfactor=x1.FinDataRollingZScores(len1)
    f1expanding=x1.FactorExpanding()
    factorDF_zscores=x1.ExpandedFactor2DataFrame(f1expanding)
    return factorDF,factorDF_zscores

def QuickRatio(x1,len1=8):
    x1.Title('fzb','TOT_CUR_ASSETS')
    x1.initfactor=x1.InitFactor()
    tca=x1.apply_initial_data('avg')#流动资产
    
    x1.Title('fzb','INVENTORIES')
    x1.initfactor=x1.InitFactor()
    INVENTORIES=x1.apply_initial_data('avg')#存货

    x1.Title('fzb','PREPAY')
    x1.initfactor=x1.InitFactor()
    PREPAY=x1.apply_initial_data('avg')#预付款项

    x1.Title('fzb','TOT_CUR_LIAB')
    x1.initfactor=x1.InitFactor()
    tcl=x1.apply_initial_data('avg')#流动负债
    x1.initfactor=(tca-INVENTORIES-PREPAY)/tcl
    f1expanding=x1.FactorExpanding()
    factorDF=x1.ExpandedFactor2DataFrame(f1expanding)
    x1.initfactor=x1.FinDataRollingZScores(len1)
    f1expanding=x1.FactorExpanding()
    factorDF_zscores=x1.ExpandedFactor2DataFrame(f1expanding)
    return factorDF,factorDF_zscores

    
def GOODWILLRatio(x1,len1=8):
    x1.Title('fzb','GOODWILL')
    x1.initfactor=x1.InitFactor()
    GOODWILL=x1.apply_initial_data('avg')#商誉
    x1.Title('fzb','TOT_ASSETS')
    x1.initfactor=x1.InitFactor()
    TOT_ASSETS=x1.apply_initial_data('avg')#总资产
    x1.initfactor=GOODWILL/TOT_ASSETS
    f1expanding=x1.FactorExpanding()
    factorDF=x1.ExpandedFactor2DataFrame(f1expanding)
    x1.initfactor=x1.FinDataRollingZScores(len1)
    f1expanding=x1.FactorExpanding()
    factorDF_zscores=x1.ExpandedFactor2DataFrame(f1expanding)
    return factorDF,factorDF_zscores

def ACCT_RCV__NET_CASH_FLOWS_OPER_ACT(x1,len1=8):
    x1.Title('fzb','ACCT_RCV')
    x1.initfactor=x1.InitFactor()
    ACCT_RCV=x1.apply_initial_data('ttm')#应收账款
    x1.Title('xjlb','NET_CASH_FLOWS_OPER_ACT')
    x1.initfactor=x1.InitFactor()
    NET_CASH_FLOWS_OPER_ACT=x1.apply_initial_data('ttm')#经营活动产生的现金流量净额
    x1.initfactor=ACCT_RCV/NET_CASH_FLOWS_OPER_ACT
    f1expanding=x1.FactorExpanding()
    factorDF=x1.ExpandedFactor2DataFrame(f1expanding)
    x1.initfactor=x1.FinDataRollingZScores(len1)
    f1expanding=x1.FactorExpanding()
    factorDF_zscores=x1.ExpandedFactor2DataFrame(f1expanding)
    return factorDF,factorDF_zscores

def wet_profit_ratio(x1,len1=8):
    x1.Title('xjlb','STOT_CASH_INFLOWS_OPER_ACT')
    x1.initfactor=x1.InitFactor()
    CASH_RECP_SG_AND_RS=x1.apply_initial_data('ttm')#经营活动现金流入小计
    x1.Title('xjlb','STOT_CASH_OUTFLOWS_OPER_ACT')
    x1.initfactor=x1.InitFactor()
    NET_INCR_LENDING_FUND=x1.apply_initial_data('ttm')#经营活动现金流出小计
    x1.initfactor=CASH_RECP_SG_AND_RS/NET_INCR_LENDING_FUND
    f1expanding=x1.FactorExpanding()
    factorDF=x1.ExpandedFactor2DataFrame(f1expanding)
    x1.initfactor=x1.FinDataRollingZScores(len1)
    f1expanding=x1.FactorExpanding()
    factorDF_zscores=x1.ExpandedFactor2DataFrame(f1expanding)
    return factorDF,factorDF_zscores

def net_profit_FIN_EXP_CS_ratio(x1,len1=8):
    x1.Title('xjlb','FIN_EXP_CS')
    x1.initfactor=x1.InitFactor()
    FIN_EXP_CS=x1.apply_initial_data('ttm')#财务费用,但是只有年报和半年报
    x1.Title('xjlb','NET_PROFIT_CS')
    x1.initfactor=x1.InitFactor()
    NETPROFIT=x1.apply_initial_data('ttm')#净利润,但是只有年报和半年报
    x1.initfactor=FIN_EXP_CS/NETPROFIT
    f1expanding=x1.FactorExpanding()
    factorDF=x1.ExpandedFactor2DataFrame(f1expanding)
    x1.initfactor=x1.FinDataRollingZScores(len1)
    f1expanding=x1.FactorExpanding()
    factorDF_zscores=x1.ExpandedFactor2DataFrame(f1expanding)
    return factorDF,factorDF_zscores

""" def financialdatadates(x1):
    datadate0=x1.ReportData.lrb[x1.Report_idx_Stock3d,:,0]#公布日期
    datadate1=x1.ReportData.lrb[x1.Report_idx_Stock3d,:,1]#报告期
    qnum=x1.ReportData.lrb[x1.Report_idx_Stock3d,:,-3]#季报标签
    ynum=x1.ReportData.lrb[x1.Report_idx_Stock3d,:,-2]#年份
    StockCodes=x1.StockCodes.repeat(np.shape(datadate1)[1])
    index1=pd.MultiIndex.from_arrays([StockCodes,pd.to_datetime(datadate1.reshape(-1),format='%Y%m%d')])
    realesed_dates_df=pd.DataFrame(pd.to_datetime(datadate0.reshape(-1),format='%Y%m%d'),index=index1,columns=['ReleasedDates'])
    qnum_df=pd.DataFrame(qnum.reshape(-1),index=index1,columns=['Quater'])
    ynum_df=pd.DataFrame(ynum.reshape(-1),index=index1,columns=['Year'])
    realesed_dates_df=pd.merge(realesed_dates_df,qnum_df,left_index=True, right_index=True)
    realesed_dates_df=pd.merge(realesed_dates_df,ynum_df,left_index=True, right_index=True)
    realesed_dates_df.index.names=['StockCodes','ReportDates']
    return realesed_dates_df

def ReleasedDatesCount(realesed_dates_df,TradingDates):
    reportenddate=realesed_dates_df.index.get_level_values(1).unique()
    realesed_dates_df.groupby('Quater')
    def row_calc(row,TradingDates):
        indices=np.searchsorted(row,TradingDates.values.flatten(),side='right')-1
        time_diffs=(TradingDates.values.flatten()-row.iloc[indices].values)/np.timedelta64(1,'D')
        time_diffs=pd.Series(time_diffs,index=TradingDates.iloc[:,0])
        return time_diffs 

    for i in range(4):
        print(i)
        quater1data=realesed_dates_df.loc[realesed_dates_df['Quater']==1+i]
        rd=quater1data['ReleasedDates'].to_frame().unstack()
        time_diffs=rd.apply(row_calc,axis=1,args=(TradingDates,))
        df=time_diffs.reset_index()
        df=df.melt(id_vars='StockCodes',var_name='TradingDates')
        df.set_index(['TradingDates','StockCodes'],inplace=True)
        columnnames='Quater'+str(i+1)
        df.columns=[columnnames]
        if i ==0:
            DateCount_df=df
        else:
            DateCount_df=pd.merge(DateCount_df,df,left_index=True,right_index=True)
    return DateCount_df """


def half_decay_factor(DateCount_df,factorDF,para=20,show=False):
    logbase=np.exp(-np.log(2)/para)
    minvalues= DateCount_df.min(axis=1)
    minvalues[minvalues<0]=np.nan
    indice=logbase**minvalues
    df=factorDF*indice
    return df








if __name__=='__main__':
    print('制作因子 start')
    
    #SDP.Main_Data_Renew()#StockDataPrepairing中的数据更新
    datasavepath=r'E:\Documents\PythonProject\StockProject\StockData'
    datapath=r'E:\Documents\PythonProject\StockProject\StockData'
    realead_dates=pd.read_pickle(datapath+'\\'+'realesed_dates_df.pkl')
    realead_dates_count_df=pd.read_pickle(datapath+'\\'+'realesed_dates_count_df.pkl')
    today=datetime.today()
    int_today=int(today.strftime('%Y%m%d'))

    PriceDf=pd.read_pickle(datapath+'\\'+'Price.pkl')
    Price_reset=PriceDf.reset_index()
    Price_reset=Price_reset.rename(columns={'TradingDates':'tradingday','StockCodes':'code'})
    StockData3dMatrix=SDP.StockDataDF2Matrix(Price_reset)
    cla=FinData.财报数据()
    cla.复制三张表()
    begindate=20140101
    cla.读取财报数据(begindate, int_today)
    x1=SDP.FactorMatrix_Report(StockData3dMatrix,cla)
    TradingDates=PriceDf.index.get_level_values(0).to_frame()
    PriceDf=None


    factorDF=RoE(x1)
    pd.to_pickle(factorDF,datasavepath+r'\ROE.pkl')
    factorDF=RoE_zscores(x1,r_method='ttm',e_method='avg',len1=8)
    pd.to_pickle(factorDF,datasavepath+r'\ROEzscores_8.pkl')
    factorDF=RoE_zscores(x1,r_method='ttm',e_method='avg',len1=4)
    pd.to_pickle(factorDF,datasavepath+r'\ROEzscores_4.pkl')
    factorDF=SUE_ss(x1,4)
    pd.to_pickle(factorDF,datasavepath+r'\SUE_ss_4.pkl')
    factorDF=SUE_ttm(x1,4)
    pd.to_pickle(factorDF,datasavepath+r'\SUE_ttm_4.pkl')
    factorDF=SUE_yoy(x1,4)
    pd.to_pickle(factorDF,datasavepath+r'\SUE_yoy_4.pkl')
    factorDF,factorDF_zscores=CurrentRatio(x1,8)
    pd.to_pickle(factorDF,datasavepath+r'\CurrentRatio.pkl')
    pd.to_pickle(factorDF_zscores,datasavepath+r'\CurrentRatio_zscores_8.pkl')
    factorDF,factorDF_zscores=QuickRatio(x1,8)
    pd.to_pickle(factorDF,datasavepath+r'\QuickRatio.pkl')
    pd.to_pickle(factorDF_zscores,datasavepath+r'\QuickRatio_zscores_8.pkl')
    factorDF,factorDF_zscores=GOODWILLRatio(x1,8)
    pd.to_pickle(factorDF,datasavepath+r'\GOODWILLRatio.pkl')
    pd.to_pickle(factorDF_zscores,datasavepath+r'\GOODWILLRatio_zscores_8.pkl')
    factorDF,factorDF_zscores=ACCT_RCV__NET_CASH_FLOWS_OPER_ACT(x1,8)
    pd.to_pickle(factorDF,datasavepath+r'\ACCT_RCV__NET_CASH_FLOWS_OPER_ACT.pkl')
    pd.to_pickle(factorDF_zscores,datasavepath+r'\ACCT_RCV__NET_CASH_FLOWS_OPER_ACT_zscores_8.pkl')
    factorDF,factorDF_zscores=wet_profit_ratio(x1,8)
    pd.to_pickle(factorDF,datasavepath+r'\wet_profit_ratio.pkl')
    pd.to_pickle(factorDF_zscores,datasavepath+r'\wet_profit_ratio_zscores_8.pkl')
 
 
    SUE_ss_4=pd.read_pickle(datasavepath+r'\SUE_ss_4.pkl')
    SUE_ss_4_hd20= half_decay_factor(realead_dates_count_df,SUE_ss_4,para=5,show=False)
    pd.to_pickle(SUE_ss_4_hd20,datasavepath+r'\SUE_ss_4_hd5.pkl')

    testdata1=pd.read_pickle(datasavepath+r'\ROE.pkl')
    testdata2=pd.read_pickle(datasavepath+r'\CurrentRatio.pkl')
    ROE_d_Currentratio=testdata1/testdata2
    pd.to_pickle(ROE_d_Currentratio,datasavepath+r'\ROE_d_Currentratio.pkl')

    gw=pd.read_pickle(datasavepath+r'\GOODWILLRatio.pkl')
    qr=pd.read_pickle(datasavepath+r'\QuickRatio.pkl')
    gw_divide_qr=gw/qr
    pd.to_pickle(gw_divide_qr,datasavepath+r'\GOODWILLRatio_divide_QuickRatio.pkl')