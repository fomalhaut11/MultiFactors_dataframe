import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import MultiFactorsTool 
import gc
if __name__ == '__main__':
    BeginDate='20180101'
    EndDate='20230731'
    backtesttype='weekly'
    datasavepath=r'E:\Documents\PythonProject\StockProject\StockData'
    tempdatapath=r'E:\Documents\PythonProject\StockProject\StockData\TestTempData'
    factor_name='YoYZscores4_DEDUCTEDPROFIT'
    factor_name='ma_5'
    base_name=['MarketCap'] 
    matrixname='NearestIndice250_60' 
    c_name='classification_one_hot'
    nearestnums=50
    groupnums=10
    NetralBase='True'
    test=MultiFactorsTool.SingleFactorTest(BeginDate,EndDate,datasavepath)
    '''配置数据'''
    #test.quickstart(factor_name,base_name,matrixname,c_name,groupnums,c_type='one_hot',NetralBase)
    #test.quickstart(factor_name,base_name,matrixname,nearestnums,groupnums,c_type='nearest',NetralBase)#初次启动 读取数据

    test.preparation(factor_name,base_name,matrixname,nearestnums,groupnums,c_type='nearest')
    test.SingleFactorTest_NearestMatrix(groupnums,NetralBase)#单因子检验
    gc.collect()
    test.SingleFactorTest_NearestMatrix(groupnums,NetralBase,ifplot=True)#单因子检验
    