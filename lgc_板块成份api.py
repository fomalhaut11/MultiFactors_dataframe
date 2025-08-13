import time
import os
from sqlalchemy import create_engine
import pandas as pd
import urllib.parse
"""
from lgc_板块成份api import lgc_板块成份股
cla_板块成份股 = lgc_板块成份股("申万行业板块")
成分股_data = cla.读取每日成分股(20210415, 20220415)
成分股_data = cla.读取每日成分股(20220415, 20220415)
"""
class lgc_板块成份股():
    def __init__(self, 指数类型="申万行业板块", 指数code='all', industry='all'):
        pwd = urllib.parse.quote_plus('Sy88sjk')
        str_sql = 'mssql+pymssql://%s:%s@%s/%s' % ('sa', pwd, '198.16.102.88', 'stock_data')
        self.engine = create_engine(str_sql, echo=False)
        str_ = "select * from tradingday order by tradingday"
        self.days = pd.read_sql(str_, self.engine)['tradingday'].values
        if 指数code != 'all':
            str_ = "select * from lgc_板块进出表 where concept_code='%s' order by sel_day" % 指数code
        else:
            if industry != 'all':
                str_ = "select * from lgc_板块进出表 where industry='%s' order by sel_day" % industry
            else:
                str_ = "select * from lgc_板块进出表 where 指数类型='%s' order by sel_day" % 指数类型
        self.进出表 = pd.read_sql(str_, self.engine)
        self.数据初始化()

    def 数据初始化(self):
        self.进出表['临时'] = 1
        self.进出表.loc[self.进出表['纳入_剔除'] == "剔除", '临时'] = -1
        self.进出表['纳入_剔除'] = self.进出表['临时']
        self.进出表['cum_纳入_剔除'] = self.进出表.groupby(by=['industry', 'concept_code', 'code'])['纳入_剔除'].cumsum()
        判断数据错误 = self.进出表['cum_纳入_剔除'].unique()
        if len(判断数据错误) >= 3:
            print("数据错误...............")
            y = self.进出表[~self.进出表['cum_纳入_剔除'].isin([0, 1])]
            # y1 = self.进出表[self.进出表['code'] == '000333']
            print(y)
            os._exit(0)

    def 读取每日成分股(self, st_day, end_day):
        days = self.days[(self.days >= st_day) & (self.days <= end_day)]
        进出表 = self.进出表[self.进出表['sel_day'] <= days[-1]]
        成分股_data = []
        for day in days:
            进出表_i = 进出表[进出表['sel_day'] <= day]
            最新进出表 = 进出表_i.groupby(by=['industry', 'concept_code', 'code']).last()
            最新进出表 = 最新进出表[最新进出表['cum_纳入_剔除'] != 0].reset_index()
            最新进出表['tradingday'] = day
            成分股_data.append(最新进出表[['tradingday', 'sel_day', 'industry', 'concept_code', 'concept_name', 'code', 'code_cn', 'exchange_id']])
        成分股_data = pd.concat(成分股_data)
        return 成分股_data

    def 写入数据库(self, st_day, end_day, 成分股_data, 指数类型):
        成分股_data['指数类型'] = 指数类型
        成分股_data['写入时间'] = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        str_ = "delete from lgc_板块成份股 where tradingday>=%d and tradingday<=%d and 指数类型='%s'" % (st_day, end_day, 指数类型)
        self.engine.execute(str_)
        pd.io.sql.to_sql(成分股_data, 'lgc_板块成份股', con=self.engine, if_exists='append', index=False)

if __name__ == '__main__':
    """
    ['沪深交易所核心指数', '申万行业板块', '申万300行业', '申万风格行业', 'wind主题策略指数', 'wind热门概念指数', 'wind基金重仓指数']
    指数code='all': 读取这个板块对应所有指数成份股
    指数code='000300': 读取沪深交易所核心指数
    """
    t1 = time.time()
    cla = lgc_板块成份股(指数类型="有code,指数类型可以不填", 指数code='851441')
    # cla = lgc_板块成份股(指数类型="沪深交易所核心指数", 指数code='000905')
    # 成分股_data = cla.读取每日成分股(20220415, 20220415)
    # cla = lgc_板块成份股(指数类型="有industry,指数类型可以不填", industry='sw_l2')
    # cla = lgc_板块成份股(指数类型="申万行业板块")
    成分股_data = cla.读取每日成分股(20211210, 20211216)
    print(time.time() - t1, '1')
    x = 1













