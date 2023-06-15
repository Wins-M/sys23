import numpy as np
import pandas as pd
import os
from typing import List

# Config
PATH = '/Users/winston/mygitrep/sys23/htf/{}'
DATA_FMT = '/Users/winston/mygitrep/sys23/cache/time_mom/{}'
# DATA_FMT = 'D:/time_mom/{}'
BD0, ED0 = '2020-10-01', '2022-12-31'
BD1, ED1 = '2022-01-01', '2022-12-31'


# %% TODO: Events
"""
因变量

T+1 日 到 T+61 日之间、经风险模型调整后的股票残差收益率

1. 平时/公告发生时/公告子类发生时
2. 全市场/成份股/特征选股


"""

# %% Feature construction
"""
自变量

公告当天 T 日的量价指标和公告前 5 天的量价指标

"""
os.listdir(DATA_FMT.format(''))

ind_tradingdate = pd.read_pickle(DATA_FMT.format('return_c2c.pkl')).loc[BD0: ED0].index

# def load_base_data(kw_list: List[str]):
base_data = {}

for f in ['return_c2o', 'return_o2c', 'return_c2c',
          'quote_close', 'quote_high', 'quote_low',
          'quote_preclose', 'quote_amount',
          'marketcap_freefloat', 'indu_citic1']:
    kw, file = f, DATA_FMT.format(f + '.pkl')

    if f[:5] == 'indu_':
        df = pd.read_pickle(file).loc[BD0: ED0].iloc[:, 0]
        df = df.unstack()
        df.reindex(ind_tradingdate).fillna(method='ffill')
        base_data[kw] = df
        del df
    else:
        base_data[kw] = pd.read_pickle(file).loc[BD0: ED0]

# All 2D frames are of same shape
for k, v in base_data.items():
    v.index.name = 'tradingdate'
    v.columns.name = 'stockcode'
    # base_data[k] = v
    base_data[k] = v.loc[BD1: ED1]  # TODO
    print(k, '\t', base_data[k].shape)


# ========= 变量 ==========

# 开盘跳空幅度
oprs = base_data['return_c2o']

# 日内涨幅
cors = base_data['return_o2c']

# 日收益
res = base_data['return_c2c']

# 最高价至收盘价幅度
chrs = base_data['quote_close'] / base_data['quote_high'] - 1

# 最低价至收盘价幅度
clrs = base_data['quote_close'] / base_data['quote_low'] - 1

# 最低价至最高价幅度
rhld = base_data['quote_high'] / base_data['quote_low'] - 1

# True Range, TR = MAX(H - L, H - C_1, C_1 - L)
ltrrg = pd.concat([(base_data['quote_high'] / base_data['quote_low'] - 1).stack(),
                   (base_data['quote_high'] / base_data['quote_preclose'] - 1).stack(),
                   (base_data['quote_preclose'] / base_data['quote_low'] - 1).stack()],
                  axis=1).max(axis=1).unstack()

# 换手率
turn = base_data['quote_amount'] / base_data['marketcap_freefloat']

# 对数换手率
rltrn = np.log(turn)


# %% 预处理
def risk_adjust(this: pd.DataFrame, kind='i', indus=None, lnsize=None):
    """TODO: 风险调整"""
    return this


indus = base_data['marketcap_freefloat']
lnsize = base_data['indu_citic1']


# %% 计算函数
def negof(df):
    """相反数"""
    return -df


def csregres(df, df1, is1d=False):
    """线性残差 cross-section regression"""
    from statsmodels.api import OLS

    # df = tsdelay(tslma(rltrn,5),1).stack()
    # df1 = rltrn.stack()

    pn = pd.DataFrame()
    if is1d:
        pn = pd.concat((df.rename('Y'), df1.rename('X')), axis=1)
    else:
        pn = pd.concat((df.stack().rename('Y'), df1.stack().rename('X')), axis=1)

    pn = pn.dropna()
    # TODO: df/df1 截距项 or 中心化？
    pn['C'] = 1
    resid = pn.groupby(pn.index.get_level_values(0), as_index=False).apply(lambda s: OLS(s['Y'], s[['X', 'C']], missing='drop').fit().resid)
    resid.index = resid.index.droplevel(0)

    if is1d:
        resid = resid.reindex(df.index)
    else:
        resid = resid.unstack().reindex_like(df)
    return resid


def tsdelay(df, p, is1d=False):
    """时序滞后"""
    if is1d or df.shape[1] == 1:
        df.unstack().shift(p).stack().reindex(df.index)
    return df.shift(p)


def divide(df, df1):
    """相除"""
    return df.div(df1)


def substract(df, df1):
    """相减"""
    return df.add(negof(df1))


def absl(df):
    """求绝对值"""
    return df.abs()


def ma(df, p, is1d=False):
    """移动平均"""
    if is1d or df.shape[1] == 1:
        return df.unstack().rolling(p).mean().stack().reindex(df.index)
    return df.rolling(p).mean()


def tslma(df, p, is1d=False):
    """TODO: ???"""
    return ma(df, p, is1d)


def tsstd(df, p, is1d=False):
    """时序标准差"""
    if is1d or df.shape[1] == 1:
        return df.unstack().rolling(p).std().stack().reindex(df.index)
    return df.rolling(p).std()


# %% TODO: 计算因子
"""
公告当天指标/公告前指标
收益率类/振幅类/换手率类

"""


def name2formula(s: str):
    return s.replace('[', '(').replace(']', ')').replace(';', ',')


factor_tror_all = {}
src_file = PATH.format('fac_tror.xlsx')
fac_name_all = pd.read_excel(src_file)['指标名称'].to_list()

for i in range(len(fac_name_all)):
    fac_name = fac_name_all[i]
    print(fac_name)

    df = eval(name2formula(fac_name))
    df.name = fac_name
    print(df.describe())

    factor_tror_all[fac_name] = df
del i


# %% 因子选股能力评估
"""
统计量：

平时的信息系数：交易指标平时与之后 60 个交易日收益率的 Pearson 相关系数，该 统计量衡量交易指标作为因子的长期选股能力；

公告时的信息系数：公告发生时交易指标与股票 60 个交易日后收益率的 Pearson 相 关系数，该统计量衡量交易指标对于公告发生后股票长期收益的预测性；

公告时与盈利指标的相关性：公告发生时交易指标与 ROE 同比的 Pearson 相关系数， 该统计量衡量了财务公告信息对于当日该交易指标的影响

"""


# %%
def main():
    pass


if __name__ == '__main__':
    main()
