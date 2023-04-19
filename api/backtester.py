import os
import numpy as np
import pandas as pd
from pprint import pprint
from tqdm import tqdm
import sys

# from api import load_tushare
# from plt_head import *
_PATH = '//'


def cal_result_stat(df: pd.DataFrame, save_path: str = None, kind='cumsum', freq='D', lang='EN') -> pd.DataFrame:
    """
    对日度收益序列df计算相关结果
    :param lang:
    :param df: 值为日收益率小r，列index为日期DateTime
    :param save_path: 存储名（若有）
    :param kind: 累加/累乘
    :param freq: M D W Y
    :return: 结果面板
    """
    if kind == 'cumsum':
        df1 = df.cumsum() + 1
    elif kind == 'cumprod':
        df1 = df.add(1).cumprod()
    else:
        raise ValueError(f"""Invalid kind={kind}, only support('cumsum', 'cumprod')""")

    if freq == 'D':
        freq_year_adj = 242
    elif freq == 'W':
        freq_year_adj = 48
    elif freq == 'M':
        freq_year_adj = 12
    elif freq == 'Y':
        freq_year_adj = 1
    else:
        raise ValueError(f"""Invalid freq={freq}, only support('D', 'W', 'M', 'Y')""")

    data = df.copy()
    data['Date'] = data.index
    data['SemiYear'] = data['Date'].apply(lambda s: f'{s.year}-H{s.month // 7 + 1}')
    res: pd.DataFrame = data.groupby('SemiYear')[['Date']].last().reset_index()
    res.index = res['Date']
    res['Cash'] = (2e7 * df1.loc[res.index]).round(1)
    res['UnitVal'] = df1.loc[res.index]
    res['TRet'] = res['UnitVal'] - 1
    res['PRet'] = res['UnitVal'].pct_change()
    res.iloc[0, -1] = res['UnitVal'].iloc[0] - 1
    res['PSharpe'] = df.groupby(data.SemiYear).apply(lambda s: s.mean() / s.std() * np.sqrt(freq_year_adj)).values
    mdd = df1 / df1.cummax() - 1 if kind == 'cumprod' else df1 - df1.cummax()
    res['PMaxDD'] = mdd.groupby(data.SemiYear).min().values
    res['PCalmar'] = res['PRet'] / res['PMaxDD'].abs()
    res['PWinR'] = df.groupby(data['SemiYear']).apply(lambda s: (s > 0).mean()).values
    res['TMaxDD'] = mdd.min().values[0]
    res['TSharpe'] = (df.mean() / df.std() * np.sqrt(freq_year_adj)).values[0]
    res['TCalmar'] = res['TSharpe'] / res['TMaxDD'].abs()
    res['TWinR'] = (df > 0).mean().values[0]
    res['TAnnRet'] = (df1.iloc[-1] ** (freq_year_adj / len(df1)) - 1).values[0]

    res['Date'] = res['Date'].apply(lambda x: x.strftime('%Y-%m-%d'))
    if lang == 'CH':
        res1 = pd.DataFrame(columns=res.columns, index=['CH'])
        res1.loc['CH', :] = ['日期', '年度', '资金', '净值', '累计收益',
                             '收益', '夏普', '回撤', '卡玛', '胜率',
                             '总回撤', '总夏普', '总卡玛', '总胜率',
                             '年化收益', ]
        res = pd.concat([res, res1], ignore_index=True)

    res = res.set_index('SemiYear')
    if save_path is not None:
        table_save_safe(res, save_path)
    return res


class Backtester(object):

    def __init__(self, config: dict):
        self.conf: dict = config
        self._rtn: pd.DataFrame = pd.DataFrame()

    def rtn(self):
        if len(self._rtn) == 0:
            self._load_rtn()
        return self._rtn.copy()

    def _load_rtn(self, kind='ctc'):
        if kind == 'ctc':
            self._rtn = pd.read_csv(
                self.conf['path']['closeAdj'],
                index_col=0, parse_dates=True
            ).pct_change().iloc[1:]
        elif kind == 'oto':
            self._rtn = pd.read_csv(
                self.conf['path']['openAdj'],
                index_col=0, parse_dates=True
            ).pct_change().iloc[1:]


# %%
def main():
    # %%
    config_path = f'{_PATH}/config_stk.yaml'
    conf = load_tushare.conf_init(conf_path=config_path)
    print(conf['status'])
    # pprint(conf)
    backtester = Backtester(conf)
    rtn = backtester.rtn()

    # %%
    tmp = pd.read_csv(conf['path']['circ_mv'], index_col=0, parse_dates=True)
    tmp = tmp.apply(lambda s: (s - s.mean()) / s.std(), axis=1)
    fval = tmp.shift(1).iloc[1:]


    # %%
    # 20日反转因子
    winLen = 20
    fval = (-rtn).rolling(winLen).mean().iloc[winLen:].apply(
        lambda s: (s-s.mean()) / s.std(), axis=1).dropna(how='all', axis=1)

    # %%
    # 100日动量因子
    winLen = 100
    skipLen = 20
    fval = rtn.rolling(winLen).mean().shift(skipLen).iloc[winLen + skipLen:].apply(
        lambda s: (s-s.mean()) / s.std(), axis=1).dropna(how='all', axis=1)
    fval = fval.shift(1).iloc[1:]

    # %%
    begin_date = '2017-01-01'
    group_n = 10

    tmp = fval.loc[begin_date:]
    tmp = (tmp.rank(axis=1, pct=True) * group_n).round(0)
    tmp = pd.DataFrame(
        {f"Group{k}": (rtn.reindex_like(tmp) * (tmp == k)).mean(axis=1)
         for k in range(1, group_n+1)})
    # %%
    tmp.mean().plot.bar(width=.9, rot=0)
    plt.tight_layout()
    plt.show()

    # %%
    tmp.cumsum().add(1).plot(linewidth=3)
    # tmp.add(1).cumprod().plot(linewidth=3)
    plt.tight_layout(); plt.show()

    # %%
    print(0)


if __name__ == '__main__':
    main()
