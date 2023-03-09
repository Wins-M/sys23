import os
import numpy as np
import pandas as pd
from pprint import pprint
from tqdm import tqdm
import sys

import load_tushare
from plt_head import *
_PATH = '//'


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
