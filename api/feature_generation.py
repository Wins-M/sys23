import os
import sys
from datetime import datetime
import numpy as np
import pandas as pd

import warnings
warnings.simplefilter("ignore")

import matplotlib.pyplot as plt
import seaborn
seaborn.set_style("darkgrid")
plt.rc("figure", figsize=(6, 3))
plt.rc("savefig", dpi=90)
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']
plt.rc("font", size=10)
plt.rcParams["date.autoformatter.hour"] = "%H:%M:%S"

_PATH = '/Users/winston/mygitrep/sys23/'  # TODO
_VERSION = '0512'
_DIR = '/Users/winston/Documents/BA/5-2/8.机器学习与商业数据挖掘/TermPaper/cache/'


def target_shop(conf, save_path, force_update=False):
    """准备预测目标"""

    if os.path.exists(save_path) and not force_update:
        return pd.read_pickle(save_path)

    # Time range
    from api.load_tushare import next_calendar_date, TushareLoader
    begin_date, end_date = conf['begin_date'], conf['end_date']
    tds = TushareLoader(conf).get_tradedates(
        start_date=next_calendar_date(begin_date, -120, lfmt='%Y-%m-%d', rfmt='%Y-%m-%d'),
        end_date=end_date,
        filename=conf['path']['tradedates'],
        fmt='%Y-%m-%d'
    )  # 交易日

    # Tradeable stock pool
    from api.backtester import get_suspend_sgn, get_updown_status, get_new_share, get_st_status, get_large_mv
    bd_1 = next_calendar_date(begin_date, delta=-20, lfmt='%Y-%m-%d', rfmt='%Y-%m-%d')
    ed1 = next_calendar_date(end_date, delta=20, lfmt='%Y-%m-%d', rfmt='%Y-%m-%d')
    suspend_d = get_suspend_sgn(conf, bd_1, ed1, tds, delay=3)
    updown_status = get_updown_status(conf, bd_1, ed1, tds, delay=2)
    new_share = get_new_share(conf, bd_1, ed1, tds, delay=30)
    st_status = get_st_status(conf, bd_1, ed1, tds)
    large_mv = get_large_mv(conf, bd_1, ed1, tds, bar=0.20)
    del bd_1, ed1
    cond_list = [new_share, updown_status, suspend_d, st_status, large_mv]

    # Target: T0开盘买入, T+9收盘卖出
    from backtester import get_return, get_winsorize, get_standardize
    kind, hd = 'otc', 9
    ntds = len(tds[(tds >= begin_date) & (tds <= end_date)])  # 样本时期内交易日天数（int）
    y = get_return(conf, begin_date, end_date, kind=kind, hd=hd, cond_list=cond_list, len_assert=ntds)
    y = y.loc['2016-01-04':][::5]
    # y = get_standardize(get_winsorize(y.T, nsigma=3)).T  # 截面去极值 + 标准化
    y = get_standardize(y.T).T  # 截面标准化
    y = y.stack().rename(f'rtn_{kind}_{hd}')
    y.to_pickle(save_path)

    return y


def feature_shop(conf, kws, center, rm_win=20, save_path=None, force_update=False, lags=None):
    """根据 center index (tradingdate + stockcode) 返回 featrues"""

    if lags is None:
        lags = [1]
    if isinstance(kws, str):
        kws = [kws]

    if save_path and os.path.exists(save_path) and not force_update:
        df0 = pd.read_pickle(save_path)
        kw0 = [f"{kw}_L{lag}" for kw in kws for lag in lags]
        kw1 = set(kw0).difference(df0.columns)
        kw2 = set([kw.rsplit('_', maxsplit=1)[0] for kw in kw1])
        if not kw1:
            df1 = feature_shop(conf, kw2, center, rm_win, None, True, lags)
            df0 = pd.concat([df0, df1[kw1]], axis=1)
            df0.to_pickle(save_path)
        return df0[kw0]

    begin_date, end_date = conf['begin_date'], conf['end_date']

    def get_signal(kw, lag=1):
        sr = pd.read_csv(conf['path'][kw], index_col=0, parse_dates=True).loc[:end_date]
        sr = sr.shift(lag)  # lag X day for prediction concern
        sr = sr.dropna(how='all').drop_duplicates()
        sr = sr.fillna(method='ffill', limit=rm_win // 2)
        sr = sr.sub(sr.rolling(rm_win).mean()).loc[begin_date:]
        sr = sr.stack().reindex(center)
        sr = sr.groupby(sr.index.get_level_values(0)).apply(lambda x: (x - np.nanmean(x)) / np.nanstd(x))  # Zscore
        return sr

    # Panel
    df = pd.DataFrame()
    for kw in kws:
        for lag in lags:
            print(f'\rFeature={kw}, Delay={lag}', end='')
            df[f"{kw}_L{lag}"] = get_signal(kw, lag=lag)
    if save_path:
        df.to_pickle(save_path)
    return df


def main():
    # %%
    print(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    os.chdir(_PATH)
    sys.path.append(_PATH)

    # Config file
    from api.load_tushare import conf_init
    config_path = f'{_PATH}/config_stk.yaml'
    conf = conf_init(conf_path=config_path)
    print(conf.keys())

    target_path = _DIR + 'target.pkl'
    target = target_shop(conf, save_path=target_path, force_update=False)

    feature_path = _DIR + 'features.pkl'
    kw = ['openAdj', 'highAdj', 'lowAdj', 'closeAdj', 'amount', 'vol',
          # 'volume_ratio', 'turnover_rate', 'turnover_rate_f',
          # 'pb', 'pe', 'pe_ttm', 'ps', 'ps_ttm', 'dv_ratio', 'dv_ttm',
          # 'total_share', 'float_share', 'free_share',
          # 'total_mv', 'circ_mv'
          ]
    features = feature_shop(conf, kws=kw, center=target.index, save_path=feature_path, force_update=False, lags=list(range(1, 21)))
    feature_description(features)
    pass


def feature_description(fea: pd.DataFrame):
    fea = fea.copy()
    fea = fea.dropna()

    # Cross section asset numbers
    fea.groupby(fea.index.get_level_values(0)).count().mean(axis=1).plot()
    plt.tight_layout(); plt.show()


if __name__ == '__main__':
    main()
