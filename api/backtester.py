import numpy as np
import pandas as pd
from tqdm import tqdm

# %matplotlib inline
import warnings

from api import load_tushare

warnings.simplefilter("ignore")
import seaborn
seaborn.set_style("darkgrid")
import matplotlib.pyplot as plt
plt.rc("figure", figsize=(9, 5))
plt.rc("font", size=12)
plt.rcParams['axes.autolimit_mode'] = 'round_numbers'
plt.rcParams['axes.xmargin'] = 0
plt.rcParams['axes.ymargin'] = 0
plt.rc("savefig", dpi=90)
plt.rcParams["date.autoformatter.hour"] = "%H:%M:%S"

# from api import load_tushare
# from plt_head import *
_PATH = '//'


def get_ic_stat(left: pd.DataFrame, rtns: dict, ranked=True, save_path=None) -> pd.DataFrame:
    """NEW: 计算IC，传入2D表格：因子值+Dict[收益率]"""
    mtd = 'spearman' if ranked else 'pearson'
    ic_stat = pd.DataFrame()
    for label, other in rtns.items():
        sr_name = label + (' rank IC' if ranked else ' IC')
        ic_stat[sr_name] = left.corrwith(other, axis=1, method=mtd)
    if save_path:
        ic_stat.to_csv(save_path)
    return ic_stat


def cal_cum_ic(df_ic: pd.DataFrame, save_path=None, plot_path=None, ishow=True, ylen=242):
    """NEW: 由截面IC计算累积IC"""
    df = df_ic.cumsum()
    df.columns = [f"{c}, IR: {df_ic[c].mean()/df_ic[c].std() * np.sqrt(ylen):.3f}" for c in df_ic.columns]
    if save_path:
        df.to_csv(save_path)
    if plot_path or ishow:
        df.plot()
        plt.tight_layout()
        if plot_path:
            plt.savefig(plot_path)
        if ishow:
            plt.show()
        else:
            plt.close()


# 收益率
def sift_tradeable(rtn: pd.DataFrame, cond: pd.DataFrame, kind='01', dropna=True) -> pd.DataFrame:
    """筛选收益率，只保留可行的"""
    # assert rtn.shape[0] == cond.shape[0]
    assert rtn.index.dtype == cond.index.dtype
    if kind == '01':
        res: pd.DataFrame = rtn * cond.reindex_like(rtn).fillna(1).replace(0, np.nan)
    else:
        raise Exception
    if dropna:
        res = res.dropna(how='all', axis=1)
    return res


def sift_tradeable_matrics(left: pd.DataFrame, cond_list: list, silence=False) -> pd.DataFrame:
    """多次进行筛选"""
    shape0 = left.shape
    for cond in cond_list:
        left = sift_tradeable(rtn=left, cond=cond)
    if not silence:
        print(f"After {len(cond_list)} tradeable sift: {shape0} -> {left.shape}")
    return left

    
# 去除T-59, ..., T0新上市
def get_new_share(conf, bd, ed, tds, delay=60, kind='01'):
    """新股标识，1为非新股，0为新股"""
    df = pd.read_csv(conf['path']['new_share'], parse_dates=['ipo_date'])[['ts_code', 'ipo_date']]
    df = df.drop_duplicates()
    df.columns = ['stockcode', 'tradingdate']
    df = df[df['tradingdate'] >= tds[0]]
    df['issue_date'] = 1
    df = df.pivot(index='tradingdate', columns='stockcode', values='issue_date')
    df = df.isna().astype(int)
    df = df.reindex(pd.to_datetime(tds)).fillna(1)
    df = (df.rolling(delay).sum() == delay).iloc[delay-1:]
    if kind == '01':
        df = df.astype(int)
    else:
        raise AttributeError(f'Unsupported kind={kind}!') 
    return df.loc[bd: ed]


# 去除T-1, T0涨跌停
def get_updown_status(conf, bd, ed, tds, delay=2, kind='01'):
    """涨跌停标识，1为正常，0为出现涨跌停
    
    * 2020-02-03: 因为新冠，春节后延迟一天开盘，大量跌停
    """
    df = pd.read_csv(conf['path']['updown_status'], parse_dates=['tradingdate'])
    
    # 诊断
    # df[df.iloc[:, :2].duplicated(keep=False)].to_csv('/Users/winston/Desktop/涨跌停异常.csv', index=False)
    
    # ou od cu cd 全部去除
    df['updown_status'] = 1
    df = df.drop_duplicates()
    df = df.pivot('tradingdate', 'stockcode', 'updown_status').isna().astype(int)
    df = df.reindex(pd.to_datetime(tds)).fillna(1)
    df = (df.rolling(delay).sum() == delay).iloc[delay-1:]
    
    if kind == '01':
        df = df.astype(int)
    else:
        raise AttributeError(f'Unsupported kind={kind}!') 
    df = df.loc[bd: ed]
    
    return df


# 去除T-2, T-1, T0停牌
def get_suspend_sgn(conf, bd, ed, tds, delay=3, kind='01'):
    """停牌标识，1为正常，0为出现停牌"""
    sus = pd.read_csv(conf['path']['suspend_d'], parse_dates=['tradingdate'])
    sus['suspend_d'] = 1
    sus = sus.pivot(index='tradingdate', columns='stockcode', values='suspend_d')
    sus = sus.isna().astype(int)
    sus = sus.reindex(pd.to_datetime(tds)).fillna(1)
    sus = (sus.rolling(delay).sum() == delay).iloc[delay-1:]  # delay=3: T0, T-1, T-2 not S/R
    if kind == '01':
        sus = sus.astype(int)
    else:
        raise AttributeError(f'Unsupported kind={kind}!') 
    sus = sus.loc[bd: ed]
    return sus


# 去除ST
def get_st_status(conf, bd, ed, tds, kind='01'):
    """ST标识，1为正常，0为出现ST"""
    df = pd.read_csv(conf['path']['st_status'], parse_dates=['tradingdate'])
    df = df.pivot(index='tradingdate', columns='stockcode', values='st_status')
    df.iloc[0] = df.iloc[0].fillna(1)
    df = df.fillna(method='ffill')
    df = df.reindex(pd.to_datetime(tds))
    df.iloc[0] = df.iloc[0].fillna(1)
    df = df.fillna(method='ffill').astype(int)
    if kind == '01':
        df = df.astype(int)
    else:
        raise AttributeError(f'Unsupported kind={kind}!') 
    df = df.loc[bd: ed]
    return df


# =========== OLD ============


def get_neutralize_sector_size(fval: pd.DataFrame, stdlnsize: pd.DataFrame, indus: pd.DataFrame) -> pd.DataFrame:
    """单因子行业和市值中性化"""
    from statsmodels.regression.linear_model import OLS

    factoro = fval[~fval.index.duplicated()].copy()
    factoro[np.isinf(factoro)] = np.nan
    cols = [i for i in factoro.columns if (i in stdlnsize.columns) and (i in indus.columns)]
    factoro = factoro.loc[:, cols]
    dic = {}
    date = factoro.index[100]
    for date in tqdm(factoro.index):
        try:
            industry = indus.loc[date, cols]
        except:
            break
        #indus = indus.loc[date, cols]
        z = pd.get_dummies(industry)
        s = stdlnsize.loc[date, cols]

        x = pd.concat([z, s], axis=1, sort=True)
        y = factoro.loc[date].sort_index(ascending=True)
        mask = (y.notnull()) & (x.notnull().all(axis=1))

        x1 = x[mask]
        y1 = y[mask]

        if len(y1) == 0:
            continue
        else:
            est = OLS(y1, x1).fit()
            dic[date] = est.resid
    #
    newfval = get_standardize(get_winsorize(pd.DataFrame.from_dict(dic, 'index')))
    newfval = newfval.reindex_like(fval)
    newfval[fval.isnull()] = np.nan
    #
    return newfval


def get_neutralize_sector(fval: pd.DataFrame, indus: pd.DataFrame) -> pd.DataFrame:
    """单因子面板的行业中性化"""
    factoro = fval.copy()
    dic = {}
    date = factoro.index[0]
    for date in tqdm(factoro.index):
        #未控制factoro的日期切片#
        try:
            x_indus = indus.loc[date, :]
        except:
            break  # indus里不含有该日期，即已经超出了范围（不限制enddate，日期为最新的交易）
        # mask
        x = pd.get_dummies(x_indus)
        y = factoro.loc[date, :]
        mask = y.notnull() & x.notnull().all(axis=1)
        x1 = x.sort_index()[mask]
        y1 = y.sort_index()[mask].astype('float')
        if len(y1) == 0:
            continue
        # 用行业所无法解释的residual部分
        from statsmodels.regression.linear_model import OLS
        est = OLS(y1, x1).fit()
        dic[date] = est.resid
        #
    newfactor = get_standardize(get_winsorize(pd.DataFrame.from_dict(dic, 'index')))# .reindex_like(factoro)
    newfactor[factoro.reindex_like(newfactor).isnull()] = np.nan

    return newfactor


def cal_result_stat(df: pd.DataFrame, save_path: str = None, kind='cumsum', freq='D', lang='EN') -> pd.DataFrame:
    """
    对日度收益序列df计算相关结果
    :param df: 值为日收益率小r，列index为日期DateTime
    :param save_path: 存储名（若有）
    :param kind: 累加/累乘
    :param freq: M D W Y
    :param lang: EN CH
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

            
def get_standardize(df):
    """按日标准化"""
    if (df.notnull().sum(axis=1) <= 1.).any():
        import warnings
        warnings.warn('there are {} days only has one notna data'.format((df.notnull().sum(axis=1) <= 1.).sum()))
    #
    return df.sub(df.mean(axis=1), axis=0).div(df.std(axis=1).replace(0, 1), axis=0)
        # 注意：有时会出现某一天全部为0的情况。此时标准化会带来问题


def get_winsorize(df, nsigma=3):
    """去极值缩尾"""
    df = df.copy().astype(float)
    md = df.median(axis=1)
    mad = 1.483 * (df.sub(md, axis=0)).abs().median(axis=1)
    up = df.apply(lambda k: k > md + mad * nsigma)
    down = df.apply(lambda k: k < md - mad * nsigma)
    df[up] = df[up].rank(axis=1, pct=True).multiply(mad * 0.5, axis=0).add(md + mad * (0.5 + nsigma), axis=0)
        # mad*nsigma后0.5mad大小分布
    df[down] = df[down].rank(axis=1, pct=True).multiply(mad * 0.5, axis=0).add(md - mad * (0.5 + nsigma), axis=0)
        # -1*mad*nsigma前0.5mad大小分布
    return df
    
            
def factor_neutralization(fv: pd.DataFrame, neu_mtd='n', ind_path=None, mv_path=None) -> pd.DataFrame:
    """
    返回单因子fval中性化后的结果。依次进行
    - n/i/iv: 去极值（缩尾）｜标准化
    - i/iv: 行业中性化
    - iv: 市值中性化
    (2022.3.7后,行业和市值也滞后1日,同因子对齐)

    :param neu_mtd: 中性化方式，仅标准化(n)，按行业(i)，行业+市值(iv)
    :param ind_path: 行业分类2D面板，由get_data.py获取，tradingdate,stockcode,<str>
    :param mv_path: 市值2D面板，由get_data.py获取，tradingdate,stockcode,<float>
    :param fv: 待中性化的因子值面板
    :return: 中性化后的因子

    """
    fv0 = get_standardize(get_winsorize(fv))  # standardized factor panel
    fv1 = pd.DataFrame()  # result factor panel
    # plt.hist(fv0.iloc[-1]); plt.show()
    if neu_mtd == 'n':
        return fv0
    elif 'i' in neu_mtd:
        indus = pd.read_csv(ind_path, index_col=0, parse_dates=True, dtype=str)
        indus = indus.shift(1).reindex_like(fv).fillna(method='ffill').fillna(method='bfill')  # 月调整,延用昨日(首月不准)
        if neu_mtd == 'i':  # 行业中性化
            print('NEU INDUS...')
            fv1 = get_neutralize_sector(fval=fv0, indus=indus)
        elif neu_mtd == 'iv':  # 行业&市值中性化
            size = pd.read_csv(mv_path, index_col=0, parse_dates=True, dtype=float)
            size = size.shift(1).reindex_like(fv)
            size_ln = size.apply(np.log)
            size_ln_std = size_ln  # size_ln_std = get_standardize(get_winsorize(size_ln))
            print('NEU INDUS & MKT_SIZE...')
            fv1 = get_neutralize_sector_size(fval=fv0, indus=indus, stdlnsize=size_ln_std)
        return fv1

    return fv1


def cal_ic(fv_l1, ret, lag=1, ranked=False, ishow=False, save_path=None) -> pd.DataFrame:
    """计算IC：昨日因子值与当日收益Pearson相关系数"""
    mtd = 'spearman' if ranked else 'pearson'
    ic = fv_l1.shift(lag - 1).corrwith(ret, axis=1, drop=False, method=mtd)  # lag-1: fv_l1是滞后的因子
    ic = pd.DataFrame(ic)
    ic.columns = ['IC']

    if save_path is not None:
        ic.plot.hist(figsize=(9, 5), bins=50, title='IC distribution')
        plt.tight_layout()
        plt.savefig(save_path, transparent=False)
        if ishow:
            plt.show()
        plt.close()

    return ic


def cal_ic_stat(data):
    """获取IC的统计指标"""
    from scipy import stats
    
    data = data.dropna()
    t_value, p_value = stats.ttest_1samp(data, 0)  # 计算ic的t
    pdata = data[data >= 0]
    ndata = data[data < 0]
    data_stat = list(zip(
        data.mean(), data.std(), data.skew(), data.kurt(), t_value, p_value,
        pdata.mean(), ndata.mean(), pdata.std(), ndata.std(),
        ndata.isna().mean(), pdata.isna().mean(), data.mean() / data.std()
    ))
    data_stat = pd.DataFrame(data_stat).T
    data_stat.columns = data.columns
    data_stat.index = ['mean', 'std', 'skew', 'kurt', 't_value', 'p_value',
                       'mean+', 'mean-', 'std+', 'std-', 'pos ratio', 'neg ratio', 'IR']
    #
    return data_stat


def cal_ic_decay(fval_neutralized, ret, maxlag=20, ishow=False, save_path=None) -> pd.DataFrame:
    """计算IC Decay，ret为滞后一期的因子值，ret为当日收益率"""
    from tqdm import tqdm
    ic_decay = {0: np.nan}
    print("Calculating IC Decay...")
    for k in tqdm(range(1, maxlag + 1)):
        ic_decay[k] = cal_ic(fval_neutralized, ret, lag=k, ranked=True, ishow=False).mean().values[0]
    res = pd.DataFrame.from_dict(ic_decay, orient='index', columns=['IC_mean'])

    if save_path is not None:
        res.plot.bar(figsize=(9, 5), title='IC Decay')
        plt.tight_layout()
        plt.savefig(save_path, transparent=False)
        if ishow:
            plt.show()
        plt.close()

    return res


def get_long_short_group(df: pd.DataFrame, ngroups: int, save_path=None) -> pd.DataFrame:
    """
    因子值替换为分组标签
    :param df: 因子值（经过标准化、中性化）
    :param ngroups: 分组数(+)；多头/空头内资产数量取负值(-)，若股池不够大，重叠部分不持有
    :param save_path: 若有，分组情况存到本地
    :return: 因子分组，nan仍为nan，其余为分组编号 0~(分组数-1)
    """
    res = None
    if ngroups < 0:
        cnt = df.count(axis=1)
        nm = (cnt + 2 * ngroups).abs()
        lg = (cnt - nm) / 2  # 上阈值
        hg = lg + nm  # 下阈值
        rnk = df.rank(axis=1)  # , method='first')
        res = rnk * 0 + 1  # 中间组
        res[rnk.apply(lambda s: s <= lg, axis=0)] = 0  # 空头
        res[rnk.apply(lambda s: s >= hg, axis=0)] = 2  # 多头
        # res = rnk * 0  # 一般
        # res[rnk.apply(lambda s: s >= hg, axis=0)] = 1  # 多头
    elif ngroups == 1:
        res = df
    else:
        res = df.rank(axis=1, pct=True).applymap(lambda x: x // (1 / ngroups))

    if save_path is not None:
        res.to_csv(save_path)

    return res


def cal_long_short_group_rtns(long_short_group, ret, idx_weight, ngroups, save_path=None) -> pd.DataFrame:
    """
    计算各组收益并存储
    :param long_short_group:
    :param ret:
    :param idx_weight: 只用多空分组标签下，股池内股票权重计算分组收益率
    :param ngroups:
    :param save_path:
    :return:
    """

    long_short_group = long_short_group.reindex_like(ret)
    if idx_weight is None:
        idx_weight = ret * 0 + 1
    else:
        idx_weight = idx_weight.reindex_like(ret)

    gn = 3 if ngroups < 0 else 2 if ngroups == 1 else ngroups
    ret_group = pd.DataFrame(index=ret.index)
    gi = 0
    for gi in range(gn):
        mask = (long_short_group == gi)
        ret1 = ret[mask]
        w1 = idx_weight[mask].apply(lambda s: s / np.nansum(s), axis=1)
        ret_group['$Group_{' + str(gi) + '}$'] = (ret1 * w1).sum(axis=1)

    if save_path is not None:  # 保存多空分组收益
        ret_group.to_csv(save_path)

    return ret_group


def plot_rtns_group(ret_group: pd.DataFrame, ishow=False, save_path=None, cumsum=True):
    """分层收益情况"""
    if cumsum:
        ret_group_cumulative = ret_group.cumsum()
    else:
        ret_group_cumulative = ret_group.add(1).cumprod()

    if save_path is not None:
        ret_group_cumulative.plot(grid=True, figsize=(9, 5), linewidth=1, title="Group Test Result")
        plt.tight_layout()
        plt.savefig(save_path, transparent=False)
        if ishow:
            plt.show()
        plt.close()

    return ret_group_cumulative


def cal_total_ret_group(ret_group, ishow=False, save_path=None) -> pd.DataFrame:
    """由分组收益面板，计算分组总收益"""
    ret_group_total = ret_group.sum()  # add(1).prod().add(-1)  改成累加

    if save_path is not None:
        ret_group_total.plot(figsize=(9, 5), kind='bar', title="Total Return of Group")
        plt.tight_layout()
        plt.savefig(save_path, transparent=False)
        if ishow:
            plt.show()
        plt.close()

    return ret_group_total


def portfolio_statistics_from_weight(weight, cost_rate, all_ret, save_path=None):
    """对持仓计算结果"""
    res = pd.DataFrame(index=weight.index)
    res['NStocks'] = (weight.abs() > 0).sum(axis=1)
    res['Turnover'] = weight.diff().abs().sum(axis=1)
    res['Return'] = (all_ret.reindex_like(weight) * weight).sum(axis=1)
    res['Return_wc'] = res['Return'] - res['Turnover'] * cost_rate
    res['Wealth(cumsum)'] = res['Return'].cumsum().add(1)
    res['Wealth_wc(cumsum)'] = res['Return_wc'].cumsum().add(1)
    res['Wealth(cumprod)'] = res['Return'].add(1).cumprod()
    res['Wealth_wc(cumprod)'] = res['Return_wc'].add(1).cumprod()
    if save_path:
        res.to_csv(save_path)
    return res


def cal_sr_max_drawdown(df: pd.Series, ishow=False, title=None, save_path=None, kind='cumprod') -> pd.DataFrame:
    """计算序列回撤"""
    cm = df.cummax()
    mdd = pd.DataFrame(index=df.index)
    mdd[f'{df.name}_maxdd'] = (df / cm - 1) if kind == 'cumprod' else (df - cm)

    if save_path is not None:
        try:
            mdd.plot(kind='area', figsize=(9, 5), grid=True, color='y', alpha=.5, title=title)
        except ValueError:
            mdd[mdd > 0] = 0
            mdd.plot(kind='area', figsize=(9, 5), grid=True, color='y', alpha=.5, title=title)
        finally:
            plt.tight_layout()
            plt.savefig(save_path, transparent=False)
            if ishow:
                plt.show()
            plt.close()

    return mdd


def table_save_safe(df: pd.DataFrame, tgt: str, kind=None, notify=False, **kwargs):
    """
    安全更新已有表格（当tgt在磁盘中被打开，5秒后再次尝试存储）
    :param df: 表格
    :param tgt: 目标地址
    :param kind: 文件类型，暂时仅支持csv
    :param notify: 是否
    :return:
    """
    kind = tgt.rsplit(sep='.', maxsplit=1)[-1] if kind is None else kind

    if kind == 'csv':
        func = df.to_csv
    elif kind == 'xlsx':
        func = df.to_excel
    elif kind == 'pkl':
        func = df.to_pickle
    elif kind == 'h5':
        if 'key' in kwargs:
            hdf_k = kwargs['key']
        elif 'k' in kwargs:
            hdf_k = kwargs['k']
        else:
            raise Exception('Save FileType hdf but key not given in table_save_safe')

        def func():
            df.to_hdf(tgt, key=hdf_k)
    else:
        raise ValueError(f'Save table filetype `{kind}` not supported.')

    try:
        func(tgt)
    except PermissionError:
        print(f'Permission Error: saving `{tgt}`, retry in 5 seconds...')

        import time
        time.sleep(5)
        table_save_safe(df, tgt, kind)
    finally:
        if notify:
            print(f'{df.shape} saved in `{tgt}`.')


class Signal(object):

    def __init__(self, data: pd.DataFrame, bd=None, ed=None, neu=None):
        self.fv = data
        self.bd = bd
        self.ed = ed
        self.__update_bd_ed__()
        self.neu_status = neu  # 中性化情况
        self.ic = None
        self.ic_rank = None
        self.ic_stat = None
        self.ic_decay = None
        self.ic_ir_cum = None

    def __update_bd_ed__(self):
        self.bd = self.fv.index[0] if self.bd is None else max(self.bd, self.fv.index[0])
        self.ed = self.fv.index[-1] if self.ed is None else min(self.ed, self.fv.index[-1])

    def shift_1d(self, d_shifted=1):
        self.fv = self.fv.shift(d_shifted).iloc[d_shifted:]
        self.__update_bd_ed__()

    def keep_tradeable(self, mul: pd.DataFrame):
        self.fv = self.fv.reindex_like(mul.loc[self.bd: self.ed])
        self.fv = self.fv * mul.loc[self.bd: self.ed]
        self.fv = self.fv.astype(float)

    def neutralize_by(self, mtd, p_ind, p_mv):
        self.fv = factor_neutralization(self.fv, mtd, p_ind, p_mv)
        self.neu_status = mtd  # 已做中性化

    def cal_ic(self, all_ret):
        if self.neu_status is None:
            raise AssertionError('Neutralize fval before IC calculation!')
        else:
            ret = all_ret.loc[self.bd: self.ed]
            self.ic = cal_ic(fv_l1=self.fv, ret=ret, ranked=False)
            self.ic_rank = cal_ic(fv_l1=self.fv, ret=ret, ranked=True)

    def cal_ic_statistics(self):
        if self.ic is None:
            raise AssertionError('Calculate IC before IC statistics!')
        ic_stat = pd.DataFrame()
        ic_stat['IC'] = cal_ic_stat(data=self.ic)
        ic_stat['Rank IC'] = cal_ic_stat(data=self.ic_rank)
        self.ic_stat = ic_stat.astype('float16')

    def cal_ic_decay(self, all_ret, lag):
        ret = all_ret.loc[self.bd: self.ed]
        self.ic_decay = cal_ic_decay(fval_neutralized=self.fv, ret=ret, maxlag=lag)

    def cal_ic_ir_cum(self):
        pass

    def plot_ic(self, ishow: bool, path_f: str):
        self.ic.plot.hist(figsize=(9, 5), bins=50, title='IC distribution')
        plt.tight_layout()
        plt.savefig(path_f.format('IC.png'), transparent=False)
        if ishow:
            plt.show()
        else:
            plt.close()
        self.ic_rank.plot.hist(figsize=(9, 5), bins=50, title='IC distribution')
        plt.tight_layout()
        plt.savefig(path_f.format('ICRank.png'), transparent=False)
        if ishow:
            plt.show()
        else:
            plt.close()

    def plot_ic_decay(self, ishow, path):
        self.ic_decay.plot.bar(figsize=(9, 5), title='IC Decay')
        plt.tight_layout()
        plt.savefig(path, transparent=False)
        if ishow:
            plt.show()
        else:
            plt.close()

    def get_fv(self, bd=None, ed=None) -> pd.DataFrame:
        bd = self.bd if bd is None else bd
        ed = self.ed if ed is None else ed
        return self.fv.loc[bd: ed].copy()

    def get_fbegin(self):
        return self.bd

    def get_fend(self):
        return self.ed

    def get_ic(self, ranked=True, path=None) -> pd.DataFrame:
        tgt = self.ic_rank if ranked else self.ic
        if tgt is None:
            raise AssertionError('Calculate IC first!')
        if path is not None:
            tgt.to_csv(path)
        return tgt.copy()

    def get_ic_stat(self, path=None) -> pd.DataFrame:
        if path is not None:
            self.ic_stat.to_csv(path)
        return self.ic_stat.copy()

    def get_ic_mean(self, ranked=True) -> float:
        if self.ic_stat is None:
            raise AssertionError('Calculate IC statistics before `get_ic_mean`')
        return self.ic_stat.loc['mean', ['IC', 'Rank IC'][ranked]]

    def get_ic_decay(self, path=None) -> pd.DataFrame:
        if path is not None:
            self.ic_decay.to_csv(path)
        return self.ic_decay


class Portfolio(object):
    r"""
    w: DataFrame of shape (n_views, n_assets), holding weight row sum 1
    cr: float, cost rate

    ------
    port = Portfolio(w,[ cr, ret])
    port.cal_panel_result(cr: float, ret: pd.DataFrame)
    port.cal_half_year_stat(wc: bool)

    """

    def __init__(self, w: pd.DataFrame = None, **kwargs):
        self.w_2d = w
        self.views = w.index.to_list() if isinstance(w, pd.DataFrame) else list()

        self.panel: pd.DataFrame = pd.DataFrame(
            index=self.views, columns=['NStocks', 'Turnover', 'Return', 'Return_wc',
                                       'Wealth(cumsum)', 'Wealth_wc(cumsum)',
                                       'Wealth(cumprod)', 'Wealth_wc(cumprod)'])
        self.cost_rate = None
        if ('cr' in kwargs) and ('ret' in kwargs):
            self.cal_panel_result(cr=kwargs['cr'], ret=kwargs['ret'])

        self.stat = {True: pd.DataFrame(), False: pd.DataFrame()}  # half year statistics
        self.mdd = {}

    def cal_panel_result(self, cr: float, ret: pd.DataFrame):
        """From hist ret and hold weight, cal panel: NStocks, Turnover, Return(), Wealth()"""
        self.cost_rate = cr
        self.panel = portfolio_statistics_from_weight(weight=self.w_2d, cost_rate=cr, all_ret=ret)

    def cal_half_year_stat(self, wc=False, freq='D', lang='EN'):
        """cal half year statistics from `panel`"""
        if self.panel.dropna().__len__() == 0:
            raise Exception('Empty self.panel')
        col = 'Return_wc' if wc else 'Return'
        self.stat[wc] = cal_result_stat(self.panel[[col]], freq=freq, lang=lang)

    def get_position_weight(self, path=None) -> pd.DataFrame:
        if path is not None:
            table_save_safe(df=self.w_2d, tgt=path)
        return self.w_2d.copy()

    def get_panel_result(self, path=None) -> pd.DataFrame:
        if path is not None:
            table_save_safe(df=self.panel, tgt=path)
        return self.panel.copy()

    def get_half_year_stat(self, wc=False, path=None) -> pd.DataFrame:
        if wc not in self.stat.keys() or len(self.stat[wc]) == 0:
            print('Calculate half-year statistics before get_half_year_stat...')
            self.cal_half_year_stat(wc=wc)
        if path is not None:
            table_save_safe(df=self.stat[wc], tgt=path)
        return self.stat[wc]

    def plot_turnover(self, ishow, path, title='Turnover'):
        if self.panel is None:
            raise Exception('Calculate panel result before plot turnover!')

        sr = self.panel['Turnover']
        sr.plot(figsize=(9, 5), grid=True, title=title+f', M={sr.mean() * 100:.2f}%')
        plt.tight_layout()
        plt.savefig(path, transparent=False)
        if ishow:
            plt.show()
        plt.close()

    def plot_cumulative_returns(self, ishow, path, kind='cumsum', title=None):
        title = f'Portfolio Absolute Result ({kind})' if title is None else title
        self.panel[[f'Wealth({kind})', f'Wealth_wc({kind})']].plot(figsize=(9, 5), grid=True, title=title)
        plt.tight_layout()
        plt.savefig(path, transparent=False)
        if ishow:
            plt.plot()
        plt.close()

    def plot_max_drawdown(self, ishow, path, wc=False, kind='cumsum', title=None):
        col = f'Wealth_wc({kind})' if wc else f'Wealth({kind})'
        title = f'MaxDrawdown {col}' if title is None else title
        df = self.panel[col].copy()
        df = df + 1 if df.iloc[0] < .6 else df
        cal_sr_max_drawdown(df=df, ishow=ishow, title=title, save_path=path, kind=kind)

    def plot_weight_hist(self, path, ishow=False, title='Weight Distribution'):
        plt.hist(self.w_2d.values.flatten(), bins=100)
        plt.title(title)
        plt.tight_layout()
        plt.savefig(path, transparent=False)
        if ishow:
            plt.show()
        plt.close()

    def plot_port_asset_num(self, path, ishow=False, rw=None):
        if rw is None:
            rw = {'D': 1, '20D': 20, '60D': 60}
        n_stk = self.panel['NStocks']
        tmp = pd.DataFrame()
        for k, v in rw.items():
            tmp[k] = n_stk.rolling(v).mean()
        tmp.plot(title=f'Portfolio Asset Numbers (rolling mean), M={n_stk.mean():.2f}', linewidth=1)
        plt.tight_layout()
        plt.savefig(path, transparent=False)
        if ishow:
            plt.show()
        plt.close()

    def plot_asset_weight(self, path, ishow=False, title='Asset Weight'):
        tmp = pd.DataFrame()
        tmp['w-MAX'] = self.w_2d.max(axis=1)
        tmp['w-MEDIAN'] = self.w_2d.median(axis=1)
        tmp['w-AVERAGE'] = self.w_2d.mean(axis=1)
        tmp.plot(title=title, linewidth=1)
        plt.tight_layout()
        plt.savefig(path, transparent=False)
        if ishow:
            plt.show()
        plt.close()

    def get_stock_number(self) -> pd.Series:
        return self.panel['NStocks'].copy()

    def get_turnover(self) -> pd.Series:
        return self.panel['Turnover'].copy()

    def get_daily_ret(self, wc=False) -> pd.Series:
        return self.panel['Return_wc' if wc else 'Return'].copy()

    def get_wealth(self, wc=False, kind='cumsum') -> pd.Series:
        return self.panel[f'Wealth_wc({kind})' if wc else f'Wealth({kind})'].copy()


class Strategy(object):

    def __init__(self, sgn: Signal, ng: int):
        self.sgn = sgn  # factor value after preprocessing
        self.ng = ng  # number of long-short groups
        self.ls_group = None
        self.ls_g_rtns = None
        self.holddays = None
        from typing import Dict
        self.portfolio: Dict[str, Portfolio] = {}
        # self.all_panels: Dict[str, pd.DataFrame] = {}

    def cal_long_short_group(self):
        self.ls_group = get_long_short_group(df=self.sgn.fv, ngroups=self.ng)

    def cal_group_returns(self, ret, idx_w=None):
        """TODO: idx_w指定其他股池时，先进行分组标签，再用指数成分内权重计算标签收益率"""
        ret = ret.loc[self.sgn.get_fbegin(): self.sgn.get_fend()]  # ?冗余
        self.ls_g_rtns = cal_long_short_group_rtns(
            long_short_group=self.ls_group, ret=ret, idx_weight=idx_w, ngroups=self.ng)

    def cal_long_short_panels(self, idx_w, hd, rvs, cr, ret):
        """由self.ls_group获得long, short, long_short, baseline的Portfolio，并计算序列面板"""
        if idx_w is None:
            idx_w = ret * 0 + 1
        else:
            idx_w = idx_w.reindex_like(ret)
        for kind in ['long_short', 'long', 'short', 'baseline']:
            self.portfolio[kind] = self.get_holding_position(idx_w=idx_w, hd=hd, rvs=rvs, kind=kind)
            self.portfolio[kind].cal_panel_result(cr=cr, ret=ret)
            # self.all_panels[kind] = self.portfolio[kind].get_panel()

    def plot_group_returns(self, ishow, path):
        plot_rtns_group(self.ls_g_rtns, ishow, path)

    def plot_group_returns_total(self, ishow, path):
        cal_total_ret_group(self.ls_g_rtns, ishow, path)

    def plot_long_short_turnover(self, ishow, path, roll=1):
        long_short_turnover = pd.concat([self.portfolio[k].get_turnover().rename(k) for k in ['long', 'short']], axis=1)
        if roll > 1:
            long_short_turnover = long_short_turnover.rolling(roll).mean()
            kw = f', rolling {roll} day mean'
        else:
            kw = ''
        long_short_turnover.plot(figsize=(9, 5), grid=True, title='Turnover' + kw)
        plt.tight_layout()
        plt.savefig(path, transparent=False)
        if ishow:
            plt.show()
        else:
            plt.close()

    def plot_cumulative_returns(self, ishow, path, wc=False, kind='cumsum', excess=False):
        df = pd.concat(
            [self.portfolio[k].get_wealth(wc, kind).rename(k) for k in ['baseline', 'long_short', 'long', 'short']],
            axis=1)
        if excess:
            df[['long', 'short']] -= df['baseline'].values.reshape(-1, 1)
            df = df[['long', 'short']]
        title = f'Long-Short {["Absolute", "Excess"][excess]} Result({kind}) {["No Cost", "With Cost"][wc]}'
        df.plot(figsize=(9, 5), grid=True, title=title)
        plt.tight_layout()
        plt.savefig(path, transparent=False)
        if ishow:
            plt.show()
        else:
            plt.close()

    def plot_annual_return_bars(self, ishow, path, wc=False):
        pass

    def plot_annual_sharpe(self, ishow, path, wc=False):
        pass

    def get_ls_group(self, path=None) -> pd.DataFrame:
        if path is not None:
            self.ls_group.to_csv(path)
        return self.ls_group

    def get_group_returns(self, path=None) -> pd.DataFrame:
        if path is not None:
            self.ls_g_rtns.to_csv(path)
        return self.ls_g_rtns

    def get_holding_position(self, idx_w, hd=1, rvs=False, kind='long') -> Portfolio:
        """
        由2d分组序号self.ls_group获得持仓组合
        :param idx_w: 2d权重，日截面上各股权重配比，不要求行和为一
        :param hd: 持仓长度（日），调仓周期
        :param rvs: 是否取反(分组依据因子值越大表现越好)。若False，组序号最大long，组序号最小short
        :param kind: 支持long, short, long_short
        :return: Portfolio(weight)
        """

        if (kind == 'long' and not rvs) or (kind == 'short' and rvs):
            _position = (self.ls_group == (self.ng if self.ng == 1 else self.ng - 1)).astype(int)
            _position *= idx_w.loc[self.sgn.bd: self.sgn.ed]
        elif (kind == 'short' and not rvs) or (kind == 'long' and rvs):
            _position = (self.ls_group == 0).astype(int)
            _position *= idx_w.loc[self.sgn.bd: self.sgn.ed]
        elif kind == 'long_short':
            _position_l = (self.ls_group == (self.ng if self.ng == 1 else self.ng - 1)).astype(int)
            _position_s = (self.ls_group == 0).astype(int)
            _position_l, _position_s = (_position_s, _position_l) if rvs else (_position_l, _position_s)
            _position_l *= idx_w.loc[self.sgn.bd: self.sgn.ed]
            _position_s *= idx_w.loc[self.sgn.bd: self.sgn.ed]
            _position_l = _position_l.apply(lambda s: s / s.abs().sum(), axis=1)
            _position_s = _position_s.apply(lambda s: s / s.abs().sum(), axis=1)
            _position = (_position_l - _position_s)
        elif kind == 'baseline':
            _position = idx_w.loc[self.sgn.bd: self.sgn.ed].copy()
        else:
            raise ValueError(f'Invalid portfolio kind: `{kind}`')

        if hd > 1:
            cache = _position.fillna(0)
            for d in range(1, hd):
                _position += cache.shift(d).fillna(method='bfill')
                # _position = _position.iloc[::hd].reindex_like(_position).fillna(method='ffill')  # 连续持仓
            _position /= hd
        _position = _position.apply(lambda s: s / s.abs().sum(), axis=1)
        assert round(_position.dropna(how='all').abs().sum(axis=1).prod(), 4) == 1

        self.holddays = hd
        return Portfolio(w=_position)

    def get_ls_panels(self, path_f: str = None) -> dict:
        if path_f is not None:
            self.portfolio['long_short'].get_panel_result(path_f.format('PanelLongShort.csv'))
            self.portfolio['long'].get_panel_result(path_f.format('PanelLong.csv'))
            self.portfolio['short'].get_panel_result(path_f.format('PanelShort.csv'))
        return {k: self.portfolio[k].get_panel_result() for k in self.portfolio.keys()}

    def get_portfolio_statistics(self, kind='long', wc=False, path_f=None):
        """获取半年表现面板"""
        _kind = kind.replace('long', 'Long').replace('short', 'Short') + ['NC', 'WC'][wc]
        path = None if path_f is None else path_f.format(f'Res{_kind}.csv')
        return self.portfolio[kind].get_half_year_stat(wc=wc, path=path)


# %
def main():
    # %
    config_path = f'{_PATH}/config_stk.yaml'
    conf = load_tushare.conf_init(conf_path=config_path)
    print(conf['status'])
    # pprint(conf)
    backtester = Backtester(conf)
    rtn = backtester.rtn()

    # %
    tmp = pd.read_csv(conf['path']['circ_mv'], index_col=0, parse_dates=True)
    tmp = tmp.apply(lambda s: (s - s.mean()) / s.std(), axis=1)
    fval = tmp.shift(1).iloc[1:]


    # %
    # 20日反转因子
    winLen = 20
    fval = (-rtn).rolling(winLen).mean().iloc[winLen:].apply(
        lambda s: (s-s.mean()) / s.std(), axis=1).dropna(how='all', axis=1)

    # %
    # 100日动量因子
    winLen = 100
    skipLen = 20
    fval = rtn.rolling(winLen).mean().shift(skipLen).iloc[winLen + skipLen:].apply(
        lambda s: (s-s.mean()) / s.std(), axis=1).dropna(how='all', axis=1)
    fval = fval.shift(1).iloc[1:]

    # %
    begin_date = '2017-01-01'
    group_n = 10

    tmp = fval.loc[begin_date:]
    tmp = (tmp.rank(axis=1, pct=True) * group_n).round(0)
    tmp = pd.DataFrame(
        {f"Group{k}": (rtn.reindex_like(tmp) * (tmp == k)).mean(axis=1)
         for k in range(1, group_n+1)})
    # %
    tmp.mean().plot.bar(width=.9, rot=0)
    plt.tight_layout()
    plt.show()

    # %
    tmp.cumsum().add(1).plot(linewidth=1)
    # tmp.add(1).cumprod().plot(linewidth=1)
    plt.tight_layout(); plt.show()

    # %
    print(0)


if __name__ == '__main__':
    main()
