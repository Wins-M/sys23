import os
import pandas as pd
import numpy as np
from tqdm import tqdm
import load_tushare

_PATH = '/Users/winston/mygitrep/sys23/'  # TODO


class CacheCombiner(object):

    def __init__(self, conf: dict):
        self.conf = conf
        self.tds = load_tushare.TushareLoader(conf).get_tradedates(
            start_date='20140630', filename=conf['path']['tradedates'])

    def combine_index_weight(self):
        """合并股指成分股权重并补全交易日"""

        # Cache path
        cache_path = self.conf['path']['cache']
        # Source path
        basic_path = cache_path + 'index_weight/'

        # Iterate distinct stock index
        for kind, code in self.conf['csi_pool'].items():

            # Current index code (6-digit)
            code1 = code.split('.')[0]

            stack = []
            # Iterate all years
            for year in range(int(self.tds.iloc[0][:4]), int(self.tds.iloc[-1][:4]) + 1):
                src_file = basic_path + f'{code1}_{year}.csv'
                if not os.path.exists(src_file):
                    continue
                df = pd.read_csv(src_file)
                df = df.pivot(index='trade_date', columns='con_code', values='weight')
                stack.append(df)

            # Concat year-by-year weight for current index
            stack = pd.concat(stack, axis=0)
            # Fill with NA for method `ffill` in the next step
            stack.fillna(0, inplace=True)
            # Reindex columns with the whole tradedates (though disk-space-consuming)
            stack = stack.reindex(self.tds.astype(int), method='ffill')

            # If row sum less than 98%, we'll set index weight as NA
            bad_weight_sum = (stack.sum(axis=1) < 98)
            stack.loc[bad_weight_sum] = np.nan

            # Save
            stack.to_csv(self.conf['path'][kind])
            print(f"Updated in `{self.conf['path'][kind]}`")

    def combine_st_status(self):
        df = pd.read_csv(self.conf['path']['cache'] + '/namechange.csv')
        # print(df['change_reason'].value_counts())
        # print(df.groupby('change_reason').first())
        mask_begin = (df['change_reason'] == '*ST') | (df['change_reason'] == 'ST')
        mask_end = (df['change_reason'] == '撤销*ST') | (df['change_reason'] == '撤销ST') | (
                    df['change_reason'] == '摘星改名') | (df['change_reason'] == '终止上市')
        df['isST'] = (mask_begin * 2 + mask_end * 1).replace(0, np.nan)
        df = df[['ann_date', 'change_reason', 'ts_code', 'isST']].drop_duplicates().dropna()
        print(df[df[['ann_date', 'ts_code']].duplicated(keep=False)])
        df = df.groupby(['ann_date', 'ts_code'])['isST'].sum().astype(int)
        print(df.value_counts().to_dict())
        df = df.replace(3, 0)  # TODO: 3 - 当天有ST又取消ST，则视作之后一直是ST
        df = df.replace(2, 0)  # 0 - 从公告日开始，标记为ST/*ST; 1 - 从公告日开始，不再是ST/*ST（或退市）
        print(df.value_counts().to_dict())
        df = df.reset_index()
        df = df.rename(columns={'ann_date': 'tradingdate', 'ts_code': 'stockcode', 'isST': 'st_status'})
        df.to_csv(self.conf['path']['st_status'], index=False)
        print(f"Save {len(df)} rows in `{self.conf['path']['st_status']}`.")

    def infer_tds_todo(self, last_tgt_table: str, is1d=False) -> str:
        """根据目标CSV地址，查询末尾日期，确定需要新读取的开始日期；若CSV不存在，选用self.tds开始日期"""
        if os.path.exists(last_tgt_table):
            if is1d:  # a 1d table with columns - `tradingdate`
                # Infer latest combined date from existing combined table
                latest_date = pd.read_csv(last_tgt_table)['tradingdate'].dropna(how='all').iloc[-1]
                # Next date after the latest date
                begin_date = load_tushare.next_calendar_date(latest_date, lfmt='%Y-%m-%d')
            else:
                # Infer latest combined date from existing combined table
                latest_date = pd.read_csv(last_tgt_table, index_col=0).dropna(how='all').index[-1]
                # Next date after the latest date
                begin_date = load_tushare.next_calendar_date(latest_date, lfmt='%Y-%m-%d')
        else:
            # Next date is the beginning of `self.td
            begin_date = pd.to_datetime(self.tds[0]).strftime('%Y-%m-%d')
        return begin_date

    def combine_daily(self):
        """合并`daily/daily/`中的日度行情指标并进行后复权调整"""
        cache_path = self.conf['path']['cache']
        daily_path = cache_path + 'daily/'  # TODO: download cache - daily
        adj_path = cache_path + 'adj_factor/'

        # Next date to update
        begin_date = self.infer_tds_todo(
            last_tgt_table=self.conf['path']['circ_mv'])  # TODO

        ohlc = ['open', 'high', 'low', 'close']
        volamt = ['vol', 'amount']

        tds_todo = self.tds[self.tds >= begin_date]
        if len(tds_todo) > 0:
            cached = {}
            for td in tqdm(tds_todo):
                d = pd.read_csv(f'{daily_path}/{td}.csv', index_col=0).sort_index()
                a = pd.read_csv(f'{adj_path}/{td}.csv', index_col=0)
                try:
                    a = a.loc[d.index, 'adj_factor']
                except KeyError as e:
                    a = a.loc[a.index.intersection(d.index), 'adj_factor']
                    print(f"{td}: {len(set(d.index) - set(a.index))} index in 'daily' found" 
                          f" but in 'adj_factor' not found:\n    original error message: {e}\n")

                # TODO: 20230421: 1 index in 'daily' found but in 'adj_factor' not found:
                # original error message: "['689009.SH'] not in index"
                # assert not set(d.index) - set(a.index)  # 所有日期内所含个股都有对应adjfactor

                for c in d.columns:
                    if c not in ohlc + volamt:
                        continue

                    sr = d[c]
                    sr.name = pd.to_datetime(td, format='%Y%m%d')
                    if c not in cached:
                        cached[c] = [sr]
                    else:
                        cached[c].append(sr)

                    if c in ohlc:  # 复权调整
                        csr = (d[c] * a).round(9)
                        csr.name = pd.to_datetime(td, format='%Y%m%d')
                        cc = c + 'Adj'
                        if cc not in cached:
                            cached[cc] = [csr]
                        else:
                            cached[cc].append(csr)

            for k, v in cached.items():  # 暂时必须延续之前，补充新增交易日
                file = self.conf['path'][k]
                if os.path.exists(file):
                    df0 = pd.read_csv(file, index_col=0, parse_dates=True)
                else:
                    df0 = pd.DataFrame()
                df1 = pd.concat(v, axis=1).T
                df2 = pd.concat([df0, df1], axis=0)
                df2.to_csv(file)
                print(f"Updated in `{self.conf['path'][k]}`")
                print(f"Updated {len(tds_todo)} rows in `{self.conf['path'][k]}`")

        else:
            print(f"Skip up-to-time: `{daily_path}`")

    def combine_suspend(self):
        """合并停牌信息，最后一天为复牌（R），之前连续的停牌（S），当日又停又复牌（SR or RS），默认NA"""
        cache_path = self.conf['path']['cache']
        src_path = cache_path + 'suspend_d/'  # TODO: download cache path - suspend_d
        tgt_path = self.conf['path']['suspend_d']

        begin_date = self.infer_tds_todo(tgt_path, is1d=True)

        tds_todo = self.tds[self.tds >= begin_date]
        if len(tds_todo) > 0:
            cached = []
            for td in tqdm(tds_todo):
                s = pd.read_csv(f"{src_path}{td}.csv")[['ts_code', 'suspend_type']]
                s = s.groupby('ts_code').sum()
                s = s['suspend_type'].rename(pd.to_datetime(td, format='%Y%m%d'))
                cached.append(s)

            if os.path.exists(tgt_path):
                df0 = pd.read_csv(tgt_path, index_col=0, parse_dates=True)
            else:
                df0 = pd.DataFrame()
            df1 = pd.concat(cached, axis=1).T
            df1 = df1.stack()
            df1 = df1.reset_index()
            df1.columns = ['tradingdate', 'stockcode', 'suspend_d']
            df2 = pd.concat([df0, df1], axis=0)
            df2.to_csv(tgt_path, index=False)
            print(f"Updated {len(tds_todo)} rows in `{tgt_path}`")

        else:
            print(f"Skip up-to-time: `{tgt_path}`")

    def combine_stk_limit(self):
        """合并每日的涨跌停价格"""
        src_path = self.conf['path']['cache'] + 'stk_limit/'
        begin_date = self.infer_tds_todo(self.conf['path']['down_limit'])

        tds_todo = self.tds[self.tds >= begin_date]
        if len(tds_todo) > 0:
            cached = {}
            for td in tqdm(self.tds[self.tds >= begin_date]):
                s = pd.read_csv(f"{src_path}{td}.csv", index_col=[1])
                td1 = pd.to_datetime(td, format='%Y%m%d')

                for c in ['up_limit', 'down_limit']:
                    if c in cached:
                        cached[c].append(s[c].rename(td1))
                    else:
                        cached[c] = [s[c].rename(td1)]

            for k, v in cached.items():
                tgt_path = self.conf['path'][k]
                df0 = pd.read_csv(tgt_path, index_col=0, parse_dates=True) \
                    if os.path.exists(tgt_path) else pd.DataFrame()
                df1 = pd.concat(cached[k], axis=1).T
                df2 = pd.concat([df0, df1], axis=0)
                df2.to_csv(tgt_path)
                print(f"Updated {len(tds_todo)} rows in `{tgt_path}`")

        else:
            print(f"Skip up-to-time: `{src_path}`")

    def infer_up_down_status(self, force_update_all=False):
        """直接从日度价格（原始）和日度涨跌停上下界确定涨跌停状态 OU/OD/CU/CD"""

        tgt_path = self.conf['path']['updown_status']

        if force_update_all:
            tds_todo = self.tds
        else:
            tds_todo = self.tds[self.tds >= self.infer_tds_todo(tgt_path)]

        if len(tds_todo) > 0:
            cached = []
            for td in tqdm(tds_todo):
                df = pd.read_csv(f"{self.conf['path']['cache']}daily/{td}.csv", index_col=0)[['open', 'close']]
                po, pc = df['open'], df['close']
                df = pd.read_csv(f"{self.conf['path']['cache']}stk_limit/{td}.csv", index_col=1)[['up_limit', 'down_limit']]
                bu, bd = df['up_limit'], df['down_limit']
                del df

                open_up = po[po >= bu.reindex_like(po)]  # TODO: =?
                open_down = po[po <= bd.reindex_like(po)]
                close_up = pc[pc >= bu.reindex_like(pc)]
                close_down = pc[pc <= bd.reindex_like(pc)]

                open_up[:] = 'OU'
                open_down[:] = 'OD'
                close_up[:] = 'CU'
                close_down[:] = 'CD'

                sr = pd.concat([
                    open_up,
                    open_down,
                    close_up[close_up.index.difference(open_up.index)],
                    close_down[close_down.index.difference(open_down.index)]
                ]).reset_index()
                sr['tradingdate'] = pd.to_datetime(td, format='%Y%m%d')
                sr.columns = ['stockcode', 'updown_status', 'tradingdate']
                sr = sr[['tradingdate', 'stockcode', 'updown_status']]
                sr = sr.sort_values('stockcode')

                cached.append(sr)

            if (not force_update_all) and (os.path.exists(tgt_path)):
                df0 = pd.read_csv(tgt_path, parse_dates=['tradingdate'])
            else:
                df0 = pd.DataFrame()
            df1 = pd.concat(cached, axis=0)
            df2 = pd.concat([df0, df1], axis=0)

            df2.to_csv(tgt_path, index=False)
            print(f"Updated {len(tds_todo)} rows in `{tgt_path}`")
        else:
            print(f"Skip up-to-time: `{tgt_path}`")

    def combine_daily_basic(self):
        """合并`./cache/daily_basic/`中每日的基本面指标"""

        # Cache path
        cache_path = self.conf['path']['cache']
        # Source path
        basic_path = cache_path + 'daily_basic/'

        # Next date to update
        begin_date = self.infer_tds_todo(
            last_tgt_table=self.conf['path']['circ_mv'])

        tds_todo = self.tds[self.tds >= begin_date]
        if len(tds_todo) > 0:

            # Container for new daily data loaded
            cached = {}
            # Iterate for all new dates
            for td in tqdm(tds_todo):
                # Load one-day basic data
                d = pd.read_csv(f'{basic_path}/{td}.csv', index_col=0).loc[:, 'turnover_rate':]
                # Reorder by stock-code
                d.sort_index(inplace=True)

                # Cache one-day indices
                for cc in d.columns:
                    sr = d[cc]
                    sr.name = pd.to_datetime(td, format='%Y%m%d')
                    if cc not in cached:
                        cached[cc] = [sr]
                    else:
                        cached[cc].append(sr)

            # Concat and save
            for k, v in cached.items():
                file = self.conf['path'][k]

                if os.path.exists(file):
                    df0 = pd.read_csv(file, index_col=0, parse_dates=True)
                else:
                    df0 = pd.DataFrame()
                df1 = pd.concat(v, axis=1).T
                df2 = pd.concat([df0, df1], axis=0)
                df2.to_csv(file)

                print(f"Updated {len(tds_todo)} rows in `{self.conf['path'][k]}`")
        else:
            print(f"Skip up-to-time: `{basic_path}`")

    def generate_demo(self, length=30):
        """将所有已有表格都生成demo作为示例数据"""
        for k, file in tqdm(self.conf['path'].items()):
            if file[-4:] != '.csv':
                continue
            pd.read_csv(file, index_col=0).tail(length).to_csv(
                self.conf['path']['cache_demo'] + file.rsplit('/', maxsplit=1)[-1])


def main():
    config_path = f'{_PATH}/config_stk.yaml'
    conf = load_tushare.conf_init(conf_path=config_path)
    if conf['status'] == 1:
        combiner = CacheCombiner(conf)
        combiner.combine_suspend()
        combiner.combine_st_status()
        combiner.combine_daily()
        combiner.combine_stk_limit()
        combiner.infer_up_down_status(force_update_all=False)
        combiner.combine_daily_basic()
        combiner.combine_index_weight()
        if conf['update_cache_demo'] == 1:
            combiner.generate_demo(length=10)


if __name__ == '__main__':
    main()
