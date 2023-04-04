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

    def infer_date_to_update_from_existing_table(self, last_tgt_table: str) -> str:
        """根据目标CSV地址，查询末尾日期，确定需要新读取的开始日期；若CSV不存在，选用self.tds开始日期"""
        if os.path.exists(last_tgt_table):
            # Infer latest combined date from existing combined table - `circ_mv.csv`
            latest_date = pd.read_csv(last_tgt_table, index_col=0).index[-1]
            # Next date after the latest date
            begin_date = load_tushare.next_calendar_date(latest_date, lfmt='%Y-%m-%d')
        else:
            # Next date is the beginning of `self.td
            begin_date = pd.to_datetime(self.tds[0]).strftime('%Y-%m-%d')
        return begin_date

    def combine_daily_basic(self):
        """合并`./cache/daily_basic/`中每日的基本面指标"""

        # Cache path
        cache_path = self.conf['path']['cache']
        # Source path
        basic_path = cache_path + 'daily_basic/'

        # Next date to update
        begin_date = self.infer_date_to_update_from_existing_table(
            last_tgt_table=self.conf['path']['circ_mv'])

        # Container for new daily data loaded
        cached = {}
        # Iterate for all new dates
        for td in tqdm(self.tds[self.tds >= begin_date]):
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

            print(f"Updated in `{self.conf['path'][k]}`")

    def combine_daily(self):
        """合并`daily/daily/`中的日度行情指标并进行后复权调整"""
        cache_path = self.conf['path']['cache']
        daily_path = cache_path + 'daily/'
        adj_path = cache_path + 'adj_factor/'

        # Next date to update
        begin_date = self.infer_date_to_update_from_existing_table(
            last_tgt_table=self.conf['path']['circ_mv'])

        ohlc = ['open', 'high', 'low', 'close']
        volamt = ['vol', 'amount']

        cached = {}
        for td in tqdm(self.tds[self.tds >= begin_date]):
            d = pd.read_csv(f'{daily_path}/{td}.csv', index_col=0).sort_index()
            a = pd.read_csv(f'{adj_path}/{td}.csv', index_col=0).loc[d.index, 'adj_factor']
            assert not set(d.index) - set(a.index)  # 所有日期内所含个股都有对应adjfactor

            for c in d.columns:
                if c not in ohlc + volamt:
                    continue
                sr = (d[c] * a).round(9) if c in ohlc else d[c]
                cc = c + 'Adj' if c in ohlc else c
                sr.name = pd.to_datetime(td, format='%Y%m%d')
                if cc not in cached:
                    # cached[cc] = pd.DataFrame(sr).T
                    cached[cc] = [sr]
                else:
                    # cached[cc] = pd.concat([cached[cc], sr], axis=0)
                    cached[cc].append(sr)

        for k, v in cached.items():  # 暂时必须延续之前，补充新增交易日
            file = self.conf['path'][k]
            df0 = pd.read_csv(file, index_col=0, parse_dates=True)
            df1 = pd.concat(v, axis=1).T
            df2 = pd.concat([df0, df1], axis=0)
            df2.to_csv(file)
            print(f"Updated in `{self.conf['path'][k]}`")

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
        combiner.combine_daily()
        combiner.combine_daily_basic()
        combiner.combine_index_weight()
        combiner.generate_demo(length=30)


if __name__ == '__main__':
    main()
