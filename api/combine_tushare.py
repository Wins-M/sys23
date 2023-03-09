import os
import pandas as pd
from tqdm import tqdm
import load_tushare

_PATH = '/Users/winston/mygitrep/sys23/'  # TODO

class CacheCombiner(object):

    def __init__(self, conf: dict):
        self.conf = conf
        self.tds = load_tushare.TushareLoader(conf).get_tradedates(
            start_date='20140630', filename=conf['path']['tradedates'])

    def combine_daily_basic(self):
        cache_path = self.conf['path']['cache']
        basic_path = cache_path + 'daily_basic/'

        latest_date = pd.read_csv(self.conf['path']['circ_mv'], index_col=0).index[-1]
        begin_date = load_tushare.next_calendar_date(latest_date, lfmt='%Y-%m-%d')

        cached = {}
        for td in tqdm(self.tds):
            if td < begin_date:
                continue
            d = pd.read_csv(f'{basic_path}/{td}.csv', index_col=0).sort_index().loc[:, 'turnover_rate':]
            for cc in d.columns:
                sr = d[cc]
                sr.name = pd.to_datetime(td, format='%Y%m%d')
                if cc not in cached:
                    # cached[cc] = pd.DataFrame(sr).T
                    cached[cc] = [sr]
                else:
                    # cached[cc] = pd.concat([cached[cc], sr], axis=0)
                    cached[cc].append(sr)

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

    def combine_daily(self):
        cache_path = self.conf['path']['cache']
        daily_path = cache_path + 'daily/'
        adj_path = cache_path + 'adj_factor/'

        latest_date = pd.read_csv(self.conf['path']['amount'], index_col=0).index[-1]
        begin_date = load_tushare.next_calendar_date(latest_date, lfmt='%Y-%m-%d')

        ohlc = ['open', 'high', 'low', 'close']
        volamt = ['vol', 'amount']

        cached = {}
        for td in tqdm(self.tds):
            if td < begin_date:
                continue
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


def main():
    config_path = f'{_PATH}/config_stk.yaml'
    conf = load_tushare.conf_init(conf_path=config_path)
    if conf['status'] == 1:
        combiner = CacheCombiner(conf)
        combiner.combine_daily()
        combiner.combine_daily_basic()


if __name__ == '__main__':
    main()
