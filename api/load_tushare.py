import os
import pandas as pd
# import numpy as np
import datetime
import time
from tqdm import tqdm
import tushare as ts
# import importlib
from typing import Dict

_PATH = '/Users/winston/mygitrep/sys23/'  # TODO


def next_calendar_date(lend: str, delta=1,
                       lfmt='%Y%m%d', rfmt='%Y%m%d') -> str:
    """下一个自然日"""
    lend = datetime.datetime.strptime(lend, lfmt)
    rbegin = lend + datetime.timedelta(delta)
    rbegin = rbegin.strftime(rfmt)
    return rbegin


def mkdir_safe(path: str):
    """Safe make local dir. TODO: interactive"""
    if os.path.exists(path):
        print(f'Dir `{path}` already exists!')
    else:
        os.makedirs(path, exist_ok=True)
        print(f'Make new dir `{path}`.')


def conf_init(conf_path: str) -> dict:
    """Load config file from a path"""
    import yaml
    conf = yaml.safe_load(open(conf_path, encoding='utf-8'))

    # Tushare token
    conf['tushare_token'] = open(conf['tushare_token']).read().strip('\n')
    
    # Path fill
    for k in conf['path'].keys():
        conf['path'][k] = conf['path'][k].format(conf['api_path'])

    return conf


def main():
    conf_path = '/Users/winston/mygitrep/sys23/config_stk.yaml'
    conf = conf_init(conf_path)
    tushare_loader = TushareLoader(conf)
    tushare_loader.update_local_cache()


class TushareLoader(object):
    """Get Tushare data"""

    def __init__(self, conf: dict):
        # self.conf: dict = conf
        self.start_date: str = '20140630'  # tushare最开始从2015
        self.end_date: str = datetime.date.today().strftime('%Y%m%d')
        self.pro = ts.pro_api(conf['tushare_token'])
        self.csi_pool: Dict[str] = conf['csi_pool']

    def update_local_cache(self):
        """"""

        # % 获取交易日期

        # calendar_dates = data.api.wget_tradedates(start_date, end_date, pro, path=_PATH)
        tradedates = self.get_tradedates(
            self.start_date, self.end_date, path=_PATH, fmt='%Y%m%d')

        # % 获取日线行情

        # df = pro.daily(trade_date=tradedates.iloc[0])
        api_kws = [
            'daily', 'daily_basic', 'adj_factor',
            'suspend_d', 'moneyflow', 'stk_limit',
            # 'index_dailybasic', # 大盘指数每日指标
            # moneyflow_hsgt,
        ]

        for api_kw in api_kws:
            cache_path = _PATH + f'cache/{api_kw}/'
            mkdir_safe(cache_path)

            bar = tqdm(tradedates)
            for td in bar:
                file = cache_path + td + '.csv'
                if os.path.exists(file):  # 当前目录下存在同名文件，跳过 O(T)
                    continue
                else:  # 从Tushare获得数据
                    # break
                    for _ in range(3):  # 尝试3次
                        try:
                            df = self.pro.query(api_kw, trade_date=td)
                        except:
                            time.sleep(1)
                        else:
                            df.to_csv(file, index=False)
                            break  # 找到并存储
                bar.set_description(api_kw + ': ' + td)

            df = pd.read_csv(cache_path + tradedates.iloc[-1] + '.csv')
            print(df.tail())

        # % IPO新股列表 最大2000条
        new_share = self.wget_new_share(
            path=_PATH + 'cache/',
            lfy=2008, rfy=2023, step=5
        )
        print(new_share.head())

        # % 指数成分和权重
        for kind in self.csi_pool.keys():  # ['CSI300', 'CSI500', 'CSI800', 'CSI1000']:
            index_code = self.csi_pool[kind]
            print(kind, index_code)

            df = self.wget_index_weight(
                index_code, 2014, 2024,
                path=_PATH + 'cache/'
            )
            print(df.tail(3))

    def wget_index_weight(self,
                          index_code, lfy, rfy,
                          path='./', fpath='index_weight') -> pd.DataFrame:
        """指数成分和权重"""
        cache_path = path + '/' + fpath + '/'
        os.makedirs(cache_path, exist_ok=True)

        def get_index_weight(code, year) -> pd.DataFrame:
            qt = list()
            qt.append(self.pro.index_weight(index_code=code, start_date=f'{year}0101', end_date=f'{year}0331'))
            qt.append(self.pro.index_weight(index_code=code, start_date=f'{year}0401', end_date=f'{year}0630'))
            qt.append(self.pro.index_weight(index_code=code, start_date=f'{year}0701', end_date=f'{year}0930'))
            qt.append(self.pro.index_weight(index_code=code, start_date=f'{year}1001', end_date=f'{year}1231'))
            for i in range(4):
                if len(qt[i]) == 5000:  # [300, 500, 800]*(3*2) < 5000 quarterly
                    print(f"{code} FY{year} Q{i + 1}: exceeds limit 5000")
                elif len(qt[i]) == 0:  # [300, 500, 800]*(3*2) < 5000 quarterly
                    print(f"{code} FY{year} Q{i + 1}: no index_weight records")
            return pd.concat(qt)

        res = pd.DataFrame()
        for fy in range(lfy, rfy):
            file = cache_path + f"""{index_code.split('.')[0]}_{fy}.csv"""
            if fy != 2023 and os.path.exists(file):
                res = pd.concat([res, pd.read_csv(file, dtype={'trade_date': str})])
                continue
            # break
            for _ in range(3):  # 尝试3次
                try:
                    df = get_index_weight(index_code, fy)
                except:
                    time.sleep(1)
                else:
                    df.to_csv(file, index=False)
                    res = pd.concat([res, df])
                    break  # 找到并存储
        return res

    def wget_new_share(self, path='./',
                       lfy=2008, rfy=2023, step=5,
                       filename='new_share.csv') -> pd.DataFrame:
        """更新IPO新股列表 最大2000条
        - tushare数据最早开始于2008
        """

        file = path + filename
        if os.path.exists(file):  # 目录下已有文件，则直接更新所有新增日期
            df = pd.read_csv(file, dtype={'ipo_date': str, 'issue_date': str})
            print(f"""Load {len(df)} rows from `{file}`""")
            rbegin = next_calendar_date(df.ipo_date.max())
            for _ in range(3):
                try:
                    tmp = self.pro.new_share(start_date=rbegin, end_date=f'{rfy}1231')
                except:
                    time.sleep(1)
                else:
                    if len(tmp) == 2000:
                        raise Exception(f'长度超限(2000) {rbegin} {rfy}1231 更新已有文件失败')
                    elif len(tmp) > 0:
                        df = pd.concat([tmp[tmp['ipo_date'] >= rbegin], df])
                        df.to_csv(file, index=False)
                        print(f"""Save {len(df)} rows in `{file}`""")
                    break
        else:  # 全历史
            df = pd.DataFrame()
            for fy in range(lfy, rfy + 1, step):
                tmp = pd.DataFrame()
                for _ in range(3):
                    try:
                        tmp = self.pro.new_share(start_date=f'{fy}0101', end_date=f'{fy + 4}1231')
                    except:
                        time.sleep(1)
                    else:
                        break
                if len(tmp) > 2000:
                    # raise Exception(f'长度超限(2000) {fy} {fy+4}')
                    print('长度超限(2000)', fy, fy + 4)
                    tmp = pd.concat([
                        self.pro.new_share(start_date=f'{fy + i}0101', end_date=f'{fy + i}1231')
                        for i in range(5)
                    ])
                df = pd.concat([tmp, df])

            df.to_csv(file, index=False)
            print(f"""Save {len(df)} rows in `{file}`""")

        return df

    def get_tradedates(self, start_date=None, end_date=None,
                       path='', filename='tradedates.csv',
                       fmt='%Y%m%d', silent=False) -> pd.Series:
        """获取交易日序列"""
        if not start_date:
            start_date = self.start_date
        if not end_date:
            end_date = self.end_date
        calendar_dates = self.wget_tradedates(start_date, end_date,
                                              path=path, filename=filename)

        tmp = calendar_dates[calendar_dates.is_open == 1]['cal_date']
        tmp = pd.to_datetime(tmp.reset_index(drop=True))
        tmp = tmp.apply(lambda x: x.strftime(fmt))
        if not silent:
            print('get_tradedates:', tmp.iloc[0], tmp.iloc[-1])
        return tmp

    def wget_tradedates(self, start_date=None, end_date=None,
                        path='', filename: str = 'tradedates.csv'):
        """获取交易日历信息，时间正序"""

        if not start_date:
            start_date = self.start_date
        if not end_date:
            end_date = self.end_date

        # filename='tradedates.csv'
        def get_td(start, end):
            if not self.pro:
                raise Exception('tushare pro missing')
            df = self.pro.trade_cal(
                exchange='SSE',
                start_date=start,
                end_date=end,
                fields='exchange,cal_date,is_open,pretrade_date',
                # is_open='0'
            )
            # for k in ['cal_date', 'pretrade_date']:
            #     df[k] = pd.to_datetime(df[k])
            return df[::-1]

        file = path + '/' + filename if path else filename
        if file and os.path.exists(file):
            updated = False
            tradedates = pd.read_csv(file, dtype={
                'exchange': str, 'cal_date': str,
                'is_open': int, 'pretrade_date': str})
            rstart, rend = tradedates['cal_date'].iloc[0], tradedates['cal_date'].iloc[-1]
            if start_date < rstart:  # 指定开始日期在现有之前
                tmp = get_td(start_date, rstart).iloc[:-1]
                tradedates = pd.concat([tmp, tradedates])
                updated = True
            if end_date > rend:  # 指定结束日期在现有之后
                tmp = get_td(rend, end_date).iloc[1:]
                tradedates = pd.concat([tradedates, tmp])
                updated = True
            if file and updated:
                print(f'Update `{file}`')
                tradedates.to_csv(file, index=False)
            tradedates = tradedates[(tradedates['cal_date'] >= start_date) &
                                    (tradedates['cal_date'] <= end_date)]
        else:
            if self.pro is None:
                raise Exception('需要更新，但缺少tushare接口pro')
            tradedates = get_td(start_date, end_date)
            if file:
                tradedates.to_csv(f'{file}', index=None)
            print(f'Create `{file}`')

        return tradedates


if __name__ == '__main__':
    main()

# def wget_index_weight(pro,
#                       index_code, lfy, rfy,
#                       path='./', fpath='index_weight') -> pd.DataFrame:
#     """指数成分和权重"""
#     cache_path =  path + '/' + fpath + '/'
#     os.makedirs(cache_path, exist_ok=True)
#
#     def get_index_weight(index_code, fy) -> pd.DataFrame:
#         qt = []
#         qt.append(pro.index_weight(index_code=index_code, start_date=f'{fy}0101', end_date=f'{fy}0331'))
#         qt.append(pro.index_weight(index_code=index_code, start_date=f'{fy}0401', end_date=f'{fy}0630'))
#         qt.append(pro.index_weight(index_code=index_code, start_date=f'{fy}0701', end_date=f'{fy}0930'))
#         qt.append(pro.index_weight(index_code=index_code, start_date=f'{fy}1001', end_date=f'{fy}1231'))
#         for i in range(4):
#             if len(qt[i]) == 5000:  # [300, 500, 800]*(3*2) < 5000 quarterly
#                 print(f"{index_code} FY{fy} Q{i+1}: exceeds limit 5000")
#             elif len(qt[i]) == 0:  # [300, 500, 800]*(3*2) < 5000 quarterly
#                 print(f"{index_code} FY{fy} Q{i+1}: no index_weight records")
#         return pd.concat(qt)
#
#     res = pd.DataFrame()
#     for fy in range(lfy, rfy):
#         file = cache_path + f"""{index_code.split('.')[0]}_{fy}.csv"""
#         if fy != 2023 and os.path.exists(file):
#             res = pd.concat([res, pd.read_csv(file, dtype={'trade_date': str})])
#             continue
#         # break
#         for _ in range(3):  # 尝试3次
#             try:
#                 df = get_index_weight(index_code, fy)
#             except:
#                 time.sleep(1)
#             else:
#                 df.to_csv(file, index=False)
#                 res = pd.concat([res, df])
#                 break  # 找到并存储
#     return res
#
#
# def wget_new_share(pro, path='./',
#                    lfy=2008, rfy=2023, step=5,
#                    filename='new_share.csv') -> pd.DataFrame:
#     """更新IPO新股列表 最大2000条
#     - tushare数据最早开始于2008
#     """
#
#     file = path + filename
#     if os.path.exists(file):  # 目录下已有文件，则直接更新所有新增日期
#         df = pd.read_csv(file, dtype={'ipo_date': str, 'issue_date': str})
#         print(f"""Load {len(df)} rows from `{file}`""")
#         rbegin = next_calendar_date(df.ipo_date.max())
#         for _ in range(3):
#             try:
#                 tmp = pro.new_share(start_date=rbegin, end_date=f'{rfy}1231')
#             except:
#                 time.sleep(1)
#             else:
#                 if len(tmp) == 2000:
#                     raise Exception(f'长度超限(2000) {rbegin} {rfy}1231 更新已有文件失败')
#                 elif len(tmp) > 0:
#                     df = pd.concat([tmp[tmp['ipo_date'] >= rbegin], df])
#                     df.to_csv(file, index=False)
#                     print(f"""Save {len(df)} rows in `{file}`""")
#                 break
#     else:  # 全历史
#         df = pd.DataFrame()
#         for fy in range(lfy, rfy+1, step):
#             tmp = pd.DataFrame()
#             for _ in range(3):
#                 try:
#                     tmp = pro.new_share(start_date=f'{fy}0101', end_date=f'{fy+4}1231')
#                 except:
#                     time.sleep(1)
#                 else:
#                     break
#             if len(tmp) > 2000:
#                 # raise Exception(f'长度超限(2000) {fy} {fy+4}')
#                 print('长度超限(2000)', fy, fy+4)
#                 tmp = pd.concat([
#                     pro.new_share(start_date=f'{fy+i}0101', end_date=f'{fy+i}1231')
#                     for i in range(5)
#                 ])
#             df = pd.concat([tmp, df])
#
#         df.to_csv(file, index=False)
#         print(f"""Save {len(df)} rows in `{file}`""")
#
#     return df
#
#
# def get_tradedates(start_date, end_date, pro=None,
#                    path='', filename='tradedates.csv',
#                    fmt = '%Y%m%d', silent=False) -> pd.Series:
#     """获取交易日序列"""
#     calendar_dates = wget_tradedates(start_date, end_date, pro,
#                                      path=path, filename=filename)
#
#     tmp = calendar_dates[calendar_dates.is_open==1]['cal_date']
#     tmp = pd.to_datetime(tmp.reset_index(drop=True))
#     tmp = tmp.apply(lambda x: x.strftime(fmt))
#     if not silent:
#         print('get_tradedates:', tmp.iloc[0], tmp.iloc[-1])
#     return tmp
#
#
# def wget_tradedates(start_date, end_date, pro=None,
#                     path='', filename: str = 'tradedates.csv'):
#     """获取交易日历信息"""
#     # filename='tradedates.csv'
#     def get_td(start, end):
#         if not pro:
#             raise Exception('tushare pro missing')
#         df = pro.trade_cal(
#             exchange='SSE',
#             start_date=start,
#             end_date=end,
#             fields='exchange,cal_date,is_open,pretrade_date',
#             # is_open='0'
#         )
#         # for k in ['cal_date', 'pretrade_date']:
#         #     df[k] = pd.to_datetime(df[k])
#         return df
#
#     file = path + '/' + filename if path else filename
#     if file and os.path.exists(file):
#         updated = False
#         tradedates = pd.read_csv(file, dtype={
#             'exchange': str, 'cal_date': str,
#             'is_open': int, 'pretrade_date': str})
#         rstart, rend = tradedates['cal_date'].iloc[0], tradedates['cal_date'].iloc[-1]
#         if start_date < rstart:  # 指定开始日期在现有之前
#             tmp = get_td(start_date, rstart).iloc[:-1]
#             tradedates = pd.concat([tmp, tradedates])
#             updated = True
#         if end_date > rend:  # 指定结束日期在现有之后
#             tmp = get_td(rend, end_date).iloc[1:]
#             tradedates = pd.concat([tradedates, tmp])
#             updated = True
#         if file and updated:
#             print(f'Update `{file}`')
#             tradedates.to_csv(file, index=False)
#         tradedates = tradedates[(tradedates['cal_date'] >= start_date) &
#                                 (tradedates['cal_date'] <= end_date)]
#     else:
#         if pro is None:
#             raise Exception('需要更新，但缺少tushare接口pro')
#         tradedates = get_td(start_date, end_date)
#         if file:
#             tradedates.to_csv(f'{file}', index=None)
#         print(f'Create `{file}`')
#
#     return tradedates
#
#
