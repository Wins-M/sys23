import os
import datetime
import pandas as pd


def main():
    src_path = '/Users/winston/mygitrep/sys23/cache/'

    correct_duplicated_lines(src_path)
    correct_blank_lines(src_path)


def correct_blank_lines(src_path):
    for file in os.listdir(src_path):
        if file[-4:] == '.csv' and file[:3] != 'BAK':
            df = pd.read_csv(src_path + file, index_col=0)
            if df.index.name == 'Unnamed: 0':
                df.index.name = 'tradingdate'
            df1 = df.dropna(how='all')
            di = df.index.difference(df1.index)

            if len(df) - len(df1) > 0:
                print(file, len(df) - len(df1), len(df), len(df1), di)

                # cmd = input('\treplace file? (Y/n)')
                # if cmd in ['N', 'n']:
                #     continue
                # else:
                #     df.to_csv(src_path + f'BAK{datetime.datetime.now()}' + file, index=False)
                #     df1.to_csv(src_path + file, index=False)


def correct_duplicated_lines(src_path):
    for file in os.listdir(src_path):
        if file[-4:] == '.csv' and file[:3] != 'BAK':

            # df = pd.read_csv(src_path + file, index_col=0)
            df = pd.read_csv(src_path + file)
            df1 = df.drop_duplicates()
            # df1 = df1.dropna(how='all')
            dl = len(df) - len(df1)
            if dl > 0:
                print(f'Find ({dl}) lines duplicated in `{file}` with {len(df)} lines, ')
                # cmd = 'Y'
                cmd = input('\treplace file? (Y/n)')
                if cmd in ['N', 'n']:
                    continue
                else:
                    df.to_csv(src_path + f'BAK{datetime.datetime.now()}' + file, index=False)
                    df1.to_csv(src_path + file, index=False)


if __name__ == '__main__':
    main()
