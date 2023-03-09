# sys23
投研平台（自用）

自用的回测框架，只限个人使用。

- ***目录[`api`](api): 接口***
- [`load_tushare`](data/load_tushare.py): 从Tushare获取基础数据
- 
- ***目录[`data`](data): 数据相关***
- 
- ***目录[`demo`](demo): 功能/策略的展示***
- [`mkt_dvg_min_amt`](demo/mkt_dvg_min_amt.ipynb): 成交额市场分歧度择时【因子】
- 
- ***目录[`idea`](./idea/): 研报等参考思路***


---

## [本地数据](./cache)

| 名称                | 类型  | 描述                                   |
| :------------------ | :---- | :------------------------------------- |
| tradedates          | str   | 交易日期                               |
| [daily, adj_factor] |       |                                        |
| openAdj             | float | 开盘价 x 复权因子                      |
| highAdj             | float | 最高价 x 复权因子                      |
| lowAdj              | float | 最低价 x 复权因子                      |
| closeAdj            | float | 收盘价 x 复权因子                      |
| vol                 | float | 成交量 （手）                          |
| amount              | float | 成交额 （千元）                        |
| [daily_basic]       |       |                                        |
| turnover_rate       | float | 换手率（%）                            |
| turnover_rate_f     | float | 换手率（自由流通股）                   |
| volume_ratio        | float | 量比                                   |
| pe                  | float | 市盈率（总市值/净利润， 亏损的PE为空） |
| pe_ttm              | float | 市盈率（TTM，亏损的PE为空）            |
| pb                  | float | 市净率（总市值/净资产）                |
| ps                  | float | 市销率                                 |
| ps_ttm              | float | 市销率（TTM）                          |
| dv_ratio            | float | 股息率 （%）                           |
| dv_ttm              | float | 股息率（TTM）（%）                     |
| total_share         | float | 总股本 （万股）                        |
| float_share         | float | 流通股本 （万股）                      |
| free_share          | float | 自由流通股本 （万）                    |
| total_mv            | float | 总市值 （万元）                        |
| circ_mv             | float | 流通市值（万元）                       |
|                     |       |                                        |
|                     |       |                                        |
|                     |       |                                        |
|                     |       |                                        |
|                     |       |                                        |
