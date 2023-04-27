# sys23
投研平台（自用）

自用的回测框架，只限个人使用。

- ***目录[`api`](api): 接口***
- [`load_tushare`](api/load_tushare.py): 从[Tushare](https://tushare.pro/)获取基础数据
- [`combine_tushare`](api/combine_tushare.py): 将日度流量数据整合成面板数据
- 
- ***目录[`data`](data): 数据相关***
- 
- ***目录[`demo`](demo): 功能/策略的展示***
- [`mkt_dvg_min_amt`](demo/mkt_dvg_min_amt.ipynb): 成交额市场分歧度择时【因子】
- [`lucky_number`](demo/lucky_number/): 价格中的数字【实证】
- [`prospect`](demo/prospect.ipynb): 前景理论【TODO】
- 
- ***目录[`idea`](idea): 想法***
- 


---

## [本地数据](./cache)

 [示例数据](./demo/cache_demo/)

| 名称                                                                                                         | 类型          | 描述                                                             |
|:-----------------------------------------------------------------------------------------------------------|:------------|:---------------------------------------------------------------|
| [tradedates.csv](tradedates.csv)                                                                           | str         | 交易日期                                                           |
| [new_share](https://tushare.pro/document/2?doc_id=123)                                                     | IPO新股列表     | 【描述：获取新股上市列表数据】                                                |
| [new_share.csv](demo/cache_demo/new_share.csv)                                                             | object, XD  | ipo_date 上网发行日期；issue_date 上市日期                                |
| [suspend_d](https://tushare.pro/document/2?doc_id=214)                                                     | 每日停复牌信息     | 【更新时间：不定期】【描述：按日期方式获取股票每日停复牌信息】                                |
| [suspend_d.csv](demo/cache_demo/suspend_d.csv)                                                             | str, 1D     | 停牌 R S RS SR                                                   |
| [namechange](https://tushare.pro/document/2?doc_id=100)                                                    | 股票曾用名       | 【历史名称变更记录】【一次更新全部】                                             |
| [st_status.csv](demo/cache_demo/st_status.csv)                                                             | str, 1D     | ST or *ST: 0 1                                                 |
| [stk_limit](https://tushare.pro/document/2?doc_id=214)                                                     | 每日涨跌停价格     | 获取全市场（包含A/B股和基金）每日涨跌停价格，包括涨停价格，跌停价格等】【每个交易日8点40左右更新当日股票涨跌停价格。】 |
| [up_limit.csv](demo/cache_demo/up_limit.csv)                                                               | float       | 涨停价                                                            |
| [down_limit.csv](demo/cache_demo/down_limit.csv)                                                           | float       | 跌停价                                                            |
| [updown_status.csv](demo/cache_demo/updown_status.csv)                                                     | str, 1D     | 涨跌停状态 OU OD CU CD 有重复；可能不准确                                    |
| [daily](https://tushare.pro/document/2?doc_id=27) + [adj_factor](https://tushare.pro/document/2?doc_id=28) | 日线行情 + 复权因子 | 【交易日每天15点～16点之间入库】【复权因子更新时间：早上9点30分】                           |
| [openAdj.csv](demo/cache_demo/openAdj.csv)                                                                 | float       | 开盘价 x 复权因子                                                     |
| [highAdj.csv](demo/cache_demo/highAdj.csv)                                                                 | float       | 最高价 x 复权因子                                                     |
| [lowAdj.csv](demo/cache_demo/lowAdj.csv)                                                                   | float       | 最低价 x 复权因子                                                     |
| [closeAdj.csv](demo/cache_demo/closeAdj.csv)                                                               | float       | 收盘价 x 复权因子                                                     |
| [vol.csv](demo/cache_demo/vol.csv)                                                                         | float       | 成交量 （手）                                                        |
| [amount.csv](demo/cache_demo/amount.csv)                                                                   | float       | 成交额 （千元）                                                       |
| [daily_basic](https://tushare.pro/document/2?doc_id=32)                                                    | 每日指标        | 获取全部股票每日重要的基本面指标，可用于选股分析、报表展示等。【更新时间：交易日每日15点～17点之间】           |
| [turnover_rate.csv](demo/cache_demo/turnover_rate.csv)                                                     | float       | 换手率（%）                                                         |
| [turnover_rate_f.csv](demo/cache_demo/turnover_rate_f.csv)                                                 | float       | 换手率（自由流通股）                                                     |
| [volume_ratio.csv](demo/cache_demo/volume_ratio.csv)                                                       | float       | 量比                                                             |
| [pe.csv](demo/cache_demo/pe.csv)                                                                           | float       | 市盈率（总市值/净利润， 亏损的PE为空）                                          |
| [pe_ttm.csv](demo/cache_demo/pe_ttm.csv)                                                                   | float       | 市盈率（TTM，亏损的PE为空）                                               |
| [pb.csv](demo/cache_demo/pb.csv)                                                                           | float       | 市净率（总市值/净资产）                                                   |
| [ps.csv](demo/cache_demo/ps.csv)                                                                           | float       | 市销率                                                            |
| [ps_ttm.csv](demo/cache_demo/ps_ttm.csv)                                                                   | float       | 市销率（TTM）                                                       |
| [dv_ratio.csv](demo/cache_demo/dv_ratio.csv)                                                               | float       | 股息率 （%）                                                        |
| [dv_ttm.csv](demo/cache_demo/dv_ttm.csv)                                                                   | float       | 股息率（TTM）（%）                                                    |
| [total_share.csv](demo/cache_demo/total_share.csv)                                                         | float       | 总股本 （万股）                                                       |
| [float_share.csv](demo/cache_demo/float_share.csv)                                                         | float       | 流通股本 （万股）                                                      |
| [free_share.csv](demo/cache_demo/free_share.csv)                                                           | float       | 自由流通股本 （万）                                                     |
| [total_mv.csv](demo/cache_demo/total_mv.csv)                                                               | float       | 总市值 （万元）                                                       |
| [circ_mv.csv](demo/cache_demo/circ_mv.csv)                                                                 | float       | 流通市值（万元）                                                       |
| [idx_weight](https://tushare.pro/document/2?doc_id=96)                                                     | 指数成分和权重     | 获取各类指数成分和权重，**月度数据** 。来源：指数公司网站公开数据                            |
| [index_weight_CSI300.csv](demo/cache_demo/index_weight_CSI300.csv)                                         | float       | CSI300 : 399300.SZ                                             |
| [index_weight_CSI500.csv](demo/cache_demo/index_weight_CSI500.csv)                                         | float       | CSI500 : 000905.SH                                             |
| [index_weight_CSI800.csv](demo/cache_demo/index_weight_CSI800.csv)                                         | float       | CSI800 : 000906.SH                                             |
| [index_weight_CSI1000.csv](demo/cache_demo/index_weight_CSI1000.csv)                                       | float       | CSI1000 : 000852.SH                                            |
