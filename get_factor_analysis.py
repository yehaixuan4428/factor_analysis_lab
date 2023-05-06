#
# Created on Tue Apr 18 2023
#
# Copyright (c) 2023 China Galaxy Security Co., Ltd.
# Author: Haixuan Ye
#
# 因子分析架构包括：
# 按 全市场 、 分行业（基于最新行业，虽然有lookahead bias，但是只是单因子分析的话，影响不大),  分市值（0~33， 33~66， 66~100）来看以下指标：
# IC相关： 与未来1D, 5D, 10D 的IC均值， 显著性t-value, ICIR, IC绝对值大于0.02的比例。
# 分层收益情况 (summary table + plot)


import pandas as pd
from alphalens import utils
from utils.load_data import get_stock_price, load_preprocessor
from scipy import stats
import empyrical as ep


def get_clean_factor_and_forward_returns(
    factor_data, price_data, industry_info, mv_info, periods, n_groups=10
):
    factor_returns = utils.get_clean_factor_and_forward_returns(
        factor_data,
        price_data,
        quantiles=n_groups,
        groupby=industry_info,
        binning_by_group=True,
        periods=periods,
    )

    factor_returns["factor_quantile_by_industry"] = factor_returns.groupby(
        level=0, group_keys=False
    )["factor"].apply(lambda x: pd.qcut(x, q=n_groups, labels=False))

    factor_returns["mv"] = mv_info.reindex(factor_returns.index)
    factor_returns["mv"].fillna(factor_returns["mv"].median(), inplace=True)
    factor_returns["mv_quantile"] = factor_returns.groupby(level=0, group_keys=False)[
        "mv"
    ].apply(lambda x: pd.qcut(x, q=3, labels=["Small", "Medium", "Big"]))
    factor_returns["factor_quantile_by_mv"] = (
        factor_returns.reset_index()
        .groupby(["date", "mv_quantile"], group_keys=False)["factor"]
        .apply(lambda x: pd.qcut(x, q=n_groups, labels=False))
        .values
    )

    return factor_returns


# 全市场分析
def analyze_by_whole_market(factor_returns, ic_threshold=0.02):
    # IC analysis
    forward_return_columns = utils.get_forward_returns_columns(factor_returns.columns)
    grouper = factor_returns.groupby(level=0)

    #### IC calculation
    IC_ts = grouper.apply(
        lambda x: x[forward_return_columns].corrwith(x["factor"], method="spearman")
    )
    # icir
    icir = IC_ts.mean() / IC_ts.std()
    # ic t-value
    ic_tvalue = pd.Series(stats.ttest_1samp(IC_ts, 0).statistic, index=icir.index)
    # prob of abs(IC) > 0.02
    prob = (IC_ts.abs() > ic_threshold).sum() / len(IC_ts)
    ic_mean = IC_ts.mean()

    ic_summary = pd.concat(
        [
            ic_mean.rename("IC_mean"),
            icir.rename("ICIR"),
            ic_tvalue.rename("IC t-value"),
            prob.rename(f"IC prob (abs > {ic_threshold})"),
        ],
        axis=1,
    )
    ic_monthly_mean = IC_ts.rolling(20).mean().dropna()
    #######
    # return analysis
    group_returns = grouper.apply(
        lambda x: x.groupby("factor_quantile")[forward_return_columns].mean()
    )

    multiplier = [int(x[:-1]) for x in forward_return_columns]

    group_returns_1d = group_returns.copy()
    for col, m in zip(forward_return_columns, multiplier):
        group_returns_1d[col] = (1.0 + group_returns_1d[col]) ** (1 / m) - 1.0

    annual_return = group_returns_1d.groupby(level=1).apply(ep.annual_return)
    annual_vol = pd.DataFrame(
        group_returns_1d.groupby(level=1).apply(ep.annual_volatility).tolist(),
        index=annual_return.index,
        columns=annual_return.columns,
    )
    annual_mdd = group_returns_1d.groupby(level=1).apply(ep.max_drawdown)
    group_return_summary = pd.concat(
        [
            pd.concat([annual_return], keys=["Annual_return"], names=["Type"], axis=1),
            pd.concat([annual_vol], keys=["Annual_vol"], names=["Type"], axis=1),
            pd.concat([annual_mdd], keys=["Annual_MDD"], names=["Type"], axis=1),
            pd.concat(
                [group_returns.groupby(level=1).mean()],
                keys=["Mean_ret"],
                names=["Type"],
                axis=1,
            ),
        ],
        axis=1,
    )
    group_cum_return = group_returns_1d.groupby(level=1, group_keys=False).apply(
        lambda x: (1.0 + x).cumprod()
    )
    return {
        "ic_summary": ic_summary,
        "ic_monthly_mean": ic_monthly_mean,
        "group_return_summary": group_return_summary,
        "group_cum_return": group_cum_return,
    }


def add_multiindex(df, new_name):
    return pd.concat([df], keys=[new_name], names=["Type"], axis=1)


# 分行业分析
def analyze_by_industry(factor_returns, ic_threshold=0.02):
    # IC analysis
    forward_return_columns = utils.get_forward_returns_columns(factor_returns.columns)
    factor_returns = factor_returns.reset_index()
    grouper = factor_returns.groupby(["date", "group"])

    #### IC calculation
    IC_ts = grouper.apply(
        lambda x: x[forward_return_columns].corrwith(x["factor"], method="spearman")
    )

    # icir
    industry_grouper = IC_ts.groupby(level=1)
    icir = industry_grouper.apply(lambda x: x.mean() / x.std())

    # ic t-value
    # ic_tvalue = pd.Series(stats.ttest_1samp(IC_ts, 0).statistic, index=icir.index)
    ic_tvalue = pd.DataFrame(
        industry_grouper.apply(lambda x: stats.ttest_1samp(x, 0.0).statistic).tolist(),
        index=icir.index,
        columns=icir.columns,
    )
    # prob of abs(IC) > 0.02
    # prob = (IC_ts.abs() > ic_threshold).sum() / len(IC_ts)
    prob = industry_grouper.apply(lambda x: (x.abs() > ic_threshold).sum() / len(x))

    ic_mean = industry_grouper.mean()

    ic_summary = pd.concat(
        [
            add_multiindex(ic_mean, "IC_mean"),
            add_multiindex(icir, "ICIR"),
            add_multiindex(ic_tvalue, "IC t-value"),
            add_multiindex(prob, f"IC prob (abs > {ic_threshold})"),
        ],
        axis=1,
    )
    ic_monthly_mean = (
        IC_ts.groupby(level=1, group_keys=False).rolling(20).mean().dropna()
    )
    #######

    # return analysis
    group_returns = factor_returns.groupby(
        ["date", "group", "factor_quantile_by_industry"]
    )[forward_return_columns].mean()

    multiplier = [int(x[:-1]) for x in forward_return_columns]

    group_returns_1d = group_returns.copy()
    for col, m in zip(forward_return_columns, multiplier):
        group_returns_1d[col] = (1.0 + group_returns_1d[col]) ** (1 / m) - 1.0

    annual_return = group_returns_1d.groupby(level=[1, 2]).apply(ep.annual_return)

    annual_vol = pd.DataFrame(
        group_returns_1d.groupby(level=[1, 2]).apply(ep.annual_volatility).tolist(),
        index=annual_return.index,
        columns=annual_return.columns,
    )
    annual_mdd = group_returns_1d.groupby(level=[1, 2]).apply(ep.max_drawdown)
    group_return_summary = pd.concat(
        [
            add_multiindex(annual_return, "Annual_return"),
            add_multiindex(annual_vol, "Annual_vol"),
            add_multiindex(annual_mdd, "Annual_MDD"),
            add_multiindex(group_returns.groupby(level=[1, 2]).mean(), "Mean_ret"),
        ],
        axis=1,
    )
    group_cum_return = group_returns_1d.groupby(level=[1, 2], group_keys=False).apply(
        lambda x: (1.0 + x).cumprod()
    )
    return {
        "ic_summary": ic_summary,
        "ic_monthly_mean": ic_monthly_mean,
        "group_return_summary": group_return_summary,
        "group_cum_return": group_cum_return,
    }


# 分市值规模分析
def analyze_by_market_value(factor_returns, ic_threshold=0.02):
    # IC analysis
    forward_return_columns = utils.get_forward_returns_columns(factor_returns.columns)
    factor_returns = factor_returns.reset_index()
    grouper = factor_returns.groupby(["date", "mv_quantile"])

    #### IC calculation
    IC_ts = grouper.apply(
        lambda x: x[forward_return_columns].corrwith(x["factor"], method="spearman")
    )

    # icir
    industry_grouper = IC_ts.groupby(level=1)
    icir = industry_grouper.apply(lambda x: x.mean() / x.std())

    # ic t-value
    # ic_tvalue = pd.Series(stats.ttest_1samp(IC_ts, 0).statistic, index=icir.index)
    ic_tvalue = pd.DataFrame(
        industry_grouper.apply(lambda x: stats.ttest_1samp(x, 0.0).statistic).tolist(),
        index=icir.index,
        columns=icir.columns,
    )
    # prob of abs(IC) > 0.02
    # prob = (IC_ts.abs() > ic_threshold).sum() / len(IC_ts)
    prob = industry_grouper.apply(lambda x: (x.abs() > ic_threshold).sum() / len(x))

    ic_mean = industry_grouper.mean()

    ic_summary = pd.concat(
        [
            add_multiindex(ic_mean, "IC_mean"),
            add_multiindex(icir, "ICIR"),
            add_multiindex(ic_tvalue, "IC t-value"),
            add_multiindex(prob, f"IC prob (abs > {ic_threshold})"),
        ],
        axis=1,
    )
    ic_monthly_mean = (
        IC_ts.groupby(level=1, group_keys=False).rolling(20).mean().dropna()
    )
    #######

    # return analysis
    group_returns = factor_returns.groupby(
        ["date", "mv_quantile", "factor_quantile_by_mv"]
    )[forward_return_columns].mean()

    multiplier = [int(x[:-1]) for x in forward_return_columns]

    group_returns_1d = group_returns.copy()
    for col, m in zip(forward_return_columns, multiplier):
        group_returns_1d[col] = (1.0 + group_returns_1d[col]) ** (1 / m) - 1.0

    annual_return = group_returns_1d.groupby(level=[1, 2]).apply(ep.annual_return)

    annual_vol = pd.DataFrame(
        group_returns_1d.groupby(level=[1, 2]).apply(ep.annual_volatility).tolist(),
        index=annual_return.index,
        columns=annual_return.columns,
    )
    annual_mdd = group_returns_1d.groupby(level=[1, 2]).apply(ep.max_drawdown)
    group_return_summary = pd.concat(
        [
            add_multiindex(annual_return, "Annual_return"),
            add_multiindex(annual_vol, "Annual_vol"),
            add_multiindex(annual_mdd, "Annual_MDD"),
            add_multiindex(group_returns.groupby(level=[1, 2]).mean(), "Mean_ret"),
        ],
        axis=1,
    )
    group_cum_return = group_returns_1d.groupby(level=[1, 2], group_keys=False).apply(
        lambda x: (1.0 + x).cumprod()
    )
    return {
        "ic_summary": ic_summary,
        "ic_monthly_mean": ic_monthly_mean,
        "group_return_summary": group_return_summary,
        "group_cum_return": group_cum_return,
    }


if __name__ == "__main__":
    start_time = pd.to_datetime("20220101")
    end_time = pd.to_datetime("20220930")
    stock_price = get_stock_price(
        start_time=start_time, end_time=end_time, price_type="adjopen"
    )
    factor_data = (
        pd.read_feather(
            "E:/google_drive/factor_database/pfactors/alphaP1015_20.feather"
        )
        .set_index("dt")
        .loc[start_time : end_time - pd.DateOffset(months=1)]
        .stack()
    )
    preprocessor = load_preprocessor(start_time=start_time, end_time=end_time)
    del preprocessor["mv"]

    mv = get_stock_price(start_time, end_time, "mv_total").stack()

    preprocessor = preprocessor.stack()
    preprocessor = preprocessor.loc[preprocessor != 0].sort_index()
    preprocessor = (
        preprocessor.reset_index()
        .set_index(["dt", "code"])["level_2"]
        .astype("category")
    )

    factor_returns = get_clean_factor_and_forward_returns(
        factor_data=factor_data,
        price_data=stock_price.shift(-1),
        industry_info=preprocessor,
        mv_info=mv,
        n_groups=10,
    )

    results = analyze_by_market_value(factor_returns)
    print(results)
