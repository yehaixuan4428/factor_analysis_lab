import numpy as np
import pandas as pd
import os
from utils.load_data import get_stock_price
from utils.price_vol_api import MData
from tqdm import tqdm
from parallel_factor_generation.clean_factor import (
    clean_factor_from_raw,
    load_support_data,
)


def create_amp_weighted_intraday_mmt(mData, params):
    close = mData.get("close")
    high = mData.get("high")
    low = mData.get("low")
    open = mData.get("open")
    amp = (high - low) / close

    stock_ret = np.log(close / open)
    # stock_ret = np.log(1.0 + close.pct_change())

    results = pd.DataFrame(np.nan, index=amp.index, columns=amp.columns)

    upper_cap = 0.8
    lower_cap = 0.2
    period = params[0]

    amp_q_upper = amp.rolling(period).quantile(upper_cap)
    amp_q_lower = amp.rolling(period).quantile(lower_cap)

    for i in tqdm(range(period - 1, len(results.index))):
        tmp = amp.iloc[i - period + 1 : i + 1, :]
        tmp_upper = amp_q_upper.iloc[i - period + 1 : i + 1, :]
        tmp_lower = amp_q_lower.iloc[i - period + 1 : i + 1, :]
        tmp_close_pct = stock_ret.iloc[i - period + 1 : i + 1, :]

        upper = tmp.subtract(tmp_upper, axis=1)
        lower = tmp.subtract(tmp_lower, axis=1)
        weight = pd.DataFrame(
            0.0, index=amp.index[i - period + 1 : i + 1], columns=amp.columns
        )
        weight = weight.mask(upper >= 0, other=1.0)
        weight = weight.mask(lower <= 0, other=-1.0)
        results.iloc[i, :] = (weight * tmp_close_pct).sum()

    stock_ret_overnight = np.log(open / close.shift())
    mmt_overnight_120 = stock_ret_overnight.rolling(120).sum()
    stock_ret_5 = stock_ret.rolling(5).sum()

    results = results.shift(5).subtract(mmt_overnight_120, fill_value=0.0)
    results = results.add(stock_ret_5, fill_value=0.0)

    return results


if __name__ == "__main__":
    mData = MData()

    start_time = pd.to_datetime("20120101")
    end_time = pd.to_datetime("20230430")

    close = get_stock_price(
        start_time=start_time, end_time=end_time, price_type="adjclose"
    )
    low = get_stock_price(start_time=start_time, end_time=end_time, price_type="adjlow")
    high = get_stock_price(
        start_time=start_time, end_time=end_time, price_type="adjhigh"
    )
    open = get_stock_price(
        start_time=start_time, end_time=end_time, price_type="adjopen"
    )

    mData.load_data(close, "close")
    mData.load_data(low, "low")
    mData.load_data(high, "high")
    mData.load_data(open, "open")

    raw_factor = create_amp_weighted_intraday_mmt(mData=mData, params=[20])
    raw_factor = raw_factor.dropna(how="all", axis=1).dropna(how="all", axis=0)

    preprocessors, mask = load_support_data(start_time, end_time)
    output_folder = "E:/google_drive/develop_strategies/momentum"

    cleaned_factors = clean_factor_from_raw(raw_factor, preprocessors, mask)
    output_filename = os.path.join(output_folder, f"amp_weighted_mmt_final.feather")
    cleaned_factors.reset_index().to_feather(output_filename, compression="lz4")
