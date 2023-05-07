import numpy as np
import os
from utils.load_data import get_stock_price
import pandas as pd
from parallel_factor_generation.clean_factor import (
    clean_factor_from_raw,
    load_support_data,
)
from tqdm import tqdm


def create_stock_ret(start_time, end_time, ret_type):
    stock_close = get_stock_price(start_time, end_time, "adjclose").dropna(
        how="all", axis=1
    )
    if ret_type == "close":
        return stock_close.pct_change()
    elif ret_type == "overnight":
        stock_open = get_stock_price(start_time, end_time, "adjopen").dropna(
            how="all", axis=1
        )
        return stock_open / stock_close.shift() - 1.0
    elif ret_type == "intraday":
        stock_open = get_stock_price(start_time, end_time, "adjopen").dropna(
            how="all", axis=1
        )
        return stock_close / stock_open - 1.0
    else:
        raise ValueError(f"wrong ret_type: {ret_type}")


if __name__ == "__main__":
    ret_types = ["close", "overnight", "intraday"]
    window_sizes = [5, 20, 120, 240]

    start_time = pd.to_datetime("20120101")
    end_time = pd.to_datetime("20230430")

    preprocessors, mask = load_support_data(start_time, end_time)
    output_folder = "E:/google_drive/develop_strategies/momentum"

    pbar = tqdm(ret_types)
    for ret_type in pbar:
        pbar.set_description(ret_type)
        pbar2 = tqdm(window_sizes)
        for window_size in pbar2:
            pbar2.set_description(str(window_size))
            stock_ret = create_stock_ret(
                start_time=start_time, end_time=end_time, ret_type=ret_type
            )
            if window_size == 5:
                raw_factor = np.log(1.0 + stock_ret).rolling(window_size).sum()
            else:
                raw_factor = (
                    np.log(1.0 + stock_ret).rolling(window_size).sum()
                    - np.log(1.0 + stock_ret).rolling(5).sum()
                )

            raw_factor = raw_factor.dropna(how="all", axis=1).dropna(how="all", axis=0)

            cleaned_factors = clean_factor_from_raw(raw_factor, preprocessors, mask)
            output_filename = os.path.join(
                output_folder, f"mom_{ret_type}_w{window_size}.feather"
            )
            cleaned_factors.reset_index().to_feather(output_filename, compression="lz4")
