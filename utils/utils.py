import pandas as pd
import numpy as np

def aggregate_weekly_data(df):
    return_cols = [col for col in df.columns if 'yield' not in col]
    additive_cols = ["yield_delta"]
    last_value_cols = ["yield_level"]
    log_returns = np.log1p(df[return_cols])
    compounded = np.expm1(log_returns.resample("W-FRI").sum())
    summed = df[additive_cols].resample("W-FRI").sum()
    last_vals = df[last_value_cols].resample("W-FRI").last()
    result = pd.concat([compounded, summed, last_vals], axis=1)
    return result.dropna()