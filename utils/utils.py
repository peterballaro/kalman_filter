import pandas as pd
import numpy as np

def aggregate_weekly_data(df, additive_cols=[]):
    for col in df:
        if "OAS" in col or "yield" in col and col not in additive_cols:
            print(f"Warning: Column '{col}' contains 'OAS' or 'yield', which may not be suitable for additive aggregation.")
    return_cols = [col for col in df.columns if col not in additive_cols]
    last_value_cols = [col for col in df.columns if col not in return_cols and col not in additive_cols]
    log_returns = np.log1p(df[return_cols])
    compounded = np.expm1(log_returns.resample("W-FRI").sum())
    summed = df[additive_cols].resample("W-FRI").sum()
    last_vals = df[last_value_cols].resample("W-FRI").last()
    result = pd.concat([compounded, summed, last_vals], axis=1)
    return result.dropna()