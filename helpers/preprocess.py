import numpy as np
import pandas as pd
from pandas import DataFrame


def preprocess(data: DataFrame, sort_by: list[str], r_nan: str, r_columns: list[str] = [], ascending: bool = False) -> DataFrame:
    # Remove columns
    if len(r_columns) > 0:
        data = data.drop(r_columns, axis=1)

    # Remove NaNs
    if r_nan:
        data = data.replace(r_nan, np.nan)
        data = data.dropna()

    # Sort
    if sort_by:
        data = data.sort_values(by=sort_by, ascending=ascending)

    return data
