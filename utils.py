from __future__ import annotations
import pandas as pd
from pandas.core.frame import DataFrame as DataFrame
from typing import Any
import json
import os
from dataclasses import dataclass, field
import numpy as np
from sklearn.preprocessing import LabelEncoder


class MutualExclusivityError(Exception):
    pass


@dataclass
class ScriptConfig:
    config_file: str = "config.json"

    # Default configuration keys and their corresponding default values
    config_defaults: dict[str, Any] = field(
        default_factory=lambda: {
            "default_path": os.getcwd(),  # Default working path (location of data sets)
            "dataset_file_names": [],  # List containing strings representing file names of DFs
        }
    )

    # Config dictionary to hold loaded configuration
    config: dict[str, Any] = field(init=False)

    # This will hold the values for default_path, log_level, and timeout
    default_path: str = field(init=False)
    log_level: str = field(init=False)
    timeout: int = field(init=False)
    dataset_file_names: list[str] = field(init=False)

    def __post_init__(self):
        # Load the configuration when the object is created
        self.config = self.load_config()

        # Load the key values from config or default if they don't exist
        self.default_path = self.get_config_value("default_path")
        self.dataset_file_names = self.get_config_value("dataset_file_names")

    def load_config(self) -> dict[str, Any]:
        """
        Load the configuration file if it exists, otherwise return default values.

        Returns:
        - dict: Configuration or default values if the file does not exist or fails to load.
        """
        if os.path.exists(self.config_file):
            try:
                with open(self.config_file, "r") as file:
                    return json.load(file)
            except json.JSONDecodeError:
                print(
                    f"Error parsing {self.config_file}, using default configuration.")
                return self.config_defaults
        else:
            print(f"{self.config_file} not found, using default configuration.")
            return self.config_defaults

    def get_config_value(self, key: str) -> Any:
        """
        Retrieve a value from the config dictionary, defaulting to the value in config_defaults
        if the key is not found.

        Args:
        - key (str): The key to retrieve from the config.

        Returns:
        - Any: The value associated with the key, or the default value if the key is not found.
        """
        return self.config.get(key, self.config_defaults[key])


# Example usage:

# Create an instance of the ScriptConfig class
script_config = ScriptConfig()


def load_datasets(base_path: str = script_config.default_path,
                  script_config_: ScriptConfig = script_config,
                  clean: bool = False,
                  merge: bool = False,
                  concat: bool = False,
                  label_encode: list[str] | None = None,
                  replace_encode: bool = False,
                  sample_size: float | None = None
) -> list[DataFrame] | DataFrame:
    """
    Loads the datasets given by the configuration
    Args:
        base_path: path to datasets
        script_config_: ScriptConfig object
        clean: Cleans up the data if True
        merge: Merges the datasets into one using ´combine_first´ if True
        concat: Concatenates the datasets into one if True
        sample_size: Select a fraction of the dataset to be loaded
        label_encode: Label encode provided columns
        replace_encode: Replace non-encoded columns with teh encoded columns
    Returns:
        A list containing the three loaded datasets if 'merge' is False, else a single DataFrame
        containing the merged data
    """
    if concat and merge:
        raise MutualExclusivityError("Select either merge or concat. Both option cannot be True")

    dfs: list[DataFrame] = []
    for df in script_config_.dataset_file_names:
        _df: DataFrame = pd.read_csv(f"{base_path}/{df}", delimiter=";")
        if sample_size is not None:
            _df = _df.sample(frac=sample_size)
        dfs.append(_df)

    # Label encoding
    if label_encode:
        label_encoder = LabelEncoder()
        for i, d in enumerate(dfs):
            for label in label_encode:
                if label in d:
                    new_label = f"{label}_le" if not replace_encode else label
                    dfs[i][new_label] = label_encoder.fit_transform(dfs[i][label])

    if clean:
        # Clean up column names
        dfs[0] = dfs[0].rename(columns={'field': 'sensor',
                                        'value': 'temperature'})
        # Clean up sensor values (Keep only what's contained within the underscores)
        dfs[0]['sensor'] = dfs[0]['sensor'].str.extract(r'_(.*?)_')

        # Extract powerclass (W/dBm) from single column
        dfs[1]['power_dbm'] = dfs[1]["value"].str.extract(r"\[(\d+\.\d+) dBm\]").astype(float)
        dfs[1]["powerclass_watt"] = dfs[1]["value"].str.extract(r"(\d+)").astype(int)
        dfs[1] = dfs[1].drop(columns=['value'])

        # Remove any rows in 'unit' column != C
        # dfs[0] = filter_df(dfs[0], 'unit', value=['C'])

        # Drop unnecessary columns
        dfs[1] = dfs[1].drop(columns=['field', 'id_trx_status'])
        dfs[0] = dfs[0].drop(columns=['unit', 'id_field_values', 'id_ftp'])

    if merge:
        for d in dfs:
            d.set_index('id_audit')
        df: DataFrame = dfs[0].combine_first(dfs[1]).combine_first(dfs[2])
        return df
    if concat:
        df: DataFrame = pd.concat(dfs)
        return df

    return dfs


def df_pivot_get(
    df: DataFrame,
    column: str,
    value: str,
    index: list[str] | None = None
) -> DataFrame:
    """
    Returns the dataframe reshaped as a pivot table, setting whatever is provided as parameter
    'column' as columns and placing parameter value 'values' as values. Default behavior is to
    reshape the dataframe as a pivot table for each sensor

    Args:
        df: Target dataframe
        column: The new column header
        value: Value to fill the matrix for the corresponding 'column' for each row
        index: Specifies which columns to keep as index (to be retained in the dataframe)

    Returns:
        A dataframe where each 'column' is a column and 'values' placed accordingly
    """
    if index is None:
        index = ['customer', 'id_audit', 'serial', 'branch_header']

    return df.pivot(index=index, columns=column, values=value)


def filter_count_std_95(df: DataFrame | pd.Series,
                        axis: int = 0,
                        column: str | None = None) -> DataFrame:
    """
    Filter dataframe to keep only the columns with a count above 2 STDs (I.e,
    remove the values that fall outside 95%)
    Args:
        df: Target dataframe
        axis: Target axis in dataframe
    Returns:
        The filtered dataframe
    """
    if column is not None:
        df = df[column]
    readings_per_column: pd.Series = df.count()
    mean_r: int = readings_per_column.mean()
    std_r: int = readings_per_column.std()
    region_95: tuple[int, int] = ((mean_r - 2 * std_r), (mean_r + 2 * std_r))
    return df.dropna(axis=axis, thresh=region_95[0])



def remove_outliers_range(
    df: pd.DataFrame,
    column: str,
    limits: tuple[int | float, int | float]
) -> pd.DataFrame:
    """
    Remove any rows from dataframe that is outside specified range for a given column
    Args:
        df: Target dataframe
        column: Name of column
        limits: Tuple with elements [start, stop] (inclusive)

    Returns:
        The cleaned dataframe
    """
    start, stop = limits
    return df[(df[column] >= start) & (df[column] <= stop)]


# Set outliers to None for all numerical columns in the DataFrame
def remove_outliers_IQR(
    col: pd.DataFrame | pd.Series | np.ndarray,
    limits: tuple[float, float] = (0.25, 0.75)
) -> pd.DataFrame:
    """
    Remove outliers for all numerical columns using the IQR method.
    Can handle both pandas DataFrame/Series and NumPy ndarray.
    """

    # If input is NumPy ndarray, convert to pandas DataFrame
    if isinstance(col, np.ndarray):
        col = pd.DataFrame(col)

    # Ensure all columns are numeric for DataFrame
    if isinstance(col, pd.DataFrame):
        if not col.apply(lambda c: np.issubdtype(c.dtype, np.number)).all():
            raise TypeError("All columns in the DataFrame must be numeric.")

    # Ensure the input is numeric for Series
    if isinstance(col, pd.Series):
        if not np.issubdtype(col.dtype, np.number):
            raise TypeError("Input column must be numeric.")

    # Apply IQR-based outlier removal column-wise
    def apply_iqr(column: pd.Series) -> pd.Series:
        Q1 = column.quantile(limits[0])
        Q3 = column.quantile(limits[1])
        IQR = Q3 - Q1
        return column.where((column >= (Q1 - 1.5 * IQR)) & (column <= (Q3 + 1.5 * IQR)))

    # If it's a DataFrame, apply the IQR function to each column
    col_cleaned = col.apply(apply_iqr) if isinstance(col, pd.DataFrame) else apply_iqr(col)

    # If the original input was a NumPy array, convert back to NumPy after cleaning
    if not isinstance(col_cleaned, pd.DataFrame):
        return pd.DataFrame(col_cleaned)

    return col_cleaned


def filter_df(df: DataFrame, column_name: str, value: list[str | int]) -> DataFrame:
    """
    Return a dataframe with only rows that match the column/value
    Example:
        filter_df(df, 'id', [147, 200, 4]) -> DF with only rows containing id = 147 | 200 | 4
    """
    res = df[df[column_name].isin(value)]
    assert isinstance(res, DataFrame)
    return res


def n_unique_get(df: DataFrame) -> DataFrame:
    """
    Show number of unique column for a dataframe
    Args:
        df: DataFrame to be analyzed
    Returns:
        A pandas dataframe containing two columns:
            'Column': Name of column
            'Unique': Number of unique rows for a perticular column
            'Total': Number of non-NaN values
    """
    col_info: list[int] = []

    # Loop through each column and gather the unique count and non-NaN count
    for col in df.columns:
        unique_count = df[col].nunique()
        non_nan_count = df[col].count()  # count() gives the number of non-NaN entries
        nan_count = df[col].isna().sum()
        col_info.append([col, unique_count, non_nan_count, nan_count])

    # Create and return the DataFrame
    return pd.DataFrame(col_info, columns=[
        "Column", "Unique", "Total (non-Nan)", "NaN"]).to_string(index=False)


def unique_values_sample_get(df: pd.DataFrame, n: int = 5) -> pd.DataFrame:
    """
    Get a sample of 'n' unique values from all columns for a dataset
    Args:
        df: Target pandas dataframe
        n: How many samples to get

    Returns:
        A DataFrame holding 'n' unique values from the provided DataFrame. If 'n' is lower than
        the number of unique values, the remaining values will be padded with NaN
    """
    data: dict[str, Any] = {}

    for column in df.columns:
        unique_values = df[column].unique()
        n_unique = len(unique_values)  # Get the number of unique values

        # Pick n random unique values
        if n_unique <= n:
            column_values = list(unique_values)
        else:
            column_values = np.random.choice(unique_values, size=n, replace=False)

        # Pad with NaN if there are fewer than 'n' unique values
        column_values = list(column_values)
        column_values.extend([np.nan] * (n - len(column_values)))

        data[column] = column_values

    return pd.DataFrame(data)


def main():
    # Load the three data sets
    dfs: list[DataFrame] = load_datasets()

    # Merge the datasets using 'id_audit' as the common key
    df_merged = pd.merge(
        dfs[0], dfs[1], on="id_audit", how="inner"
    )  # First merge temp and power
    df_merged = pd.merge(
        df_merged, dfs[2], on="id_audit", how="inner"
    )  # Then merge with radio

    print(df_merged)


if __name__ == "__main__":
    main()
