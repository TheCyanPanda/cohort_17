import pandas as pd
from pandas.core.frame import DataFrame as DataFrame
from typing import Any
import json
import os
from dataclasses import dataclass, field
import numpy as np


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
                  merge_how: str = 'outer') -> list[DataFrame] | DataFrame:
    """
    Loads the datasets given by the configuration
    Args:
        base_path: path to datasets
        script_config_: ScriptConfig object
        clean: Cleans up the data if True
    Returns:
        A list containing the three loaded datasets if 'merge' is False, else a single DataFrame
        containing the merged data
    """
    dfs: list[DataFrame] = []
    for df in script_config_.dataset_file_names:
        dfs.append(pd.read_csv(f"{base_path}/{df}", delimiter=";"))

    if clean:
        # Clean up column names
        dfs[0] = dfs[0].rename(columns={'field': 'sensor',
                                        'value': 'temperature'})

        # Extract powerclass (W/dBm) from single column
        dfs[1]['power_dbm'] = dfs[1]["value"].str.extract(r"\[(\d+\.\d+) dBm\]").astype(float)
        dfs[1]["powerclass_watt"] = dfs[1]["value"].str.extract(r"(\d+)").astype(int)
        dfs[1] = dfs[1].drop(columns=['value'])

        # Remove any rows in 'unit' column != C
        dfs[0] = filter_df(dfs[0], 'unit', value=['C'])
        dfs[0] = dfs[0].drop(columns=['unit'])

        # Remove 'field' column (Only contains one unique value)
        dfs[1] = dfs[1].drop(columns=['field'])

    if merge:
        df_merged: DataFrame = pd.merge(dfs[0], dfs[1], on='id_audit', how=merge_how)
        df_merged = pd.merge(df_merged, dfs[2], on='id_audit', how=merge_how)
        return df_merged

    return dfs


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
