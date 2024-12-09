from __future__ import annotations
import os
import pandas as pd


def load_dataset():
    PATH="/home/jovyan/Project_5_Temp_Sensors/"
    
    ''' load dataset and do non destruction conversions '''
    temp = pd.read_csv(PATH+'dataset_1_radioTemperatures_20210303.csv', delimiter=';')
    power = pd.read_csv(PATH+'dataset_2_powerClass_20210303.csv', delimiter=';')
    radio = pd.read_csv(PATH+'dataset_3_radioId_20210303.csv', delimiter=';')
    temp.rename(columns={'value':'temp_value','field':'sensor_label'}, inplace=True)
    power.rename(columns={'value':'power_class'}, inplace=True)
    return temp,power,radio

def drop_irrelevant_columns(temp,power):
    '''drop irrelevant columns, as indivated by data owner'''
    temp.drop(['id_field_values','id_ftp','unit'], axis=1,inplace=True)
    power.drop(['id_trx_status','field'], axis=1,inplace=True)
    
def drop_illegal_temp_values(df):
    '''drop any temp values out of range of [-40, 125] as per data owner'''
    df.drop(df[df['temp_value'] < -40].index, inplace=True)
    df.drop(df[df['temp_value'] > 125].index, inplace=True)

def split_power_class(df):
    '''split power_class into Watts and dBm and then drop the power_class column'''
    '''extract alphanumeric code for branch headers and sensor labels to encode later'''
    df['watts'] = df['power_class'].str.extract(r'(\d+)W').astype(int)
    df['dbm'] = df['power_class'].str.extract(r'\[([^ ]*)').astype(float)
    df.drop(['power_class'], axis=1,inplace=True)

def extract_sensor_label(df):
    df['sensor_label'] = df['sensor_label'].str.extract(r'\_(\D)')

def extract_sensor_label_full(df):
    df['sensor_label'] = df['sensor_label'].str.extract(r'\_([^_]*)')

def extract_branch_header(df):
    df['branch_header'] = df['branch_header'].str.extract(r'\ (\D)')

def label_encode_sensor_label(df):
    '''Encode categorical feature (branch headers and sensor labels)'''
    label_encoder = LabelEncoder()
    df['sensor'] = label_encoder.fit_transform(df['sensor_label'])
    df.drop(['sensor_label'], axis=1,inplace=True)

def label_encode_id_audit(temp,power,radio,le):
    '''label encode in all three dataframes using the provided label encoder'''
    df = pd.concat([temp,power,radio])
    le.fit(df['id_audit'])
    temp['id_audit_le'] = le.transform(temp['id_audit'])
    temp.drop(['id_audit'], axis=1,inplace=True)
    power['id_audit_le'] = le.transform(power['id_audit'])
    power.drop(['id_audit'], axis=1,inplace=True)
    radio['id_audit_le'] = le.transform(radio['id_audit'])
    radio.drop(['id_audit'], axis=1,inplace=True)

def group_encode_branch_header(df):
    '''branch header is grouped as A-D --> 0 and E-H --> 1'''
    df['branch_header'] = power['branch_header'].replace ({'A','B','C', 'D'}, '0')
    df['branch_header'] = power['branch_header'].replace ({'E','F','G', 'H'}, '1')
    df['branch_header_enc'] = label_encoder.fit_transform(df['branch_header'])
    df.drop(['branch_header'], axis=1,inplace=True)

def transposeSensorLabels_move(x):
    '''within a groupby group, move the temp_value to a column identified by sensor_label'''
    '''such that a full group (id_audit) becomes a single row with many columns (one per sensor)'''
    '''https://pandas.pydata.org/docs/reference/api/pandas.core.groupby.DataFrameGroupBy.apply.html'''
    q ={}
    index = []
    for label in x['sensor_label']:
        q[label] = x['temp_value'][x[x['sensor_label'] == label].index.tolist()].tolist()
    return pd.DataFrame(q)

def transposeSensorLabels(df):
    x = df.groupby('id_audit').parallel_apply(transposeSensorLabels_move).reset_index()
    x.drop(['level_1'],axis=1,inplace=True)
    return x
    
def transposePower_move(df):
    q ={}
    index = []
    for branch in df['branch_header']:
        q[branch+"watts"] = df['watts'][df[df['branch_header'] == branch].index.tolist()].tolist()
        q[branch+"dbm"] = df['dbm'][df[df['branch_header'] == branch].index.tolist()].tolist()
    return pd.DataFrame(q)
    
def transposePower(df):
    x = df.groupby('id_audit').parallel_apply(transposePower_move).reset_index()
    x.drop(['level_1'],axis=1,inplace=True)
    return x

def tranpose_dataframe(df: pd.DataFrame,
                       key_column: str | list[str],
                       value_columns: list[str],
                       suffixes: dict[str, str] | None = None,
                       group_column: str = 'id_audit'
) -> pd.DataFrame:
    """
    Returns a transposed dataframe
    
    Args:
        df: Target dataframe to be transposed
        key_column: Name opf column to be split up in a pivot table to become new columns
        value_columns: List of columns to be transposed into the new columns
    """
    # Create a copy of the dataframe for each value column to pivot
    pivoted_dfs = []
    
    # Create pivot table for each target value_column
    for value_column in value_columns:
        pivot_df = df.pivot_table(index=group_column, columns=key_column, values=value_column, dropna=False)
    
        if suffixes:
            pivot_df.columns = [f"{col}{suffixes.get(value_column, '')}" for col in pivot_df.columns]
        
        pivoted_dfs.append(pivot_df)
    
    # Concatenate all pivoted DataFrames along the columns
    result_df = pd.concat(pivoted_dfs, axis=1)
    
    return result_df.reset_index()

def clean_datasets(dfs: tuple[pd.DataFrame]) -> None:
    """
    Cleans the dataset in place.
    """
    drop_irrelevant_columns(dfs[0], dfs[1])
    drop_illegal_temp_values(dfs[0])
    split_power_class(dfs[1])
    extract_sensor_label_full(dfs[0])
    extract_branch_header(dfs[1])

def merge_datasets(dfs: tuple[pd.DataFrame]) -> pd.DataFrame:
    """
    Returns the merged dataset where sensors/temperature and branch/power as been transformed into separate columns
    """
    df_temp_transposed = tranpose_dataframe(dfs[0], key_column='sensor_label',
                                               value_columns=['temp_value'])
    df_power_transposed = tranpose_dataframe(dfs[1], key_column='branch_header',
                                                value_columns=['watts', 'dbm'],
                                                suffixes={'watts': 'watts', 'dbm': 'dbm'})
    df = pd.merge(dfs[2], df_temp_transposed)
    df = pd.merge(df, df_power_transposed)
    return df