import pandas as pd
import utils
from pandas import DataFrame
from sklearn.preprocessing import StandardScaler, LabelEncoder
import seaborn as sns
import matplotlib.pyplot as plt


def main():
    # Load datasets - cleaned and merged into one
    df: DataFrame = utils.load_datasets(clean=True, merge=True, merge_how='inner')
    df = df.dropna()
    df = df.sample(200000)
    # Print quantities
    print(utils.n_unique_get(df))

    # Encode categorical features
    label_encoder = LabelEncoder()
    df['sensor_id'] = label_encoder.fit_transform(df['sensor'])
    df['branch_id'] = label_encoder.fit_transform(df['branch_header'])
    df = df.drop(columns=['sensor', 'branch_header'])
    # print(utils.unique_values_sample_get(df))

    # Select numeric columns
    features = ['power_dbm', 'id_audit', 'sensor_id', 'branch_id',
                'temperature', 'powerclass_watt']

    # Standardize the data
    standard_scaler = StandardScaler()
    scaled_features = standard_scaler.fit_transform(df[features])

    # Remove outliers (temperature range outside [-40 - 125] +- error_margin

    sns.pairplot(df[features], hue='branch_id', palette='viridis')
    plt.suptitle("Pair plot of raw data", y=1.02)
    plt.show()


if __name__ == '__main__':
    main()

