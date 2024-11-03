import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.impute import SimpleImputer
from sklearn.neighbors import KNeighborsRegressor
from sklearn.cluster import DBSCAN


class NumericalPreprocessingToolkit:
    """
    A class for performing various preprocessing techniques on numerical data.

    Techniques:
    1. Standardization: Scale data to have a mean of 0 and a standard deviation of 1.
    2. Normalization: Scale data to a range between 0 and 1 or -1 and 1.
    3. Handling Missing Values: Impute missing values using methods like mean, median, or mode.
    4. Outlier Detection and Removal: Identify and remove outliers using Z-score, IQR, or DBSCAN.
    5. Binning: Convert continuous numerical variables into discrete bins.

    Attributes:
    df : pandas.DataFrame
        Input data.
    columns : list
        List of columns to be processed.
    """

    def __init__(self, df: pd.DataFrame):
        """
        Initialize with a pandas DataFrame.

        Parameters:
        df (pandas.DataFrame): The input DataFrame.
        """
        self.df = df

    def _replace_nan(self, value: float = 0) -> pd.DataFrame:
        self.df.replace('', np.nan, inplace=True)
        self.df.replace(' ', np.nan, inplace=True)
        self.df.replace('?', np.nan, inplace=True)
        self.df.replace('NA', np.nan, inplace=True)
        self.df.replace('N/A', np.nan, inplace=True)
        self.df.replace('na', np.nan, inplace=True)
        self.df.replace('n/a', np.nan, inplace=True)
        self.df.replace('nan', np.nan, inplace=True)
        self.df.replace('NaN', np.nan, inplace=True)
        self.df.replace('Nan', np.nan, inplace=True)
        self.df.replace('NAN', np.nan, inplace=True)
        self.df.replace(np.nan, value, inplace=True)
        return self.df

    def handle_missing_values(self, columns: list, method: str = 'mean') -> pd.DataFrame:
        """
        Handle missing values in the specified columns using the given imputation method.

        Parameters:
        columns (list): List of column names to impute missing values.
        method (str): Imputation method - 'mean', 'median', 'mode', or 'knn'.

        Returns:
        pandas.DataFrame: DataFrame with missing values imputed.
        """
        self._replace_nan()
        if method == 'mean':
            imputer = SimpleImputer(strategy='mean')
        elif method == 'median':
            imputer = SimpleImputer(strategy='median')
        elif method == 'mode':
            imputer = SimpleImputer(strategy='most_frequent')
        elif method == 'knn':

            for col in columns:
                missing_idx = self.df[self.df[col].isna()].index
                non_missing_idx = self.df[self.df[col].notna()].index
                knn = KNeighborsRegressor(n_neighbors=5)
                knn.fit(self.df.loc[non_missing_idx, columns].dropna(
                ), self.df.loc[non_missing_idx, col].dropna())
                self.df.loc[missing_idx, col] = knn.predict(
                    self.df.loc[missing_idx, columns].dropna())
            return self.df
        else:
            raise ValueError(
                "Invalid method. Choose 'mean', 'median', 'mode', or 'knn'.")

        self.df[columns] = imputer.fit_transform(self.df[columns])
        return self.df

    def standardize(self, columns: list) -> pd.DataFrame:
        """
        Standardize the specified columns (mean = 0, standard deviation = 1).

        Parameters:
        columns (list): List of column names to standardize.

        Returns:
        pandas.DataFrame: DataFrame with standardized columns.
        """
        self.handle_missing_values(columns)
        scaler = StandardScaler()
        self.df[columns] = scaler.fit_transform(self.df[columns])
        return self.df

    def normalize(self, columns: list, feature_range: tuple = (0, 1)) -> pd.DataFrame:
        """
        Normalize the specified columns to a given range.

        Parameters:
        columns (list): List of column names to normalize.
        feature_range (tuple): The desired range of the transformed data (default is (0, 1)).

        Returns:
        pandas.DataFrame: DataFrame with normalized columns.
        """
        self.handle_missing_values(columns)
        scaler = MinMaxScaler(feature_range=feature_range)
        self.df[columns] = scaler.fit_transform(self.df[columns])
        return self.df

    def detect_and_remove_outliers(self, columns: list, method: str = 'zscore', threshold: float = 3.0) -> pd.DataFrame:
        """
        Detect and remove outliers from the specified columns.

        Parameters:
        columns (list): List of column names to check for outliers.
        method (str): Method to use - 'zscore', 'iqr', or 'dbscan'.
        threshold (float): The threshold for outlier detection (used in Z-score and IQR).

        Returns:
        pandas.DataFrame: DataFrame with outliers removed.
        """
        self.handle_missing_values(columns)

        if method == 'zscore':
            z_scores = np.abs(
                (self.df[columns] - self.df[columns].mean()) / self.df[columns].std())
            return self.df[(z_scores < threshold).all(axis=1)]

        elif method == 'iqr':
            Q1 = self.df[columns].quantile(0.25)
            Q3 = self.df[columns].quantile(0.75)
            IQR = Q3 - Q1
            return self.df[~((self.df[columns] < (Q1 - 1.5 * IQR)) | (self.df[columns] > (Q3 + 1.5 * IQR))).any(axis=1)]

        elif method == 'dbscan':
            dbscan = DBSCAN(eps=0.5, min_samples=5)
            labels = dbscan.fit_predict(self.df[columns])
            return self.df[labels != -1]

        else:
            raise ValueError(
                "Invalid method. Choose 'zscore', 'iqr', or 'dbscan'.")

    def binning(self, columns: list, bins: int = 5, labels: list = None) -> pd.DataFrame:
        """
        Bin continuous numerical variables into discrete intervals.

        Parameters:
        columns (list): List of column names to bin.
        bins (int): Number of bins to create.
        labels (list): Optional labels for the bins.

        Returns:
        pandas.DataFrame: DataFrame with binned columns.
        """
        self.handle_missing_values(columns)
        for col in columns:
            self.df[col] = pd.cut(self.df[col], bins=bins, labels=labels)
        return self.df


if __name__ == "__main__":

    df = pd.read_csv(
        '../data/airbnb/Airbnb Data/Listings.csv', encoding='ISO-8859-1')

    print(df.head())

    columns_to_process = [
        'accommodates', 'bedrooms', 'price', 'minimum_nights', 'maximum_nights',
        'review_scores_rating', 'review_scores_accuracy', 'review_scores_cleanliness',
        'review_scores_checkin', 'review_scores_communication', 'review_scores_location',
        'review_scores_value'
    ]

    toolkit = NumericalPreprocessingToolkit(df)

    df_missing_handled = toolkit.handle_missing_values(
        columns=columns_to_process, method='mean')
    print("DataFrame after Handling Missing Values:")
    print(df_missing_handled)

    df_standardized = toolkit.standardize(columns=columns_to_process)
    print("\nDataFrame after Standardization:")
    print(df_standardized)

    df_normalized = toolkit.normalize(
        columns=columns_to_process, feature_range=(0, 1))
    print("\nDataFrame after Normalization:")
    print(df_normalized)

    df_outliers_removed = toolkit.detect_and_remove_outliers(
        columns=columns_to_process, method='zscore', threshold=3.0)
    print("\nDataFrame after Outlier Detection and Removal (Z-score):")
    print(df_outliers_removed)

    df_binned = toolkit.binning(columns=['price'], bins=3)
    print("\nDataFrame after Binning 'price' column:")
    print(df_binned)
