import pandas as pd
from sklearn.preprocessing import LabelEncoder


class CategoricalUtilityToolkit:
    """
    A toolkit to handle and preprocess categorical variables in a pandas DataFrame.

    This toolkit includes methods for handling missing values, encoding categorical
    variables, and managing rare categories.
    """

    def __init__(self, df: pd.DataFrame):
        """
        Initializes the class with a DataFrame.

        Parameters:
            df (pd.DataFrame): Input DataFrame containing categorical variables.
        """
        self.df = df

    def handle_missing(self, columns: list = None, fill_value: str = 'missing') -> pd.DataFrame:
        """
        Handles missing values in the specified columns by filling with a given value.

        Parameters:
            columns (list): List of columns to handle. If None, it processes all columns.
            fill_value (str): The value to fill missing entries with. Default is 'missing'.

        Returns:
            pd.DataFrame: The DataFrame with missing values filled.
        """
        if columns is None:
            columns = self.df.select_dtypes(include=['object']).columns

        df_filled = self.df.copy()
        df_filled[columns] = df_filled[columns].fillna(fill_value)
        return df_filled

    def label_encode(self, columns: list) -> pd.DataFrame:
        df_encoded = self.df.copy()
        le = LabelEncoder()

        for col in columns:
            df_encoded[col] = le.fit_transform(df_encoded[col].astype(str))

        return df_encoded
