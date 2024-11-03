import pandas as pd
from sklearn.preprocessing import LabelEncoder

class CategoricalUtilityToolkit:

    def __init__(self, df: pd.DataFrame):
        self.df = df

    def handle_missing(self, columns: list = None, fill_value: str = 'missing') -> pd.DataFrame:

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