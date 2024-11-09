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

    def label_encode(self, columns: list) -> pd.DataFrame:

        df_encoded = self.df.copy()
        le = LabelEncoder()

        for col in columns:
            df_encoded[col] = le.fit_transform(df_encoded[col].astype(str))

        return df_encoded

    def one_hot_encode(self, columns: list) -> pd.DataFrame:

        df_encoded = pd.get_dummies(self.df, columns=columns, drop_first=True)
        return df_encoded

    def handle_rare_categories(self, columns: list, threshold: float = 0.05) -> pd.DataFrame:

        df_processed = self.df.copy()

        for col in columns:
            freq = df_processed[col].value_counts(normalize=True)
            rare_labels = freq[freq < threshold].index
            df_processed[col] = df_processed[col].apply(
                lambda x: 'Rare' if x in rare_labels else x)

        return df_processed

