from typing import Callable, List, Tuple
import pandas as pd
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.tag import pos_tag
from nltk.chunk import ne_chunk
import string

# Download necessary NLTK data
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('maxent_ne_chunker')
nltk.download('words')
nltk.download('wordnet')
nltk.download('vader_lexicon')
nltk.download('maxent_ne_chunker_tab')


class NLPUtilityToolkit:
    def __init__(self) -> None:
        """
        Initializes the NLPUtilityToolkit with necessary components for NLP tasks.
        """
        self.stemmer = PorterStemmer()
        self.lemmatizer = WordNetLemmatizer()
        self.sia = SentimentIntensityAnalyzer()

    def _apply_to_column(self, df: pd.DataFrame, column: str, func: Callable[[str], any]) -> pd.DataFrame:
        """
        Applies a given function to a specified column in the DataFrame.

        Args:
            df (pd.DataFrame): The DataFrame containing the column to be processed.
            column (str): The name of the column to be processed.
            func (Callable[[str], any]): The function to apply to each value in the column.

        Returns:
            pd.DataFrame: The DataFrame with a new column containing the processed values.
        """
        df[f'{func.__name__}_{column}'] = df[column].apply(
            lambda x: func(x) if isinstance(x, str) else [])
        return df

    def tokenize(self, df: pd.DataFrame, column: str) -> pd.DataFrame:
        """
        Tokenizes the text in the specified column.

        Args:
            df (pd.DataFrame): The DataFrame containing the column to be tokenized.
            column (str): The name of the column to be tokenized.

        Returns:
            pd.DataFrame: The DataFrame with a new column containing the tokenized text.
        """
        return self._apply_to_column(df, column, word_tokenize)

    def pos_tagging(self, df: pd.DataFrame, column: str) -> pd.DataFrame:
        """
        Performs Part-of-Speech (POS) tagging and Named Entity Recognition (NER) on the specified column.

        Args:
            df (pd.DataFrame): The DataFrame containing the column to be processed.
            column (str): The name of the column to be processed.

        Returns:
            pd.DataFrame: The DataFrame with a new column containing POS tags and named entities.
        """
        def get_named_entities(text: str) -> List[Tuple[str, str]]:
            tokens = word_tokenize(text)
            pos_tags = pos_tag(tokens)
            named_entities = ne_chunk(pos_tags)
            return [(ne.label(), ' '.join(word for word, tag in ne.leaves()))
                    for ne in named_entities if isinstance(ne, nltk.Tree)]

        return self._apply_to_column(df, column, get_named_entities)

    def stemming(self, df: pd.DataFrame, column: str) -> pd.DataFrame:
        """
        Applies stemming to the text in the specified column.

        Args:
            df (pd.DataFrame): The DataFrame containing the column to be stemmed.
            column (str): The name of the column to be stemmed.

        Returns:
            pd.DataFrame: The DataFrame with a new column containing the stemmed text.
        """
        return self._apply_to_column(df, column, lambda x: [self.stemmer.stem(token) for token in word_tokenize(x)])

    def lemmatization(self, df: pd.DataFrame, column: str) -> pd.DataFrame:
        """
        Applies lemmatization to the text in the specified column.

        Args:
            df (pd.DataFrame): The DataFrame containing the column to be lemmatized.
            column (str): The name of the column to be lemmatized.

        Returns:
            pd.DataFrame: The DataFrame with a new column containing the lemmatized text.
        """
        return self._apply_to_column(df, column, lambda x: [self.lemmatizer.lemmatize(token) for token in word_tokenize(x)])

    def parse(self, df: pd.DataFrame, column: str) -> pd.DataFrame:
        """
        Parses the text in the specified column and returns tokens with their POS tags.

        Args:
            df (pd.DataFrame): The DataFrame containing the column to be parsed.
            column (str): The name of the column to be parsed.

        Returns:
            pd.DataFrame: The DataFrame with a new column containing the parsed text (tokens and POS tags).
        """
        def simple_parse(text: str) -> List[dict]:
            tokens = word_tokenize(text)
            pos_tags = pos_tag(tokens)
            return [{'token': token, 'pos': tag} for token, tag in pos_tags]

        return self._apply_to_column(df, column, simple_parse)

    def sentiment_analysis(self, df: pd.DataFrame, column: str) -> pd.DataFrame:
        """
        Performs sentiment analysis on the text in the specified column.

        Args:
            df (pd.DataFrame): The DataFrame containing the column to be analyzed.
            column (str): The name of the column to be analyzed.

        Returns:
            pd.DataFrame: The DataFrame with a new column containing sentiment scores.
        """
        def get_sentiment(text: str) -> float:
            scores = self.sia.polarity_scores(text)
            return scores['compound']

        df[f'sentiment_{column}'] = df[column].apply(
            lambda x: get_sentiment(x) if isinstance(x, str) else None)
        return df

    def preprocess_text(self, text: str) -> str:
        """
        Preprocesses text by converting to lowercase, removing punctuation, and removing stopwords.

        Args:
            text (str): The text to preprocess.

        Returns:
            str: The preprocessed text.
        """
        # Convert to lowercase
        text = text.lower()
        # Remove punctuation
        text = text.translate(str.maketrans('', '', string.punctuation))
        # Tokenize
        tokens = word_tokenize(text)
        # Remove stopwords
        stopwords = set(nltk.corpus.stopwords.words('english'))
        tokens = [token for token in tokens if token not in stopwords]
        # Join tokens back into a string
        return ' '.join(tokens)

    def preprocess_column(self, df: pd.DataFrame, column: str) -> pd.DataFrame:
        """
        Applies text preprocessing to the specified column.

        Args:
            df (pd.DataFrame): The DataFrame containing the column to be processed.
            column (str): The name of the column to be preprocessed.

        Returns:
            pd.DataFrame: The DataFrame with a new column containing the preprocessed text.
        """
        return self._apply_to_column(df, column, self.preprocess_text)


def main():
    # Create a sample DataFrame
    data = {
        'id': [1, 2, 3],
        'text': [
            "The quick brown fox jumps over the lazy dog.",
            "Natural language processing is fascinating!",
            "Python is a versatile programming language for data science."
        ]
    }
    df = pd.DataFrame(data)

    # Initialize the NLP Utility Toolkit
    nlp_toolkit = NLPUtilityToolkit()

    # Apply various NLP operations
    df = nlp_toolkit.tokenize(df, 'text')
    df = nlp_toolkit.pos_tagging(df, 'text')
    df = nlp_toolkit.stemming(df, 'text')
    df = nlp_toolkit.lemmatization(df, 'text')
    df = nlp_toolkit.parse(df, 'text')
    df = nlp_toolkit.sentiment_analysis(df, 'text')
    df = nlp_toolkit.preprocess_column(df, 'text')

    # Display results
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    pd.set_option('display.max_colwidth', None)
    df.to_excel('some.xlsx')


if __name__ == "__main__":
    main()
