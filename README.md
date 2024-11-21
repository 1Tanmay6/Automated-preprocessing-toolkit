# Modular Preprocessing and EDA Toolkit

This project is a modular toolkit for automated preprocessing of numerical, categorical, and NLP data, with a dedicated EDA module for generating visualizations. Designed to streamline the preprocessing pipeline, this toolkit enables efficient data cleaning, transformation, and feature extraction while offering optional EDA for visual insights.

## Table of Contents

- [Features](#features)
- [Project Structure](#project-structure)
- [Setup](#setup)
- [Usage](#usage)
  - [Numerical Preprocessing](#numerical-preprocessing)
  - [Categorical Preprocessing](#categorical-preprocessing)
  - [NLP Preprocessing](#nlp-preprocessing)
  - [EDA and Visualization](#eda-and-visualization)
- [Contributing](#contributing)
- [License](#license)

## Features

- **Automated Preprocessing**: For structured (numerical, categorical) and unstructured (NLP) data.
- **Data Cleaning and Transformation**: Simplifies handling of missing values, scaling, encoding, text cleaning, and vectorization.
- **EDA through Visualizations**: Generate exploratory visualizations for insights on data distribution, correlations, categorical summaries and much more.

## Project Structure

The repository is structured into individual modules, each specializing in a different data preprocessing task. Only the `visualizer_utility_toolkit.py` module is specific to EDA.

```
root(/)
├── preprocessing_utilites
    ├── numerical_preprocessor_toolkit.py
    ├── categorical_preprocessor_toolkit.py
    ├── global_utlility.py 
    ├── nlp_preprocessor_toolkit.py 
    └──visualizer_utility_toolkit.py 
└── example_notebook.ipynb
```

Each module can be used independently or in combination with others, allowing flexible integration based on your specific data requirements.

## Setup

1. **Clone the repository**:
   ```bash
   git clone https://github.com/1Tanmay6/Automated-preprocessing-toolkit.git
   cd Automated-preprocessing-toolkit
   ```

2. **Install dependencies**:
   Install the necessary dependencies listed in `requirements.txt`.
   ```bash
   pip install -r requirements.txt
   ```

3. **Launch Jupyter Notebook** (optional):
   For an interactive example, open the `example_notebook.ipynb`.
   ```bash
   jupyter notebook example_notebook.ipynb
   ```

## Usage

Below is a quick overview of each module's functionality. Please refer to the `example_notebook.ipynb` for a more detailed guide.

### Numerical Preprocessing

The `numerical_preprocessor_toolkit.py` module provides automated preprocessing for numerical data, including handling missing values, scaling, and remove outliers.

```python
from preprocessing_utilites import NumericalPreprocessingToolkit

# Initialize with numerical data
num_prep = NumericalPreprocessingToolkit(data=numeric_data)

# Handle missing values
num_prep.handle_missing_values(columns: list, method: str)

# Scale data
num_prep.standardize(columns: list)

# Remove Outliers
num_prep.detect_and_remove_outliers(columns: list, method: str = 'zscore', threshold: float = 3.0)
```

### Categorical Preprocessing

The `categorical_preprocessor_toolkit.py` module handles preprocessing for categorical data, including encoding and missing value treatment.

```python
from preprocessing_utilites import CategoricalUtilityToolkit

# Initialize with categorical data
cat_prep = CategoricalUtilityToolkit(data=categorical_data)

# Encode categorical features
cat_prep.label_encode(columns: list)

# Handle missing values
cat_prep.handle_missing(columns: list = None, fill_value: str = 'missing')

# Handle rare categories of categorical data
cat_prep.handle_rare_categories(columns: list, threshold: float = 0.05)
```

### NLP Preprocessing

The `nlp_preprocessor_toolkit.py` module automates preprocessing tasks for text data, such as cleaning, tokenization, and POS Tagging.

```python
from preprocessing_utilites import NLPUtilityToolkit

# Initialize with text data
nlp_prep = NLPUtilityToolkit(text_data=text_data)

# Preprocess text data
nlp_prep.preprocess_text(text: str)

# Tokenize text
nlp_prep.tokenize(df: pd.DataFrame, column: str)

# POS Tagging text data
nlp_prep.pos_tagging(df: pd.DataFrame, column: str)
```

### EDA and Visualization

The `visualizer_utility_toolkit.py` module provides tools for exploratory data analysis through visualizations. This module generates graphs to help you understand patterns and relationships in your data, and let's you save the results as well.

```python
from preprocessing_utilites import VisualizerUtilityToolkit

# Initialize with parameters for EDA
viz = VisualizerUtilityToolkit(save_dir: str = 'plots', display: bool = False)

# Plot single distribution of numerical data
viz.line_plot(x: Union[List, np.ndarray], y: Union[List, np.ndarray],
                  x_label: str = 'X', y_label: str = 'Y', title: str = 'Line Plot')

# Plot Multiple type of graphs
viz.generate_all_plots(x: Union[List, np.ndarray], y: Union[List, np.ndarray],
                           x_label: str = 'X', y_label: str = 'Y')

# Visualize categorical data as well with various options (one given below, check out file for more)
viz.bar_plot(x: Union[List, np.ndarray], y: Union[List, np.ndarray],
                 x_label: str = 'X', y_label: str = 'Y', title: str = 'Bar Plot')
```

## Contributing

Contributions are welcome! Please check the [CONTRIBUTING.md](CONTRIBUTING.md) file for details on how to get involved. Suggestions for improvements, new features, or bug reports can be submitted via issues or pull requests.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more information.