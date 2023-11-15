"""

    Helper functions for the pretrained model to be used within our API.

    Author: Explore Data Science Academy.

    Note:
    ---------------------------------------------------------------------
    Please follow the instructions provided within the README.md file
    located within this directory for guidance on how to use this script
    correctly.

    Importantly, you will need to modify this file by adding
    your own data preprocessing steps within the `_preprocess_data()`
    function.
    ----------------------------------------------------------------------

    Description: This file contains several functions used to abstract aspects
    of model interaction within the API. This includes loading a model from
    file, data preprocessing, and model prediction.  

"""

# Helper Dependencies
import numpy as np
import pandas as pd
import pickle
import json
from sklearn.impute import SimpleImputer
from sklearn.feature_extraction import FeatureHasher
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.preprocessing import StandardScaler

def _preprocess_data(data, df, categorical_columns, numerical_columns):
    """Private helper function to preprocess data for model prediction.

    Parameters
    ----------
    data : str
        The data payload received within POST requests sent to our API.
    df : pd.DataFrame
        The dataset for preprocessing.
    categorical_columns : list
        List of column names that are categorical.
    numerical_columns : list
        List of column names that are numerical.

    Returns
    -------
    pd.DataFrame
        The preprocessed data, ready to be used by our model for prediction.
    """
    # Convert the json string to a python dictionary object
    feature_vector_dict = json.loads(data)
    # Load the dictionary as a Pandas DataFrame.
    feature_vector_df = pd.DataFrame.from_dict([feature_vector_dict])

    # Impute missing values.
    imputer = SimpleImputer(strategy='mean')
    df[numerical_columns] = imputer.fit_transform(df[numerical_columns])

    # Hash categorical columns.
    hasher = FeatureHasher()
    categorical_data = hasher.fit_transform(df[categorical_columns])
    numerical_data = df[numerical_columns]

    # Combine the categorical and numerical data.
    df = pd.concat([numerical_data, pd.DataFrame(categorical_data.toarray())], axis=1)

    # Standardize numerical columns.
    scaler = StandardScaler()
    df[numerical_columns] = scaler.fit_transform(df[numerical_columns])

    # Select features using chi-squared selection.
    selector = SelectKBest(chi2, k=10)
    selector.fit(df, df['target'])
    selected_features = selector.get_support(indices=True)

    # Select the selected features.
    df = df.iloc[:, selected_features]

    return df

def load_model(path_to_model: str):
    """Adapter function to load our pretrained model into memory.

    Parameters
    ----------
    path_to_model : str
        The relative path to the model weights/schema to load.
        Note that unless another file format is used, this needs to be a
        .pkl file.

    Returns
    -------
    <class: sklearn.estimator>
        The pretrained model loaded into memory.
    """
    return pickle.load(open(path_to_model, 'rb'))

def make_prediction(data, model, df, categorical_columns, numerical_columns):
    """Prepare request data for model prediction.

    Parameters
    ----------
    data : str
        The data payload received within POST requests sent to our API.
    model : <class: sklearn.estimator>
        An sklearn model object.
    df : pd.DataFrame
        The dataset for preprocessing.
    categorical_columns : list
        List of column names that are categorical.
    numerical_columns : list
        List of column names that are numerical.

    Returns
    -------
    list
        A 1-D python list containing the model prediction.
    """
    # Data preprocessing.
    prep_data = _preprocess_data(data, df, categorical_columns, numerical_columns)
    # Perform prediction with model and preprocessed data.
    prediction = model.predict(prep_data)
    # Format as list for output standardization.
    return prediction[0].tolist()
