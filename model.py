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

def _preprocess_data(data):
    """Private helper function to preprocess data for model prediction.

    Parameters
    ----------
    data : str
        The data payload received within POST requests sent to our API.

    Returns
    -------
    pd.DataFrame
        The preprocessed data, ready to be used our model for prediction.
    """
    # Convert the json string to a python dictionary object
    feature_vector_dict = json.loads(data)
    # Load the dictionary as a Pandas DataFrame.
    feature_vector_df = pd.DataFrame.from_dict([feature_vector_dict])
    # Ensure numerical columns are present in the DataFrame
    missing_columns = set(numerical_columns) - set(feature_vector_df.columns)
    if missing_columns:
        raise KeyError(f"{missing_columns} not in index")

    # Specify numerical and categorical columns
    numerical_columns = ['Madrid_wind_speed', 'Valencia_wind_deg', 'Bilbao_rain_1h', 'Valencia_wind_speed',
                          'Seville_humidity', 'Madrid_humidity', 'Bilbao_clouds_all', 'Bilbao_wind_speed',
                          'Seville_clouds_all', 'Bilbao_wind_deg', 'Barcelona_wind_speed', 'Barcelona_wind_deg',
                          'Madrid_clouds_all', 'Seville_wind_speed', 'Barcelona_rain_1h', 'Seville_pressure',
                          'Seville_rain_1h', 'Bilbao_snow_3h', 'Barcelona_pressure', 'Seville_rain_3h',
                          'Madrid_rain_1h', 'Barcelona_rain_3h', 'Valencia_snow_3h', 'Madrid_weather_id',
                          'Barcelona_weather_id', 'Bilbao_pressure', 'Seville_weather_id', 'Valencia_pressure',
                          'Seville_temp_max', 'Madrid_pressure', 'Valencia_temp_max', 'Valencia_temp',
                          'Bilbao_weather_id', 'Seville_temp', 'Valencia_humidity', 'Valencia_temp_min',
                          'Barcelona_temp_max', 'Madrid_temp_max', 'Barcelona_temp', 'Bilbao_temp_min',
                          'Bilbao_temp', 'Barcelona_temp_min', 'Bilbao_temp_max', 'Seville_temp_min',
                          'Madrid_temp', 'Madrid_temp_min','Day', 'Month' ,'Year', 'Hour','Start_minute','start_seconds',
                          'Start_weekend','Start_week_of_year','winter','spring','summer','autumn' ]

    categorical_columns = ['Madrid_weather_id', 'Barcelona_weather_id', 'Bilbao_weather_id', 'Seville_weather_id']

    # Ensure numerical columns are treated as numeric
    feature_vector_df[numerical_columns] = feature_vector_df[numerical_columns].apply(pd.to_numeric, errors='coerce')

    # Impute missing values in numerical columns with mean
    feature_vector_df[numerical_columns] = feature_vector_df[numerical_columns].fillna(feature_vector_df[numerical_columns].mean())

    # Handle categorical columns (you may need to encode them if required)
    feature_vector_df[categorical_columns] = feature_vector_df[categorical_columns].astype('category')

    # Add any additional preprocessing steps as needed

    return feature_vector_df
    # ------------------------------------------------------------------------
def load_model(path_to_model:str):
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


""" You may use this section (above the make_prediction function) of the python script to implement 
    any auxiliary functions required to process your model's artifacts.
"""

def make_prediction(data, model):
    """Prepare request data for model prediction.

    Parameters
    ----------
    data : str
        The data payload received within POST requests sent to our API.
    model : <class: sklearn.estimator>
        An sklearn model object.

    Returns
    -------
    list
        A 1-D python list containing the model prediction.

    """
    # Data preprocessing.
    prep_data = _preprocess_data(data)
    # Perform prediction with model and preprocessed data.
    prediction = model.predict(prep_data)
    # Format as list for output standardisation.
    return prediction[0].tolist()
