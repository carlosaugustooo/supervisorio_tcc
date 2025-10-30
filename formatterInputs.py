from datetime import datetime, timedelta
import streamlit as st
import pandas as pd
from connections import *
from session_state import get_session_variable

def keys2DatetimeObj(datetime_dict):

    datetimeList = list(datetime_dict.keys())

    # Define the format of your date string
    date_format = "%Y-%m-%d %H:%M:%S.%f"

    # Parse the string into a datetime object
    date_object = [datetime.strptime(datetimeElement, date_format)
                   for datetimeElement in datetimeList]
    return date_object


def datetime_obj_to_elapsed_time(variable:str)->dict:

    sensor_data_dict = get_session_variable(variable)
  
    if not sensor_data_dict:
        return None
    
    date_object = keys2DatetimeObj(sensor_data_dict)

    time_interval = [0] + [(date_object[i] - date_object[0]).total_seconds()
                           for i in range(1, len(date_object))]

    # sensor_formatted2Hours_dict = {
    #     key.split()[1]: value
    #     for key, value in sensor_dict.items()
    # }

    elapsed_time_in_sec = {time_interval[i]: sensor_data_dict[key]
                           for i, key in enumerate(sensor_data_dict)}
    

    return elapsed_time_in_sec

def dictionary_to_pandasDataframe(variable:dict,variable_name_column:str)->pd.DataFrame:

    # Convert the inner dictionary to a list of dictionaries
    data_list = [{"Time (s)": float(timestamp), f"{variable_name_column}": float(value)} for timestamp, value in variable.items()]

    # Create a DataFrame from the list of dictionaries
    variable_dataframe = pd.DataFrame(data_list)
    return variable_dataframe

def insertReferenceInDataframe(variable_dataframe:pd.DataFrame,reference_col:list)->pd.DataFrame:
    # Get the length of the DataFrame
    df_length = len(variable_dataframe)
    if not reference_col:
        reference_col = [0]*df_length
        
    # Convert the inner dictionary to a list of dictionaries
    variable_dataframe['Reference'] = reference_col[:df_length]

    return variable_dataframe

def dataframeToPlot(variable_dict:str,variable_name_to_plot:str, second_variable:str) -> pd.DataFrame:
    variable_with_time = datetime_obj_to_elapsed_time(variable_dict)
    if not datetime_obj_to_elapsed_time(variable_dict):
        return None
    
    process_dictionary = dictionary_to_pandasDataframe(variable_with_time,variable_name_to_plot)
    return insertReferenceInDataframe(process_dictionary,get_session_variable(second_variable))


def calculate_time_limit()->float:
    # Receber os valores de tempo de amostragem e número de amostras da sessão
    sampling_time = get_session_variable('sampling_time')
    samples_number = get_session_variable('samples_number')
    time_max_value = samples_number*sampling_time
    return time_max_value


def get_sample_position(sampling_time:float, samples_number:int, time_instant:float) -> int:
    """
    Calculate the position (index) of a given time instant in the sampled data.

    Parameters:
    - sampling_time: The time between each sample.
    - samples_number: The total number of samples.
    - time_instant: The time instant for which you want to find the sample position.

    Returns:
    - The position (index) of the sample corresponding to the given time instant.
    """
    if time_instant < 0:
        return 0  # Handle negative time instant

    # Calculate the position based on the time instant and sampling time
    position = int(time_instant / sampling_time)

    # Ensure the position is within the valid range [0, samples_number-1]
    position = max(0, min(position, samples_number - 1))

    return position