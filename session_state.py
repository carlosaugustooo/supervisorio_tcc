import streamlit as st


def loadSessionStates():
    if 'connected' not in st.session_state:
        st.session_state.connected = {}

    if 'sensor' not in st.session_state:
        st.session_state.sensor = {}

    if 'controller_parameters' not in st.session_state:
        st.session_state.controller_parameters = {}

    if 'sampling_time' not in st.session_state.controller_parameters:
        st.session_state.controller_parameters['sampling_time'] = 0.1

    if 'samples_number' not in st.session_state.controller_parameters:
        st.session_state.controller_parameters['samples_number'] = 100

    if 'control_signal_1' not in st.session_state.controller_parameters:
        st.session_state.controller_parameters['control_signal_1'] = {}

    if 'control_signal_2' not in st.session_state.controller_parameters:
        st.session_state.controller_parameters['control_signal_2'] = {}
        
    if 'reference_input' not in st.session_state.controller_parameters:
        st.session_state.controller_parameters['reference_input'] = {}
        
    if 'saturation_max_value' not in st.session_state.controller_parameters:
        st.session_state.controller_parameters['saturation_max_value'] = {}
        
    if 'saturation_min_value' not in st.session_state.controller_parameters:
        st.session_state.controller_parameters['saturation_min_value'] = {}
        
    if 'process_output_sensor' not in st.session_state.controller_parameters:
        st.session_state.controller_parameters['process_output_sensor'] = {}
    
    if 'iae_metric' not in st.session_state.controller_parameters:
        st.session_state.controller_parameters['iae_metric'] = 0
    
    if 'tvc_1_metric' not in st.session_state.controller_parameters:
        st.session_state.controller_parameters['tvc_1_metric'] = 0
        
    if 'tvc_2_metric' not in st.session_state.controller_parameters:
        st.session_state.controller_parameters['tvc_2_metric'] = 0
    

session_list = [
        "samples_number",
        "sampling_time",
        "control_signal_1",
        "control_signal_2",
        "reference_input",
        "saturation_max_value",
        "saturation_min_value",
        "process_output_sensor",
        "iae_metric",
        "tvc_1_metric",
        "tvc_2_metric"
        ]

def get_session_variable(variable:str)-> dict|float:
    """
    Function to get the session states variables

    Parameters:
    variable (str):
        "samples_number",
        "sampling_time",
        "control_signal_1",
        "control_signal_2",
        "reference_input",
        "saturation_max_value",
        "saturation_min_value",
        "process_output_sensor"
    Returns:
    dict
    """
    
    session_variable = {

        "samples_number":           st.session_state.controller_parameters['samples_number'],
        "sampling_time":            st.session_state.controller_parameters['sampling_time'],
        "control_signal_1":         st.session_state.controller_parameters['control_signal_1'],
        "control_signal_2":         st.session_state.controller_parameters['control_signal_2'],
        "reference_input":          st.session_state.controller_parameters['reference_input'],
        "saturation_max_value":     st.session_state.controller_parameters['saturation_max_value'],
        "saturation_min_value":     st.session_state.controller_parameters['saturation_min_value'],
        "process_output_sensor":    st.session_state.controller_parameters['process_output_sensor'],
        "iae_metric":               st.session_state.controller_parameters['iae_metric'],
        "tvc_1_metric":             st.session_state.controller_parameters['tvc_1_metric'],
        "tvc_2_metric":             st.session_state.controller_parameters['tvc_2_metric'],
    }
    
    return session_variable[variable]

def set_session_controller_parameter(controller_parameter:str, new_data) ->None:
    """
    Function to get the session states controller parameters variables

    Parameters:
    variable (str):
        "samples_number",
        "sampling_time",
        "control_signal_1",
        "control_signal_2",
        "reference_input",
        "saturation_max_value",
        "saturation_min_value",
        "process_output_sensor"
    Returns:
    None
    """
    st.session_state.controller_parameters[controller_parameter] = new_data
   