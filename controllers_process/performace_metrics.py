from session_state import get_session_variable
import numpy as np

def integrated_absolute_error()->float:
    
    sampling_time = get_session_variable('sampling_time') 
    reference_input = get_session_variable('reference_input') 
    process_output_sensor_dict = get_session_variable('process_output_sensor').values() 
    process_output_sensor = np.array(list(process_output_sensor_dict))
    process_output_sensor_length = len(process_output_sensor)
    # Calculate absolute difference element-wise
    absolute_error = np.abs(reference_input[:process_output_sensor_length] - process_output_sensor)
    sum_absolute_error = np.sum(absolute_error)*sampling_time
    return sum_absolute_error
    

def total_variation_control(control_signal:str)->float:
    
   
    control_signal_dict  = get_session_variable(control_signal).values()
    control_signal_values = np.array(list(control_signal_dict))
    control_variation = np.abs(np.diff(control_signal_values))
    total_control_variation = np.sum(control_variation)
    return total_control_variation