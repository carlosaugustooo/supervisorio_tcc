import streamlit as st
from formatterInputs import *
from numpy import exp, ones, zeros,array, dot, convolve
from connections import *
import datetime
from session_state import get_session_variable
from controllers_process.validations_functions import *


def imcControlProcessSISO(transfer_function_type:str,num_coeff:str,den_coeff:str,
                          imc_mr_tau_mf1:float, 
                          imc_multiple_reference1:float, imc_multiple_reference2:float, imc_multiple_reference3:float,
                          change_ref_instant2 = 1, change_ref_instant3 = 1):
    
    if num_coeff == '':
        return st.error('Coeficientes incorretos no Numerador.')
    
    if den_coeff =='':
        return st.error('Coeficientes incorretos no Denominador.')

    # Receber os valores de tempo de amostragem e número de amostras da sessão
    sampling_time = get_session_variable('sampling_time')
    samples_number = get_session_variable('samples_number')

    # IMC Controller Project

    # Initial Conditions
    process_output = zeros(samples_number)
    model_output_1 = zeros(samples_number)
    erro1 = zeros(samples_number)
    output_model_comparation = zeros(samples_number)

    # Take the index of time to change the referencee
    instant_sample_2 = get_sample_position(sampling_time, samples_number, change_ref_instant2)
    instant_sample_3 = get_sample_position(sampling_time, samples_number, change_ref_instant3)

    reference_input = imc_multiple_reference1*ones(samples_number)
    reference_input[instant_sample_2:instant_sample_3] = imc_multiple_reference2
    reference_input[instant_sample_3:] = imc_multiple_reference3
    
    st.session_state.controller_parameters['reference_input'] = reference_input.tolist()

    # Power Saturation
    max_pot = get_session_variable('saturation_max_value')
    min_pot = get_session_variable('saturation_min_value')

    # Manipulated variable
    manipulated_variable_1 = zeros(samples_number)
    motors_power_packet = "0"

    # Model transfer Function
    A_coeff, B_coeff = convert_tf_2_discrete(num_coeff,den_coeff,transfer_function_type)
    
    # print(A_coeff)
    # print(B_coeff)
    A_order = len(A_coeff)-1
    B_order = len(B_coeff) # Zero holder aumenta um grau
    
    # Close Loop Tau Calculation
    # tau_mf1 = ajuste1*tausmith1
    tau_mf1 = imc_mr_tau_mf1
    alpha1 = exp(-sampling_time/tau_mf1)
    
    # Perform polynomial multiplication using np.convolve
    alpha_delta = [1,-alpha1]
    B_delta = convolve(B_coeff,alpha_delta)
    
    
    # Receive the Arduino object from the session
    arduinoData = st.session_state.connected['arduinoData']

    # clear previous control signal values
    st.session_state.controller_parameters['control_signal_1']= dict()
    control_signal_1 = st.session_state.controller_parameters['control_signal_1']
    
    # clear previous control signal values
    st.session_state.controller_parameters['process_output_sensor'] = dict()
    process_output_sensor = st.session_state.controller_parameters['process_output_sensor']

    # inicializar  o timer
    start_time = time.time()
    kk = 0

    # Inicializar a barra de progresso
    progress_text = "Operation in progress. Please wait."
    my_bar = st.progress(0, text=progress_text)
    
    # receive the first mesure 
    sendToArduino(arduinoData, "0")
    

    while kk < samples_number:
        current_time = time.time()
        if current_time - start_time > sampling_time:
            start_time = current_time
            
            # -----  Angle Sensor Output
            # print(f'kk = {kk}')
            process_output[kk] = readFromArduino(arduinoData)

            
            if kk <= A_order:
                # Store the output process values and control signal
                sendToArduino(arduinoData, '0')
          
            # ---- Motor Model Output
            elif kk == 1 and A_order == 1:
                model_output_1[kk] = dot(-A_coeff[1:], model_output_1[kk-1::-1])\
                                        + dot(B_coeff, manipulated_variable_1[kk-1::-1])
                # Determine uncertainty
                output_model_comparation[kk] = process_output[kk] - model_output_1[kk]

                # Determine Error
                erro1[kk] = reference_input[kk] - output_model_comparation[kk]

                # Control Signal
                manipulated_variable_1[kk] = dot(-B_delta[1:],manipulated_variable_1[kk-1::-1])+ (1-alpha1)*dot(A_coeff,erro1[kk::-1])
                manipulated_variable_1[kk] /= B_delta[0]
                
            elif kk >A_order:
                
                # print(f'kk == {kk}')
                # print(f'model_output_1: {model_output_1[kk-1:kk-A_order-1:-1]}')
                # print(f'manipulated_variable_1: {manipulated_variable_1[kk-1:kk-B_order-1:-1]}')
                model_output_1[kk] = dot(-A_coeff[1:], model_output_1[kk-1:kk-A_order-1:-1])\
                                        + dot(B_coeff, manipulated_variable_1[kk-1:kk-B_order-1:-1])
                

                # Determine uncertainty
                output_model_comparation[kk] = process_output[kk] - model_output_1[kk]

                # Determine Error
                erro1[kk] = reference_input[kk] - output_model_comparation[kk]

                # Control Signal
                manipulated_variable_1[kk] = dot(-B_delta[1:],manipulated_variable_1[kk-1:kk-B_order-1:-1])+ (1-alpha1)*dot(A_coeff,erro1[kk:kk-A_order-1:-1])
                manipulated_variable_1[kk] /= B_delta[0]
            
            
            # Control Signal Saturation
            manipulated_variable_1[kk] = max(min_pot, min(manipulated_variable_1[kk], max_pot))

            # Motor Power String Formatation
            motors_power_packet = f"{manipulated_variable_1[kk]}\r"
            sendToArduino(arduinoData, motors_power_packet)
                
            # Store the output process values and control signal
            current_timestamp = datetime.now()
            process_output_sensor[str(current_timestamp)] = float(process_output[kk])
            control_signal_1[str(current_timestamp)] = float(manipulated_variable_1[kk])
            kk += 1

            percent_complete = kk / (samples_number)
            my_bar.progress(percent_complete, text=progress_text)

    # Turn off the motor
    sendToArduino(arduinoData, '0')


def imcControlProcessTISO(transfer_function_type:str,num_coeff_1:str,den_coeff_1:str, num_coeff_2:str,den_coeff_2:str,
                          imc_mr_tau_mf1:float, imc_mr_tau_mf2:float,
                          imc_multiple_reference1:float, imc_multiple_reference2:float, imc_multiple_reference3:float,
                          change_ref_instant2 = 1, change_ref_instant3 = 1):

    if num_coeff_1 == '':
        return st.error('Coeficientes incorretos no Numerador 1.')
    
    if den_coeff_1 =='':
        return st.error('Coeficientes incorretos no Denominador 1.')
    
    if num_coeff_2 == '':
        return st.error('Coeficientes incorretos no Numerador 2.')
    
    if den_coeff_2 =='':
        return st.error('Coeficientes incorretos no Denominador 2.')

    # Receber os valores de tempo de amostragem e número de amostras da sessão
    sampling_time = get_session_variable('sampling_time')
    samples_number = get_session_variable('samples_number')

    # IMC Controller Project

    # Initial Conditions
    process_output = zeros(samples_number)
    model_output_1 = zeros(samples_number)
    model_output_2 = zeros(samples_number)
    erro1 = zeros(samples_number)
    erro2 = zeros(samples_number)
    output_model_comparation_1 = zeros(samples_number)
    output_model_comparation_2 = zeros(samples_number)

    # Take the index of time to change the referencee
    instant_sample_2 = get_sample_position(sampling_time, samples_number, change_ref_instant2)
    instant_sample_3 = get_sample_position(sampling_time, samples_number, change_ref_instant3)

    reference_input = imc_multiple_reference1*ones(samples_number)
    reference_input[instant_sample_2:instant_sample_3] = imc_multiple_reference2
    reference_input[instant_sample_3:] = imc_multiple_reference3
    st.session_state.controller_parameters['reference_input'] = reference_input.tolist()


    # Power Saturation
    max_pot = get_session_variable('saturation_max_value')
    min_pot = get_session_variable('saturation_min_value')


    # Manipulated variable
    manipulated_variable_1 = zeros(samples_number)
    manipulated_variable_2 = zeros(samples_number)
    motors_power_packet = "0,0"

    # Model transfer Function 1
    A_coeff_1, B_coeff_1 = convert_tf_2_discrete(num_coeff_1,den_coeff_1,transfer_function_type)
    
    # print(A_coeff)
    # print(B_coeff)
    A_order = len(A_coeff_1)-1
    B_order = len(B_coeff_1) # Zero holder aumenta um grau

    ## Model transfer Function 2
    A_coeff_2, B_coeff_2 = convert_tf_2_discrete(num_coeff_2,den_coeff_2,transfer_function_type)
    
    # print(A_coeff)
    # print(B_coeff)
    
    # Close Loop Tau Calculation
    tau_mf1 = imc_mr_tau_mf1
    alpha1 = exp(-sampling_time/tau_mf1)

    tau_mf2 = imc_mr_tau_mf2
    alpha2 = exp(-sampling_time/tau_mf2)
    
    # Perform polynomial multiplication using np.convolve
    alpha_delta_1 = [1,-alpha1]
    B_delta_1 = convolve(B_coeff_1,alpha_delta_1)
    
    # Perform polynomial multiplication using np.convolve
    alpha_delta_2 = [1,-alpha2]
    B_delta_2 = convolve(B_coeff_2,alpha_delta_2)
    

    # Receber o objeto arduino da sessão
    arduinoData = st.session_state.connected['arduinoData']

    # limpar os valores anteriores do sensor
    st.session_state.sensor = dict()
    sensor = st.session_state.sensor

    
    # clear previous control signal values
    st.session_state.controller_parameters['control_signal_1']= dict()
    control_signal_1 = st.session_state.controller_parameters['control_signal_1']
    
    
    # clear previous control signal values
    st.session_state.controller_parameters['control_signal_2']= dict()
    control_signal_2 = st.session_state.controller_parameters['control_signal_2']
    
    # clear previous control signal values
    st.session_state.controller_parameters['process_output_sensor'] = dict()
    process_output_sensor = st.session_state.controller_parameters['process_output_sensor']
    
    
    # inicializar  o timer
    start_time = time.time()
    kk = 0

    # Inicializar a barra de progresso
    progress_text = "Operation in progress. Please wait."
    my_bar = st.progress(0, text=progress_text)
    
    # receive the first mesure 
    sendToArduino(arduinoData, '0,0')

    while kk < samples_number:
        current_time = time.time()
        if current_time - start_time > sampling_time:
            start_time = current_time
            
            # -----  Angle Sensor Output
            process_output[kk] = readFromArduino(arduinoData)
            
            if kk <= A_order:
                              
                sendToArduino(arduinoData, '0,0')
                
            # ---- Motor Model Output
            elif kk == 1 and A_order == 1:
                model_output_1[kk] = dot(-A_coeff_1[1:], model_output_1[kk-1::-1]) + dot(B_coeff_1, manipulated_variable_1[kk-1::-1])
                model_output_2[kk] = dot(-A_coeff_2[1:], model_output_2[kk-1::-1]) - dot(B_coeff_2, manipulated_variable_2[kk-1::-1])
                
                # Determine uncertainty
                output_model_comparation_1[kk] = process_output[kk] - model_output_1[kk]
                output_model_comparation_2[kk] = -(process_output[kk] - model_output_2[kk])

                # Determine Error
                erro1[kk] = reference_input[kk] - output_model_comparation_1[kk]
                erro2[kk] = -(reference_input[kk] + output_model_comparation_2[kk])

                # Control Signal
                manipulated_variable_1[kk] = dot(-B_delta_1[1:],manipulated_variable_1[kk-1::-1]) + (1-alpha1)*dot(A_coeff_1,erro1[kk::-1])
                manipulated_variable_1[kk] /= B_delta_1[0]
                
                manipulated_variable_2[kk] = dot(-B_delta_2[1:],manipulated_variable_2[kk-1::-1]) + (1-alpha2)*dot(A_coeff_2,erro2[kk::-1])
                manipulated_variable_2[kk] /= B_delta_2[0]
                
                            
            elif kk > A_order:
                model_output_1[kk] = dot(-A_coeff_1[1:], model_output_1[kk-1:kk-A_order-1:-1]) + dot(B_coeff_1, manipulated_variable_1[kk-1:kk-B_order-1:-1])
                model_output_2[kk] = dot(-A_coeff_2[1:], model_output_2[kk-1:kk-A_order-1:-1]) - dot(B_coeff_2, manipulated_variable_2[kk-1:kk-B_order-1:-1])
                
                # Determine uncertainty
                output_model_comparation_1[kk] = process_output[kk] - model_output_1[kk]
                output_model_comparation_2[kk] = -(process_output[kk] - model_output_2[kk])

                # Determine Error
                erro1[kk] = reference_input[kk] - output_model_comparation_1[kk]
                erro2[kk] = -(reference_input[kk] + output_model_comparation_2[kk])

                # Control Signal
                manipulated_variable_1[kk] = dot(-B_delta_1[1:],manipulated_variable_1[kk-1:kk-B_order-1:-1]) + (1-alpha1)*dot(A_coeff_1,erro1[kk:kk-A_order-1:-1])
                manipulated_variable_1[kk] = manipulated_variable_1[kk]/B_delta_1[0]
                
                manipulated_variable_2[kk] = dot(-B_delta_2[1:],manipulated_variable_2[kk-1:kk-B_order-1:-1])+ (1-alpha2)*dot(A_coeff_2,erro2[kk:kk-A_order-1:-1])
                manipulated_variable_2[kk] = manipulated_variable_2[kk]/B_delta_2[0]
                
            # Control Signal Saturation
            manipulated_variable_1[kk] = max(min_pot, min(manipulated_variable_1[kk], max_pot))
            manipulated_variable_2[kk] = max(min_pot, min(manipulated_variable_2[kk], max_pot))
        

            # Motor Power String Formatation
            motors_power_packet = f"{manipulated_variable_1[kk]},{manipulated_variable_2[kk]}\r"

            sendToArduino(arduinoData, motors_power_packet)
            
            # Store the output process values and control signal
            current_timestamp = datetime.now()
            process_output_sensor[str(current_timestamp)] = float(process_output[kk])
            control_signal_1[str(current_timestamp)] = float(manipulated_variable_1[kk])
            control_signal_2[str(current_timestamp)] = float(manipulated_variable_2[kk])
            kk += 1

            percent_complete = kk / (samples_number)
            my_bar.progress(percent_complete, text=progress_text)
            
                


    # Turn off the motor
    sendToArduino(arduinoData, '0,0')

