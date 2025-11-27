
import streamlit as st
from formatterInputs import *
from numpy import exp, ones, zeros,array, dot, convolve
from connections import *
import datetime
from session_state import get_session_variable, set_session_controller_parameter
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
    
    if 'arduinoData' not in st.session_state.connected:
        return st.warning('Arduino não conectado!')

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
    
    set_session_controller_parameter('reference_input', reference_input.tolist())

    # Power Saturation
    max_pot = get_session_variable('saturation_max_value')
    min_pot = get_session_variable('saturation_min_value')
    
    # Validação para o erro 'NoneType' na Saturação
    if max_pot is None or min_pot is None:
        st.error("FALHA (Back-end): Valores de Saturação (Máx/Mín) não definidos. Configure-os na Sidebar.")
        return

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
    set_session_controller_parameter('control_signal_1',dict())
    control_signal_1 = get_session_variable('control_signal_1')
    
    # clear previous control signal values
    set_session_controller_parameter('process_output_sensor',dict())
    process_output_sensor = get_session_variable('process_output_sensor')

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

# A função imcControlProcessTISO foi removida.
