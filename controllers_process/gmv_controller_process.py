import streamlit as st
from formatterInputs import *
import numpy as np
from connections import *
import datetime
from session_state import get_session_variable, set_session_controller_parameter
from controllers_process.validations_functions import *


def create_P1_poly(ne,ns,filter=False):
    P1 = np.zeros((ne + ns + 2,1))
    P1[0] = 1
    if filter:
        P1[1] = -0.98
    return P1


def E_S_poly_calculation(ne,ns,Am1,P):
    delta = [1, -1]
    Am1_barra1 = np.convolve(Am1, delta)
    mat_Scoef1 = np.vstack((np.zeros((ne+1,ns+1)), np.eye(ns+1)))
    mat_EAcoef1= np.zeros((ne + ns + 2, ne+1))

    am1_barra_len = len(Am1_barra1)

    for k in range(ne+1):
        mat_EAcoef1[k:k+am1_barra_len,k] = Am1_barra1.T

    mat_SEcoef1 = np.concatenate((mat_EAcoef1,mat_Scoef1),axis=1)  

    mat_SEcoef1_inv = np.linalg.inv(mat_SEcoef1)
    EScoef1_array = np.dot(mat_SEcoef1_inv,P)


    epoly1 = EScoef1_array[0:ne+1].T[0]
    spoly1 = EScoef1_array[ne+1:].T[0]
    return epoly1,spoly1


# Calculo do polinômio R1(z^-1)
def r_poly_calculation(Bm1,epoly1,q0):
    BE_poly = np.convolve(Bm1,epoly1)
    BE_poly[0] += q0
    rpoly1 = BE_poly
    return rpoly1


def gmvControlProcessSISO(transfer_function_type:str,num_coeff:str,den_coeff:str,
                          gmv_q01:float, 
                          gmv_multiple_reference1:float, gmv_multiple_reference2:float, gmv_multiple_reference3:float,
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
    
    # Receber o objeto arduino da sessão
    arduinoData = st.session_state.connected['arduinoData']

    # gmv Controller Project

    # Initial Conditions
    process_output = np.zeros(samples_number)
    delta_control_signal = np.zeros(samples_number)

    # Take the index of time to change the referencee
    instant_sample_2 = get_sample_position(sampling_time, samples_number, change_ref_instant2)
    instant_sample_3 = get_sample_position(sampling_time, samples_number, change_ref_instant3)

    reference_input = gmv_multiple_reference1*np.ones(samples_number)
    reference_input[instant_sample_2:instant_sample_3] = gmv_multiple_reference2
    reference_input[instant_sample_3:] = gmv_multiple_reference3
    
    st.session_state.controller_parameters['reference_input'] = reference_input.tolist()

    # Power Saturation
    max_pot = get_session_variable('saturation_max_value')
    min_pot = get_session_variable('saturation_min_value')

    # Manipulated variable
    manipulated_variable_1 = np.zeros(samples_number)
    serial_data_pack = "0"

    # Model transfer Function
    A_coeff, B_coeff = convert_tf_2_discrete(num_coeff,den_coeff,transfer_function_type)
    
    # print(A_coeff)
    # print(B_coeff)
    A_order = len(A_coeff)-1
    B_order = len(B_coeff)-1 
    # Close Loop Tau Calculation
    # tau_mf1 = ajuste1*tausmith1
    q01 = gmv_q01
    
    
    d = 1
    na1 = A_order
    nb1 = B_order
    ns1 = na1
    ne1 = d-1


    # Definir o polinômio P1(z)
    P1 = create_P1_poly(ne1,ns1)
    # Encontrar o polinômio E1(z) e S1(z)
    e_poly_1,s_poly_1 = E_S_poly_calculation(ne1,ns1,A_coeff,P1)
    
        
    # Encontrar o polinômio R por meio do Q1(z^-1) e B1(z^-1)
    r_poly_1 = r_poly_calculation(B_coeff,e_poly_1,q01)
    nr1 = len(r_poly_1)-1

    
    # Calculo do polinômio T1(z)
    t01 = sum(P1)[0]
    

    
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
                current_timestamp = datetime.now()
                process_output_sensor[str(current_timestamp)] = float(process_output[kk])
                control_signal_1[str(current_timestamp)] = float(manipulated_variable_1[kk])
                kk += 1

                percent_complete = kk / (samples_number)
                my_bar.progress(percent_complete, text=progress_text)
                
                sendToArduino(arduinoData, '0')
                

            # ---- Motor Model Output
            elif kk == 1 and A_order == 1:
                
                # GMV Control Signal
                delta_control_signal[kk] = np.dot(r_poly_1[1:],delta_control_signal[kk-1::-1]) \
                                            + t01*reference_input[kk] - np.dot(s_poly_1,process_output[kk::-1])
                delta_control_signal[kk]/=r_poly_1[0]
                
                manipulated_variable_1[kk] = manipulated_variable_1[kk-1] +  delta_control_signal[kk]
               
                # Control Signal Saturation
                manipulated_variable_1[kk] = max(min_pot, min(manipulated_variable_1[kk], max_pot))
            

                # Motor Power String Formatation
                serial_data_pack = f"{manipulated_variable_1[kk]}\r"

                sendToArduino(arduinoData, serial_data_pack)
                
                # Store the output process values and control signal
                current_timestamp = datetime.datetime.now()
                process_output_sensor[str(current_timestamp)] = float(process_output[kk])
                control_signal_1[str(current_timestamp)] = float(manipulated_variable_1[kk])
                kk += 1

                percent_complete = kk / (samples_number)
                my_bar.progress(percent_complete, text=progress_text)
                                        
            elif kk >A_order:
                
                # GMV Control Signal
                delta_control_signal[kk] = np.dot(r_poly_1[1:],delta_control_signal[kk-1:kk-nb1-1:-1]) + t01*reference_input[kk] - np.dot(s_poly_1,process_output[kk:kk-ns1-1:-1])
                delta_control_signal[kk]/=r_poly_1[0]

                # Control Signal
                manipulated_variable_1[kk] = manipulated_variable_1[kk-1] +  delta_control_signal[kk]
                
                # Control Signal Saturation
                manipulated_variable_1[kk] = max(min_pot, min(manipulated_variable_1[kk], max_pot))
            

                # Motor Power String Formatation
                serial_data_pack = f"{manipulated_variable_1[kk]}\r"
                sendToArduino(arduinoData, serial_data_pack)
                
                # Store the output process values and control signal
                current_timestamp = datetime.now()
                process_output_sensor[str(current_timestamp)] = float(process_output[kk])
                control_signal_1[str(current_timestamp)] = float(manipulated_variable_1[kk])
                kk += 1

                percent_complete = kk / (samples_number)
                my_bar.progress(percent_complete, text=progress_text)

    # Turn off the motor
    sendToArduino(arduinoData, '0')


def gmvControlProcessTISO(transfer_function_type:str,num_coeff_1:str,den_coeff_1:str, num_coeff_2:str,den_coeff_2:str,
                          gmv_q01:float,gmv_q02:float,
                          gmv_multiple_reference1:float, gmv_multiple_reference2:float, gmv_multiple_reference3:float,
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

    if not st.session_state.connected['arduinoData']:
        return st.toast('Arduino não conectado!')
    # Receber o objeto arduino da sessão
    arduinoData = st.session_state.connected['arduinoData']
    
    # gmv Controller Project

    # Initial Conditions
    process_output = np.zeros(samples_number)
    delta_control_signal_1 = np.zeros(samples_number)
    delta_control_signal_2 = np.zeros(samples_number)

    # Take the index of time to change the referencee
    instant_sample_2 = get_sample_position(sampling_time, samples_number, change_ref_instant2)
    instant_sample_3 = get_sample_position(sampling_time, samples_number, change_ref_instant3)

    reference_input = gmv_multiple_reference1*np.ones(samples_number)
    reference_input[instant_sample_2:instant_sample_3] = gmv_multiple_reference2
    reference_input[instant_sample_3:] = gmv_multiple_reference3
    st.session_state.controller_parameters['reference_input'] = reference_input.tolist()


    # Power Saturation
    max_pot = get_session_variable('saturation_max_value')
    min_pot = get_session_variable('saturation_min_value')


    # Manipulated variable
    manipulated_variable_1 = np.zeros(samples_number)
    manipulated_variable_2 = np.zeros(samples_number)
    motors_power_packet = "0,0"

    # Model transfer Function 1
    A_coeff_1, B_coeff_1 = convert_tf_2_discrete(num_coeff_1,den_coeff_1,transfer_function_type)
    
    # print(A_coeff)
    # print(B_coeff)
    A_order = len(A_coeff_1)-1
    B_order = len(B_coeff_1)-1 # Zero holder aumenta um grau

    ## Model transfer Function 2
    A_coeff_2, B_coeff_2 = convert_tf_2_discrete(num_coeff_2,den_coeff_2,transfer_function_type)
    q01 = gmv_q01
    q02 = gmv_q02
    
    # print(A_coeff)
    # print(B_coeff)
    d = 1
    na1 = A_order
    nb1 = B_order
    ns1 = na1
    ne1 = d-1
    
    
    # Definir o polinômio P1(z)
    P1 = create_P1_poly(ne1,ns1,filter=True)
    P2 = create_P1_poly(ne1,ns1,filter=True)
    
    # Encontrar o polinômio E1(z) e S1(z)
    e_poly_1,s_poly_1 = E_S_poly_calculation(ne1,ns1,A_coeff_1,P1)
    e_poly_2,s_poly_2 = E_S_poly_calculation(ne1,ns1,A_coeff_2,P2)
    
        
    # Encontrar o polinômio R por meio do Q1(z^-1) e B1(z^-1)
    r_poly_1 = r_poly_calculation(B_coeff_1,e_poly_1,q01)
    r_poly_2 = r_poly_calculation(B_coeff_2,e_poly_2,q02)
    nr1 = len(r_poly_1)-1
    nr2 = len(r_poly_1)-1

    
    # Calculo do polinômio T1(z)
    t01 = sum(P1)[0]
    t02 = sum(P2)[0]
    


    
    # clear previous control signal values
    set_session_controller_parameter('control_signal_1',dict())
    control_signal_1 = get_session_variable('control_signal_1')
    
    
    # clear previous control signal values
    set_session_controller_parameter('control_signal_2',dict())
    control_signal_2 = get_session_variable('control_signal_2')
    
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
    sendToArduino(arduinoData, '0,0')

    while kk < samples_number:
        current_time = time.time()
        if current_time - start_time > sampling_time:
            start_time = current_time
            
            # -----  Angle Sensor Output
            process_output[kk] = readFromArduino(arduinoData)
            
            if kk <= A_order:
                # Store the output process values and control signal
                current_timestamp = datetime.now()
                process_output_sensor[str(current_timestamp)] = float(process_output[kk])
                control_signal_1[str(current_timestamp)] = float(manipulated_variable_1[kk])
                control_signal_2[str(current_timestamp)] = float(manipulated_variable_2[kk])
                kk += 1

                percent_complete = kk / (samples_number)
                my_bar.progress(percent_complete, text=progress_text)
                
                sendToArduino(arduinoData, '0,0')
                
            # ---- Motor Model Output
            elif kk == 1 and A_order == 1:
                # ---- GMV Control Signal
                delta_control_signal_1[kk] = np.dot(r_poly_1[1:],delta_control_signal_1[kk-1::-1])\
                                            + t01*reference_input[kk] - np.dot(s_poly_1,process_output[kk::-1])
                delta_control_signal_1[kk]/=r_poly_1[0]
                manipulated_variable_1[kk] = manipulated_variable_1[kk-1] +  delta_control_signal_1[kk]
                
                delta_control_signal_2[kk] = np.dot(r_poly_2[1:],delta_control_signal_2[kk-1::-1])\
                                            + t02*reference_input[kk] - np.dot(s_poly_2,process_output[kk::-1])
                delta_control_signal_2[kk]/=r_poly_2[0]
                manipulated_variable_2[kk] = manipulated_variable_2[kk-1] -  delta_control_signal_2[kk]
               
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
                
            elif kk > A_order:
                
                # GMV Control Signal
                delta_control_signal_1[kk] = np.dot(r_poly_1[1:],delta_control_signal_1[kk-1:kk-nb1-1:-1])\
                                            + t01*reference_input[kk] - np.dot(s_poly_1,process_output[kk:kk-ns1-1:-1])
                delta_control_signal_1[kk]/=r_poly_1[0]
                delta_control_signal_2[kk] = np.dot(r_poly_2[1:],delta_control_signal_2[kk-1:kk-nb1-1:-1])\
                                            + t02*reference_input[kk] - np.dot(s_poly_2,process_output[kk:kk-ns1-1:-1])
                delta_control_signal_2[kk]/=r_poly_2[0]

                # Control Signal
                manipulated_variable_1[kk] = manipulated_variable_1[kk-1] +  delta_control_signal_1[kk]
                manipulated_variable_2[kk] = manipulated_variable_2[kk-1] -  delta_control_signal_2[kk]
                
               
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

