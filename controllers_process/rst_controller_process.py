import streamlit as st
import time
from formatterInputs import *
from numpy import exp, ones, zeros, array, dot, convolve
from connections import *
from datetime import datetime
# CORRIGIDO: Importar as duas funções de session_state
from session_state import get_session_variable, set_session_controller_parameter 
from controllers_process.validations_functions import *


def rstControlProcessIncrementalSISO(transfer_function_type:str, 
                                     num_coeff:str, # Polinômio B
                                     den_coeff:str, # Polinômio A
                                     tau_ml_input:float, # Constante de Tempo de Malha Fechada (tau_mf)
                                     rst_single_reference:float,
                                     rst_siso_multiple_reference2:float, 
                                     rst_siso_multiple_reference3:float,
                                     change_ref_instant2 = 1, 
                                     change_ref_instant3 = 1):
    
    # 1. Limpa o erro anterior a cada nova execução
    set_session_controller_parameter('debug_error', None)

    # --- VALIDAÇÕES COM MENSAGEM DE ERRO PERSISTENTE ---
    if num_coeff == '':
        set_session_controller_parameter('debug_error', 'FALHA (Back-end): Coeficientes do Numerador B estão vazios.')
        return 
    
    if den_coeff =='':
        set_session_controller_parameter('debug_error', 'FALHA (Back-end): Coeficientes do Denominador A estão vazios.')
        return

    if 'arduinoData' not in st.session_state.connected:
        set_session_controller_parameter('debug_error', 'FALHA (Back-end): Arduino não conectado. Conecte na Sidebar primeiro.')
        return

    st.info("Validação OK. Iniciando cálculos do RST...")
    
    # 2. TRY/EXCEPT PARA CAPTURAR ERROS DE LÓGICA
    try:
        # --- SETUP E INICIALIZAÇÕES ---

        sampling_time = get_session_variable('sampling_time')
        samples_number = get_session_variable('samples_number')
        arduinoData = st.session_state.connected['arduinoData']

        # Condições Iniciais
        process_output = zeros(samples_number)
        delta_control_signal = zeros(samples_number) # du(k)
        manipulated_variable_1 = zeros(samples_number) # u(k)
        
        # Geração da Referência
        instant_sample_2 = get_sample_position(sampling_time, samples_number, change_ref_instant2)
        instant_sample_3 = get_sample_position(sampling_time, samples_number, change_ref_instant3)

        reference_input = rst_single_reference * ones(samples_number)
        reference_input[instant_sample_2:instant_sample_3] = rst_siso_multiple_reference2
        reference_input[instant_sample_3:] = rst_siso_multiple_reference3
        
        set_session_controller_parameter('reference_input', reference_input.tolist())

        # Saturação
        max_pot = get_session_variable('saturation_max_value')
        min_pot = get_session_variable('saturation_min_value')

        # --- CÁLCULO DOS COEFICIENTES RST ---
        
        A_coeff_all, B_coeff_all = convert_tf_2_discrete(num_coeff, den_coeff, transfer_function_type)
        
        # Validação para o modelo de 1ª ordem discreta (como assumido pelo MATLAB):
        if A_coeff_all.size < 2 or B_coeff_all.size < 1:
            # ERRO PERSISTENTE:
            set_session_controller_parameter('debug_error', f'FALHA (Back-end): O modelo (A={A_coeff_all}, B={B_coeff_all}) não é de 1ª ordem. Verifique os inputs.')
            return
            
        a1 = A_coeff_all[1] # Coeficiente a1 (do A(z^-1))
        b0 = B_coeff_all[0] # Coeficiente b0 (do B(z^-1))
        r0 = 1.0 # Coeficiente R(z^-1) = r0 (fixo)
        
        # Cálculo dos Polinômios
        tau_mf = tau_ml_input
        P1 = exp(-sampling_time / tau_mf)
        
        s0 = -(P1 + (a1 - 1.0) * r0) / b0
        s1 = -(a1 * r0) / b0
        t0 = s0 + s1
        
        # --- LIMPEZA DE ESTADO DE SESSÃO ---
        
        set_session_controller_parameter('control_signal_1', dict())
        control_signal_1 = get_session_variable('control_signal_1')
        
        set_session_controller_parameter('process_output_sensor', dict())
        process_output_sensor = get_session_variable('process_output_sensor')

        start_time = time.time()
        kk = 0

        progress_text = "Operation in progress. Please wait."
        my_bar = st.progress(0, text=progress_text)
        
        sendToArduino(arduinoData, "0")
        
        # --- LOOP DE CONTROLE ---

        while kk < samples_number:
            current_time = time.time()
            if current_time - start_time > sampling_time:
                start_time = current_time
                
                # 1. Medição da Saída
                process_output[kk] = readFromArduino(arduinoData)
                
                # 2. Período Inicial (k=0)
                if kk == 0: 
                    manipulated_variable_1[kk] = 0.0
                    sendToArduino(arduinoData, '0')

                # 3. Lógica de Controle (k >= 1)
                elif kk >= 1: 
                    
                    # LEI DE CONTROLE RST INCREMENTAL
                    ref_term = (t0 / r0) * reference_input[kk]
                    
                    output_term = (s0 / r0) * process_output[kk] + \
                                  (s1 / r0) * process_output[kk-1]
                    
                    delta_control_signal[kk] = ref_term - output_term
                    
                    # Sinal de Controle Total: u(k) = u(k-1) + du(k)
                    manipulated_variable_1[kk] = manipulated_variable_1[kk-1] + delta_control_signal[kk]
                    
                
                # 4. SATURAÇÃO E SERIAL WRITE
                
                manipulated_variable_1[kk] = max(min_pot, min(manipulated_variable_1[kk], max_pot))

                serial_data_pack = f"{manipulated_variable_1[kk]}\r"
                sendToArduino(arduinoData, serial_data_pack)
                
                # 5. Armazenamento e Progresso
                current_timestamp = datetime.now()
                process_output_sensor[str(current_timestamp)] = float(process_output[kk])
                control_signal_1[str(current_timestamp)] = float(manipulated_variable_1[kk])
                
                kk += 1
                percent_complete = kk / (samples_number)
                my_bar.progress(percent_complete, text=progress_text)
                

        # Turn off the motor
        sendToArduino(arduinoData, '0')

    except NameError as e:
        # ERRO PERSISTENTE:
        set_session_controller_parameter('debug_error', f"Erro de Nome (Back-end): Variável faltando na importação ou no escopo. Detalhe: {e}")
        try: sendToArduino(arduinoData, '0') 
        except: pass
        
    except Exception as e:
        # ERRO PERSISTENTE:
        set_session_controller_parameter('debug_error', f"Erro Inesperado durante o Processamento RST. O loop falhou. Detalhe: {e}")
        try: sendToArduino(arduinoData, '0') 
        except: pass

# 
# --- FUNÇÃO MIMO (COPIADA DO IMC) ---
#
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

    sampling_time = get_session_variable('sampling_time')
    samples_number = get_session_variable('samples_number')
    process_output = zeros(samples_number)
    model_output_1 = zeros(samples_number)
    model_output_2 = zeros(samples_number)
    erro1 = zeros(samples_number)
    erro2 = zeros(samples_number)
    output_model_comparation_1 = zeros(samples_number)
    output_model_comparation_2 = zeros(samples_semples_number)
    instant_sample_2 = get_sample_position(sampling_time, samples_number, change_ref_instant2)
    instant_sample_3 = get_sample_position(sampling_time, samples_number, change_ref_instant3)
    reference_input = imc_multiple_reference1*ones(samples_number)
    reference_input[instant_sample_2:instant_sample_3] = imc_multiple_reference2
    reference_input[instant_sample_3:] = imc_multiple_reference3
    st.session_state.controller_parameters['reference_input'] = reference_input.tolist()
    max_pot = get_session_variable('saturation_max_value')
    min_pot = get_session_variable('saturation_min_value')
    manipulated_variable_1 = zeros(samples_number)
    manipulated_variable_2 = zeros(samples_number)
    motors_power_packet = "0,0"
    A_coeff_1, B_coeff_1 = convert_tf_2_discrete(num_coeff_1,den_coeff_1,transfer_function_type)
    A_order = len(A_coeff_1)-1
    B_order = len(B_coeff_1) 
    A_coeff_2, B_coeff_2 = convert_tf_2_discrete(num_coeff_2,den_coeff_2,transfer_function_type)
    tau_mf1 = imc_mr_tau_mf1
    alpha1 = exp(-sampling_time/tau_mf1)
    tau_mf2 = imc_mr_tau_mf2
    alpha2 = exp(-sampling_time/tau_mf2)
    alpha_delta_1 = [1,-alpha1]
    B_delta_1 = convolve(B_coeff_1,alpha_delta_1)
    alpha_delta_2 = [1,-alpha2]
    B_delta_2 = convolve(B_coeff_2,alpha_delta_2)
    arduinoData = st.session_state.connected['arduinoData']
    set_session_controller_parameter('control_signal_1', dict())
    control_signal_1 = get_session_variable('control_signal_1')
    set_session_controller_parameter('control_signal_2', dict())
    control_signal_2 = get_session_variable('control_signal_2')
    set_session_controller_parameter('process_output_sensor', dict())
    process_output_sensor = get_session_variable('process_output_sensor')
    start_time = time.time()
    kk = 0
    progress_text = "Operation in progress. Please wait."
    my_bar = st.progress(0, text=progress_text)
    sendToArduino(arduinoData, '0,0')

    while kk < samples_number:
        current_time = time.time()
        if current_time - start_time > sampling_time:
            start_time = current_time
            process_output[kk] = readFromArduino(arduinoData)
            
            if kk <= A_order:
                sendToArduino(arduinoData, '0,0')
                
            elif kk == 1 and A_order == 1:
                model_output_1[kk] = dot(-A_coeff_1[1:], model_output_1[kk-1::-1]) + dot(B_coeff_1, manipulated_variable_1[kk-1::-1])
                model_output_2[kk] = dot(-A_coeff_2[1:], model_output_2[kk-1::-1]) - dot(B_coeff_2, manipulated_variable_2[kk-1::-1])
                output_model_comparation_1[kk] = process_output[kk] - model_output_1[kk]
                output_model_comparation_2[kk] = -(process_output[kk] - model_output_2[kk])
                erro1[kk] = reference_input[kk] - output_model_comparation_1[kk]
                erro2[kk] = -(reference_input[kk] + output_model_comparation_2[kk])
                manipulated_variable_1[kk] = dot(-B_delta_1[1:],manipulated_variable_1[kk-1::-1]) + (1-alpha1)*dot(A_coeff_1,erro1[kk::-1])
                manipulated_variable_1[kk] /= B_delta_1[0]
                manipulated_variable_2[kk] = dot(-B_delta_2[1:],manipulated_variable_2[kk-1::-1]) + (1-alpha2)*dot(A_coeff_2,erro2[kk::-1])
                manipulated_variable_2[kk] /= B_delta_2[0]
                
            elif kk > A_order:
                model_output_1[kk] = dot(-A_coeff_1[1:], model_output_1[kk-1:kk-A_order-1:-1]) + dot(B_coeff_1, manipulated_variable_1[kk-1:kk-B_order-1:-1])
                model_output_2[kk] = dot(-A_coeff_2[1:], model_output_2[kk-1:kk-A_order-1:-1]) - dot(B_coeff_2, manipulated_variable_2[kk-1:kk-B_order-1:-1])
                output_model_comparation_1[kk] = process_output[kk] - model_output_1[kk]
                output_model_comparation_2[kk] = -(process_output[kk] - model_output_2[kk])
                erro1[kk] = reference_input[kk] - output_model_comparation_1[kk]
                erro2[kk] = -(reference_input[kk] + output_model_comparation_2[kk])
                manipulated_variable_1[kk] = dot(-B_delta_1[1:],manipulated_variable_1[kk-1:kk-B_order-1:-1]) + (1-alpha1)*dot(A_coeff_1,erro1[kk:kk-A_order-1:-1])
                manipulated_variable_1[kk] = manipulated_variable_1[kk]/B_delta_1[0]
                manipulated_variable_2[kk] = dot(-B_delta_2[1:],manipulated_variable_2[kk-1:kk-B_order-1:-1])+ (1-alpha2)*dot(A_coeff_2,erro2[kk:kk-A_order-1:-1])
                manipulated_variable_2[kk] = manipulated_variable_2[kk]/B_delta_2[0]
                
            manipulated_variable_1[kk] = max(min_pot, min(manipulated_variable_1[kk], max_pot))
            manipulated_variable_2[kk] = max(min_pot, min(manipulated_variable_2[kk], max_pot))
        
            motors_power_packet = f"{manipulated_variable_1[kk]},{manipulated_variable_2[kk]}\r"
            sendToArduino(arduinoData, motors_power_packet)
            
            current_timestamp = datetime.now()
            process_output_sensor[str(current_timestamp)] = float(process_output[kk])
            control_signal_1[str(current_timestamp)] = float(manipulated_variable_1[kk])
            control_signal_2[str(current_timestamp)] = float(manipulated_variable_2[kk])
            kk += 1

            percent_complete = kk / (samples_number)
            my_bar.progress(percent_complete, text=progress_text)
            
    # Turn off the motor
    sendToArduino(arduinoData, '0,0')