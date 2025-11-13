import streamlit as st
import time
from formatterInputs import *
from numpy import exp, ones, zeros, array, dot, convolve
from connections import *
from datetime import datetime
# CORRIGIDO: Importar as duas funções de session_state
from session_state import *
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
    

    # --- VALIDAÇÕES COM MENSAGEM DE ERRO PERSISTENTE ---
    if num_coeff == '':
        return st.error('FALHA (Back-end): Coeficientes do Numerador B estão vazios')
    
    if den_coeff =='':
        return st.error('FALHA (Back-end): Coeficientes do Denominador A estão vazios.')

    if 'arduinoData' not in st.session_state.connected:
        return st.error('FALHA (Back-end): Arduino não conectado. Conecte na Sidebar primeiro')
        
    # Validação para o erro 'NoneType'
    sampling_time = get_session_variable('sampling_time')
    samples_number = get_session_variable('samples_number')

    if sampling_time is None or samples_number is None:
        st.error("FALHA (Back-end): Tempo de amostragem (Ts) ou Quantidade de amostras (N) não definidos. Configure-os na Sidebar.")
        return


    st.info("Validação OK. Iniciando cálculos do RST...")
    
    # 2. TRY/EXCEPT PARA CAPTURAR ERROS DE LÓGICA
    try:
        # --- SETUP E INICIALIZAÇÕES ---
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

        # Validação para o erro 'NoneType' na Saturação
        if max_pot is None or min_pot is None:
            st.error("FALHA (Back-end): Valores de Saturação (Máx/Mín) não definidos. Configure-os na Sidebar.")
            return
# --- FIM DA CORREÇÃO ---

        # --- CÁLCULO DOS COEFICIENTES RST ---
        
        A_coeff_all, B_coeff_all = convert_tf_2_discrete(num_coeff, den_coeff, transfer_function_type)
        
        # Validação para o modelo de 1ª ordem discreta (como assumido pelo MATLAB):
        if A_coeff_all.size < 2 or B_coeff_all.size < 1:
            # ERRO CORRIGIDO (Padrão GMV):
            st.error(f'FALHA (Back-end): O modelo (A={A_coeff_all}, B={B_coeff_all}) não é de 1ª ordem. Verifique os inputs.')
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
        # ERRO CORRIGIDO (Padrão GMV):
        st.error(f"Erro de Nome (Back-end): Variável faltando na importação ou no escopo. Detalhe: {e}")
        
    except Exception as e:
        # ERRO CORRIGIDO (Padrão GMV):
        st.error(f"Erro Inesperado durante o Processamento RST. O loop falhou. Detalhe: {e}")

# 
# --- FUNÇÃO MIMO (COPIADA DO IMC) ---
#
def imcControlProcessTISO(transfer_function_type:str,num_coeff_1:str,den_coeff_1:str, num_coeff_2:str,den_coeff_2:str,
                          imc_mr_tau_mf1:float, imc_mr_tau_mf2:float,
                          imc_multiple_reference1:float, imc_multiple_reference2:float, imc_multiple_reference3:float,
                          change_ref_instant2 = 1, change_ref_instant3 = 1):

    st.warning("Função TISO (IMC) não implementada neste ficheiro.")
    pass

 