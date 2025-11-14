import streamlit as st
import time
from formatterInputs import *
# ADICIONAR 'eye' para a matriz de covariância
from numpy import exp, ones, zeros, array, dot, convolve, eye
from connections import *
from datetime import datetime
# CORRIGIDO: Importar as duas funções de session_state
from session_state import *
from controllers_process.validations_functions import *


def rstControlProcessIncrementalSISO(transfer_function_type:str, 
                                     num_coeff:str, # Polinômio B
                                     den_coeff:str, # Polinômio A
                                     tau_ml_input:float, # Constante de Tempo de Malha Fechada (tau_mf)
                                     pid_structure:str, # <-- NOVO PARÂMETRO
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
        # --- INÍCIO DA ADIÇÃO ---
        e = zeros(samples_number) # Erro e(k)
        # --- FIM DA ADIÇÃO ---
        
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
        
        # --- INÍCIO DA ADIÇÃO (Linhas 97-109) ---
        # Parâmetros PID derivados do RST (para 1ª ordem)
        # sampling_time é pego da sessão (Linha 32)
        
        kc = -s1
        ki_rst = t0  # Este é o ki da fórmula RST (t0)
        
        # Evita divisão por zero se t0 ou kc forem 0
        if ki_rst == 0 or kc == 0:
            st.error(f"FALHA (Back-end): Parâmetros RST (t0={ki_rst}, s1={-kc}) causam divisão por zero no cálculo do PID.")
            return

        # Parâmetros PID
        ti = (kc * sampling_time) / ki_rst
        td = 0.0 # O seu exemplo usa td=0. Você pode tornar isso um input no futuro.
        # --- FIM DA ADIÇÃO ---

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
                    # --- INÍCIO DA ADIÇÃO ---
                    e[kk] = reference_input[kk] - process_output[kk]
                    # --- FIM DA ADIÇÃO ---

                # 3. Lógica de Controle (k >= 1)
                elif kk >= 1: 
                    
                    # --- CÁLCULO DE VARIÁVEIS COMUNS ---
                    ts = sampling_time
                    e[kk] = reference_input[kk] - process_output[kk] # e(k)
                    
                    # --- LÓGICA DE CONTROLE BASEADA NA ESTRUTURA ---
                    
                    if pid_structure == 'RST Incremental Puro':
                        # LEI DE CONTROLE RST INCREMENTAL (Original)
                        ref_term = (t0 / r0) * reference_input[kk]
                        output_term = (s0 / r0) * process_output[kk] + \
                                      (s1 / r0) * process_output[kk-1]
                        
                        delta_control_signal[kk] = ref_term - output_term
                        manipulated_variable_1[kk] = manipulated_variable_1[kk-1] + delta_control_signal[kk]

                    # --- ESTRUTURAS PID (I+PD, PI+D, Ideal, Paralelo) ---
                    # Todas estas estruturas precisam de valores passados (k-1, k-2)
                    # Vamos tratar k=1 como um caso especial, e a lógica principal só roda em k >= 2

                    elif kk == 1: 
                        # Não temos e(k-2) ou y(k-2), então apenas seguramos o controle
                        manipulated_variable_1[kk] = manipulated_variable_1[kk-1]
                        delta_control_signal[kk] = 0.0
                    
                    elif kk >= 2:
                        # Variáveis comuns para k >= 2
                        e_k = e[kk]
                        e_k1 = e[kk-1]
                        # e_k2 = e[kk-2] # (Não é usado se td=0)
                        y_k = process_output[kk]
                        y_k1 = process_output[kk-1]
                        # y_k2 = process_output[kk-2] # (Não é usado se td=0)

                        delta_u = 0.0 # Inicializa

                        if pid_structure == 'I + PD':
                            # Fórmula: u(k) = u(k-1) + kc*((e(k)*ts)/ti - y(k) + y(k-1)) (com td=0)
                            termo_I = (e_k * ts) / ti
                            termo_P = -y_k
                            termo_D = y_k1
                            delta_u = kc * (termo_I + termo_P + termo_D)

                        elif pid_structure == 'PI + D':
                            # Fórmula: u(k) = u(k-1) + kc*((1+ts/ti)*e(k) - e(k-1)) (com td=0)
                            termo_PI_1 = (1.0 + ts / ti) * e_k
                            termo_PI_2 = -e_k1
                            delta_u = kc * (termo_PI_1 + termo_PI_2)
                        
                        elif pid_structure == 'PID Ideal':
                            # Fórmula: u(k) = u(k-1) + kc*(1 + ts/ti)*e(k) - kc*e(k-1) (com td=0)
                            termo_q0 = kc * (1.0 + ts / ti) * e_k
                            termo_q1 = -kc * e_k1
                            delta_u = termo_q0 + termo_q1
                            
                        elif pid_structure == 'PID Paralelo':
                            # Fórmula: delta_u = Kp*(e(k) - e(k-1)) + Ki*ts*e(k)
                            # Kp = kc, Ki = ki_rst (t0), Kd = 0
                            termo_P = kc * (e_k - e_k1)
                            termo_I = ki_rst * ts * e_k
                            delta_u = termo_P + termo_I
                        
                        else:
                            delta_u = 0.0 # Estrutura desconhecida, não faz nada
                        
                        # Aplica a lei de controle
                        manipulated_variable_1[kk] = manipulated_variable_1[kk-1] + delta_u
                        delta_control_signal[kk] = delta_u
                    
                    # --- FIM DA ALTERAÇÃO ---
                
                
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
# --- INÍCIO DA NOVA FUNÇÃO ADAPTATIVA ---
#

def rstControlProcessAdaptiveSISO(tau_ml_input:float, # Constante de Tempo de Malha Fechada (tau_mf)
                                  pid_structure:str, # Estrutura (I+PD, etc.)
                                  a1_initial:float,  # Parâmetro inicial a1
                                  b0_initial:float,  # Parâmetro inicial b0
                                  p0_initial:float,  # Fator de esquecimento (P inicial)
                                  rst_single_reference:float,
                                  rst_siso_multiple_reference2:float, 
                                  rst_siso_multiple_reference3:float,
                                  change_ref_instant2 = 1, 
                                  change_ref_instant3 = 1):
    

    # --- 1. VALIDAÇÕES INICIAIS ---
    if 'arduinoData' not in st.session_state.connected:
        return st.error('FALHA (Back-end): Arduino não conectado. Conecte na Sidebar primeiro')
        
    sampling_time = get_session_variable('sampling_time')
    samples_number = get_session_variable('samples_number')

    if sampling_time is None or samples_number is None:
        st.error("FALHA (Back-end): Tempo de amostragem (Ts) ou Quantidade de amostras (N) não definidos. Configure-os na Sidebar.")
        return

    max_pot = get_session_variable('saturation_max_value')
    min_pot = get_session_variable('saturation_min_value')

    if max_pot is None or min_pot is None:
        st.error("FALHA (Back-end): Valores de Saturação (Máx/Mín) não definidos. Configure-os na Sidebar.")
        return

    st.info("Validação OK. Iniciando RST Adaptativo...")
    
    # 2. TRY/EXCEPT PARA CAPTURAR ERROS DE LÓGICA
    try:
        # --- 3. SETUP E INICIALIZAÇÕES ---
        arduinoData = st.session_state.connected['arduinoData']

        # Vetores de dados
        process_output = zeros(samples_number)
        delta_control_signal = zeros(samples_number) # du(k)
        manipulated_variable_1 = zeros(samples_number) # u(k)
        e = zeros(samples_number) # Erro e(k)
        
        # Parâmetros do MQR (RLS)
        teta = array([a1_initial, b0_initial]) # teta = [a1, b0]
        p_matrix = eye(2) * p0_initial         # Matriz de covariância P
        
        # Parâmetros RST
        r0 = 1.0
        tau_mf = tau_ml_input
        p1 = exp(-sampling_time / tau_mf) # p1 é fixo (baseado no tau desejado)

        # Ganhos PID (serão atualizados no loop)
        kc = 0.0
        ti = 1.0
        ki_rst = 1.0

        # Geração da Referência
        instant_sample_2 = get_sample_position(sampling_time, samples_number, change_ref_instant2)
        instant_sample_3 = get_sample_position(sampling_time, samples_number, change_ref_instant3)

        reference_input = rst_single_reference * ones(samples_number)
        reference_input[instant_sample_2:instant_sample_3] = rst_siso_multiple_reference2
        reference_input[instant_sample_3:] = rst_siso_multiple_reference3
        
        set_session_controller_parameter('reference_input', reference_input.tolist())

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
        
        # --- 4. LOOP DE CONTROLE ADAPTATIVO ---
        while kk < samples_number:
            current_time = time.time()
            if current_time - start_time > sampling_time:
                start_time = current_time
                
                # 1. Medição da Saída
                process_output[kk] = readFromArduino(arduinoData)
                
                # 2. Período Inicial (k=0, k=1)
                if kk < 2: 
                    manipulated_variable_1[kk] = 0.0
                    sendToArduino(arduinoData, '0')
                    e[kk] = reference_input[kk] - process_output[kk]

                # 3. Lógica Adaptativa (k >= 2)
                elif kk >= 2: 
                    
                    # --- A. ESTIMADOR MQR (RLS) ---
                    # Usa y(k), y(k-1) e u(k-1) para estimar teta(k)
                    
                    # Vetor regressor fi = [-y(k-1), u(k-1)]
                    fi = array([-process_output[kk-1], manipulated_variable_1[kk-1]]) 
                    
                    # Previsão da saída yest(k) = fi * teta(k-1)
                    y_est = dot(fi, teta) 
                    
                    # Erro de estimação est(k) = y(k) - yest(k)
                    est_k = process_output[kk] - y_est
                    
                    # Atualização do Ganho K
                    fi_T = fi.reshape(2, 1)
                    fi = fi.reshape(1, 2)
                    
                    # Proteção contra divisão por zero (denominador muito pequeno)
                    ganho_den_val = 1.0 + dot(dot(fi, p_matrix), fi_T)
                    if abs(ganho_den_val) < 1e-6:
                        ganho_den_val = 1e-6 # Evita divisão por zero
                        
                    ganho_den = ganho_den_val
                    ganho_num = dot(p_matrix, fi_T)
                    ganho = ganho_num / ganho_den # Vetor ganho (2, 1)
                    
                    # Atualização dos Parâmetros teta(k) = teta(k-1) + K * est(k)
                    teta = teta + ganho.T * est_k
                    teta = teta.flatten() # Mantém teta como (2,)
                    
                    # Atualização da Matriz de Covariância P
                    p_matrix = p_matrix - dot(dot(ganho, (1.0 + dot(dot(fi, p_matrix), fi_T))), ganho.T)
                    
                    # Parâmetros estimados
                    a1_k = teta[0]
                    b0_k = teta[1]

                    # Proteção contra divisão por zero (b0_k muito pequeno)
                    if abs(b0_k) < 1e-5:
                        b0_k = 1e-5 if b0_k >= 0 else -1e-5
                    
                    # --- B. CÁLCULO DOS GANHOS (ADAPTATIVOS) ---
                    s0 = (-p1 - (a1_k - 1.0) * r0) / b0_k
                    s1 = -a1_k / b0_k
                    t0 = s0 + s1
                    
                    kc = -s1
                    ki_rst = t0
                    
                    # Proteção contra divisão por zero (PID)
                    if abs(ki_rst) < 1e-5 or abs(kc) < 1e-5:
                        # Reusa os ganhos antigos se os novos forem instáveis
                        pass 
                    else:
                        ti = (kc * sampling_time) / ki_rst
                    
                    ts = sampling_time
                    e[kk] = reference_input[kk] - process_output[kk] # e(k)

                    # --- C. LEI DE CONTROLE (com ganhos adaptativos) ---
                    
                    # Variáveis comuns para k >= 2
                    e_k = e[kk]
                    e_k1 = e[kk-1]
                    y_k = process_output[kk]
                    y_k1 = process_output[kk-1]

                    delta_u = 0.0 # Inicializa

                    if pid_structure == 'RST Incremental Puro':
                        ref_term = (t0 / r0) * reference_input[kk]
                        output_term = (s0 / r0) * process_output[kk] + \
                                      (s1 / r0) * process_output[kk-1]
                        delta_u = ref_term - output_term

                    elif pid_structure == 'I + PD':
                        termo_I = (e_k * ts) / ti
                        termo_P = -y_k
                        termo_D = y_k1
                        delta_u = kc * (termo_I + termo_P + termo_D)

                    elif pid_structure == 'PI + D':
                        termo_PI_1 = (1.0 + ts / ti) * e_k
                        termo_PI_2 = -e_k1
                        delta_u = kc * (termo_PI_1 + termo_PI_2)
                    
                    elif pid_structure == 'PID Ideal':
                        termo_q0 = kc * (1.0 + ts / ti) * e_k
                        termo_q1 = -kc * e_k1
                        delta_u = termo_q0 + termo_q1
                        
                    elif pid_structure == 'PID Paralelo':
                        termo_P = kc * (e_k - e_k1)
                        termo_I = ki_rst * ts * e_k
                        delta_u = termo_P + termo_I
                    
                    else:
                        delta_u = 0.0 
                    
                    # Aplica a lei de controle
                    manipulated_variable_1[kk] = manipulated_variable_1[kk-1] + delta_u
                    delta_control_signal[kk] = delta_u

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
                
        # --- FIM DO LOOP ---
        sendToArduino(arduinoData, '0')

    except NameError as e:
        st.error(f"Erro de Nome (Back-end): Variável faltando na importação ou no escopo. Detalhe: {e}")
        
    except Exception as e:
        st.error(f"Erro Inesperado durante o Processamento RST Adaptativo. O loop falhou. Detalhe: {e}")

# --- FIM DA FUNÇÃO ADAPTATIVA ---

# --- A FUNÇÃO MIMO FOI REMOVIDA DAQUI ---