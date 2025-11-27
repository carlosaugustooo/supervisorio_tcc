import streamlit as st
import time
import traceback
import serial 
from formatterInputs import *
from numpy import exp, ones, zeros, array, dot, convolve, eye, trim_zeros, sum as np_sum, roots, abs as np_abs, clip, linalg
from connections import *
from datetime import datetime
from session_state import *
from controllers_process.validations_functions import *

def rstControlProcessIncrementalSISO(transfer_function_type:str, 
                                     num_coeff:str, 
                                     den_coeff:str, 
                                     tau_ml_input:float, 
                                     pid_structure:str,
                                     rst_single_reference:float,
                                     rst_siso_multiple_reference2:float, 
                                     rst_siso_multiple_reference3:float,
                                     change_ref_instant2 = 1, 
                                     change_ref_instant3 = 1):
    
    # --- 1. VALIDAÇÕES INICIAIS ---
    if num_coeff == '' or den_coeff == '':
        return st.error('FALHA: Coeficientes vazios.')
    
    if 'arduinoData' not in st.session_state.connected:
        return st.error('FALHA: Arduino não conectado.')

    try:
        sampling_time = float(get_session_variable('sampling_time'))
        samples_number = int(get_session_variable('samples_number'))
        max_pot = float(get_session_variable('saturation_max_value'))
        min_pot = float(get_session_variable('saturation_min_value'))
        
        # Limitador de Taxa para suavizar PID
        delta_u_max = (max_pot - min_pot) * 0.20 
        
    except Exception as e:
        return st.error(f"FALHA CONFIG: Verifique parâmetros. Erro: {e}")

    st.info("Iniciando RST (Correção PID)...")

    try:
        # --- 2. MODELAGEM MATEMÁTICA ---
        arduinoData = st.session_state.connected['arduinoData']

        # Converte TF
        A_coeff_all, B_coeff_all = convert_tf_2_discrete(num_coeff, den_coeff, transfer_function_type)
        
        B_stripped = [b for b in B_coeff_all if abs(b) > 1e-12]
        if len(B_stripped) == 0: return st.error("FALHA: Numerador nulo.")
            
        b0 = float(B_stripped[0])
        a1_raw = float(A_coeff_all[1]) if len(A_coeff_all) > 1 else 0.0
        
        if a1_raw > 0:
            a1 = -abs(a1_raw)
            msg_aviso = f"⚠️ AVISO: a1 invertido para {a1} (estabilidade)."
        else:
            a1 = a1_raw
            msg_aviso = None

        A_order = len(A_coeff_all) - 1
        if A_order < 1: A_order = 1

        # --- CÁLCULO DOS GANHOS ---
        tau_mf = tau_ml_input
        P1 = exp(-sampling_time / tau_mf)
        
        # Observador Ajustado (0.1 para menos agressividade)
        alpha_obs = 0.1
        
        # Equações de Bezout
        s0 = (1.0 - P1 - alpha_obs - a1) / b0
        s1 = (a1 + P1 * alpha_obs) / b0
        t0 = ((1.0 - P1) * (1.0 - alpha_obs)) / b0
        
        # Parâmetros PID Equivalentes
        kc = -s1
        ki_rst = t0 
        ti = (kc * sampling_time) / ki_rst if abs(ki_rst) > 1e-9 else 9999.0

        with st.expander("Diagnóstico", expanded=True):
            if msg_aviso: st.warning(msg_aviso)
            st.write(f"**Ganhos RST:** $t_0={t0:.4f}, s_0={s0:.4f}, s_1={s1:.4f}$")
            
            if "PID" in pid_structure:
                st.info(f"Modo PID Selecionado ({pid_structure}).")

        # --- 3. PREPARAÇÃO (RESET COMPLETO) ---
        process_output = zeros(samples_number)
        delta_control_signal = zeros(samples_number)
        manipulated_variable_1 = zeros(samples_number)
        e = zeros(samples_number)

        # Referência
        instant_sample_2 = get_sample_position(sampling_time, samples_number, change_ref_instant2)
        instant_sample_3 = get_sample_position(sampling_time, samples_number, change_ref_instant3)
        reference_input = rst_single_reference * ones(samples_number)
        reference_input[instant_sample_2:instant_sample_3] = rst_siso_multiple_reference2
        reference_input[instant_sample_3:] = rst_siso_multiple_reference3
        
        set_session_controller_parameter('reference_input', reference_input.tolist())
        set_session_controller_parameter('control_signal_1', dict())
        control_signal_1 = get_session_variable('control_signal_1')
        set_session_controller_parameter('process_output_sensor', dict())
        process_output_sensor = get_session_variable('process_output_sensor')

        # --- 4. LOOP DE CONTROLE ---
        
        # === LIMPEZA CRÍTICA DE BUFFER SERIAL ===
        if arduinoData and arduinoData.is_open:
            arduinoData.reset_input_buffer()
            arduinoData.reset_output_buffer()
            try:
                sendToArduino(arduinoData, "0")
            except Exception:
                pass # Ignora erro no reset inicial
            time.sleep(0.2) 
            if arduinoData.is_open:
                arduinoData.reset_input_buffer() 
        else:
            return st.error("Erro Crítico: Porta Serial desconectada antes do início.")
        # ========================================

        start_time = time.time()
        kk = 0
        progress_text = "Controlador Ativo..."
        my_bar = st.progress(0, text=progress_text)
        
        # Filtro Mínimo
        alpha_filter = 0.1 
        last_valid_output = 0.0
        
        while kk < samples_number:
            current_time = time.time()
            if current_time - start_time > sampling_time:
                start_time = current_time 
                
                # Leitura Rápida e Segura
                try:
                    if arduinoData.is_open:
                        raw_val = readFromArduino(arduinoData)
                        if raw_val is not None:
                            curr_read = float(raw_val)
                        else:
                            curr_read = last_valid_output
                    else:
                        raise serial.SerialException("Porta Fechada")
                except (ValueError, serial.SerialException):
                    curr_read = last_valid_output
                except Exception:
                    curr_read = last_valid_output
                
                # Filtro Mínimo
                if kk > 0:
                    filtered_val = alpha_filter * process_output[kk-1] + (1.0 - alpha_filter) * curr_read
                else:
                    filtered_val = curr_read
                
                process_output[kk] = filtered_val
                last_valid_output = filtered_val 
                
                e[kk] = reference_input[kk] - process_output[kk]

                if kk <= A_order:
                    manipulated_variable_1[kk] = 0.0
                    try:
                        if arduinoData.is_open: sendToArduino(arduinoData, "0")
                    except: pass
                else:
                    ts = sampling_time
                    delta_u = 0.0
                    
                    y_k = process_output[kk]
                    y_k1 = process_output[kk-1]
                    ref_k = reference_input[kk]
                    e_k = e[kk]
                    e_k1 = e[kk-1]

                    # --- SELEÇÃO DE ESTRUTURA ---
                    # CORREÇÃO: ki_rst JÁ CONTÉM O TEMPO DE AMOSTRAGEM (ki_rst = t0 = Ki*Ts)
                    # NÃO MULTIPLICAR POR ts NOVAMENTE!

                    if pid_structure == 'RST Incremental Puro':
                        delta_u = (t0 * ref_k) - (s0 * y_k) - (s1 * y_k1)

                    elif pid_structure == 'I + PD':
                        # I no erro, PD na saída
                        termo_I = ki_rst * e_k  # REMOVIDO * ts
                        termo_P = kc * (y_k1 - y_k) 
                        delta_u = termo_I + termo_P
                    
                    elif pid_structure == 'PI + D':
                        delta_u = kc * (e_k - e_k1) + (ki_rst * e_k) # REMOVIDO * ts

                    elif pid_structure == 'PID Ideal':
                        delta_u = kc * (e_k - e_k1) + (ki_rst * e_k) # REMOVIDO * ts
                        
                    elif pid_structure == 'PID Paralelo':
                        delta_u = kc * (e_k - e_k1) + (ki_rst * e_k) # REMOVIDO * ts

                    # --- ANTI-WINDUP ---
                    u_prev = manipulated_variable_1[kk-1]
                    u_candidate = u_prev + delta_u
                    
                    if u_candidate > max_pot:
                        u_sat = max_pot
                        if delta_u > 0: delta_u = 0 
                    elif u_candidate < min_pot:
                        u_sat = min_pot
                        if delta_u < 0: delta_u = 0
                    else:
                        u_sat = u_candidate
                    
                    # UNIFICAÇÃO DA LÓGICA DE SATURAÇÃO (RST e PID usam u_sat da mesma forma)
                    if pid_structure == 'RST Incremental Puro':
                         manipulated_variable_1[kk] = u_sat 
                    else:
                         manipulated_variable_1[kk] = u_prev + delta_u 

                    # Saturação Final
                    manipulated_variable_1[kk] = max(min_pot, min(manipulated_variable_1[kk], max_pot))

                    serial_data = f"{manipulated_variable_1[kk]:.4f}\r"
                    try:
                        if arduinoData.is_open:
                            sendToArduino(arduinoData, serial_data)
                    except serial.SerialException:
                        st.error("Perda de conexão serial durante o envio.")
                        break

                timestamp_str = str(datetime.now())
                process_output_sensor[timestamp_str] = float(process_output[kk])
                control_signal_1[timestamp_str] = float(manipulated_variable_1[kk])
                
                kk += 1
                my_bar.progress(kk / samples_number, text=progress_text)

        if arduinoData.is_open:
            try: sendToArduino(arduinoData, "0")
            except: pass
            
        st.success("Ensaio Finalizado.")

    except Exception as e:
        err_msg = traceback.format_exc()
        try:
            if 'arduinoData' in locals() and arduinoData.is_open: sendToArduino(arduinoData, "0")
        except: pass
        st.error(f"ERRO: {e}")
        st.code(err_msg)

def rstControlProcessAdaptiveSISO(tau_ml_input:float, 
                                  pid_structure:str, 
                                  a1_initial:float, 
                                  b0_initial:float, 
                                  p0_initial:float, 
                                  rst_single_reference:float,
                                  rst_siso_multiple_reference2:float, 
                                  rst_siso_multiple_reference3:float,
                                  change_ref_instant2 = 1, 
                                  change_ref_instant3 = 1):
    
    # --- 1. VALIDAÇÕES E SETUP ---
    if 'arduinoData' not in st.session_state.connected:
        return st.error('FALHA: Arduino não conectado.')

    try:
        sampling_time = float(get_session_variable('sampling_time'))
        samples_number = int(get_session_variable('samples_number'))
        max_pot = float(get_session_variable('saturation_max_value'))
        min_pot = float(get_session_variable('saturation_min_value'))
    except Exception as e:
        return st.error(f"FALHA CONFIG: {e}")

    st.info("Iniciando RST Adaptativo (Versão Estável Restaurada)...")

    try:
        arduinoData = st.session_state.connected['arduinoData']
        
        # --- PARÂMETROS ADAPTATIVOS ---
        # Garante que a1 inicial seja negativo (estável)
        if a1_initial > 0: a1_initial = -abs(a1_initial)
        
        theta = array([a1_initial, b0_initial])
        
        P_cov = eye(2) * p0_initial 
        # Lambda de 0.98: Adapta rápido mas sem ser nervoso
        rls_lambda = 0.98 

        # --- DEFINIÇÃO DA GAIOLA DE SEGURANÇA (BOUNDS) ---
        a1_min = -0.999 
        a1_max = -0.001
        
        b0_nominal = abs(b0_initial)
        # Limites amplos mas seguros (evita b0 -> 0 ou infinito)
        b0_min = max(0.0001, b0_nominal * 0.1)
        b0_max = max(0.001, b0_nominal * 5.0)

        tau_mf = tau_ml_input
        P1 = exp(-sampling_time / tau_mf)
        alpha_obs = 0.05
        
        # Zona Morta Simples
        dead_zone_threshold = 1.0 
        
        # --- BUFFERS E VARIÁVEIS ---
        process_output = zeros(samples_number)
        manipulated_variable_1 = zeros(samples_number)
        e = zeros(samples_number)
        
        a1_est_hist = zeros(samples_number)
        b0_est_hist = zeros(samples_number)

        # Referência
        instant_sample_2 = get_sample_position(sampling_time, samples_number, change_ref_instant2)
        instant_sample_3 = get_sample_position(sampling_time, samples_number, change_ref_instant3)
        reference_input = rst_single_reference * ones(samples_number)
        reference_input[instant_sample_2:instant_sample_3] = rst_siso_multiple_reference2
        reference_input[instant_sample_3:] = rst_siso_multiple_reference3
        
        set_session_controller_parameter('reference_input', reference_input.tolist())
        set_session_controller_parameter('control_signal_1', dict())
        control_signal_1 = get_session_variable('control_signal_1')
        set_session_controller_parameter('process_output_sensor', dict())
        process_output_sensor = get_session_variable('process_output_sensor')

        # --- LIMPEZA E INICIALIZAÇÃO SEGURA ---
        if arduinoData and arduinoData.is_open:
            arduinoData.reset_input_buffer()
            arduinoData.reset_output_buffer()
            try:
                sendToArduino(arduinoData, "0")
            except Exception:
                pass
            time.sleep(0.2)
            if arduinoData.is_open: arduinoData.reset_input_buffer()
        else:
            return st.error("Erro Crítico: Porta Serial desconectada.")

        start_time = time.time()
        kk = 0
        progress_text = "Adaptando Controlador..."
        my_bar = st.progress(0, text=progress_text)
        
        # Filtro de leitura leve (sem atraso)
        alpha_filter = 0.2
        last_valid_output = 0.0

        # Ganhos iniciais
        s0, s1, t0 = 0.0, 0.0, 0.0
        kc, ki_rst = 0.0, 0.0

        # Limites de Segurança de Ganho
        MAX_GAIN_LIMIT = 50.0

        while kk < samples_number:
            current_time = time.time()
            if current_time - start_time > sampling_time:
                start_time = current_time 
                
                # Leitura Segura
                try:
                    if arduinoData.is_open:
                        raw_val = readFromArduino(arduinoData)
                        if raw_val is not None: curr_read = float(raw_val)
                        else: curr_read = last_valid_output
                    else:
                        raise serial.SerialException("Porta Fechada")
                except (ValueError, serial.SerialException):
                    curr_read = last_valid_output
                except Exception:
                    curr_read = last_valid_output
                
                if kk > 0:
                    y_k = alpha_filter * process_output[kk-1] + (1.0 - alpha_filter) * curr_read
                else:
                    y_k = curr_read
                
                process_output[kk] = y_k
                last_valid_output = y_k
                e[kk] = reference_input[kk] - y_k

                # --- LÓGICA ADAPTATIVA ---
                
                if kk >= 2:
                    phi = array([-process_output[kk-1], manipulated_variable_1[kk-1]])
                    phi = phi.reshape(2, 1) 
                    y_hat = dot(theta, phi)
                    erro_pred = y_k - y_hat
                    
                    # ZONA MORTA (Sem Leakage)
                    is_saturated = (manipulated_variable_1[kk-1] >= max_pot) or (manipulated_variable_1[kk-1] <= min_pot)
                    if abs(erro_pred) > dead_zone_threshold and not is_saturated:
                        numerador = dot(P_cov, phi)
                        denominador = rls_lambda + dot(dot(phi.T, P_cov), phi)
                        if denominador < 1e-3: denominador = 1e-3
                        K_rls = numerador / denominador
                        
                        theta = theta + (K_rls * erro_pred).flatten()
                        
                        termo_correcao = dot(dot(K_rls, phi.T), P_cov)
                        P_cov = (P_cov - termo_correcao) / rls_lambda
                    
                    # --- PROJEÇÃO DE PARÂMETROS (GAIOLA) ---
                    # Sem filtro theta_smooth! Atualização direta mas presa nos limites.
                    
                    a1_est = clip(theta[0], a1_min, a1_max)
                    b0_est = clip(theta[1], b0_min, b0_max)
                    
                    # Atualiza o vetor theta para não derivar fora dos limites
                    theta[0] = a1_est
                    theta[1] = b0_est
                    
                    a1_est_hist[kk] = a1_est
                    b0_est_hist[kk] = b0_est
                    
                    # Recálculo Ganhos
                    s0 = (1.0 - P1 - alpha_obs - a1_est) / b0_est
                    s1 = (a1_est + P1 * alpha_obs) / b0_est
                    t0 = ((1.0 - P1) * (1.0 - alpha_obs)) / b0_est
                    
                    # TRAVA DE GANHOS
                    s0 = clip(s0, -MAX_GAIN_LIMIT, MAX_GAIN_LIMIT)
                    s1 = clip(s1, -MAX_GAIN_LIMIT, MAX_GAIN_LIMIT)
                    t0 = clip(t0, -MAX_GAIN_LIMIT, MAX_GAIN_LIMIT)
                    
                    y_k1 = process_output[kk-1]
                    ref_k = reference_input[kk]
                    e_k = e[kk]
                    e_k1 = e[kk-1]
                    ts = sampling_time
                    
                    kc = -s1
                    ki_rst = t0
                    
                    # --- LEI DE CONTROLE CORRIGIDA PARA TODAS AS ESTRUTURAS ---
                    
                    if pid_structure == 'RST Incremental Puro':
                        delta_u = (t0 * ref_k) - (s0 * y_k) - (s1 * y_k1)
                    
                    elif pid_structure == 'I + PD':
                        # P atua na saída (y) -> Suave
                        termo_I = ki_rst * e_k 
                        termo_P = kc * (y_k1 - y_k)
                        delta_u = termo_I + termo_P
                    
                    elif pid_structure == 'PI + D':
                        # PI no erro (Pode ser agressivo se Kc alto)
                        # Sugestão: Manter P no erro para seguir a teoria, mas limitar Delta U depois
                        delta_u = kc * (e_k - e_k1) + (ki_rst * e_k)

                    elif pid_structure == 'PID Ideal':
                        delta_u = kc * (e_k - e_k1) + (ki_rst * e_k)
                        
                    elif pid_structure == 'PID Paralelo':
                        delta_u = kc * (e_k - e_k1) + (ki_rst * e_k)
                    
                    else:
                        delta_u = (t0 * ref_k) - (s0 * y_k) - (s1 * y_k1)

                else:
                    delta_u = 0.0
                
                # --- APLICAÇÃO DO CONTROLE ---
                if kk > 0:
                    u_prev = manipulated_variable_1[kk-1]
                else:
                    u_prev = 0.0
                
                # ZONA MORTA DE ATUAÇÃO
                if abs(delta_u) < 0.05: delta_u = 0.0
                    
                delta_u_max = (max_pot - min_pot) * 0.15 
                delta_u = clip(delta_u, -delta_u_max, delta_u_max)
                u_candidate = u_prev + delta_u
                
                if u_candidate > max_pot:
                    u_sat = max_pot
                    if delta_u > 0: delta_u = 0
                elif u_candidate < min_pot:
                    u_sat = min_pot
                    if delta_u < 0: delta_u = 0
                else:
                    u_sat = u_candidate
                
                # UNIFICAÇÃO DA LÓGICA DE SATURAÇÃO
                if pid_structure == 'RST Incremental Puro':
                    manipulated_variable_1[kk] = u_sat
                else:
                    manipulated_variable_1[kk] = u_prev + delta_u
                
                manipulated_variable_1[kk] = max(min_pot, min(manipulated_variable_1[kk], max_pot))
                
                serial_data = f"{manipulated_variable_1[kk]:.4f}\r"
                try:
                    if arduinoData.is_open: sendToArduino(arduinoData, serial_data)
                except serial.SerialException:
                    st.error("Perda de conexão serial.")
                    break

                timestamp_str = str(datetime.now())
                process_output_sensor[timestamp_str] = float(process_output[kk])
                control_signal_1[timestamp_str] = float(manipulated_variable_1[kk])
                
                kk += 1
                my_bar.progress(kk / samples_number, text=progress_text)

        if arduinoData.is_open:
            try: sendToArduino(arduinoData, "0")
            except: pass
        st.success("Ensaio Adaptativo Finalizado.")

    except Exception as e:
        err_msg = traceback.format_exc()
        try:
            if 'arduinoData' in locals() and arduinoData.is_open: sendToArduino(arduinoData, "0")
        except: pass
        st.error(f"ERRO ADAPTATIVO: {e}")
        st.code(err_msg)