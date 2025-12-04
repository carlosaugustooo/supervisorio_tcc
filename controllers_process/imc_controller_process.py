import streamlit as st
import time
import numpy as np
from formatterInputs import *
from connections import *
from datetime import datetime
from session_state import *
from controllers_process.validations_functions import *

# ==============================================================================
# FUNÇÕES AUXILIARES DE MODELAGEM
# ==============================================================================

def poly_mul(p1, p2):
    return np.convolve(p1, p2)

def poly_add(p1, p2):
    l1, l2 = len(p1), len(p2)
    if l1 > l2: p2 = np.pad(p2, (l1-l2, 0), 'constant')
    elif l2 > l1: p1 = np.pad(p1, (l2-l1, 0), 'constant')
    return p1 + p2

def c2d_tustin_numpy(num_s, den_s, Ts):
    num_s = np.array(num_s, dtype=float)
    den_s = np.array(den_s, dtype=float)
    n = max(len(num_s), len(den_s)) - 1
    
    num_s = np.pad(num_s, (n + 1 - len(num_s), 0), 'constant')
    den_s = np.pad(den_s, (n + 1 - len(den_s), 0), 'constant')
    
    c = 2.0 / Ts
    num_z = np.zeros(n + 1)
    den_z = np.zeros(n + 1)
    
    for i in range(n + 1):
        k = n - i 
        term_minus = np.array([1.0])
        for _ in range(k): term_minus = np.convolve(term_minus, [1, -1])
        term_plus = np.array([1.0])
        for _ in range(n - k): term_plus = np.convolve(term_plus, [1, 1])
            
        factor = c**k
        term_poly = np.convolve(term_minus, term_plus) * factor
        
        num_z = poly_add(num_z, num_s[i] * term_poly)
        den_z = poly_add(den_z, den_s[i] * term_poly)
        
    if abs(den_z[0]) > 1e-12:
        num_z = num_z / den_z[0]
        den_z = den_z / den_z[0]
        
    return num_z, den_z

def get_pade_model_numpy(num_str, den_str, delay_time, sampling_time):
    try:
        num_plant = np.array([float(x) for x in num_str.split(',')])
        den_plant = np.array([float(x) for x in den_str.split(',')])
        
        if delay_time > 0:
            d = delay_time
            num_pade = np.array([d**2, -6*d, 12])
            den_pade = np.array([d**2,  6*d, 12])
            num_total = poly_mul(num_plant, num_pade)
            den_total = poly_mul(den_plant, den_pade)
        else:
            num_total = num_plant
            den_total = den_plant
            
        return c2d_tustin_numpy(num_total, den_total, sampling_time)
    except:
        return np.array([0.0]), np.array([1.0])

# ==============================================================================
# CONTROLADOR IMC (COM LEITURA CORRIGIDA E SALVAMENTO DE DADOS)
# ==============================================================================

def imcControlProcessSISO(transfer_function_type:str, num_coeff:str, den_coeff:str,
                          imc_mr_tau_mf1:float, 
                          pid_structure:str, 
                          imc_multiple_reference1:float, imc_multiple_reference2:float, imc_multiple_reference3:float,
                          change_ref_instant2 = 1, change_ref_instant3 = 1):
    
    if num_coeff == '': return st.error('Coeficientes inválidos.')

    sampling_time = get_session_variable('sampling_time')
    samples_number = get_session_variable('samples_number')
    
    if 'arduinoData' not in st.session_state.connected:
        return st.warning('Arduino não conectado!')
    arduinoData = st.session_state.connected['arduinoData']

    # --- 1. CONFIGURAÇÃO ---
    process_output = np.zeros(samples_number)
    manipulated_variable_1 = np.zeros(samples_number)
    
    inst2 = get_sample_position(sampling_time, samples_number, change_ref_instant2)
    inst3 = get_sample_position(sampling_time, samples_number, change_ref_instant3)
    reference_input = imc_multiple_reference1 * np.ones(samples_number)
    reference_input[inst2:inst3] = imc_multiple_reference2
    reference_input[inst3:] = imc_multiple_reference3
    
    set_session_controller_parameter('reference_input', reference_input.tolist())

    max_pot = get_session_variable('saturation_max_value') or 100.0
    min_pot = get_session_variable('saturation_min_value') or 0.0

    # --- 2. SINTONIA IMC ---
    try:
        n_temp = [float(x) for x in num_coeff.split(',')]
        d_temp = [float(x) for x in den_coeff.split(',')]
        kp_val = n_temp[-1] if len(n_temp) > 0 else 1.0
        tau_val = d_temp[0] if len(d_temp) > 0 else 1.0
        theta_val = 2.0 
    except:
        kp_val, tau_val, theta_val = 5.43, 123.0, 2.0

    tau_mf = imc_mr_tau_mf1
    denom_kc = 2 * tau_mf + theta_val
    if denom_kc == 0: denom_kc = 0.001
    
    kc_imc = (1.0 / kp_val) * ((2 * tau_val + theta_val) / denom_kc)
    ki_imc = kc_imc / (tau_val + theta_val / 2.0)
    
    denom_td = 2 * tau_val + theta_val
    if denom_td == 0: denom_td = 0.001
    kd_imc = kc_imc * ((tau_val * theta_val) / denom_td)

    # --- 3. MODELAGEM (Padé) ---
    if transfer_function_type == 'Continuo':
        B_coeff, A_coeff = get_pade_model_numpy(num_coeff, den_coeff, theta_val, sampling_time)
    else:
        A_coeff, B_coeff = convert_tf_2_discrete(num_coeff, den_coeff, transfer_function_type)

    # --- 4. DIAGNÓSTICO ---
    with st.expander("Diagnóstico IMC (Método TCC + Padé)", expanded=True):
        c1, c2 = st.columns(2)
        with c1:
            st.markdown(f"**Planta:** Kp={kp_val}, $\\tau$={tau_val}")
            st.markdown(f"**Filtro:** $\\tau_{{mf}}$={tau_mf}")
        with c2:
            st.latex(f"K_c = {kc_imc:.4f}")
            st.latex(f"K_i = {ki_imc:.4f}")
            st.latex(f"K_d = {kd_imc:.4f}")
        
        pid_clean = pid_structure.strip()

    # --- 5. LOOP DE CONTROLE ---
    set_session_controller_parameter('control_signal_1', dict())
    control_signal_1 = get_session_variable('control_signal_1')
    set_session_controller_parameter('process_output_sensor', dict())
    process_output_sensor = get_session_variable('process_output_sensor')

    if arduinoData.is_open: 
        arduinoData.reset_input_buffer()
        sendToArduino(arduinoData, "0")
        
    start_time = time.time()
    kk = 0
    my_bar = st.progress(0, text="Executando IMC...")

    y_model_hist = np.zeros(samples_number)
    y_prev = 0.0
    u_prev = 0.0
    e_prev = 0.0
    e_prev2 = 0.0
    
    # Placeholder para visualização
    metrics_ph = st.empty()

    while kk < samples_number:
        current_time = time.time()
        if current_time - start_time > sampling_time:
            start_time = current_time
            
            # --- A. LEITURA OTIMIZADA ---
            y_k = y_prev 
            
            if arduinoData.is_open:
                raw_val = None
                while arduinoData.in_waiting > 0:
                    try:
                        line = arduinoData.readline().decode('utf-8').strip()
                        if line: raw_val = float(line)
                    except: 
                        pass 
                
                if raw_val is not None:
                    y_k = raw_val
            
            process_output[kk] = y_k
            y_prev = y_k

            if kk < 3: 
                manipulated_variable_1[kk] = 0.0
                y_model_hist[kk] = 0.0
                if arduinoData.is_open: sendToArduino(arduinoData, '0')
            else:
                ref_k = reference_input[kk]
                
                y_k1 = process_output[kk-1]
                y_k2 = process_output[kk-2]
                e_k = ref_k - y_k
                
                # --- B. CÁLCULO DO CONTROLE ---
                delta_u = 0.0
                
                if pid_clean == 'IMC Padrão':
                    ay_term = 0.0
                    for i in range(1, len(A_coeff)):
                        idx = kk - i
                        if idx >= 0: ay_term += A_coeff[i] * y_model_hist[idx]
                    
                    bu_term = 0.0
                    for i in range(len(B_coeff)):
                        idx = kk - 1 - i 
                        if idx >= 0: bu_term += B_coeff[i] * manipulated_variable_1[idx]
                    
                    y_model_current = bu_term - ay_term
                    y_model_hist[kk] = y_model_current
                    
                    delta_u = (ki_imc * sampling_time * e_k) - \
                              (kc_imc * (y_k - y_k1)) - \
                              ((kd_imc/sampling_time) * (y_k - 2*y_k1 + y_k2))
                    
                else:
                    Ki_d = ki_imc * sampling_time
                    Kd_d = kd_imc / sampling_time if sampling_time > 0 else 0
                    
                    if pid_clean == 'I + PD':
                        delta_u = (Ki_d * e_k) - (kc_imc * (y_k - y_k1)) - (Kd_d * (y_k - 2*y_k1 + y_k2))
                    elif pid_clean == 'PI + D':
                        delta_u = (kc_imc * (e_k - e_prev)) + (Ki_d * e_k) - (Kd_d * (y_k - 2*y_k1 + y_k2))
                    elif pid_clean == 'PID Paralelo':
                        delta_u = (kc_imc * (e_k - e_prev)) + (Ki_d * e_k) + (Kd_d * (e_k - 2*e_prev + e_prev2))
                    elif pid_clean == 'PID Ideal':
                        if abs(kc_imc) > 1e-9:
                            term_p = (e_k - e_prev)
                            term_i = (ki_imc/kc_imc)*sampling_time * e_k
                            term_d = (kd_imc/kc_imc/sampling_time) * (e_k - 2*e_prev + e_prev2)
                            delta_u = kc_imc * (term_p + term_i + term_d)
                
                u_calc = u_prev + delta_u
                u_final = max(min_pot, min(u_calc, max_pot))
                
                manipulated_variable_1[kk] = u_final
                u_prev = u_final
                e_prev2 = e_prev
                e_prev = e_k

            # --- C. ENVIO E LOGS ---
            if arduinoData.is_open:
                sendToArduino(arduinoData, f"{manipulated_variable_1[kk]:.4f}\r")
                
            timestamp = str(datetime.now())
            process_output_sensor[timestamp] = float(process_output[kk])
            control_signal_1[timestamp] = float(manipulated_variable_1[kk])
            
            # --- D. VISUALIZAÇÃO EM TEMPO REAL ---
            if kk % 2 == 0:
                with metrics_ph.container():
                    c1, c2 = st.columns(2)
                    c1.metric("Nível Real (cm)", f"{process_output[kk]:.2f}")
                    c2.metric("Sinal Controle (V)", f"{manipulated_variable_1[kk]:.2f}")

            kk += 1
            my_bar.progress(kk/samples_number)

    # --- FINALIZAÇÃO E SALVAMENTO (CRUCIAL PARA OS DADOS NÃO SUMIREM) ---
    if arduinoData.is_open:
        for _ in range(3):
            sendToArduino(arduinoData, '0')
            time.sleep(0.05)

    # 1. Salva Parâmetros de Sintonia
    st.session_state.controller_parameters['imc_calculated_params'] = {
        'Kc': float(kc_imc),
        'Ki': float(ki_imc),
        'Kd': float(kd_imc)
    }

    # 2. Calcula e Salva Índices de Desempenho
    # Importa usando os nomes CORRETOS definidos no seu performace_metrics.py
    try:
        from controllers_process.performace_metrics import integrated_absolute_error, total_variation_control
        
        final_iae = integrated_absolute_error()
        final_tvc = total_variation_control('control_signal_1') # Passa o nome da chave do sinal
        
        st.session_state.controller_parameters['iae_metric'] = final_iae
        st.session_state.controller_parameters['tvc_1_metric'] = final_tvc
    except Exception as e:
        st.error(f"Erro ao calcular métricas: {e}")
    
    st.success("Teste IMC Finalizado.")