import streamlit as st
from formatterInputs import *
import numpy as np
from connections import *
from datetime import datetime   
from session_state import get_session_variable, set_session_controller_parameter
from controllers_process.validations_functions import *
import time

# ==============================================================================
# FUNÇÕES MATEMÁTICAS (CORE)
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
    except Exception as e:
        st.error(f"Erro no cálculo do modelo: {e}")
        return np.array([0.0]), np.array([1.0])

# ==============================================================================
# LÓGICA GMV (DIOFANTINA)
# ==============================================================================

def create_P1_poly(ne, ns):
    P1 = np.zeros((ne + ns + 2, 1))
    P1[0] = 1.0 
    return P1

def E_S_poly_calculation(ne, ns, Am1, P):
    if len(Am1) < ne + 2:
        Am1 = np.pad(Am1, (0, ne + 2 - len(Am1)), 'constant')
    mat_Scoef1 = np.vstack((np.zeros((ne+1, ns+1)), np.eye(ns+1)))
    mat_EAcoef1 = np.zeros((ne + ns + 2, ne+1))
    for k in range(ne+1):
        col_len = min(len(Am1), mat_EAcoef1.shape[0] - k)
        mat_EAcoef1[k : k + col_len, k] = Am1[:col_len].T
    mat_SEcoef1 = np.concatenate((mat_EAcoef1, mat_Scoef1), axis=1)  
    if np.linalg.cond(mat_SEcoef1) > 1e12:
        mat_SEcoef1 += np.eye(mat_SEcoef1.shape[0]) * 1e-6
    try:
        EScoef1_array = np.linalg.solve(mat_SEcoef1, P)
    except:
        EScoef1_array = np.linalg.lstsq(mat_SEcoef1, P, rcond=None)[0]
    return EScoef1_array[0:ne+1].T[0], EScoef1_array[ne+1:].T[0]

def r_poly_calculation(Bm1, epoly1, q0):
    BE_poly = np.convolve(Bm1, epoly1)
    if len(BE_poly) == 0: BE_poly = np.array([0.0])
    BE_poly[0] += q0
    return BE_poly

# ==============================================================================
# PROCESSO PRINCIPAL (GMV INCREMENTAL PADRONIZADO)
# ==============================================================================

def gmvControlProcessSISO(transfer_function_type: str, num_coeff: str, den_coeff: str,
                          gmv_q01: float,
                          pid_structure: str, 
                          gmv_multiple_reference1: float, gmv_multiple_reference2: float, gmv_multiple_reference3: float,
                          change_ref_instant2=1, change_ref_instant3=1):
    
    if num_coeff == '': return st.error('Coeficientes inválidos.')

    sampling_time = get_session_variable('sampling_time')
    samples_number = get_session_variable('samples_number')
    
    # Validação de Conexão
    if 'arduinoData' not in st.session_state.connected:
        return st.warning('Arduino não conectado!')
    arduinoData = st.session_state.connected['arduinoData']

    # --- 1. CONFIGURAÇÃO ---
    process_output = np.zeros(samples_number)
    delta_control_signal = np.zeros(samples_number) # Delta U (Memória do GMV)
    manipulated_variable_1 = np.zeros(samples_number) # U Total

    try:
        instant_sample_2 = int(change_ref_instant2 / sampling_time)
        instant_sample_3 = int(change_ref_instant3 / sampling_time)
    except:
        instant_sample_2 = int(samples_number * 0.33)
        instant_sample_3 = int(samples_number * 0.66)

    instant_sample_2 = max(0, min(instant_sample_2, samples_number))
    instant_sample_3 = max(0, min(instant_sample_3, samples_number))

    reference_input = float(gmv_multiple_reference1) * np.ones(samples_number)
    reference_input[instant_sample_2:instant_sample_3] = float(gmv_multiple_reference2)
    reference_input[instant_sample_3:] = float(gmv_multiple_reference3)
    
    set_session_controller_parameter('reference_input', reference_input.tolist())

    min_pot = get_session_variable('saturation_min_value')
    max_pot = get_session_variable('saturation_max_value')
    if min_pot is None: min_pot = 0.0
    if max_pot is None: max_pot = 5.0

    # --- 2. MODELAGEM ---
    if transfer_function_type == 'Continuo':
        B_coeff, A_coeff = get_pade_model_numpy(num_coeff, den_coeff, 2.0, sampling_time)
    else:
        A_coeff, B_coeff = convert_tf_2_discrete(num_coeff, den_coeff, transfer_function_type)

    # --- 3. CÁLCULO GMV INCREMENTAL (Polinomial) ---
    # Para garantir erro zero, incluímos um integrador no modelo de projeto.
    # Modelo: A_inc(z) = A(z) * (1 - z^-1)
    # Isso faz o controlador calcular variações (Delta U), que somadas viram Ação Integral.
    
    poly_int = np.array([1.0, -1.0]) 
    A_design = poly_mul(A_coeff, poly_int) 

    d = 1
    na = len(A_design) - 1
    ns = max(na, 2) 
    ne = d - 1 

    # Diofantina: P = E * A_design + z^-d * S
    P1 = create_P1_poly(ne, ns)
    e_poly_1, s_poly_1 = E_S_poly_calculation(ne, ns, A_design, P1)
    
    # R(z) = E*B + Q
    r_poly_1 = r_poly_calculation(B_coeff, e_poly_1, gmv_q01)
    
    # T0 (Rastreamento): No modo incremental, T(1) = S(1) é suficiente para erro zero.
    t01 = np.sum(s_poly_1)
    
    r0 = r_poly_1[0] if abs(r_poly_1[0]) > 1e-9 else 1.0

    # --- 4. EXIBIÇÃO DE GANHOS (DIAGNÓSTICO) ---
    inv_r0 = 1.0 / r0
    
    # PID Equivalente para exibição
    s0, s1, s2 = s_poly_1[0], 0.0, 0.0
    if len(s_poly_1) > 1: s1 = s_poly_1[1]
    if len(s_poly_1) > 2: s2 = s_poly_1[2]
    
    kd_disp = inv_r0 * s2
    kp_disp = -inv_r0 * (s1 + 2*s2)
    ki_disp = inv_r0 * np.sum(s_poly_1)

    with st.expander(f"Diagnóstico GMV Incremental (T0={t01:.3f})", expanded=True):
        c1, c2 = st.columns(2)
        with c1: st.latex(f"K_c \\approx {kp_disp:.3f}")
        with c2: st.latex(f"K_i \\approx {ki_disp:.3f}")

    # --- 5. LOOP DE CONTROLE ---
    set_session_controller_parameter('control_signal_1', dict())
    control_signal_1 = get_session_variable('control_signal_1')
    set_session_controller_parameter('process_output_sensor', dict())
    process_output_sensor = get_session_variable('process_output_sensor')

    start_time = time.time()
    kk = 0
    pid_clean = pid_structure.strip()
    
    if arduinoData.is_open: 
        arduinoData.reset_input_buffer()
        sendToArduino(arduinoData, "0")
    
    my_bar = st.progress(0, text="Executando GMV...")
    metrics_ph = st.empty() 
    y_prev = 0.0 
    
    # Memórias de Controle
    e_prev = 0.0
    e_prev2 = 0.0
    ref_smooth = 0.0
    alpha_smooth = 0.1

    while kk < samples_number:
        current_time = time.time()
        if current_time - start_time > sampling_time:
            start_time = current_time
            
            # --- LEITURA ROBUSTA (Limpa Buffer) ---
            y_k = y_prev 
            if arduinoData.is_open:
                raw_val = None
                while arduinoData.in_waiting > 0:
                    try:
                        line = arduinoData.readline().decode('utf-8').strip()
                        if line: raw_val = float(line)
                    except: pass 
                if raw_val is not None: y_k = raw_val
            
            process_output[kk] = y_k
            y_prev = y_k 

            # --- REFERÊNCIA ---
            target = reference_input[kk]
            ref_smooth = (1 - alpha_smooth) * ref_smooth + alpha_smooth * target

            # --- CÁLCULO ---
            u_raw = 0.0 
            delta_u = 0.0

            if kk < 3: 
                u_raw = 0.0
                if arduinoData.is_open: sendToArduino(arduinoData, '0')
            else:
                if pid_clean == 'GMV Padrão':
                    # --- LÓGICA GMV INCREMENTAL POLINOMIAL ---
                    # Equação: r0 * Delta_u(t) = T*w - S*y - R_rest * Delta_u(t-1)
                    
                    term_T = t01 * ref_smooth
                    
                    s_term = 0.0
                    for i in range(len(s_poly_1)):
                        idx = kk - i
                        if idx >= 0: s_term += s_poly_1[i] * process_output[idx]

                    r_past_term = 0.0
                    for i in range(1, len(r_poly_1)):
                        idx = kk - i
                        # Usa a variação de controle passada
                        if idx >= 0: r_past_term += r_poly_1[i] * delta_control_signal[idx]
                    
                    # Calcula o DELTA (Variação)
                    delta_u = (term_T - s_term - r_past_term) / r0
                    delta_control_signal[kk] = delta_u
                    
                    # Soma ao anterior (Integrador)
                    u_raw = manipulated_variable_1[kk-1] + delta_u

                else:
                    # --- LÓGICA PID (MANUAL) ---
                    e_k = ref_smooth - y_k 
                    y_k1 = process_output[kk-1]
                    y_k2 = process_output[kk-2]
                    
                    if pid_clean == 'I + PD':
                        delta_u = (ki_disp * e_k) - (kp_disp * (y_k - y_k1)) - (kd_disp * (y_k - 2*y_k1 + y_k2))
                    elif pid_clean == 'PI + D':
                        delta_u = (kp_disp * (e_k - e_prev)) + (ki_disp * e_k) - (kd_disp * (y_k - 2*y_k1 + y_k2))
                    elif pid_clean == 'PID Paralelo':
                        delta_u = (kp_disp * (e_k - e_prev)) + (ki_disp * e_k) + (kd_disp * (e_k - 2*e_prev + e_prev2))
                    elif pid_clean == 'PID Ideal':
                        if abs(kp_disp) > 1e-9:
                            term_p = (e_k - e_prev)
                            term_i = (ki_disp/kp_disp)*e_k
                            term_d = (kd_disp/kp_disp)*(e_k - 2*e_prev + e_prev2)
                            delta_u = kp_disp * (term_p + term_i + term_d)
                    
                    e_prev2 = e_prev
                    e_prev = e_k
                    
                    u_raw = manipulated_variable_1[kk-1] + delta_u

            # --- SATURAÇÃO ANTI-WINDUP ---
            # Corta valores negativos para 0 (tanque não enche com tensão negativa)
            if u_raw < 0: 
                u_final = 0.0
                # Zera memória para não acumular "dívida" negativa
                if pid_clean == 'GMV Padrão': delta_control_signal[kk] = 0.0 # Opcional: zera delta se travar
                u_raw = 0.0
            else:
                u_final = min(u_raw, max_pot)
                if u_raw > max_pot: u_raw = max_pot
            
            manipulated_variable_1[kk] = u_final 
            
            # --- ENVIO ---
            if arduinoData.is_open:
                sendToArduino(arduinoData, f"{u_final:.4f}\r")
            
            # --- LOGS ---
            ts_now = str(datetime.now())
            process_output_sensor[ts_now] = float(process_output[kk])
            control_signal_1[ts_now] = float(u_final)
            
            if kk % 2 == 0:
                with metrics_ph.container():
                    c1, c2 = st.columns(2)
                    c1.metric("Ref | Nível", f"{target:.1f} | {y_k:.1f}")
                    lbl = f"Calc: {u_raw:.2f}"
                    c2.metric("Controle", f"{u_final:.2f} V", help=lbl)
            
            kk += 1
            my_bar.progress(kk/samples_number)

    # --- FINALIZAÇÃO ---
    if arduinoData.is_open:
        for _ in range(3):
            sendToArduino(arduinoData, '0')
            time.sleep(0.05)

    st.session_state.controller_parameters['gmv_calculated_params'] = {
        'Rho (q0)': gmv_q01,
        'Kc': float(kp_disp),
        'Ki': float(ki_disp),
        'Kd': float(kd_disp)
    }

    try:
        from controllers_process.performace_metrics import integrated_absolute_error, total_variation_control
        st.session_state.controller_parameters['iae_metric'] = integrated_absolute_error()
        st.session_state.controller_parameters['tvc_1_metric'] = total_variation_control('control_signal_1') 
    except: pass

    st.success("Teste GMV Finalizado.")