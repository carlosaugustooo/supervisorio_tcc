import streamlit as st
import time
import numpy as np
import pandas as pd
from formatterInputs import *
from connections import *
from datetime import datetime
from session_state import *
from controllers_process.validations_functions import *

# ==============================================================================
# 1. CLASSES E FUNÇÕES AUXILIARES
# ==============================================================================

def get_pade_model_numpy(num_str, den_str, delay_time, sampling_time):
    # Função auxiliar mock caso precise
    return np.array([0.0]), np.array([1.0])

class OnlineIdentifier:
    """ 
    Algoritmo RLS para identificação em tempo real.
    Modelo: y(k) = -a1*y(k-1) + b0*u(k-d) 
    """
    def __init__(self, n_theta=2, lambda_factor=0.98):
        self.theta = np.zeros(n_theta) # [a1, b0]
        self.P = np.eye(n_theta) * 1000.0
        self.lambda_factor = lambda_factor

    def update(self, y_k, phi_k):
        y_hat = np.dot(phi_k, self.theta)
        error = y_k - y_hat
        
        p_phi = np.dot(self.P, phi_k)
        den = self.lambda_factor + np.dot(phi_k, p_phi)
        
        if abs(den) > 1e-10:
            K = p_phi / den
        else:
            K = np.zeros_like(self.theta)
        
        self.theta = self.theta + K * error
        self.P = (self.P - np.outer(K, phi_k) @ self.P) / self.lambda_factor
        
        return self.theta

# ==============================================================================
# 2. NÚCLEO DE SINTONIA (ALOCAÇÃO DE POLOS)
# ==============================================================================

def tuning_calc_notes(a1, b0, Tau_MF, Ts):
    """
    Calcula r1, s0, s1, t0 via Alocação de Polos.
    Am(z) = z - p1, com p1 = exp(-Ts/Tau_MF)
    """
    if Tau_MF < 0.05: Tau_MF = 0.05
    if abs(b0) < 1e-6: b0 = 1e-6 * np.sign(b0 if b0!=0 else 1)

    p1 = np.exp(-Ts / Tau_MF)
    
    # Solução da Diofantina para RST Incremental 1ª Ordem
    # R(z) = 1 + r1*z^-1 (incremental separado)
    # S(z) = s0 + s1*z^-1
    
    r1 = 1.0 - a1 - p1
    s0 = (a1 - r1 * (a1 - 1.0)) / b0
    s1 = (r1 * a1) / b0
    t0 = s0 + s1
    
    return r1, s0, s1, t0

def tuning_static_wrapper(Kp, Tau, Tau_MF, Ts, Theta):
    p_plant = np.exp(-Ts / Tau)
    a1 = -p_plant
    b0 = Kp * (1.0 - p_plant)
    return tuning_calc_notes(a1, b0, Tau_MF, Ts)

# ==============================================================================
# 3. CONSTRUÇÃO DOS POLINÔMIOS E GANHOS
# ==============================================================================

def build_polynomials(r1, s0, s1, t0):
    """
    Retorna os polinômios RST base e os ganhos PID equivalentes.
    A lógica de estrutura (Ideal, Paralelo, etc) será tratada no Loop de Controle.
    """
    # Polinômios Base para RST Puro
    R = np.array([1.0, r1])
    S = np.array([s0, s1, 0.0]) 
    T = np.array([t0, 0.0])

    # Cálculo de Ganhos PID Equivalentes (Baseado em PI discreto)
    # Comparação: S(z) = (Kc + Ki) - Kc*z^-1
    # s0 = Kc + Ki
    # s1 = -Kc
    Kc_equiv = -s1
    Ki_equiv = s0 + s1
    Kd_equiv = 0.0 # RST de 1ª ordem resulta em PI, sem D

    return R, S, T, (Kc_equiv, Ki_equiv, Kd_equiv)

# ==============================================================================
# 4. CONTROLADOR RST ADAPTATIVO
# ==============================================================================

def rstControlProcessAdaptiveSISO(tf_type_str, num_str, den_str,
                                  tau_ml_input:float, pid_structure:str,
                                  p0_initial:float,
                                  rst_single_reference:float, 
                                  rst_siso_multiple_reference2:float, 
                                  rst_siso_multiple_reference3:float,
                                  change_ref_instant2=20, change_ref_instant3=40):

    if 'arduinoData' not in st.session_state.connected: return st.error('Arduino desconectado.')
    arduinoData = st.session_state.connected['arduinoData']
    
    Ts = float(get_session_variable('sampling_time'))
    samples_number = int(get_session_variable('samples_number'))
    max_pot = float(get_session_variable('saturation_max_value') or 100.0)
    min_pot = float(get_session_variable('saturation_min_value') or 0.0)
    
    try:
        a1_initial = -0.9
        b0_initial = 0.1
        if num_str and den_str:
             b0_initial = float(num_str.split(',')[-1])
             den_vals = [float(x) for x in den_str.split(',')]
             if len(den_vals) > 1: a1_initial = den_vals[1] / den_vals[0]
    except:
        a1_initial = -0.9
        b0_initial = 0.1

    rls = OnlineIdentifier(n_theta=2)
    rls.theta = np.array([a1_initial, b0_initial])
    rls.P = np.eye(2) * p0_initial

    # Memórias
    hist_u = [0.0, 0.0, 0.0]
    hist_y = [0.0, 0.0]
    
    # Memórias para PID Estruturado
    e_prev = 0.0
    e_prev2 = 0.0
    y_prev = 0.0
    y_prev2 = 0.0
    delta_control_prev = 0.0
    u_prev = 0.0
    y_prev_read = 0.0

    inst2 = get_sample_position(Ts, samples_number, change_ref_instant2)
    inst3 = get_sample_position(Ts, samples_number, change_ref_instant3)
    reference_input = float(rst_single_reference) * np.ones(samples_number)
    reference_input[inst2:inst3] = float(rst_siso_multiple_reference2)
    reference_input[inst3:] = float(rst_siso_multiple_reference3)
    
    process_output = np.zeros(samples_number)
    manipulated_variable = np.zeros(samples_number)
    
    set_session_controller_parameter('reference_input', reference_input.tolist())
    set_session_controller_parameter('control_signal_1', dict())
    control_signal_dict = get_session_variable('control_signal_1')
    set_session_controller_parameter('process_output_sensor', dict())
    process_output_sensor = get_session_variable('process_output_sensor')

    if arduinoData.is_open:
        arduinoData.reset_input_buffer()
        arduinoData.reset_output_buffer()
        sendToArduino(arduinoData, "0\n")
        time.sleep(0.5)

    st.markdown(f"### 📊 Monitoramento Adaptativo ({pid_structure})")
    diag_placeholder = st.empty()
    metrics_ph = st.empty()

    start_time = time.time()
    kk = 0
    my_bar = st.progress(0, text=f"RST {pid_structure}...")

    final_kc, final_ki, final_kd = 0.0, 0.0, 0.0
    final_t0 = 0.0

    pid_clean = pid_structure.strip()

    while kk < samples_number:
        current_time = time.time()
        if current_time - start_time > Ts:
            start_time = current_time 
            
            y_curr = y_prev_read
            if arduinoData.is_open:
                raw_val = None
                while arduinoData.in_waiting > 0:
                    try:
                        line = arduinoData.readline().decode('utf-8').strip()
                        if line: raw_val = float(line)
                    except: pass
                if raw_val is not None: y_curr = raw_val
            
            y_prev_read = y_curr
            process_output[kk] = y_curr
            
            # Identificação
            if kk >= 2:
                phi = np.array([-hist_y[0], hist_u[1]]) 
                theta_est = rls.update(y_curr, phi)
                a1_est, b0_est = theta_est[0], theta_est[1]
            else:
                a1_est, b0_est = a1_initial, b0_initial

            # Sintonia
            r1, s0, s1, t0 = tuning_calc_notes(a1_est, b0_est, tau_ml_input, Ts)
            R_poly, S_poly, T_poly, gains_pid = build_polynomials(r1, s0, s1, t0)
            
            kc_pid, ki_pid, kd_pid = gains_pid
            final_kc, final_ki, final_kd = kc_pid, ki_pid, kd_pid
            final_t0 = t0

            if kk % 4 == 0:
                with diag_placeholder.container():
                    c1, c2, c3, c4 = st.columns(4)
                    c1.metric("a1 (Est)", f"{a1_est:.3f}")
                    c2.metric("b0 (Est)", f"{b0_est:.3f}")
                    c3.metric("Kc", f"{kc_pid:.3f}")
                    c4.metric("Ki", f"{ki_pid:.3f}")

            # Lei de Controle
            ref_val = reference_input[kk]
            e_k = ref_val - y_curr
            
            delta_u = 0.0
            
            if pid_clean == 'RST Incremental Puro':
                # Estrutura Polinomial Clássica: T*w - S*y - (R-1)*delta_u
                term_t = T_poly[0] * ref_val
                term_s = S_poly[0] * y_curr + S_poly[1] * hist_y[0]
                term_r = R_poly[1] * delta_control_prev
                delta_u = term_t - term_s - term_r
                
            elif pid_clean == 'I + PD':
                # I no erro, P e D na saída (derivada da saída)
                delta_u = (ki_pid * e_k) - (kc_pid * (y_curr - y_prev)) - (kd_pid * (y_curr - 2*y_prev + y_prev2))
                
            elif pid_clean == 'PI + D':
                # PI no erro, D na saída
                delta_u = (kc_pid * (e_k - e_prev)) + (ki_pid * e_k) - (kd_pid * (y_curr - 2*y_prev + y_prev2))
                
            elif pid_clean == 'PID Paralelo':
                # Tudo no erro
                delta_u = (kc_pid * (e_k - e_prev)) + (ki_pid * e_k) + (kd_pid * (e_k - 2*e_prev + e_prev2))
                
            elif pid_clean == 'PID Ideal':
                if abs(kc_pid) > 1e-9:
                    term_p = (e_k - e_prev)
                    term_i = (ki_pid / kc_pid) * e_k # Ki_paralelo / Kp_paralelo = Ts/Ti
                    term_d = (kd_pid / kc_pid) * (e_k - 2*e_prev + e_prev2) # Kd_paralelo / Kp_paralelo = Td/Ts
                    delta_u = kc_pid * (term_p + term_i + term_d)
                else:
                    delta_u = 0.0

            # Slew Rate & Saturação
            MAX_CHANGE = 50.0 
            delta_u = max(-MAX_CHANGE, min(delta_u, MAX_CHANGE))
            
            u_calc = u_prev + delta_u
            u_final = max(min_pot, min(u_calc, max_pot))
            delta_real = u_final - u_prev
            
            manipulated_variable[kk] = u_final

            if arduinoData.is_open:
                sendToArduino(arduinoData, f"{u_final:.4f}\n")
            
            # Atualização Memórias
            hist_u[2] = hist_u[1]
            hist_u[1] = hist_u[0]
            hist_u[0] = u_final
            hist_y[1] = hist_y[0]
            hist_y[0] = y_curr
            
            e_prev2 = e_prev
            e_prev = e_k
            y_prev2 = y_prev
            y_prev = y_curr
            
            u_prev = u_final
            delta_control_prev = delta_real
            
            ts_str = str(datetime.now())
            process_output_sensor[ts_str] = float(y_curr)
            control_signal_dict[ts_str] = float(u_final)
            
            with metrics_ph.container():
                m1, m2 = st.columns(2)
                m1.metric("Nível", f"{y_curr:.2f}")
                m2.metric("Controle", f"{u_final:.2f}")
            
            kk += 1
            my_bar.progress(kk/samples_number)
        else:
            time.sleep(0.001)

    if arduinoData.is_open:
        sendToArduino(arduinoData, "0\n")
        time.sleep(0.1)

    # FINALIZAÇÃO E SALVAMENTO DE DADOS (CRÍTICO PARA EXIBIÇÃO)
    set_session_controller_parameter('process_output_sensor', process_output_sensor)
    set_session_controller_parameter('control_signal_1', control_signal_dict)

    # Cálculo e Salvamento de Métricas
    iae_calc = np.sum(np.abs(reference_input[:samples_number] - process_output[:samples_number])) * Ts
    tvc_calc = np.sum(np.abs(np.diff(manipulated_variable[:samples_number])))
    
    # Salva DIRETAMENTE em controller_parameters para a interface pegar
    if 'controller_parameters' not in st.session_state: st.session_state.controller_parameters = {}
    st.session_state.controller_parameters['iae_metric'] = float(iae_calc)
    st.session_state.controller_parameters['tvc_1_metric'] = float(tvc_calc)
    
    # Salva parâmetros de sintonia
    st.session_state.controller_parameters['rst_calculated_params'] = {
        'Kc': float(final_kc),
        'Ki': float(final_ki),
        'Kd': float(final_kd),
        'Tau_MF': float(tau_ml_input),
        'T0': float(final_t0)
    }

    st.success("Teste Adaptativo Finalizado.")

# ==============================================================================
# 5. CONTROLADOR RST INCREMENTAL (ESTÁTICO - COM BUFFER FIX)
# ==============================================================================

def rstControlProcessIncrementalSISO(transfer_function_type:str, 
                                     num_coeff:str, den_coeff:str, 
                                     tau_ml_input:float, pid_structure:str,
                                     rst_single_reference:float,
                                     rst_siso_multiple_reference2:float, 
                                     rst_siso_multiple_reference3:float,
                                     change_ref_instant2=1, change_ref_instant3=1):
    
    if num_coeff == '': return st.error('Coeficientes vazios.')
    if 'arduinoData' not in st.session_state.connected: return st.error('Arduino desconectado.')
    arduinoData = st.session_state.connected['arduinoData']

    Ts = float(get_session_variable('sampling_time'))
    samples_number = int(get_session_variable('samples_number'))
    max_pot = float(get_session_variable('saturation_max_value') or 100.0)
    min_pot = float(get_session_variable('saturation_min_value') or 0.0)

    try:
        num = [float(x) for x in num_coeff.split(',')]
        den = [float(x) for x in den_coeff.split(',')]
        
        if transfer_function_type == 'Continuo':
            kp_plant = num[-1]
            tau_plant = den[0] if len(den) > 0 else 1.0
            theta_plant = 0.0 
            r1, s0, s1, t0 = tuning_static_wrapper(kp_plant, tau_plant, tau_ml_input, Ts, theta_plant)
        else:
            b0 = num[-1]
            a1 = den[1] if len(den) > 1 else 0.0
            r1, s0, s1, t0 = tuning_calc_notes(a1, b0, tau_ml_input, Ts)

    except Exception as e:
        st.error(f"Erro ao processar modelo: {e}")
        return

    R_poly, S_poly, T_poly, gains_pid = build_polynomials(r1, s0, s1, t0)
    kc_st, ki_st, kd_st = gains_pid

    with st.expander(f"Diagnóstico RST - {pid_structure}", expanded=True):
        c1, c2, c3 = st.columns(3)
        c1.code(f"S=[{S_poly[0]:.3f}, {S_poly[1]:.3f}]\nR=[1.0, {R_poly[1]:.3f}]")
        c2.code(f"T=[{T_poly[0]:.3f}, {T_poly[1]:.3f}]")
        c3.latex(f"K_c={kc_st:.3f}, K_i={ki_st:.3f}")

    process_output = np.zeros(samples_number)
    manipulated_variable_1 = np.zeros(samples_number)
    
    # Memórias PID
    delta_control_prev = 0.0
    u_prev = 0.0
    y_prev_read = 0.0
    hist_y_prev = 0.0 # apenas para RST puro se precisar
    
    e_prev = 0.0
    e_prev2 = 0.0
    y_prev = 0.0
    y_prev2 = 0.0
    
    # Referências
    inst2 = get_sample_position(Ts, samples_number, change_ref_instant2)
    inst3 = get_sample_position(Ts, samples_number, change_ref_instant3)
    reference_input = float(rst_single_reference) * np.ones(samples_number)
    reference_input[inst2:inst3] = float(rst_siso_multiple_reference2)
    reference_input[inst3:] = float(rst_siso_multiple_reference3)
    
    set_session_controller_parameter('reference_input', reference_input.tolist())
    set_session_controller_parameter('control_signal_1', dict())
    control_signal_dict = get_session_variable('control_signal_1')
    set_session_controller_parameter('process_output_sensor', dict())
    process_output_sensor = get_session_variable('process_output_sensor')

    if arduinoData.is_open:
        arduinoData.reset_input_buffer()
        sendToArduino(arduinoData, "0\n")
        time.sleep(0.5)
        
    start_time = time.time()
    kk = 0
    my_bar = st.progress(0, text=f"Executando RST {pid_structure}...")
    metrics_ph = st.empty()
    
    pid_clean = pid_structure.strip()

    while kk < samples_number:
        current_time = time.time()
        if current_time - start_time > Ts:
            start_time = current_time 
            
            y_curr = y_prev_read
            if arduinoData.is_open:
                raw_val = None
                while arduinoData.in_waiting > 0:
                    try:
                        line = arduinoData.readline().decode('utf-8').strip()
                        if line: raw_val = float(line)
                    except: pass
                if raw_val is not None: y_curr = raw_val
            
            y_prev_read = y_curr
            process_output[kk] = y_curr

            if kk < 3:
                u_final = 0.0
            else:
                ref_val = reference_input[kk]
                e_k = ref_val - y_curr
                delta_u = 0.0
                
                if pid_clean == 'RST Incremental Puro':
                    term_t = T_poly[0] * ref_val
                    term_s = S_poly[0] * y_curr + S_poly[1] * hist_y_prev
                    term_r = R_poly[1] * delta_control_prev
                    delta_u = term_t - term_s - term_r
                    
                elif pid_clean == 'I + PD':
                    delta_u = (ki_st * e_k) - (kc_st * (y_curr - y_prev)) - (kd_st * (y_curr - 2*y_prev + y_prev2))
                    
                elif pid_clean == 'PI + D':
                    delta_u = (kc_st * (e_k - e_prev)) + (ki_st * e_k) - (kd_st * (y_curr - 2*y_prev + y_prev2))
                    
                elif pid_clean == 'PID Paralelo':
                    delta_u = (kc_st * (e_k - e_prev)) + (ki_st * e_k) + (kd_st * (e_k - 2*e_prev + e_prev2))
                    
                elif pid_clean == 'PID Ideal':
                    if abs(kc_st) > 1e-9:
                        term_p = (e_k - e_prev)
                        term_i = (ki_st / kc_st) * e_k 
                        term_d = (kd_st / kc_st) * (e_k - 2*e_prev + e_prev2)
                        delta_u = kc_st * (term_p + term_i + term_d)
                
                # Slew Rate
                MAX_CHANGE = 20.0 
                delta_u = max(-MAX_CHANGE, min(delta_u, MAX_CHANGE))

                u_calc = u_prev + delta_u
                u_final = max(min_pot, min(u_calc, max_pot))
                delta_real = u_final - u_prev
                
                u_prev = u_final
                delta_control_prev = delta_real
                
            manipulated_variable_1[kk] = u_final
            
            # Atualiza memórias
            hist_y_prev = y_curr 
            e_prev2 = e_prev
            e_prev = e_k if kk >= 3 else 0.0
            y_prev2 = y_prev
            y_prev = y_curr

            if arduinoData.is_open:
                sendToArduino(arduinoData, f"{u_final:.4f}\n")

            ts_str = str(datetime.now())
            process_output_sensor[ts_str] = float(y_curr)
            control_signal_dict[ts_str] = float(u_final)
            
            with metrics_ph.container():
                m1, m2 = st.columns(2)
                m1.metric("Nível", f"{y_curr:.2f}")
                m2.metric("Controle", f"{u_final:.2f}")
                
            kk += 1
            my_bar.progress(kk/samples_number)
        else:
            time.sleep(0.001)

    if arduinoData.is_open:
        sendToArduino(arduinoData, "0\n")
        time.sleep(0.1)

    # FINALIZAÇÃO E SALVAMENTO DE DADOS (CRÍTICO)
    set_session_controller_parameter('process_output_sensor', process_output_sensor)
    set_session_controller_parameter('control_signal_1', control_signal_dict)

    # Cálculo e Salvamento de Métricas
    iae_calc = np.sum(np.abs(reference_input[:samples_number] - process_output[:samples_number])) * Ts
    tvc_calc = np.sum(np.abs(np.diff(manipulated_variable_1[:samples_number])))
    
    # Salva DIRETAMENTE em controller_parameters
    if 'controller_parameters' not in st.session_state: st.session_state.controller_parameters = {}
    st.session_state.controller_parameters['iae_metric'] = float(iae_calc)
    st.session_state.controller_parameters['tvc_1_metric'] = float(tvc_calc)

    # Salva Tabela
    st.session_state.controller_parameters['rst_calculated_params'] = {
        'Kc': float(kc_st),
        'Ki': float(ki_st),
        'Kd': float(kd_st),
        'Tau_MF': float(tau_ml_input),
        'T0': float(t0)
    }

    st.success("Teste Incremental Finalizado.")