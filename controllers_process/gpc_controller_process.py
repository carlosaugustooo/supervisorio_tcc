import streamlit as st
from formatterInputs import *
import numpy as np
from connections import *
from datetime import datetime 
from session_state import get_session_variable, set_session_controller_parameter, set_session_variable
from controllers_process.validations_functions import *
import time

# ==============================================================================
# FUNÇÕES MATEMÁTICAS PURAS (Sem Scipy)
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
        st.error(f"Erro Modelagem: {e}")
        return np.array([0.0]), np.array([1.0])

# ==============================================================================
# CLASSE GPC
# ==============================================================================

class GeneralizedPredictiveController:
    def __init__(self, Ny, Nu, lambda_, A, B):
        self.Ny = Ny
        self.Nu = Nu
        self.lambda_ = lambda_
        self.A = A
        self.B = B
        
        # Modelo Incremental: A_tild = A * (1 - z^-1)
        self.Ad = np.convolve(A, [1, -1])
        
        self.G = np.zeros((Ny, Nu))
        self.F_matrix = np.zeros((Ny, len(self.Ad) - 1))
        
        # Tamanho de H depende da ordem de B
        self.nb = len(B) - 1
        self.H_rows = max(self.nb, 1) 
        self.H_matrix = np.zeros((Ny, self.H_rows))
        
        self.Kgpc = None
        self.R_poly = None
        self.S_poly = None

    def calculate_controller(self):
        # 1. Resposta ao Degrau (G) e Livre (F) via Recursão Diofantina
        
        # Vetor impulso unitário
        impulse = np.zeros(self.Ny + len(self.Ad))
        impulse[0] = 1.0
        
        # Filtra impulso por 1/Ad para obter E (Resposta ao degrau de perturbação)
        e_step = np.zeros_like(impulse)
        for k in range(len(impulse)):
            val = impulse[k]
            for i in range(1, len(self.Ad)):
                if k - i >= 0:
                    val -= self.Ad[i] * e_step[k-i]
            e_step[k] = val
            
        # Convolução com B para obter parâmetros G (Resposta ao degrau da planta)
        g_seq = np.convolve(e_step, self.B)
        
        # Monta matriz G (Toeplitz triangular inferior)
        for i in range(self.Ny):
            for j in range(self.Nu):
                if i >= j:
                    self.G[i, j] = g_seq[i-j]
                    
        # 2. Matrizes F e H (Predição Livre)
        # Inicialização Recursão Diophantina
        f_curr = -self.Ad[1:] if len(self.Ad) > 1 else np.zeros(1)
        h_curr = self.B[1:] if len(self.B) > 1 else np.zeros(1)
        
        f_len = self.F_matrix.shape[1]
        h_len = self.H_matrix.shape[1]
        
        for j in range(self.Ny):
            # Armazena linha j
            f_store = f_curr[:f_len] if len(f_curr) >= f_len else np.pad(f_curr, (0, f_len-len(f_curr)), 'constant')
            self.F_matrix[j, :] = f_store
            
            h_store = h_curr[:h_len] if len(h_curr) >= h_len else np.pad(h_curr, (0, h_len-len(h_curr)), 'constant')
            self.H_matrix[j, :] = h_store
            
            # Recursão para j+1
            f0 = f_curr[0] if len(f_curr) > 0 else 0.0
            
            term = f0 * self.Ad
            len_max = max(len(f_curr), len(term))
            f_next = np.pad(f_curr, (0, len_max-len(f_curr))) - np.pad(term, (0, len_max-len(term)))
            f_next = f_next[1:] 
            
            # Recalcula E para passo j+1 para obter H correto
            e_vec = e_step[:j+2]
            eb = np.convolve(e_vec, self.B)
            h_next = eb[j+2:]
            
            f_curr = f_next
            h_curr = h_next

        # 3. Cálculo do Ganho K
        H_hess = np.dot(self.G.T, self.G) + self.lambda_ * np.eye(self.Nu)
        
        # Proteção contra Matriz Singular
        try:
            inv_H = np.linalg.inv(H_hess)
        except:
            inv_H = np.zeros_like(H_hess) # Fallback seguro
            
        K_tot = np.dot(inv_H, self.G.T)
        
        self.Kgpc = K_tot[0, :]
        
        # 4. Polinômios Equivalentes (R, S)
        self.S_poly = np.dot(self.Kgpc, self.F_matrix)
        kh = np.dot(self.Kgpc, self.H_matrix)
        self.R_poly = np.insert(kh, 0, 1.0)

# ==============================================================================
# PROCESSO PRINCIPAL
# ==============================================================================

def gpcControlProcessSISO(transfer_function_type:str, num_coeff:str, den_coeff:str,
                          gpc_siso_n1:int, gpc_siso_ny:int, gpc_siso_nu:int, gpc_siso_lambda:float,
                          gpc_multiple_reference1:float, gpc_multiple_reference2:float, gpc_multiple_reference3:float,
                          change_ref_instant2:float, change_ref_instant3:float,
                          future_inputs_checkbox=False,
                          pid_structure='GPC Padrão',
                          f_gpc_mimo_checkbox=False, K_alpha=0, alpha_fgpc=0):
    
    if num_coeff == '': return st.error('Coeficientes incorretos.')

    # Fallback seguro para sampling time
    sampling_time = get_session_variable('sampling_time')
    if sampling_time is None: sampling_time = 0.5
    
    samples_number = get_session_variable('samples_number') or 100
    
    if 'arduinoData' not in st.session_state.connected:
        return st.warning('Arduino não conectado!')
    arduinoData = st.session_state.connected['arduinoData']

    # --- LIMPEZA DE BUFFER (CRÍTICO) ---
    if arduinoData.is_open:
        arduinoData.reset_input_buffer()
        arduinoData.reset_output_buffer()
        # Envia zero com terminador para garantir handshake
        sendToArduino(arduinoData, "0\n")
        time.sleep(0.5) # Tempo aumentado para garantir reset do Arduino

    # --- INICIALIZAÇÃO DE ESTADOS ---
    process_output = np.zeros(samples_number)
    delta_control_signal = np.zeros(samples_number)
    manipulated_variable_1 = np.zeros(samples_number)

    # Referências
    instant_sample_2 = get_sample_position(sampling_time, samples_number, change_ref_instant2)
    instant_sample_3 = get_sample_position(sampling_time, samples_number, change_ref_instant3)

    reference_input = gpc_multiple_reference1 * np.ones(samples_number + gpc_siso_ny)
    reference_input[instant_sample_2:instant_sample_3] = gpc_multiple_reference2
    reference_input[instant_sample_3:] = gpc_multiple_reference3
    
    set_session_controller_parameter('reference_input', reference_input[:samples_number].tolist())

    min_pot = get_session_variable('saturation_min_value') or 0.0
    max_pot = get_session_variable('saturation_max_value') or 100.0

    # --- MODELAGEM ---
    if transfer_function_type == 'Continuo':
        delay_val = 2.0 
        B_coeff, A_coeff = get_pade_model_numpy(num_coeff, den_coeff, delay_val, sampling_time)
    else:
        A_coeff, B_coeff = convert_tf_2_discrete(num_coeff, den_coeff, transfer_function_type)
    
    # --- CÁLCULO GPC ---
    gpc = GeneralizedPredictiveController(Ny=gpc_siso_ny, Nu=gpc_siso_nu, lambda_=gpc_siso_lambda, 
                                          A=A_coeff, B=B_coeff)
    gpc.calculate_controller()

    # --- EXTRAÇÃO DE GANHOS PID ---
    r0 = gpc.R_poly[0] if abs(gpc.R_poly[0]) > 1e-9 else 1.0
    inv_r0 = 1.0 / r0
    
    s0_val = gpc.S_poly[0]
    s1_val = gpc.S_poly[1] if len(gpc.S_poly) > 1 else 0.0
    s2_val = gpc.S_poly[2] if len(gpc.S_poly) > 2 else 0.0
    
    kd_gpc = inv_r0 * s2_val
    s1_norm = inv_r0 * s1_val
    kc_gpc = -s1_norm - 2 * kd_gpc
    ki_gpc = inv_r0 * np.sum(gpc.S_poly)

    # --- DIAGNÓSTICO ---
    with st.expander(f"Diagnóstico GPC (Padé 2ª Ordem)", expanded=True):
        c1, c2 = st.columns(2)
        with c1:
            st.markdown(f"**Sintonia:** $N_y={gpc_siso_ny}$, $N_u={gpc_siso_nu}$, $\lambda={gpc_siso_lambda}$")
            st.caption(f"S(z): {np.round(gpc.S_poly, 3)}")
        with c2:
            st.markdown("**PID Equivalente:**")
            st.latex(f"K_c = {kc_gpc:.4f}")
            st.latex(f"K_i = {ki_gpc:.4f}")
            st.latex(f"K_d = {kd_gpc:.4f}")
    
    pid_clean = str(pid_structure).strip()
    if pid_clean != 'GPC Padrão': st.info(f"Modo Ativo: **{pid_clean}**")

    # --- LOOP DE CONTROLE ---
    set_session_controller_parameter('control_signal_1', dict())
    control_signal_1 = get_session_variable('control_signal_1')
    set_session_controller_parameter('process_output_sensor', dict())
    process_output_sensor = get_session_variable('process_output_sensor')

    start_time = time.time()
    kk = 0
    
    metrics_placeholder = st.empty()
    my_bar = st.progress(0, text="Executando GPC...")

    # Memória para leitura robusta
    y_prev = 0.0

    while kk < samples_number:
        current_time = time.time()
        
        # Verifica tempo de amostragem
        if current_time - start_time > sampling_time:
            start_time = current_time
            
            # ==========================================================
            # 1. LEITURA ROBUSTA (DRENAGEM DE BUFFER) - IGUAL AO GMV
            # ==========================================================
            y_k = y_prev # Default se não houver leitura nova
            
            if arduinoData.is_open:
                raw_val = None
                # Loop para esvaziar o buffer e pegar o dado mais recente
                while arduinoData.in_waiting > 0:
                    try:
                        line = arduinoData.readline().decode('utf-8').strip()
                        if line: raw_val = float(line)
                    except: pass
                
                # Se leu algo válido, atualiza y_k
                if raw_val is not None: 
                    y_k = raw_val
            
            # Armazena e atualiza memória anterior
            process_output[kk] = y_k
            y_prev = y_k

            # 2. CÁLCULO DE CONTROLE
            if kk < 3: 
                # Envia zero no início para estabilizar
                if arduinoData.is_open: sendToArduino(arduinoData, "0\n")
                delta_control_signal[kk] = 0.0
                manipulated_variable_1[kk] = 0.0
            else:
                y_k_curr = process_output[kk]
                y_k1 = process_output[kk-1]
                y_k2 = process_output[kk-2]
                
                ref_val = reference_input[kk]
                e_k = ref_val - y_k_curr
                e_k1 = reference_input[kk-1] - y_k1
                e_k2 = reference_input[kk-2] - y_k2

                if pid_clean == 'GPC Padrão':
                    if future_inputs_checkbox:
                        w_vec = reference_input[kk : kk + gpc_siso_ny]
                        if len(w_vec) < gpc_siso_ny:
                            w_vec = np.pad(w_vec, (0, gpc_siso_ny - len(w_vec)), 'edge')
                    else:
                        w_vec = np.ones(gpc_siso_ny) * ref_val
                    
                    # F*y
                    f_cols = gpc.F_matrix.shape[1]
                    y_vec = []
                    for i in range(f_cols):
                        idx = kk - i
                        if idx >= 0: y_vec.append(process_output[idx])
                        else: y_vec.append(0.0)
                    
                    # H*du
                    h_cols = gpc.H_matrix.shape[1]
                    du_vec = []
                    for i in range(h_cols):
                        idx = kk - 1 - i 
                        if idx >= 0: du_vec.append(delta_control_signal[idx])
                        else: du_vec.append(0.0)
                    
                    f_total = np.dot(gpc.F_matrix, y_vec) + np.dot(gpc.H_matrix, du_vec)
                    error_pred = w_vec - f_total
                    
                    delta_control_signal[kk] = np.dot(gpc.Kgpc, error_pred)
                    
                else:
                    # PID Equivalente
                    delta_u = 0.0
                    if pid_clean == 'I + PD':
                        delta_u = (ki_gpc * e_k) - (kc_gpc * (y_k_curr - y_k1)) - (kd_gpc * (y_k_curr - 2*y_k1 + y_k2))
                    elif pid_clean == 'PI + D':
                        delta_u = (kc_gpc * (e_k - e_k1)) + (ki_gpc * e_k) - (kd_gpc * (y_k_curr - 2*y_k1 + y_k2))
                    elif pid_clean == 'PID Paralelo':
                        delta_u = (kc_gpc * (e_k - e_k1)) + (ki_gpc * e_k) + (kd_gpc * (e_k - 2*e_k1 + e_k2))
                    elif pid_clean == 'PID Ideal':
                        if abs(kc_gpc) > 1e-9:
                            term_p = (e_k - e_k1)
                            term_i = (ki_gpc/kc_gpc)*e_k
                            term_d = (kd_gpc/kc_gpc)*(e_k - 2*e_k1 + e_k2)
                            delta_u = kc_gpc * (term_p + term_i + term_d)
                    
                    delta_control_signal[kk] = delta_u

            # 3. ATUALIZAÇÃO E ENVIO
            manipulated_variable_1[kk] = manipulated_variable_1[kk-1] + delta_control_signal[kk]
            manipulated_variable_1[kk] = max(min_pot, min(manipulated_variable_1[kk], max_pot))
            
            if arduinoData.is_open:
                # Usando \n para garantir que o Arduino reconheça o fim do comando
                # Formatando para evitar excesso de casas decimais na serial
                sendToArduino(arduinoData, f"{manipulated_variable_1[kk]:.4f}\n")
                
            ts_now = str(datetime.now())
            process_output_sensor[ts_now] = float(process_output[kk])
            control_signal_1[ts_now] = float(manipulated_variable_1[kk])
            
            # 4. VISUALIZAÇÃO
            with metrics_placeholder.container():
                mc1, mc2 = st.columns(2)
                mc1.metric("Nível (cm)", f"{process_output[kk]:.2f}")
                mc2.metric("Controle (V)", f"{manipulated_variable_1[kk]:.2f}")
            
            kk += 1
            my_bar.progress(kk/samples_number)
        else:
            # Libera CPU enquanto espera o tempo de amostragem
            time.sleep(0.005)

    if arduinoData.is_open: sendToArduino(arduinoData, "0\n")

    # --- 5. FINALIZAÇÃO E UPDATE DE ESTADO (CRÍTICO) ---
    # Garante que os valores de controle e saída estejam salvos na sessão
    set_session_controller_parameter('process_output_sensor', process_output_sensor)
    set_session_controller_parameter('control_signal_1', control_signal_1)

    # 5.1 Cálculo de Métricas (IAE e TVC)
    # IMPORTANTE: Escrever diretamente em controller_parameters para a UI detectar
    iae_calc = np.sum(np.abs(reference_input[:samples_number] - process_output[:samples_number])) * sampling_time
    tvc_calc = np.sum(np.abs(np.diff(manipulated_variable_1[:samples_number])))
    
    if 'controller_parameters' not in st.session_state:
        st.session_state.controller_parameters = {}
        
    st.session_state.controller_parameters['iae_metric'] = float(iae_calc)
    st.session_state.controller_parameters['tvc_1_metric'] = float(tvc_calc)

    # 5.2 Parâmetros de Sintonia (Tabela)
    # Salva no dicionário esperado pela UI (controller_parameters['gpc_calculated_params'])
    st.session_state.controller_parameters['gpc_calculated_params'] = {
        'Ny': int(gpc_siso_ny),
        'Nu': int(gpc_siso_nu),
        'Lambda': float(gpc_siso_lambda),
        'Kc': float(kc_gpc),
        'Ki': float(ki_gpc),
        'Kd': float(kd_gpc)
    }

    st.success("Teste GPC Finalizado.")