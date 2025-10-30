import streamlit as st
from formatterInputs import *
import numpy as np
from connections import *
import datetime
from session_state import get_session_variable,set_session_controller_parameter
from controllers_process.validations_functions import *


class GeneralizedPredictiveController:
    def __init__(self,nit, Ny,Nu,lambda_,ts,Am,Bm) -> None:
        self.nit = nit
        self.Ny = Ny
        self.Nu = Nu
        self.lambda_ = lambda_
        self.ts = ts
        self.Am = Am
        self.Bm = Bm
        na = len(Am)-1
        self.E = np.zeros((Ny,Ny))
        self.G = np.zeros((Ny,Nu))
        self.H = np.zeros((Ny,1))  
        self.gt = np.zeros((Ny,Ny))  
        self.Kgpc = np.zeros((1,Ny))  
        self.F = np.zeros((Ny,na+1))
    
    def create_A_til(self):
        delta = [1,-1]
        Atil = np.convolve(self.Am,delta)
        return Atil
        
    def create_C_poly(self):
        Atil = self.create_A_til()
        c_poly = np.ones(len(Atil))
        c_poly[1:] = 0
        # st.write('rr =')
        # st.write(rr)
        return c_poly
     
    def calculate_F_matrix(self,ny, rr, Atil):
        
        quotient_coeff = np.zeros(ny) 
        for k in range(ny):
            quotient_coeff[k],r = np.polydiv(rr,Atil)
            #st.write('new r =' ,r)
            self.F[k,:] = r
            rr = np.append(r,0)
            #st.write('new rr =' ,rr)
        return quotient_coeff # will generate a single element by index
    
    def calculate_E_matrix(self,ny,q):
        for j in range(ny):
            self.E[j:,j] = q[j] # ex: [[q0,0,0],[q0,q1,0],[q0,q1,q2]]
    
    def calculate_G_matrix(self,ny,nu):
        # # Diophantine Second Equation
        # B*E  = G*C + z^{-j}*H 
        """
        Primeiro, pegaremos a última linha da matriz E e, em seguida, a multiplicaremos com a matriz B, tratando ambos como um polinômio.
        Agora, vamos nos concentrar em determinar os coeficientes de G. 
        Esses coeficientes consistirão em todos os coeficientes do polinômio, exceto o último (que pertence a H).
        
        First, we'll take the last row of matrix E and then multiply it with matrix B, treating both as a polynomial.
        Now, let's focus on determining the coefficients of G. 
        These coefficients will consist of all coefficients from the matrix, except the last one.
        """
        # B_E_conv = np.convolve(self.Bm,self.E[-1])
        
        # G_aux = B_E_conv[:-1]
             
        B_aux = np.convolve(self.Bm,self.E[-1])
        #st.write('Baux =' ,B_aux)
        nb_aux =len(B_aux)
        T_BE = np.zeros(nb_aux+1)
        T_BE[0] = 1
        # st.write('T_BE =' ,T_BE)
        
        rr_BE = B_aux
        G_aux = np.zeros(ny)
        
        for k in range(ny):
            rr_BE = np.append(rr_BE,0)
            #st.write('rr_BE',rr_BE)
            q_BE,r_BE = np.polydiv(rr_BE, T_BE)
            G_aux[k] = q_BE
            rr_BE = r_BE
                
        for i in range(nu):
            for j in range(i,ny):
                self.G[j][i] = G_aux[j-i]
        

    def calculate_H_matrix(self,ny): 
        """
        Para se obter a matriz H deve-se multiplicar o polinômio B com todas as linhas de E. 
        O último elemento de cada operação será um elemento de H.
        
        """  
        for i in range(ny):
            EB_conv = np.convolve(self.E[i,0:(i+1)],self.Bm)    # B*E + j^{-1}*H
            #st.write('EB_conv',EB_conv)
            self.H[i] = EB_conv[-1]
            #st.write('H',self.H)
                    
    def calculateController(self):

        Atil =self.create_A_til()
        # st.write('Atil',Atil)
        
        # Calculate the predictor matrices (F, H, G)
        # Diophantine Fist Equation
        c_poly = self.create_C_poly()
                        
        quotient_coeff = self.calculate_F_matrix(self.Ny,c_poly,Atil)
           
        self.calculate_E_matrix(self.Ny,quotient_coeff)
        # st.write('E =' ,self.E)
                
        # # Diophantine Second Equation
        # B*E  = G*C + z^{-j}*H 
        
        self.calculate_G_matrix(self.Ny,self.Nu)
                
        self.calculate_H_matrix(self.Ny)
            
        
        ## Operação Final 
        
        G = self.G
        lambda_ = self.lambda_
        nu = self.Nu
        
        G_lambda_op = np.dot(G.T,G) + lambda_ *np.eye(nu) # G^T * G + lambda*I
       
        # Obtain the inverse matrix
        G_lambda_op_inv = np.linalg.inv(G_lambda_op) # (G^T * G + lambda*I)^{-1}
        
        # Multiply the inverse by the G transpose
        self.gt = np.dot(G_lambda_op_inv, G.T) # (G^T * G + lambda*I)^{-1} * G^{T}
        
        self.Kgpc = self.gt[0]
                

def gpcControlProcessSISO(transfer_function_type:str,num_coeff:str,den_coeff:str,
                          gpc_siso_ny:int,gpc_siso_nu:int,gpc_siso_lambda:float,future_inputs_checkbox:bool,
                          gpc_multiple_reference1:float, gpc_multiple_reference2:float, gpc_multiple_reference3:float,
                          f_gpc_mimo_checkbox=False, K_alpha=0, alpha_fgpc=0,
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

    # GPC Controller Project
    
    Ny = gpc_siso_ny
    Nu = gpc_siso_nu
    lambda_ = gpc_siso_lambda

    # Initial Conditions
    process_output = np.zeros(samples_number)
    delta_control_signal = np.zeros(samples_number)

    # Take the index of time to change the referencee
    instant_sample_2 = get_sample_position(sampling_time, samples_number, change_ref_instant2)
    instant_sample_3 = get_sample_position(sampling_time, samples_number, change_ref_instant3)

    reference_input = gpc_multiple_reference1*np.ones(samples_number+Ny)
    reference_input[instant_sample_2:instant_sample_3] = gpc_multiple_reference2
    reference_input[instant_sample_3:] = gpc_multiple_reference3
    set_session_controller_parameter('reference_input',reference_input[:samples_number].tolist())
    # st.session_state.controller_parameters['reference_input'] = reference_input.tolist()

    # Power Saturation
    max_pot = get_session_variable('saturation_max_value')
    min_pot = get_session_variable('saturation_min_value')

    # Manipulated variable
    manipulated_variable_1 = np.zeros(samples_number)
    serial_data_pack = "0"

    # Model transfer Function
    A_coeff, B_coeff = convert_tf_2_discrete(num_coeff,den_coeff,transfer_function_type,f_gpc_mimo_checkbox, K_alpha, alpha_fgpc)
    
    # print(A_coeff)
    # print(B_coeff)
    A_order = len(A_coeff)-1
    B_order = len(B_coeff)
    
    
  
    # GPC CONTROLLER
    gpc_m1 = GeneralizedPredictiveController(nit= samples_number,Ny=Ny,Nu=Nu,lambda_=lambda_,
                                             ts=sampling_time,Am=A_coeff, Bm=B_coeff)
    gpc_m1.calculateController()

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
                F_yf = np.dot(gpc_m1.F, np.zeros(A_order+1))
                sendToArduino(arduinoData, '0')
                

            # ---- Motor Model Output
            elif kk == 1 and A_order == 1:
              
                F_yf = np.dot(gpc_m1.F,process_output[kk::-1])
                
            elif kk > A_order:
                F_yf = np.dot(gpc_m1.F,process_output[kk:kk-A_order-1:-1])
                    # st.write('F*yf',F_yf)           
                                        
                
            aux_ref = future_inputs_selection(future_inputs_checkbox,kk,Ny,reference_input)
            
                # st.write('F*yf',F_yf)
                
            H_du = np.dot(gpc_m1.H,delta_control_signal[kk-1])
                # st.write('H*du',H_du)
                
            fi = np.squeeze(H_du) + F_yf
            
            delta_control_signal[kk] = np.dot(gpc_m1.Kgpc,(aux_ref-fi))
            manipulated_variable_1[kk] = manipulated_variable_1[kk-1] + delta_control_signal[kk]
            
            # Control Signal Saturation
            manipulated_variable_1[kk] = max(min_pot, min(manipulated_variable_1[kk], max_pot))
        

            # Motor Power String Formatation
            serial_data_pack = f"{manipulated_variable_1[kk]}\r"
            sendToArduino(arduinoData, serial_data_pack)
                
            
            # Store the output process values and control signal
            current_timestamp = str(datetime.now())
            process_output_sensor[current_timestamp] = float(process_output[kk])
            control_signal_1[current_timestamp] = float(manipulated_variable_1[kk])
            kk += 1

            percent_complete = kk / (samples_number)
            my_bar.progress(percent_complete, text=progress_text)

    # Turn off the motor
    sendToArduino(arduinoData, '0')



def gpcPidControlProcessSISO(transfer_function_type:str,num_coeff:str,den_coeff:str,
                          gpc_siso_ny:int,gpc_siso_nu:int,gpc_siso_lambda:float,future_inputs_checkbox:bool,
                          gpc_multiple_reference1:float, gpc_multiple_reference2:float, gpc_multiple_reference3:float,
                          f_gpc_mimo_checkbox=False, K_alpha=0, alpha_fgpc=0,
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

    # GPC Controller Project
    
    Ny = gpc_siso_ny
    Nu = gpc_siso_nu
    lambda_ = gpc_siso_lambda

    # Initial Conditions
    process_output = np.zeros(samples_number)
    delta_control_signal = np.zeros(samples_number)

    # Take the index of time to change the referencee
    instant_sample_2 = get_sample_position(sampling_time, samples_number, change_ref_instant2)
    instant_sample_3 = get_sample_position(sampling_time, samples_number, change_ref_instant3)

    reference_input = gpc_multiple_reference1*np.ones(samples_number+Ny)
    reference_input[instant_sample_2:instant_sample_3] = gpc_multiple_reference2
    reference_input[instant_sample_3:] = gpc_multiple_reference3
    set_session_controller_parameter('reference_input',reference_input[:samples_number].tolist())
    # st.session_state.controller_parameters['reference_input'] = reference_input.tolist()

    # Power Saturation
    max_pot = get_session_variable('saturation_max_value')
    min_pot = get_session_variable('saturation_min_value')

    # Manipulated variable
    manipulated_variable_1 = np.zeros(samples_number)
    serial_data_pack = "0"

    # Model transfer Function
    A_coeff, B_coeff = convert_tf_2_discrete(num_coeff,den_coeff,transfer_function_type,f_gpc_mimo_checkbox, K_alpha, alpha_fgpc)
    
    # print(A_coeff)
    # print(B_coeff)
    A_order = len(A_coeff)-1
    B_order = len(B_coeff)
    
    
  
    # GPC CONTROLLER
    gpc_m1 = GeneralizedPredictiveController(nit= samples_number,Ny=Ny,Nu=Nu,lambda_=lambda_,
                                             ts=sampling_time,Am=A_coeff, Bm=B_coeff)
    gpc_m1.calculateController()

    # clear previous control signal values
    set_session_controller_parameter('control_signal_1',dict())
    control_signal_1 = get_session_variable('control_signal_1')
    
    # clear previous control signal values
    set_session_controller_parameter('process_output_sensor',dict())
    process_output_sensor = get_session_variable('process_output_sensor')
    
    ## ---- Cáculo dos elementos do sinal de controle R, S e T
    T_1 = sum(gpc_m1.Kgpc);                       # polinômio que afeta o sinal de entrada
    S_1 = np.dot(gpc_m1.Kgpc,gpc_m1.F);           # polinômio que afeta as saída preditas
    R_1 = np.dot(gpc_m1.Kgpc,gpc_m1.H);           # polinômio que afeta a ação de controle passada du(t-1);
    R_1 = np.insert(R_1,0,1)

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
                sendToArduino(arduinoData, '0,0')     
            
            # Controle via estrutura RST
            elif kk == 1 and A_order == 1:
                delta_control_signal[kk] = -R_1[1]*delta_control_signal[kk-1] + T_1*reference_input[kk] - np.dot(S_1,process_output[kk::-1])
            
            elif kk > A_order:
                delta_control_signal[kk] = -R_1[1]*delta_control_signal[kk-1] + T_1*reference_input[kk] - np.dot(S_1,process_output[kk:kk-A_order-1:-1])
                manipulated_variable_1[kk] = manipulated_variable_1[kk-1] + delta_control_signal[kk]
            
            # Control Signal Saturation
            manipulated_variable_1[kk] = max(min_pot, min(manipulated_variable_1[kk], max_pot))
        

            # Motor Power String Formatation
            serial_data_pack = f"{manipulated_variable_1[kk]}\r"
            sendToArduino(arduinoData, serial_data_pack)
                
            
            # Store the output process values and control signal
            current_timestamp = str(datetime.now())
            process_output_sensor[current_timestamp] = float(process_output[kk])
            control_signal_1[current_timestamp] = float(manipulated_variable_1[kk])
            kk += 1

            percent_complete = kk / (samples_number)
            my_bar.progress(percent_complete, text=progress_text)

    # Turn off the motor
    sendToArduino(arduinoData, '0')



def gpcPIDControlProcessTISO(transfer_function_type:str,num_coeff_1:str,den_coeff_1:str, num_coeff_2:str,den_coeff_2:str,
                          gpc_mimo_ny_1:int,gpc_mimo_nu_1:int,gpc_mimo_lambda_1:float,
                          gpc_mimo_ny_2:int,gpc_mimo_nu_2:int,gpc_mimo_lambda_2:float,future_inputs_checkbox:bool,
                          gpc_multiple_reference1:float, gpc_multiple_reference2:float, gpc_multiple_reference3:float,
                          f_gpc_mimo_checkbox=False, K_alpha=0, alpha_fgpc=0,
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
    
    # gpc Controller Project
    Ny_1 = gpc_mimo_ny_1
    Nu_1 = gpc_mimo_nu_1
    lambda_1 = gpc_mimo_lambda_1
    Ny_2 = gpc_mimo_ny_2
    Nu_2 = gpc_mimo_nu_2
    lambda_2 = gpc_mimo_lambda_2

    # Initial Conditions
    process_output = np.zeros(samples_number)
    delta_control_signal_1 = np.zeros(samples_number)
    delta_control_signal_2 = np.zeros(samples_number)

    # Take the index of time to change the referencee
    instant_sample_2 = get_sample_position(sampling_time, samples_number, change_ref_instant2)
    instant_sample_3 = get_sample_position(sampling_time, samples_number, change_ref_instant3)
    
    ny_max = max(Ny_1,Ny_2)
    reference_input = gpc_multiple_reference1*np.ones(samples_number+ny_max)
    reference_input[instant_sample_2:instant_sample_3] = gpc_multiple_reference2
    reference_input[instant_sample_3:] = gpc_multiple_reference3
    set_session_controller_parameter('reference_input',reference_input[:samples_number].tolist())


    # Power Saturation
    max_pot = get_session_variable('saturation_max_value')
    min_pot = get_session_variable('saturation_min_value')


    # Manipulated variable
    manipulated_variable_1 = np.zeros(samples_number)
    manipulated_variable_2 = np.zeros(samples_number)
    motors_power_packet = "0,0"

    # Model transfer Function 1
    A_coeff_1, B_coeff_1 = convert_tf_2_discrete(num_coeff_1,den_coeff_1,transfer_function_type,f_gpc_mimo_checkbox, K_alpha, alpha_fgpc)
    
    # print(A_coeff)
    # print(B_coeff)
    A_order = len(A_coeff_1)-1
    B_order = len(B_coeff_1)-1 # Zero holder aumenta um grau

    ## Model transfer Function 2
    A_coeff_2, B_coeff_2 = convert_tf_2_discrete(num_coeff_2,den_coeff_2,transfer_function_type,f_gpc_mimo_checkbox, K_alpha, alpha_fgpc)
    
    # GPC CONTROLLER
    gpc_m1 = GeneralizedPredictiveController(nit= samples_number,Ny=Ny_1,Nu=Nu_1,lambda_=lambda_1,
                                             ts=sampling_time,Am=A_coeff_1, Bm=B_coeff_1)
    gpc_m1.calculateController()
    
    gpc_m2 = GeneralizedPredictiveController(nit= samples_number,Ny=Ny_2,Nu=Nu_2,lambda_=lambda_2,
                                             ts=sampling_time,Am=A_coeff_2, Bm=B_coeff_2)
    gpc_m2.calculateController()
    
   # clear previous control signal values
    set_session_controller_parameter('control_signal_1',dict())
    control_signal_1 = get_session_variable('control_signal_1')
    
    set_session_controller_parameter('control_signal_2',dict())
    control_signal_2 = get_session_variable('control_signal_2')
    
    # clear previous control signal values
    set_session_controller_parameter('process_output_sensor',dict())
    process_output_sensor = get_session_variable('process_output_sensor')
    
    ## ---- Cáculo dos elementos do sinal de controle R, S e T
    T_1 = sum(gpc_m1.Kgpc);                       # polinômio que afeta o sinal de entrada
    S_1 = np.dot(gpc_m1.Kgpc,gpc_m1.F);           # polinômio que afeta as saída preditas
    R_1 = np.dot(gpc_m1.Kgpc,gpc_m1.H);           # polinômio que afeta a ação de controle passada du(t-1);
    R_1 = np.insert(R_1,0,1)
    T_2 = sum(gpc_m2.Kgpc);                       # polinômio que afeta o sinal de entrada
    S_2 = np.dot(gpc_m2.Kgpc,gpc_m2.F);           # polinômio que afeta as saída preditas
    R_2 = np.dot(gpc_m2.Kgpc,gpc_m2.H);           # polinômio que afeta a ação de controle passada du(t-1);
    R_2 = np.insert(R_2,0,1)
    
    s_12 = 0
    s_22 = 0
# Paramêtros do PID
    if len(S_1) == 3 and len(S_2)==3:
        s_12 = S_1[2]
        s_22 = S_2[2]
        
  
    # Paramêtros do PID
    kc1 = S_1[0] - T_1 - s_12
    ti1 = (kc1)/(sum(S_1))
    td1 = -s_12/kc1
        
    kc2 = S_2[0] - T_2 - s_22
    ti2 = (kc2)/(sum(S_2))
    td2 = -s_22/kc2
        
    # inicializar  o timer
    start_time = time.time()
    kk = 2

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
            
            
            delta_control_signal_1[kk] =  + T_1*reference_input[kk] - np.dot(S_1,process_output[kk:kk-A_order-1:-1])
            manipulated_variable_1[kk] = manipulated_variable_1[kk-1] + delta_control_signal_1[kk]
            
            delta_control_signal_2[kk] = + T_2*reference_input[kk] - np.dot(S_2,process_output[kk:kk-A_order-1:-1])
            manipulated_variable_2[kk] = manipulated_variable_2[kk-1] - delta_control_signal_2[kk]
            
            # Control Signal Saturation
            manipulated_variable_1[kk] = max(min_pot, min(manipulated_variable_1[kk], max_pot))
        
            manipulated_variable_2[kk] = max(min_pot, min(manipulated_variable_2[kk], max_pot))
        

            # Motor Power String Formatation
            serial_data_pack = f"{manipulated_variable_1[kk]},{manipulated_variable_2[kk]}\r"
            sendToArduino(arduinoData, serial_data_pack)
                
            
            # Store the output process values and control signal
            current_timestamp = str(datetime.now())
            process_output_sensor[current_timestamp] = process_output[kk]
            control_signal_1[current_timestamp] = manipulated_variable_1[kk]
            control_signal_2[current_timestamp] = manipulated_variable_2[kk]
            kk += 1

            percent_complete = kk / (samples_number)
            my_bar.progress(percent_complete, text=progress_text)
            


    # Turn off the motor
    sendToArduino(arduinoData, '0,0')
    
    # st.write(vars(gpc_m1))
    # st.write(vars(gpc_m2))
    pid_parameters_1 = {
        'kc_1': kc1, 
        'ti_1': ti1,
        'td_1': td1
    }
    pid_parameters_2 = {
        'kc_2': kc2, 
        'ti_2': ti2,
        'td_2': td2
    }
    
    
    st.write(pid_parameters_1)
    st.write(pid_parameters_2)




def gpcControlProcessTISO(transfer_function_type:str,num_coeff_1:str,den_coeff_1:str, num_coeff_2:str,den_coeff_2:str,
                          gpc_mimo_ny_1:int,gpc_mimo_nu_1:int,gpc_mimo_lambda_1:float,
                          gpc_mimo_ny_2:int,gpc_mimo_nu_2:int,gpc_mimo_lambda_2:float,future_inputs_checkbox:bool,
                          gpc_multiple_reference1:float, gpc_multiple_reference2:float, gpc_multiple_reference3:float,
                          f_gpc_mimo_checkbox=False, K_alpha=0, alpha_fgpc=0,
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
    
    # gpc Controller Project
    Ny_1 = gpc_mimo_ny_1
    Nu_1 = gpc_mimo_nu_1
    lambda_1 = gpc_mimo_lambda_1
    Ny_2 = gpc_mimo_ny_2
    Nu_2 = gpc_mimo_nu_2
    lambda_2 = gpc_mimo_lambda_2

    # Initial Conditions
    process_output = np.zeros(samples_number)
    delta_control_signal_1 = np.zeros(samples_number)
    delta_control_signal_2 = np.zeros(samples_number)

    # Take the index of time to change the referencee
    instant_sample_2 = get_sample_position(sampling_time, samples_number, change_ref_instant2)
    instant_sample_3 = get_sample_position(sampling_time, samples_number, change_ref_instant3)
    
    ny_max = max(Ny_1,Ny_2)
    reference_input = gpc_multiple_reference1*np.ones(samples_number+ny_max)
    reference_input[instant_sample_2:instant_sample_3] = gpc_multiple_reference2
    reference_input[instant_sample_3:] = gpc_multiple_reference3
    set_session_controller_parameter('reference_input',reference_input[:samples_number].tolist())


    # Power Saturation
    max_pot = get_session_variable('saturation_max_value')
    min_pot = get_session_variable('saturation_min_value')


    # Manipulated variable
    manipulated_variable_1 = np.zeros(samples_number)
    manipulated_variable_2 = np.zeros(samples_number)
    motors_power_packet = "0,0"

    # Model transfer Function 1
    A_coeff_1, B_coeff_1 = convert_tf_2_discrete(num_coeff_1,den_coeff_1,transfer_function_type,f_gpc_mimo_checkbox, K_alpha, alpha_fgpc)
    
    # print(A_coeff)
    # print(B_coeff)
    A_order = len(A_coeff_1)-1
    B_order = len(B_coeff_1)-1 # Zero holder aumenta um grau

    ## Model transfer Function 2
    A_coeff_2, B_coeff_2 = convert_tf_2_discrete(num_coeff_2,den_coeff_2,transfer_function_type,f_gpc_mimo_checkbox, K_alpha, alpha_fgpc)
    
    # GPC CONTROLLER
    gpc_m1 = GeneralizedPredictiveController(nit= samples_number,Ny=Ny_1,Nu=Nu_1,lambda_=lambda_1,
                                             ts=sampling_time,Am=A_coeff_1, Bm=B_coeff_1)
    gpc_m1.calculateController()
    
    gpc_m2 = GeneralizedPredictiveController(nit= samples_number,Ny=Ny_2,Nu=Nu_2,lambda_=lambda_2,
                                             ts=sampling_time,Am=A_coeff_2, Bm=B_coeff_2)
    gpc_m2.calculateController()
    
   # clear previous control signal values
    set_session_controller_parameter('control_signal_1',dict())
    control_signal_1 = get_session_variable('control_signal_1')
    
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
                F_yf_1 = np.dot(gpc_m1.F, np.zeros(A_order+1))
                F_yf_2 = np.dot(gpc_m2.F, np.zeros(A_order+1))
                sendToArduino(arduinoData, '0,0')
                

            # ---- Motor Model Output
            elif kk == 1 and A_order == 1:
              
                F_yf_1 = np.dot(gpc_m1.F,process_output[kk::-1])
                F_yf_2 = np.dot(gpc_m2.F,process_output[kk::-1])
                
            elif kk > A_order:
                F_yf_1 = np.dot(gpc_m1.F,process_output[kk:kk-A_order-1:-1])
                F_yf_2 = np.dot(gpc_m2.F,process_output[kk:kk-A_order-1:-1])
                    # st.write('F*yf',F_yf)           
                                        
                
            aux_ref_1 = future_inputs_selection(future_inputs_checkbox,kk,Ny_1,reference_input)
            aux_ref_2 = future_inputs_selection(future_inputs_checkbox,kk,Ny_2,reference_input)
            
                # st.write('F*yf',F_yf)
                
            H_du_1 = np.dot(gpc_m1.H,delta_control_signal_1[kk-1])
            H_du_2 = np.dot(gpc_m2.H,delta_control_signal_2[kk-1])
                # st.write('H*du',H_du)
                
            fi_1 = np.squeeze(H_du_1) + F_yf_1
            fi_2 = np.squeeze(H_du_2) + F_yf_2
            
            delta_control_signal_1[kk] = np.dot(gpc_m1.Kgpc,(aux_ref_1-fi_1))
            manipulated_variable_1[kk] = manipulated_variable_1[kk-1] + delta_control_signal_1[kk]
            
            delta_control_signal_2[kk] = np.dot(gpc_m1.Kgpc,(aux_ref_2-fi_2))
            manipulated_variable_2[kk] = manipulated_variable_2[kk-1] - delta_control_signal_2[kk]
            
            # Control Signal Saturation
            manipulated_variable_1[kk] = max(min_pot, min(manipulated_variable_1[kk], max_pot))
        
            manipulated_variable_2[kk] = max(min_pot, min(manipulated_variable_2[kk], max_pot))
        

            # Motor Power String Formatation
            serial_data_pack = f"{manipulated_variable_1[kk]},{manipulated_variable_2[kk]}\r"
            sendToArduino(arduinoData, serial_data_pack)
                
            
            # Store the output process values and control signal
            current_timestamp = str(datetime.now())
            process_output_sensor[current_timestamp] = process_output[kk]
            control_signal_1[current_timestamp] = manipulated_variable_1[kk]
            control_signal_2[current_timestamp] = manipulated_variable_2[kk]
            kk += 1

            percent_complete = kk / (samples_number)
            my_bar.progress(percent_complete, text=progress_text)
            


    # Turn off the motor
    sendToArduino(arduinoData, '0,0')
    
    # st.write(vars(gpc_m1))
    # st.write(vars(gpc_m2))





def future_inputs_selection(future_inputs_checkbox,kk,Ny,reference_input):
    if future_inputs_checkbox:
    # Referência futura conhecidax
        aux_ref = reference_input[kk:kk+Ny]
    else:
    #  Referência futura desconhecida
        aux_ref = reference_input[kk]*np.ones(Ny)
        
    return aux_ref