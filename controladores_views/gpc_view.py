import streamlit as st
from formatterInputs import *
from connections import *
from controllers_process.validations_functions import *
from controllers_process.gpc_controller_process import gpcControlProcessSISO,gpcControlProcessTISO,gpcPIDControlProcessTISO

def gpc_Controller():
    st.header('Generalized Predictive Control (GPC)')
    

    graphics_col, gpc_config_col = st.columns([0.7, 0.3])

    with gpc_config_col:
        st.write('### Configurações do Controlador')

        sisoSystemTab, mimoSystemTab = st.tabs(["SISO", "MIMO"])
            #["Única Referência", "Múltiplas Referências"])
        
        with sisoSystemTab:
            gpc_siso_tab_form()

        with mimoSystemTab:
            gpc_mimo_tab_form()
           
    with graphics_col:

        if get_session_variable('process_output_sensor'):
            process_output_dataframe = dataframeToPlot('process_output_sensor','Process Output','reference_input')
            st.subheader('Resposta do Sistema')
            plot_chart_validation(process_output_dataframe, x = 'Time (s)', y = ['Reference','Process Output'],height=500)
                
        st.subheader('Sinal de Controle')
        if get_session_variable('control_signal_1'):
            control_signal_with_elapsed_time = datetime_obj_to_elapsed_time('control_signal_1')
            control_signal_1_dataframe = dictionary_to_pandasDataframe(control_signal_with_elapsed_time,'Control Signal 1')
            
            plot_chart_validation(control_signal_1_dataframe, x = 'Time (s)', y = 'Control Signal 1',height=200)
        
        
        if get_session_variable('control_signal_2'):
            control_signal_2_with_elapsed_time = datetime_obj_to_elapsed_time('control_signal_2')
            control_signal_2_dataframe = dictionary_to_pandasDataframe(control_signal_2_with_elapsed_time,'Control Signal 2')
            
            plot_chart_validation(control_signal_2_dataframe, x = 'Time (s)', y = 'Control Signal 2',height=200)
            
        
        
        
    st.write('### Índices de Desempenho')
    
    iae_col, tvc_col = st.columns([0.2,0.8])
    with iae_col:
        iae_metric_validation()
    with tvc_col:
        tvc1_validation()
    



def gpc_siso_tab_form():
    K_alpha = 0
    alpha_fgpc= 0
    controller_type = st.selectbox('Tipo do Controlador',["GPC/FGPC","PID"])
    tf_type_col, fgpc_col = st.columns(2)
    with tf_type_col:
        transfer_function_type = st.radio('**Tipo de Função de Transferência**',['Continuo','Discreto'],horizontal=True,key='gpc_siso_transfer_function_type')
    with fgpc_col:
        f_gpc_siso_checkbox = st.checkbox('Inserir filtro (FPGPC)?', key = 'f_gpc_siso_checkbox')
    
    if f_gpc_siso_checkbox:      
        k_alpha_col, alpha_col = st.columns(2)
        with k_alpha_col:
            K_alpha = st.number_input(label='$K_{\\alpha}$',key='f_gpc_mimo_k_alpha_input')

        with alpha_col:
            alpha_fgpc = st.number_input(label='$\\alpha_{fgpc}$',step=0.1, min_value=0.0, max_value=1.0,key='f_gpc_mimo_alpha_input')
            
    help_text = 'Valores decimais como **0.9** ou **0.1, 0.993**. Para múltiplos valores, vírgula é necessário.'
    st.write(' **Função de Transferência do Modelo:**')
    num_coeff = st.text_input('Coeficientes do **Numerador** :',key='siso_gpc_num_coeff',help=help_text,placeholder='0.994')
    
    coefficients_validations(num_coeff)
        
    den_coeff = st.text_input('Coeficientes do **Denominador** :',key='siso_gpc_den_coeff',help=help_text,placeholder='1.827 , 1')
    coefficients_validations(den_coeff)
    
    delay_checkbox_col, delay_input_col = st.columns(2)
    with delay_checkbox_col:
        delay_checkbox=st.checkbox('Atraso de Transporte?')
        
    with delay_input_col:
        if delay_checkbox:
            delay_input = st.number_input(label='delay',key='delay_input',label_visibility='collapsed')
            
    reference_number = st.radio('Quantidade de referências',['Única','Múltiplas'],horizontal=True,key='gpc_siso_reference_number')
    

    if reference_number == 'Única':
        gpc_single_reference = st.number_input(
        'Referência:', value=50, step=1, min_value=0, key='gpc_siso_single_reference')
        
    else:
        col21, col22, col23 = st.columns(3)
        with col23:

            gpc_siso_multiple_reference3 = st.number_input(
                'Referência 3:', value=30.0, step=1.0, min_value=0.0, key='siso_gpc_multiple_reference3')

        with col22:
            gpc_siso_multiple_reference2 = st.number_input(
                'Referência 2:', value=30.0, step=1.0, min_value=0.0, key='siso_gpc_multiple_reference2')

        with col21:
            gpc_siso_multiple_reference1 = st.number_input(
                'Referência 1:', value=30.0, step=1.0, min_value=0.0, key='siso_gpc_multiple_reference1')

        changeReferenceCol1, changeReferenceCol2 = st.columns(2)

        with changeReferenceCol2:
            siso_change_ref_instant3 = st.number_input(
                'Instante da referência 3 (s):', value=calculate_time_limit()*3/4, step=0.1, min_value=0.0, max_value=calculate_time_limit(), key='siso_change_ref_instant3')

        with changeReferenceCol1:
            siso_change_ref_instant2 = st.number_input(
                'Instante da referência 2 (s):', value=calculate_time_limit()/2,step=1.0, min_value=0.0, max_value=siso_change_ref_instant3, key='siso_change_ref_instant2')
    
    ny_col,nu_col,lambda_col = st.columns(3)
    
    with ny_col:   
        gpc_siso_ny = st.number_input('$N_y$', value=1, step=1, min_value=0, key='gpc_ny')
    with nu_col:   
        gpc_siso_nu = st.number_input('$N_u$', value=1, step=1, min_value=0, key='gpc_nu')
    with lambda_col:   
        gpc_siso_lambda = st.number_input('$\lambda$', value=1.0, step=0.1, min_value=0.0, key='gpc_lambda')
    
    future_inputs_checkbox=st.checkbox('Entradas Futuras?',key= 'gpc_siso_future_inputs_checkbox')
    
    start_col,cancel_col = st.columns([0.2,0.8])
    with cancel_col:
        
        cancel_button = st.button('Cancelar', key='gpc_cancel_siso_button')
        if cancel_button:
            st.stop()
            if 'arduinoData' in st.session_state.connected:
                arduinoData = st.session_state.connected['arduinoData']
                sendToArduino(arduinoData, '0')
            
    with start_col:        
        start_button = st.button('Iniciar', type='primary', key='gpc_siso_button')
    if start_button: 
        if controller_type == 'GPC/FGPC':      
            if reference_number == 'Única':
                
                gpcControlProcessSISO(transfer_function_type,num_coeff,den_coeff,
                                    gpc_siso_ny,gpc_siso_nu,gpc_siso_lambda,future_inputs_checkbox,
                                    gpc_single_reference, gpc_single_reference, gpc_single_reference,
                                    f_gpc_siso_checkbox, K_alpha, alpha_fgpc)
                
            elif reference_number == 'Múltiplas':
                gpcControlProcessSISO(transfer_function_type,num_coeff,den_coeff,
                                    gpc_siso_ny,gpc_siso_nu,gpc_siso_lambda,future_inputs_checkbox,
                                    gpc_siso_multiple_reference1, gpc_siso_multiple_reference2, gpc_siso_multiple_reference3, 
                                    f_gpc_siso_checkbox, K_alpha, alpha_fgpc,
                                    siso_change_ref_instant2,siso_change_ref_instant3)
            
            if cancel_button:
                st.rerun()
        
        if controller_type == 'PID':      
            if reference_number == 'Única':
                
                gpcControlProcessSISO(transfer_function_type,num_coeff,den_coeff,
                                    gpc_siso_ny,gpc_siso_nu,gpc_siso_lambda,future_inputs_checkbox,
                                    gpc_single_reference, gpc_single_reference, gpc_single_reference)
                
            elif reference_number == 'Múltiplas':
                gpcControlProcessSISO(transfer_function_type,num_coeff,den_coeff,
                                    gpc_siso_ny,gpc_siso_nu,gpc_siso_lambda,future_inputs_checkbox,
                                    gpc_siso_multiple_reference1, gpc_siso_multiple_reference2, gpc_siso_multiple_reference3, 
                                    siso_change_ref_instant2,siso_change_ref_instant3)
            
            if cancel_button:
                st.rerun()
        
    




def gpc_mimo_tab_form():
    K_alpha = 0
    alpha_fgpc= 0
    controller_type = st.selectbox('Tipo do Controlador',["GPC/FGPC","PID"],key='GPC_controller_type')
    tf_type_col, fgpc_col = st.columns(2)
    with tf_type_col:
        transfer_function_type = st.radio('**Tipo de Função de Transferência**',['Continuo','Discreto'],horizontal=True,key='gpc_mimo_transfer_function_type')
    with fgpc_col:
        f_gpc_mimo_checkbox = st.checkbox('Inserir filtro (FPGPC)?', key = 'f_gpc_mimo_checkbox')
    
    if f_gpc_mimo_checkbox:      
        k_alpha_col, alpha_col = st.columns(2)
        with k_alpha_col:
            K_alpha = st.number_input(label='$K_{\\alpha}$',key='f_gpc_mimo_k_alpha_input')

        with alpha_col:
            alpha_fgpc = st.number_input(label='$\\alpha_{fgpc}$',step=0.1, min_value=0.0, max_value=1.0,key='f_gpc_mimo_alpha_input')
    
    st.write(' **Função de Transferência do Modelo:**')        
    help_text = 'Valores decimais como **0.9** ou **0.1, 0.993**. Para múltiplos valores, vírgula é necessário.'
    
    model_1_num_col, model_1_den_col = st.columns(2)
    
    with model_1_num_col:

        num_coeff_1 = st.text_input('Coeficientes **Numerador 1**:',key='mimo_gpc_num_coeff_1',help=help_text,placeholder='7.737')
        coefficients_validations(num_coeff_1)
    with model_1_den_col:
        
        den_coeff_1 = st.text_input('Coeficientes **Denominador 1**:',key='mimo_gpc_den_coeff_1',help=help_text,placeholder='0.6 , 1')
        coefficients_validations(den_coeff_1)
    
    delay_checkbox_col_1, delay_input_col_1 = st.columns(2)
    with delay_checkbox_col_1:
        delay_checkbox_1=st.checkbox('Atraso de Transporte?', key = 'gpc_mimo_delay_checkbox_1')
        
    with delay_input_col_1:
        if delay_checkbox_1:
            delay_input_1 = st.number_input(label='delay',label_visibility='collapsed',key='gpc_mimo_delay_input_1')
            
    model_2_num_col, model_2_den_col = st.columns(2)
    
    with model_2_num_col:

        num_coeff_2 = st.text_input('Coeficientes **Numerador 2**:',key='mimo_gpc_num_coeff_2',help=help_text,placeholder='12.86')
        coefficients_validations(num_coeff_2)
    with model_2_den_col:
        
        den_coeff_2 = st.text_input('Coeficientes **Denominador 2**:',key='mimo_gpc_den_coeff_2',help=help_text,placeholder='0.66 , 1')
        coefficients_validations(den_coeff_2)
        
    delay_checkbox_col_2, delay_input_col_2 = st.columns(2)
    with delay_checkbox_col_2:
        delay_checkbox_2=st.checkbox('Atraso de Transporte?', key = 'gpc_mimo_delay_checkbox_2')
        
    with delay_input_col_2:
        if delay_checkbox_2:
            delay_input_2 = st.number_input(label='delay',label_visibility='collapsed',key='gpc_mimo_delay_input_2')
    
    
    reference_number = st.radio('Quantidade de referências',['Única','Múltiplas'],horizontal=True,key='gpc_mimo_reference_number')

    if reference_number == 'Única':
        gpc_single_reference = st.number_input(
        'Referência:', value=50, step=1, min_value=0, max_value=90, key='gpc_mimo_single_reference')
    
    elif reference_number == 'Múltiplas':
    
        col21, col22, col23 = st.columns(3)
        with col23:

            gpc_mimo_reference3 = st.number_input(
                'Referência 3:', value=30.0, step=1.0, min_value=0.0, max_value=90.0, key='gpc_mimo_reference3')

        with col22:
            gpc_mimo_reference2 = st.number_input(
                'Referência 2:', value=30.0, step=1.0, min_value=0.0, max_value=90.0, key='gpc_mimo_reference2')

        with col21:
            gpc_mimo_reference1 = st.number_input(
                'Referência:', value=30.0, step=1.0, min_value=0.0, max_value=90.0, key='gpc_mimo_reference1')

        changeReferenceCol1, changeReferenceCol2 = st.columns(2)

        with changeReferenceCol2:
            change_ref_instant3 = st.number_input(
                'Instante da referência 3 (s):', value=calculate_time_limit()*3/4, step=0.1, min_value=0.0, max_value=calculate_time_limit(), key='gpc_mimo_change_ref_instant3')

        with changeReferenceCol1:
            change_ref_instant2 = st.number_input(
                'Instante da referência 2 (s):', value=calculate_time_limit()/2, step=1.0, min_value=0.0, max_value=change_ref_instant3, key='gpc_mimo_change_ref_instant2')
    
    ny_col,nu_col,lambda_col = st.columns(3)
    
    with ny_col:   
        gpc_mimo_ny_1 = st.number_input('$N_{y_1}$', value=1, step=1, min_value=0, key='gpc_mimo_ny_1')
        gpc_mimo_ny_2 = st.number_input('$N_{y_2}$', value=1, step=1, min_value=0, key='gpc_mimo_ny_2')
    with nu_col:   
        gpc_mimo_nu_1 = st.number_input('$N_{u_1}$', value=1, step=1, min_value=0, key='gpc_mimo_nu_1')
        gpc_mimo_nu_2 = st.number_input('$N_{u_2}$', value=1, step=1, min_value=0, key='gpc_mimo_nu_2')
    with lambda_col:   
        gpc_mimo_lambda_1 = st.number_input('$\lambda_{1}$', value=1.0, step=0.1, min_value=0.0, key='gpc_mimo_lambda_1')
        gpc_mimo_lambda_2 = st.number_input('$\lambda_{2}$', value=1.0, step=0.1, min_value=0.0, key='gpc_mimo_lambda_2')
    
    future_inputs_checkbox=st.checkbox('Entradas Futuras?',key= 'gpc_mimo_future_inputs_checkbox')

    start_col,cancel_col = st.columns([0.2,0.8])
    with cancel_col:
        
        cancel_button = st.button('Cancelar', key='gpc_cancel_mimo_button')
        if cancel_button:
            st.stop()
            if 'arduinoData' in st.session_state.connected:
                arduinoData = st.session_state.connected['arduinoData']
                sendToArduino(arduinoData, '0')
            
    with start_col:        
        start_button = st.button('Iniciar', type='primary', key='gpc_mimo_button')
    
    

    if start_button: 
            if controller_type == 'GPC/FGPC':      
                if reference_number == 'Única':
                    
                   gpcControlProcessTISO(transfer_function_type,num_coeff_1,den_coeff_1, num_coeff_2,den_coeff_2,
                                  gpc_mimo_ny_1,gpc_mimo_nu_1,gpc_mimo_lambda_1,
                                  gpc_mimo_ny_2,gpc_mimo_nu_2,gpc_mimo_lambda_2,future_inputs_checkbox,
                                  gpc_single_reference, gpc_single_reference, gpc_single_reference,
                                  f_gpc_mimo_checkbox, K_alpha, alpha_fgpc)
                    
                elif reference_number == 'Múltiplas':
                    gpcControlProcessTISO(transfer_function_type,num_coeff_1,den_coeff_1, num_coeff_2,den_coeff_2,
                                  gpc_mimo_ny_1,gpc_mimo_nu_1,gpc_mimo_lambda_1,
                                  gpc_mimo_ny_2,gpc_mimo_nu_2,gpc_mimo_lambda_2,future_inputs_checkbox, 
                                  gpc_mimo_reference1, gpc_mimo_reference2,gpc_mimo_reference3,
                                  f_gpc_mimo_checkbox, K_alpha, alpha_fgpc,
                                  change_ref_instant2,change_ref_instant3)
                
                if cancel_button:
                    st.rerun()
            
            if controller_type == 'PID':      
                if reference_number == 'Única':
                    
                    gpcPIDControlProcessTISO(transfer_function_type,num_coeff_1,den_coeff_1, num_coeff_2,den_coeff_2,
                                  gpc_mimo_ny_1,gpc_mimo_nu_1,gpc_mimo_lambda_1,
                                  gpc_mimo_ny_2,gpc_mimo_nu_2,gpc_mimo_lambda_2,future_inputs_checkbox,
                                  gpc_single_reference, gpc_single_reference, gpc_single_reference,
                                  f_gpc_mimo_checkbox, K_alpha, alpha_fgpc)
                    
                elif reference_number == 'Múltiplas':
                    gpcPIDControlProcessTISO(transfer_function_type,num_coeff_1,den_coeff_1, num_coeff_2,den_coeff_2,
                                  gpc_mimo_ny_1,gpc_mimo_nu_1,gpc_mimo_lambda_1,
                                  gpc_mimo_ny_2,gpc_mimo_nu_2,gpc_mimo_lambda_2,future_inputs_checkbox, 
                                  gpc_mimo_reference1, gpc_mimo_reference2,gpc_mimo_reference3,
                                  f_gpc_mimo_checkbox, K_alpha, alpha_fgpc,
                                  change_ref_instant2,change_ref_instant3)
                
                if cancel_button:
                    st.rerun()
            
    