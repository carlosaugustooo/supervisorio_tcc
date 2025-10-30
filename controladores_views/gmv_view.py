import streamlit as st
from formatterInputs import *
from connections import *
from controllers_process.validations_functions import *
from controllers_process.gmv_controller_process import gmvControlProcessSISO,gmvControlProcessTISO

def gmv_Controller():
    st.header('Generalized Minimum Variation Incremental Control (GMV)')
    

    graphics_col, gmv_config_col = st.columns([0.7, 0.3])

    with gmv_config_col:
        st.write('### Configurações do Controlador')

        sisoSystemTab, mimoSystemTab = st.tabs(["SISO", "MIMO"])
            #["Única Referência", "Múltiplas Referências"])
        
        with sisoSystemTab:
            gmv_siso_tab_form()

        with mimoSystemTab:
            gmv_mimo_tab_form()
          
           
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
    



def gmv_siso_tab_form():
    transfer_function_type = st.radio('**Tipo de Função de Transferência**',['Continuo','Discreto'],horizontal=True,key='transfer_function_type')
            
    help_text = 'Valores decimais como **0.9** ou **0.1, 0.993**. Para múltiplos valores, vírgula é necessário.'
    st.write(' **Função de Transferência do Modelo:**')
    num_coeff = st.text_input('Coeficientes do **Numerador** :',key='siso_gmv_num_coeff',help=help_text,placeholder='0.994')
    
    coefficients_validations(num_coeff)
        
    den_coeff = st.text_input('Coeficientes do **Denominador** :',key='siso_gmv_den_coeff',help=help_text,placeholder='1.827 , 1')
    coefficients_validations(den_coeff)
    
    delay_checkbox_col, delay_input_col = st.columns(2)
    with delay_checkbox_col:
        delay_checkbox=st.checkbox('Atraso de Transporte?')
        
    with delay_input_col:
        if delay_checkbox:
            delay_input = st.number_input(label='delay',key='delay_input',label_visibility='collapsed')
            
    reference_number = st.radio('Quantidade de referências',['Única','Múltiplas'],horizontal=True,key='gmv_siso_reference_number')
    

    if reference_number == 'Única':
        gmv_single_reference = st.number_input(
        'Referência:', value=50, step=1, min_value=0, max_value=90, key='gmv_siso_single_reference')
        
    else:
        col21, col22, col23 = st.columns(3)
        with col23:

            gmv_siso_multiple_reference3 = st.number_input(
                'Referência 3:', value=30.0, step=1.0, min_value=0.0, max_value=90.0, key='siso_gmv_multiple_reference3')

        with col22:
            gmv_siso_multiple_reference2 = st.number_input(
                'Referência 2:', value=30.0, step=1.0, min_value=0.0, max_value=90.0, key='siso_gmv_multiple_reference2')

        with col21:
            gmv_siso_multiple_reference1 = st.number_input(
                'Referência 1:', value=30.0, step=1.0, min_value=0.0, max_value=90.0, key='siso_gmv_multiple_reference1')

        changeReferenceCol1, changeReferenceCol2 = st.columns(2)

        with changeReferenceCol2:
            siso_change_ref_instant3 = st.number_input(
                'Instante da referência 3 (s):', value=calculate_time_limit()*3/4, step=0.1, min_value=0.0, max_value=calculate_time_limit(), key='siso_change_ref_instant3')

        with changeReferenceCol1:
            siso_change_ref_instant2 = st.number_input(
                'Instante da referência 2 (s):', value=calculate_time_limit()/2,step=1.0, min_value=0.0, max_value=siso_change_ref_instant3, key='siso_change_ref_instant2')
        
    gmv_siso_q0 = float(st.text_input('Magnitude do Sinal de Controle $Q(z^{-1})$', value="1.0", key='gmv_q0'))
    
    
    start_col,cancel_col = st.columns([0.2,0.8])
    with cancel_col:
        
        cancel_button = st.button('Cancelar', key='gmv_cancel_siso_button')
        if cancel_button:
            st.stop()
            if 'arduinoData' in st.session_state.connected:
                arduinoData = st.session_state.connected['arduinoData']
                sendToArduino(arduinoData, '0')
            
    with start_col:        
        start_button = st.button('Iniciar', type='primary', key='gmv_siso_button')
    
    if start_button:       
        if reference_number == 'Única':
            
            gmvControlProcessSISO(transfer_function_type,num_coeff,den_coeff,gmv_siso_q0, gmv_single_reference, gmv_single_reference, gmv_single_reference)
            
        elif reference_number == 'Múltiplas':
            gmvControlProcessSISO(transfer_function_type,num_coeff,den_coeff,gmv_siso_q0, gmv_siso_multiple_reference1, gmv_siso_multiple_reference2, gmv_siso_multiple_reference3, siso_change_ref_instant2,siso_change_ref_instant3)
        
        if cancel_button:
            st.rerun()





def gmv_mimo_tab_form():
    
    transfer_function_type = st.radio('**Tipo de Função de Transferência**',['Continuo','Discreto'],horizontal=True,key='gmv_mimo_transfer_function_type')
    
    st.write(' **Função de Transferência do Modelo:**')        
    help_text = 'Valores decimais como **0.9** ou **0.1, 0.993**. Para múltiplos valores, vírgula é necessário.'
    
    model_1_num_col, model_1_den_col = st.columns(2)
    
    with model_1_num_col:

        num_coeff_1 = st.text_input('Coeficientes **Numerador 1**:',key='mimo_gmv_num_coeff_1',help=help_text,placeholder='7.737')
        coefficients_validations(num_coeff_1)
    with model_1_den_col:
        
        den_coeff_1 = st.text_input('Coeficientes **Denominador 1**:',key='mimo_gmv_den_coeff_1',help=help_text,placeholder='0.6 , 1')
        coefficients_validations(den_coeff_1)
    
    delay_checkbox_col_1, delay_input_col_1 = st.columns(2)
    with delay_checkbox_col_1:
        delay_checkbox_1=st.checkbox('Atraso de Transporte?', key = 'gmv_mimo_delay_checkbox_1')
        
    with delay_input_col_1:
        if delay_checkbox_1:
            delay_input_1 = st.number_input(label='delay',label_visibility='collapsed',key='gmv_mimo_delay_input_1')
            
    model_2_num_col, model_2_den_col = st.columns(2)
    
    with model_2_num_col:

        num_coeff_2 = st.text_input('Coeficientes **Numerador 2**:',key='mimo_gmv_num_coeff_2',help=help_text,placeholder='12.86')
        coefficients_validations(num_coeff_2)
    with model_2_den_col:
        
        den_coeff_2 = st.text_input('Coeficientes **Denominador 2**:',key='mimo_gmv_den_coeff_2',help=help_text,placeholder='0.66 , 1')
        coefficients_validations(den_coeff_2)
        
    delay_checkbox_col_2, delay_input_col_2 = st.columns(2)
    with delay_checkbox_col_2:
        delay_checkbox_2=st.checkbox('Atraso de Transporte?', key = 'gmv_mimo_delay_checkbox_2')
        
    with delay_input_col_2:
        if delay_checkbox_2:
            delay_input_2 = st.number_input(label='delay',label_visibility='collapsed',key='gmv_mimo_delay_input_2')
    
    
    reference_number = st.radio('Quantidade de referências',['Única','Múltiplas'],horizontal=True,key='gmv_mimo_reference_number')

    if reference_number == 'Única':
        gmv_single_reference = st.number_input(
        'Referência:', value=50, step=1, min_value=0, max_value=90, key='gmv_mimo_single_reference')
    
    elif reference_number == 'Múltiplas':
    
        col21, col22, col23 = st.columns(3)
        with col23:

            gmv_mimo_reference3 = st.number_input(
                'Referência 3:', value=30.0, step=1.0, min_value=0.0, max_value=90.0, key='gmv_mimo_reference3')

        with col22:
            gmv_mimo_reference2 = st.number_input(
                'Referência 2:', value=30.0, step=1.0, min_value=0.0, max_value=90.0, key='gmv_mimo_reference2')

        with col21:
            gmv_mimo_reference1 = st.number_input(
                'Referência:', value=30.0, step=1.0, min_value=0.0, max_value=90.0, key='gmv_mimo_reference1')

        changeReferenceCol1, changeReferenceCol2 = st.columns(2)

        with changeReferenceCol2:
            change_ref_instant3 = st.number_input(
                'Instante da referência 3 (s):', value=calculate_time_limit()*3/4, step=0.1, min_value=0.0, max_value=calculate_time_limit(), key='gmv_mimo_change_ref_instant3')

        with changeReferenceCol1:
            change_ref_instant2 = st.number_input(
                'Instante da referência 2 (s):', value=calculate_time_limit()/2, step=1.0, min_value=0.0, max_value=change_ref_instant3, key='gmv_mimo_change_ref_instant2')
    
    st.write('Magnitude do Sinal de Controle $Q(z^{-1})$')
    tau_mf_col1, tau_mf_col2 = st.columns(2)
    with tau_mf_col1:
        gmv_mimo_q01 = float(st.text_input('$Q_1(z^{-1})$', value="1.0", key='gmv_mr_tau_mf1'))
    with tau_mf_col2:
        gmv_mimo_q02 = float(st.text_input('$Q_2(z^{-1})$', value="1.0", key='gmv_mr_tau_mf2'))
        
    if st.button('Receber Dados', type='primary', key='gmv_mimo_setpoint_button'):
            
            
        if reference_number == 'Única':
            
            gmvControlProcessTISO(transfer_function_type,num_coeff_1,den_coeff_1, num_coeff_2,den_coeff_2,
                                  gmv_mimo_q01,gmv_mimo_q02, 
                                  gmv_single_reference, gmv_single_reference, gmv_single_reference)
            
        elif reference_number == 'Múltiplas':
            gmvControlProcessTISO(transfer_function_type,num_coeff_1,den_coeff_1, num_coeff_2,den_coeff_2,
                                  gmv_mimo_q01,gmv_mimo_q02, 
                                  gmv_mimo_reference1, gmv_mimo_reference2,gmv_mimo_reference3,
                                  change_ref_instant2,change_ref_instant3)
