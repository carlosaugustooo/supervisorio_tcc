import streamlit as st
from formatterInputs import *
from connections import *
from session_state import *
from controllers_process.validations_functions import *
# Importa apenas as funções SISO
from controllers_process.gpc_controller_process import gpcControlProcessSISO, gpcPidControlProcessSISO

def calculate_time_limit():
    sim_time = get_session_variable('simulation_time')
    return sim_time if sim_time is not None else 60.0

def gpc_Controller_Interface():
    st.header('Controlador Preditivo Generalizado (GPC)')
    graphics_col, gpc_config_col = st.columns([0.7, 0.3])

    with gpc_config_col:
        st.write('### Configurações do Controlador')
        
        # Removemos as abas sisoSystemTab e mimoSystemTab
        st.write('#### Configuração do Sistema (SISO)')
        gpc_siso_tab_form() # Chamamos o formulário SISO diretamente

    with graphics_col:
        y_max = get_session_variable('saturation_max_value')
        y_min = get_session_variable('saturation_min_value')

        if get_session_variable('process_output_sensor'):
            process_output_dataframe = dataframeToPlot('process_output_sensor','Process Output','reference_input')
            st.subheader('Resposta do Sistema')
            altair_plot_chart_validation(process_output_dataframe,
                                         y_max = y_max,y_min = y_min,
                                         x_column = 'Time (s)', y_column = ['Reference','Process Output'])    
        
        st.subheader('Sinal de Controle')
        if get_session_variable('control_signal_1'):
            control_signal_with_elapsed_time = datetime_obj_to_elapsed_time('control_signal_1')
            control_signal_1_dataframe = dictionary_to_pandasDataframe(control_signal_with_elapsed_time,'Control Signal 1')
            altair_plot_chart_validation(control_signal_1_dataframe,control= True,
                                         y_max = y_max,y_min = y_min,
                                         x_column = 'Time (s)', y_column = 'Control Signal 1',
                                         height=250)

    st.write('### Índices de Desempenho')
    iae_col, tvc_col = st.columns([0.2,0.8])
    with iae_col:
        iae_metric_validation()
    with tvc_col:
        tvc1_validation()

def gpc_siso_tab_form():
    
    gpc_controller_type = st.radio('**Tipo de Controlador GPC**',['GPC Clássico','GPC-PID'],horizontal=True,key='gpc_siso_controller_type')
    
    transfer_function_type = st.radio('**Tipo de Função de Transferência**',['Continuo','Discreto'],horizontal=True,key='gpc_siso_transfer_function_type')

    st.write(' **Função de Transferência do Modelo:**')
    help_text = 'Valores decimais como **0.9** ou **0.1, 0.993**. Para múltiplos valores, vírgula é necessário.'
    num_coeff = st.text_input('Coeficientes **Numerador**:',key='siso_gpc_num_coeff',help=help_text,placeholder='7.737')
    coefficients_validations(num_coeff)
    den_coeff = st.text_input('Coeficientes **Denominador**:',key='siso_gpc_den_coeff',help=help_text,placeholder='0.6 , 1')
    coefficients_validations(den_coeff)

    delay_checkbox_col, delay_input_col = st.columns(2)
    with delay_checkbox_col:
        delay_checkbox=st.checkbox('Atraso de Transporte?',key='siso_gpc_delay_checkbox')
    
    with delay_input_col:
        if delay_checkbox:
            delay_input = st.number_input(label='delay',label_visibility='collapsed',key='siso_gpc_delay_input')
    
    st.write('**Parâmetros de Sintonia GPC:**')
    ny_col, nu_col, lambda_col = st.columns(3)
    with ny_col:
        gpc_siso_ny = st.number_input('$N_y$', value=10, step=1, min_value=1, max_value=100, key='gpc_siso_ny')
    with nu_col:
        gpc_siso_nu = st.number_input('$N_u$', value=3, step=1, min_value=1, max_value=100, key='gpc_siso_nu')
    with lambda_col:
        gpc_siso_lambda = st.number_input('$\lambda$', value=0.9, step=0.1, min_value=0.0, max_value=1000.0, key='gpc_siso_lambda',format='%.2f')

    future_inputs_checkbox=st.checkbox('Considerar Referências Futuras?',key='siso_gpc_future_inputs_checkbox')
    
    reference_number = st.radio('Quantidade de referências',['Única','Múltiplas'],horizontal=True,key='gpc_siso_reference_number')
    
    if reference_number == 'Única':
        gpc_siso_single_reference = st.number_input(
        'Referência:', value=50, step=1, min_value=0, max_value=90, key='gpc_siso_single_reference')
    
    elif reference_number == 'Múltiplas':
        col21, col22, col23 = st.columns(3)
        with col23:
            gpc_siso_multiple_reference3 = st.number_input(
                'Referência 3:', value=30.0, step=1.0, min_value=0.0, max_value=90.0, key='gpc_siso_multiple_reference3')

        with col22:
            gpc_siso_multiple_reference2 = st.number_input(
                'Referência 2:', value=30.0, step=1.0, min_value=0.0, max_value=90.0, key='gpc_siso_multiple_reference2')

        with col21:
            gpc_siso_multiple_reference1 = st.number_input(
                'Referência 1:', value=30.0, step=1.0, min_value=0.0, max_value=90.0, key='gpc_siso_multiple_reference1')

        changeReferenceCol1, changeReferenceCol2 = st.columns(2)

        with changeReferenceCol2:
            siso_change_ref_instant3 = st.number_input(
                'Instante da referência 3 (s):', value=calculate_time_limit()*3/4, step=0.1, min_value=0.0, max_value=calculate_time_limit(), key='siso_gpc_change_ref_instant3')

        with changeReferenceCol1:
            default_instante_2 = siso_change_ref_instant3 / 2.0
            siso_change_ref_instant2 = st.number_input(
                'Instante da referência 2 (s):', 
                value=default_instante_2, 
                step=1.0, 
                min_value=0.0, 
                max_value=siso_change_ref_instant3, 
                key='siso_gpc_change_ref_instant2')

    if st.button('Iniciar', type='primary', key='gpc_siso_button'):
        
        if gpc_controller_type == 'GPC Clássico':
            if reference_number == 'Única':
                gpcControlProcessSISO(transfer_function_type,num_coeff,den_coeff,
                                    gpc_siso_ny,gpc_siso_nu,gpc_siso_lambda,future_inputs_checkbox,
                                    gpc_siso_single_reference, gpc_siso_single_reference, gpc_siso_single_reference)
            
            elif reference_number == 'Múltiplas':
                gpcControlProcessSISO(transfer_function_type, num_coeff, den_coeff,gpc_siso_ny, gpc_siso_nu, gpc_siso_lambda, future_inputs_checkbox,gpc_siso_multiple_reference1, gpc_siso_multiple_reference2, gpc_siso_multiple_reference3,change_ref_instant2=siso_change_ref_instant2, change_ref_instant3=siso_change_ref_instant3) 
        
        elif gpc_controller_type == 'GPC-PID':
            if reference_number == 'Única':
                gpcPidControlProcessSISO(transfer_function_type,num_coeff,den_coeff,
                                    gpc_siso_ny,gpc_siso_nu,gpc_siso_lambda,future_inputs_checkbox,
                                    gpc_siso_single_reference, gpc_siso_single_reference, gpc_siso_single_reference)
            
            elif reference_number == 'Múltiplas':
                gpcPidControlProcessSISO(transfer_function_type, num_coeff, den_coeff,
                                    gpc_siso_ny, gpc_siso_nu, gpc_siso_lambda, future_inputs_checkbox,
                                    gpc_siso_multiple_reference1, gpc_siso_multiple_reference2, gpc_siso_multiple_reference3,
                                    change_ref_instant2=siso_change_ref_instant2,  # <--- Uso explícito do nome do argumento
                                    change_ref_instant3=siso_change_ref_instant3)
# A função gpc_mimo_tab_form() foi removida.