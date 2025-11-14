import streamlit as st
from formatterInputs import *
from connections import *
from session_state import *
from controllers_process.validations_functions import *
# Importa apenas a função SISO
from controllers_process.gmv_controller_process import gmvControlProcessSISO 

def calculate_time_limit():
    sim_time = get_session_variable('simulation_time')
    return sim_time if sim_time is not None else 60.0

def gmv_Controller_Interface():
    st.header('Controlador Preditivo Generalizado (GMV)')
    graphics_col, gmv_config_col = st.columns([0.7, 0.3])

    with gmv_config_col:
        st.write('### Configurações do Controlador')
        
        # Removemos as abas sisoSystemTab e mimoSystemTab
        st.write('#### Configuração do Sistema (SISO)')
        gmv_siso_tab_form() # Chamamos o formulário SISO diretamente

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

def gmv_siso_tab_form():
    transfer_function_type = st.radio('**Tipo de Função de Transferência**',['Continuo','Discreto'],horizontal=True,key='gmv_siso_transfer_function_type')

    st.write(' **Função de Transferência do Modelo:**')
    help_text = 'Valores decimais como **0.9** ou **0.1, 0.993**. Para múltiplos valores, vírgula é necessário.'
    num_coeff = st.text_input('Coeficientes **Numerador**:',key='siso_gmv_num_coeff',help=help_text,placeholder='7.737')
    coefficients_validations(num_coeff)
    den_coeff = st.text_input('Coeficientes **Denominador**:',key='siso_gmv_den_coeff',help=help_text,placeholder='0.6 , 1')
    coefficients_validations(den_coeff)

    delay_checkbox_col, delay_input_col = st.columns(2)
    with delay_checkbox_col:
        delay_checkbox=st.checkbox('Atraso de Transporte?',key='siso_gmv_delay_checkbox')
    
    with delay_input_col:
        if delay_checkbox:
            delay_input = st.number_input(label='delay',label_visibility='collapsed',key='siso_gmv_delay_input')
    
    reference_number = st.radio('Quantidade de referências',['Única','Múltiplas'],horizontal=True,key='gmv_siso_reference_number')
    
    if reference_number == 'Única':
        gmv_siso_single_reference = st.number_input(
        'Referência:', value=50, step=1, min_value=0, max_value=90, key='gmv_siso_single_reference')
    
    elif reference_number == 'Múltiplas':
        col21, col22, col23 = st.columns(3)
        with col23:
            gmv_siso_multiple_reference3 = st.number_input(
                'Referência 3:', value=30.0, step=1.0, min_value=0.0, max_value=90.0, key='gmv_siso_multiple_reference3')

        with col22:
            gmv_siso_multiple_reference2 = st.number_input(
                'Referência 2:', value=30.0, step=1.0, min_value=0.0, max_value=90.0, key='gmv_siso_multiple_reference2')

        with col21:
            gmv_siso_multiple_reference1 = st.number_input(
                'Referência 1:', value=30.0, step=1.0, min_value=0.0, max_value=90.0, key='gmv_siso_multiple_reference1')

        changeReferenceCol1, changeReferenceCol2 = st.columns(2)

        with changeReferenceCol2:
            siso_change_ref_instant3 = st.number_input(
                'Instante da referência 3 (s):', value=calculate_time_limit()*3/4, step=0.1, min_value=0.0, max_value=calculate_time_limit(), key='siso_gmv_change_ref_instant3')

        with changeReferenceCol1:
            default_instante_2 = siso_change_ref_instant3 / 2.0
            siso_change_ref_instant2 = st.number_input(
                'Instante da referência 2 (s):', 
                value=default_instante_2, 
                step=1.0, 
                min_value=0.0, 
                max_value=siso_change_ref_instant3, 
                key='siso_gmv_change_ref_instant2')

    st.write('**Parâmetro de Ponderação (Q):**')
    gmv_q01 = float(st.text_input('$\lambda_1$', value="0.9", key='siso_gmv_q01'))

    if st.button('Iniciar', type='primary', key='gmv_siso_button'):
        
        if reference_number == 'Única':
            gmvControlProcessSISO(transfer_function_type,num_coeff,den_coeff,
                                  gmv_q01,
                                  gmv_siso_single_reference, gmv_siso_single_reference, gmv_siso_single_reference)
        
        elif reference_number == 'Múltiplas':
            gmvControlProcessSISO(transfer_function_type,num_coeff,den_coeff,
                                  gmv_q01,
                                  gmv_siso_multiple_reference1, gmv_siso_multiple_reference2,gmv_siso_multiple_reference3,
                                  siso_change_ref_instant2,siso_change_ref_instant3)

# A função gmv_mimo_tab_form() foi removida.