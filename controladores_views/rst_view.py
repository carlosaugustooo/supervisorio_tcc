import streamlit as st
from formatterInputs import *
from connections import *
from session_state import *
from controllers_process.validations_functions import *
# Importa apenas as funções SISO (Incremental e Adaptativa)
from controllers_process.rst_controller_process import rstControlProcessIncrementalSISO, rstControlProcessAdaptiveSISO

def calculate_time_limit():
    sim_time = get_session_variable('simulation_time')
    return sim_time if sim_time is not None else 60.0 # Retorna 60s como padrão

def rst_Controller_Interface():
    st.header('Regulador de Alocação de Polos (RST)')
    graphics_col, rst_config_col = st.columns([0.7, 0.3])

    with rst_config_col:
        st.write('### Configurações do Controlador')
        
        # Abas para os tipos de RST
        incremental_tab, adaptative_tab = st.tabs(["RST Incremental", "RST Adaptativo"])       
        
        # --- Configuração do RST Incremental ---
        with incremental_tab:
            st.write('#### Configuração do RST Incremental (SISO)')
            # Removemos a aba MIMO, chamamos o formulário SISO diretamente
            rst_incremental_siso_tab_form()           

        # --- Configuração do RST Adaptativo ---
        with adaptative_tab:
            st.write('#### Configuração do RST Adaptativo (SISO)')
            rst_adaptive_siso_tab_form()

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

# --- FORMULÁRIO DO RST INCREMENTAL ---
def rst_incremental_siso_tab_form():

    transfer_function_type = st.radio('**Tipo de Função de Transferência**',['Continuo','Discreto'],horizontal=True,key='rst_inc_transfer_function_type')
    help_text = 'Valores decimais como **0.9** ou **0.1, 0.993**. Para múltiplos valores, vírgula é necessário.'
    st.write(' **Função de Transferência do Modelo (B/A):**')

    num_coeff = st.text_input('Coeficientes do **Numerador B** :',key='rst_inc_num_coeff',help=help_text,placeholder='b0, b1, ...')
    coefficients_validations(num_coeff)
    den_coeff = st.text_input('Coeficientes do **Denominador A** :',key='rst_inc_den_coeff',help=help_text,placeholder='1, a1, a2, ...')
    coefficients_validations(den_coeff)
    
    st.write('**Constante de Tempo de malha fechada desejada (tau):**')
    tau_ml_input = st.number_input(
        'Constante de Tempo em Malha fechada (s):',
        value = 0.5,        # valor sugerido inicial
        min_value = 0.01,   # valor mínimo
        step = 0.01,        # passo de incremento
        key = 'rst_inc_tau_ml'
    )

    st.write('**Estrutura do Controlador (Sintonia PID via RST):**')
    pid_structure = st.selectbox(
        'Selecione a estrutura de controle:',
        ('RST Incremental Puro', 'I + PD', 'PI + D', 'PID Ideal', 'PID Paralelo'),
        key='rst_pid_structure',
        help='Selecione a lei de controle. As estruturas PID usam os parâmetros RST (s0, s1, t0) para calcular os ganhos.'
    )

    delay_checkbox_col, delay_input_col = st.columns(2)
    with delay_checkbox_col:
        delay_checkbox=st.checkbox('Atraso de Transporte?', key='rst_inc_delay_checkbox')

    with delay_input_col:
        if delay_checkbox:
            delay_input = st.number_input(label='delay',key='rst_inc_delay_input',label_visibility='collapsed')

    reference_number = st.radio('Quantidade de referências',['Única','Múltiplas'],horizontal=True,key='rst_inc_siso_reference_number')
   
    if reference_number == 'Única':
        rst_inc_single_reference = st.number_input(
        'Referência:', value=50, step=1, min_value=0, max_value=90, key='rst_inc_siso_single_reference')
    else:
        col21, col22, col23 = st.columns(3)
        with col23:
            rst_inc_siso_multiple_reference3 = st.number_input(
                'Referência 3:', value=30.0, step=1.0, min_value=0.0, max_value=90.0, key='siso_rst_inc_multiple_reference3')
        with col22:
            rst_inc_siso_multiple_reference2 = st.number_input(
                'Referência 2:', value=30.0, step=1.0, min_value=0.0, max_value=90.0, key='siso_rst_inc_multiple_reference2')
        with col21:
            rst_inc_siso_multiple_reference1 = st.number_input(
                'Referência 1:', value=30.0, step=1.0, min_value=0.0, max_value=90.0, key='siso_rst_inc_multiple_reference1')
        
        changeReferenceCol1, changeReferenceCol2 = st.columns(2)

        with changeReferenceCol2:
            siso_change_ref_instant3 = st.number_input(
                'Instante da referência 3 (s):', value=calculate_time_limit()*3/4, step=0.1, min_value=0.0, max_value=calculate_time_limit(), key='siso_change_ref_instant3')
        
        with changeReferenceCol1:
            default_instante_2 = siso_change_ref_instant3 / 2.0
            siso_change_ref_instant2 = st.number_input(
                'Instante da referência 2 (s):', 
                value=default_instante_2, 
                step=1.0, 
                min_value=0.0, 
                max_value=siso_change_ref_instant3, 
                key='siso_change_ref_instant2')

    if st.button('Iniciar', type='primary', key='rst_inc_siso_button'):
       
        if reference_number == 'Única':
            rstControlProcessIncrementalSISO(transfer_function_type, num_coeff, den_coeff, tau_ml_input, 
                                     pid_structure,
                                     rst_inc_single_reference,
                                     rst_inc_single_reference,
                                     rst_inc_single_reference)
      
        elif reference_number == 'Múltiplas':
           rstControlProcessIncrementalSISO(transfer_function_type,num_coeff,den_coeff,tau_ml_input, 
                                     pid_structure,
                                     rst_inc_siso_multiple_reference1, rst_inc_siso_multiple_reference2, rst_inc_siso_multiple_reference3, 
                                     siso_change_ref_instant2,siso_change_ref_instant3)

# --- FORMULÁRIO DO RST ADAPTATIVO ---
def rst_adaptive_siso_tab_form():
    
    st.write('#### Modelo Inicial para Estimação (B/A)')
    
    transfer_function_type = st.radio(
        '**Tipo de Função de Transferência**',
        ['Continuo','Discreto'],
        horizontal=True,
        key='rst_adp_transfer_function_type' # Chave única
    )
    
    help_text = 'Valores decimais como **0.9** ou **0.1, 0.993**. Para múltiplos valores, vírgula é necessário.'
    
    num_coeff = st.text_input(
        'Coeficientes do **Numerador B** :',
        key='rst_adp_num_coeff', # Chave única
        help=help_text,
        placeholder='b0, b1, ...'
    )
    coefficients_validations(num_coeff)
    
    den_coeff = st.text_input(
        'Coeficientes do **Denominador A** :',
        key='rst_adp_den_coeff', # Chave única
        help=help_text,
        placeholder='1, a1, a2, ...'
    )
    coefficients_validations(den_coeff)

    st.write('**Parâmetros Iniciais do Estimador (MQR):**')
    
    p0_exponent = st.number_input(
        'Expoente de P(0) (10^x)', 
        value=4.0, 
        step=1.0, 
        key='rst_adp_p0_exponent', 
        help="Valor do expoente 'x' para calcular P(0) = 10^x. Ex: 4 -> P(0) = 10000."
    )

    st.write('**Constante de Tempo de malha fechada desejada (tau):**')
    tau_ml_input = st.number_input(
        'Constante de Tempo em Malha fechada (s):',
        value = 0.5,
        min_value = 0.01,
        step = 0.01,
        key = 'rst_adp_tau_ml'
    )

    st.write('**Estrutura do Controlador (Sintonia PID via RST):**')
    pid_structure = st.selectbox(
        'Selecione a estrutura de controle:',
        ('RST Incremental Puro', 'I + PD', 'PI + D', 'PID Ideal', 'PID Paralelo'),
        key='rst_adp_pid_structure',
        help='Selecione a lei de controle. Os ganhos serão adaptados a cada passo.'
    )

    reference_number = st.radio('Quantidade de referências',['Única','Múltiplas'],horizontal=True,key='rst_adp_siso_reference_number')
    
    if reference_number == 'Única':
        rst_adp_single_reference = st.number_input(
        'Referência:', value=50, step=1, min_value=0, max_value=90, key='rst_adp_siso_single_reference')
    else:
        col21, col22, col23 = st.columns(3)
        with col23:
            rst_adp_multiple_reference3 = st.number_input(
                'Referência 3:', value=30.0, step=1.0, min_value=0.0, max_value=90.0, key='siso_rst_adp_multiple_reference3')
        with col22:
            rst_adp_multiple_reference2 = st.number_input(
                'Referência 2:', value=30.0, step=1.0, min_value=0.0, max_value=90.0, key='siso_rst_adp_multiple_reference2')
        with col21:
            rst_adp_multiple_reference1 = st.number_input(
                'Referência 1:', value=30.0, step=1.0, min_value=0.0, max_value=90.0, key='siso_rst_adp_multiple_reference1')
        
        changeReferenceCol1, changeReferenceCol2 = st.columns(2)

        with changeReferenceCol2:
            siso_change_ref_instant3 = st.number_input(
                'Instante da referência 3 (s):', value=calculate_time_limit()*3/4, step=0.1, min_value=0.0, max_value=calculate_time_limit(), key='siso_adp_change_ref_instant3')

        with changeReferenceCol1:
            default_instante_2 = siso_change_ref_instant3 / 2.0
            siso_change_ref_instant2 = st.number_input(
                'Instante da referência 2 (s):', 
                value=default_instante_2, 
                step=1.0, 
                min_value=0.0, 
                max_value=siso_change_ref_instant3, 
                key='siso_adp_change_ref_instant2')

    if st.button('Iniciar', type='primary', key='rst_adp_siso_button'):
        
        if num_coeff == '' or den_coeff == '':
            st.error("FALHA (Front-end): Coeficientes do Modelo Inicial (A ou B) estão vazios.")
            return

        sampling_time = get_session_variable('sampling_time')
        if sampling_time is None:
            st.error("FALHA (Front-end): Tempo de amostragem não definido na Sidebar.")
            return

        A_coeff_all, B_coeff_all = convert_tf_2_discrete(num_coeff, den_coeff, transfer_function_type)
        
        if A_coeff_all.size < 2 or B_coeff_all.size < 1:
            st.error(f'FALHA (Front-end): O modelo inicial (A={A_coeff_all}, B={B_coeff_all}) não é de 1ª ordem.')
            return
            
        a1_initial = A_coeff_all[1]
        b0_initial = B_coeff_all[0]
        
        p0_initial = 10.0 ** p0_exponent
        
        if reference_number == 'Única':
            rstControlProcessAdaptiveSISO(tau_ml_input, 
                                          pid_structure,
                                          a1_initial,
                                          b0_initial,
                                          p0_initial,
                                          rst_adp_single_reference,
                                          rst_adp_single_reference, 
                                          rst_adp_single_reference)
      
        elif reference_number == 'Múltiplas':
            rstControlProcessAdaptiveSISO(tau_ml_input, 
                                          pid_structure,
                                          a1_initial,
                                          b0_initial,
                                          p0_initial,
                                          rst_adp_multiple_reference1, 
                                          rst_adp_multiple_reference2, 
                                          rst_adp_multiple_reference3, 
                                          siso_change_ref_instant2,
                                          siso_change_ref_instant3)