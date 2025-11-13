import streamlit as st
from formatterInputs import *
from connections import *
from session_state import *
from controllers_process.validations_functions import *
from controllers_process.rst_controller_process import rstControlProcessIncrementalSISO,imcControlProcessTISO

# Esta função faltava e causaria um NameError.
def calculate_time_limit():
    sim_time = get_session_variable('simulation_time')
    return sim_time if sim_time is not None else 60.0 # Retorna 60s como padrão
# -----------------------------------------------
def rst_Controller_Interface():
    st.header('Regulador de Alocação de Polos (RST)')
    graphics_col, rst_config_col = st.columns([0.7, 0.3])

    with rst_config_col:
        st.write('### Configurações do Controlador')
        # 1. Novas Abas para as Variações do RST

        incremental_tab, adaptative_tab = st.tabs(["RST Incremental", "RST Adaptativo"])       
        # 2. RST Incremental (Primeira Versão)

        with incremental_tab:

            st.write('#### Configuração do RST Incremental')

            # Você pode manter a variação SISO/MIMO aqui dentro, ou simplificar:
            sisoSystemTab, mimoSystemTab = st.tabs(["SISO", "MIMO"])

            with sisoSystemTab:

                rst_incremental_siso_tab_form()           

            with mimoSystemTab:

            # Implementação MIMO pendente
                st.info("Implementação MIMO Incremental pendente.")
                # st_incremental_mimo_form()

        # 3. RST Adaptativo (Será implementado depois, por enquanto vazio)

        with adaptative_tab:
            st.warning("Implementação do RST Adaptativo pendente.")
            # rst_adaptativo_siso_form()  

    with graphics_col:

        # plot_chart_validation(control_signal_2_dataframe, x = 'Time (s)', y = 'Control Signal 2',height=200)

        y_max = get_session_variable('saturation_max_value')
        y_min = get_session_variable('saturation_min_value')

        if get_session_variable('process_output_sensor'):
            process_output_dataframe = dataframeToPlot('process_output_sensor','Process Output','reference_input')
            st.subheader('Resposta do Sistema')
            plot_chart_validation(process_output_dataframe, x = 'Time (s)', y = ['Reference','Process Output'],height=500)

            #altair_plot_chart_validation(process_output_dataframe,
                                        #  y_max = y_max,y_min = y_min,
                                        #  x_column = 'Time (s)', y_column = ['Reference','Process Output'],
                                        #  )    

        st.subheader('Sinal de Controle')
        if get_session_variable('control_signal_1'):
            control_signal_with_elapsed_time = datetime_obj_to_elapsed_time('control_signal_1')
            control_signal_1_dataframe = dictionary_to_pandasDataframe(control_signal_with_elapsed_time,'Control Signal 1')
         
            plot_chart_validation(control_signal_1_dataframe, x = 'Time (s)', y = 'Control Signal 1',height=200)

            # APAGUE OS ESPAÇOS A MAIS DA LINHA SEGUINTE:
            altair_plot_chart_validation(control_signal_1_dataframe,control= True,
                                         y_max = y_max,y_min = y_min,
                                         x_column = 'Time (s)', y_column = 'Control Signal 1',
                                         height=250)

       

        if get_session_variable('control_signal_2'):
            control_signal_2_with_elapsed_time = datetime_obj_to_elapsed_time('control_signal_2')
            control_signal_2_dataframe = dictionary_to_pandasDataframe(control_signal_2_with_elapsed_time,'Control Signal 2')

            altair_plot_chart_validation(control_signal_2_dataframe,control= True,

                                         y_max = y_max,y_min = y_min,

                                         x_column = 'Time (s)', y_column = 'Control Signal 2',
                                         height=250)

       

    st.write('### Índices de Desempenho')

    iae_col, tvc_col = st.columns([0.2,0.8])
    with iae_col:
        iae_metric_validation()
    with tvc_col:
        tvc1_validation()

def rst_incremental_siso_tab_form():

    transfer_function_type = st.radio('**Tipo de Função de Transferência**',['Continuo','Discreto'],horizontal=True,key='rst_inc_transfer_function_type')
    help_text = 'Valores decimais como **0.9** ou **0.1, 0.993**. Para múltiplos valores, vírgula é necessário.'
    st.write(' **Função de Transferência do Modelo (B/A):**')

 # ... (Inputs para Numerador B e Denominador A)
    num_coeff = st.text_input('Coeficientes do **Numerador B** :',key='rst_inc_num_coeff',help=help_text,placeholder='b0, b1, ...')
    coefficients_validations(num_coeff)  # <--- CORRIGIDO
    den_coeff = st.text_input('Coeficientes do **Denominador A** :',key='rst_inc_den_coeff',help=help_text,placeholder='1, a1, a2, ...')
    coefficients_validations(den_coeff)  # <--- CORRIGIDO
    st.write('**Constante de Tempo de malha fechada desejada (tau):**')

    # Input para a constante de tempo de malha fechada (tau)
    tau_ml_input = st.number_input(
        'Constante de Tempo em Malha fechada (s):',
        value = 0.5,        # valor sugerido inicial
        min_value = 0.01,   # valor mínimo
        step = 0.01,        # passo de incremento
        key = 'rst_inc_tau_ml'
    )

    delay_checkbox_col, delay_input_col = st.columns(2)
    with delay_checkbox_col:
        delay_checkbox=st.checkbox('Atraso de Transporte?')

    with delay_input_col:
        if delay_checkbox:
            delay_input = st.number_input(label='delay',key='delay_input',label_visibility='collapsed')

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
            siso_change_ref_instant2 = st.number_input(
                'Instante da referência 2 (s):', value=calculate_time_limit()/2,step=1.0, min_value=0.0, max_value=siso_change_ref_instant3, key='siso_change_ref_instant2')
       

    if st.button('Iniciar', type='primary', key='rst_inc_siso_button'):
       
        if reference_number == 'Única':
        
            rstControlProcessIncrementalSISO(transfer_function_type, num_coeff, den_coeff, tau_ml_input, rst_inc_single_reference, None, None, None, None)
      
        elif reference_number == 'Múltiplas':

           rstControlProcessIncrementalSISO(transfer_function_type,num_coeff,den_coeff,tau_ml_input, rst_inc_siso_multiple_reference1, rst_inc_siso_multiple_reference2, rst_inc_siso_multiple_reference3, siso_change_ref_instant2,siso_change_ref_instant3)
        
def imc_mimo_tab_form():


    transfer_function_type = st.radio('**Tipo de Função de Transferência**',['Continuo','Discreto'],horizontal=True,key='imc_mimo_transfer_function_type')

    st.write(' **Função de Transferência do Modelo:**')        

    help_text = 'Valores decimais como **0.9** ou **0.1, 0.993**. Para múltiplos valores, vírgula é necessário.'

    model_1_num_col, model_1_den_col = st.columns(2)


    with model_1_num_col:

        num_coeff_1 = st.text_input('Coeficientes **Numerador 1**:',key='mimo_imc_num_coeff_1',help=help_text,placeholder='7.737')

        coefficients_validations(num_coeff_1)

    with model_1_den_col:

        den_coeff_1 = st.text_input('Coeficientes **Denominador 1**:',key='mimo_imc_den_coeff_1',help=help_text,placeholder='0.6 , 1')

        coefficients_validations(den_coeff_1)

    delay_checkbox_col_1, delay_input_col_1 = st.columns(2)

    with delay_checkbox_col_1:

        delay_checkbox_1=st.checkbox('Atraso de Transporte?', key = 'imc_mimo_delay_checkbox_1')

       

    with delay_input_col_1:

        if delay_checkbox_1:

            delay_input_1 = st.number_input(label='delay',label_visibility='collapsed',key='imc_mimo_delay_input_1')

           

    model_2_num_col, model_2_den_col = st.columns(2)

   

    with model_2_num_col:



        num_coeff_2 = st.text_input('Coeficientes **Numerador 2**:',key='mimo_imc_num_coeff_2',help=help_text,placeholder='12.86')

        coefficients_validations(num_coeff_2)

    with model_2_den_col:

       

        den_coeff_2 = st.text_input('Coeficientes **Denominador 2**:',key='mimo_imc_den_coeff_2',help=help_text,placeholder='0.66 , 1')

        coefficients_validations(den_coeff_2)

       

    delay_checkbox_col_2, delay_input_col_2 = st.columns(2)

    with delay_checkbox_col_2:

        delay_checkbox_2=st.checkbox('Atraso de Transporte?', key = 'imc_mimo_delay_checkbox_2')

       

    with delay_input_col_2:

        if delay_checkbox_2:

            delay_input_2 = st.number_input(label='delay',label_visibility='collapsed',key='imc_mimo_delay_input_2')

   

   

    reference_number = st.radio('Quantidade de referências',['Única','Múltiplas'],horizontal=True,key='imc_mimo_reference_number')



    if reference_number == 'Única':

        imc_single_reference = st.number_input(

        'Referência:', value=50, step=1, min_value=0, max_value=90, key='imc_mimo_single_reference')

   

    elif reference_number == 'Múltiplas':

   

        col21, col22, col23 = st.columns(3)

        with col23:



            imc_mimo_reference3 = st.number_input(

                'Referência 3:', value=30.0, step=1.0, min_value=0.0, max_value=90.0, key='imc_mimo_reference3')



        with col22:

            imc_mimo_reference2 = st.number_input(

                'Referência 2:', value=30.0, step=1.0, min_value=0.0, max_value=90.0, key='imc_mimo_reference2')



        with col21:

            imc_mimo_reference1 = st.number_input(

                'Referência:', value=30.0, step=1.0, min_value=0.0, max_value=90.0, key='imc_mimo_reference1')



        changeReferenceCol1, changeReferenceCol2 = st.columns(2)



        with changeReferenceCol2:

            change_ref_instant3 = st.number_input(

                'Instante da referência 3 (s):', value=calculate_time_limit()*3/4, step=1.0, min_value=0.0, max_value=calculate_time_limit(), key='imc_mimo_change_ref_instant3')



        with changeReferenceCol1:

            change_ref_instant2 = st.number_input(

                'Instante da referência 2 (s):', value=calculate_time_limit()/2, step=1.0, min_value=0.0, max_value=change_ref_instant3, key='imc_mimo_change_ref_instant2')

   

    st.write('Constante de Tempo de Malha Fechada ($\\tau$)')

    tau_mf_col1, tau_mf_col2 = st.columns(2)

    with tau_mf_col1:

        imc_mimo_tau_mf1 = float(st.text_input('$\\tau_1$', value="0.9", key='imc_mr_tau_mf1'))

    with tau_mf_col2:

        imc_mimo_tau_mf2 = float(st.text_input('$\\tau_2$', value="0.9", key='imc_mr_tau_mf2'))



    if st.button('Receber Dados', type='primary', key='imc_mimo_setpoint_button'):

           

           

        if reference_number == 'Única':

           

            imcControlProcessTISO(transfer_function_type,num_coeff_1,den_coeff_1, num_coeff_2,den_coeff_2,

                                  imc_mimo_tau_mf1,imc_mimo_tau_mf2,

                                  imc_single_reference, imc_single_reference, imc_single_reference)

           

        elif reference_number == 'Múltiplas':

            imcControlProcessTISO(transfer_function_type,num_coeff_1,den_coeff_1, num_coeff_2,den_coeff_2,

                                  imc_mimo_tau_mf1,imc_mimo_tau_mf2,

                                  imc_mimo_reference1, imc_mimo_reference2,imc_mimo_reference3,

                                  change_ref_instant2,change_ref_instant3)