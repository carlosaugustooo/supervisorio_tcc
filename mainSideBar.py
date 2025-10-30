from connections import *
import streamlit as st


def mainSidebarMenu():
    st.write('## Conexão Serial')
    st.write('#### Componentes Seriais Encontrados:')

    portlist = get_ports()
    for port_info in portlist:
        st.write(
            f"Description: {port_info.description}")

    port_option = st.selectbox(
        'Qual porta deseja conectar?',
        [port_info.device for port_info in portlist])
    
    baudrate_connection = st.selectbox(
        'Baudrate:',
        [9600,115200,250000],index=2)

    col11, col12 = st.columns(2)
    with col11:
        if st.button('Conectar', type='primary'):
            serialPortValidationToConnect(port_option,baudrate_connection)

    with col12:
        if st.button('Desconectar'):
            serialPortValidationToDisconnect()
    st.divider()
    
    col1, col2 = st.columns(2)
    
    
    with col1:
        sampling_time_input = st.number_input(
            'Tempo de amostragem (s):',
            value=0.01,
            min_value=0.005)
        
      
        
        saturation_min_value = st.number_input('**Saturação mínima**:'.upper(),value=0,
            )
        
        st.session_state.controller_parameters['saturation_min_value'] = saturation_min_value
        if sampling_time_input:
            st.session_state.controller_parameters['sampling_time'] = sampling_time_input

    with col2:

        samples_number = st.number_input(
            'Quantidade de amostras:', value=1000, step=10, min_value=1)
        
        saturation_max_value = st.number_input('**Saturação Máxima**:'.upper(),value=15,
            )
        
        st.session_state.controller_parameters['saturation_max_value'] = saturation_max_value
        
        if samples_number:
            st.session_state.controller_parameters['samples_number'] = samples_number
            
    st.write(f'## Tempo total de Simulação:  {sampling_time_input*samples_number:.2f} s')