from connections import *
import streamlit as st
import pandas as pd
import datetime


def sidebarMenu():
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
        if st.button('Conectar'):
            serialPortValidationToConnect(port_option,baudrate_connection)

    with col12:
        if st.button('Desconectar'):
            serialPortValidationToDisconnect()

    textToSend = st.text_input('Send to Serial:')

    col1, col2 = st.columns(2)
    with col1:
        sampling_time_input = st.number_input(
            'Tempo de amostragem:',
            value=0.1,
            min_value=0.005)
        if sampling_time_input:
            st.session_state.sampling_time = sampling_time_input

    with col2:

        samples_number = st.number_input(
            'Quantidade de amostras:', value=100, step=10, min_value=1)

        if samples_number:
            st.session_state.samples_number = samples_number

    if st.button('Enviar'):
        if 'arduinoData' in st.session_state.connected:
            arduinoData = st.session_state.connected['arduinoData']
            sendToArduino(arduinoData, textToSend)
            dataRead = readFromArduino(arduinoData)
            st.write(f'## Resposta: {dataRead}')
        else:
            st.warning('Não há dispositivos conectados.')

    if st.button('Receber Dados'):
        if 'arduinoData' in st.session_state.connected:
            arduinoData = st.session_state.connected['arduinoData']
            st.session_state.controller_parameters['process_output_sensor'] = dict()
            process_output_sensor = st.session_state.controller_parameters['process_output_sensor']

            start_time = time.time()
            interation = 0
            sampling_time = st.session_state.sampling_time
            progress_text = "Operation in progress. Please wait."
            my_bar = st.progress(0, text=progress_text)

            while interation < samples_number:
                current_time = time.time()
                if current_time - start_time > sampling_time:
                    start_time = current_time
                    sendToArduino(arduinoData, textToSend)
                    dataRead = readFromArduino(arduinoData)
                    current_timestamp = datetime.datetime.now()
                    process_output_sensor[str(current_timestamp)] = float(dataRead)
                    interation += 1

                    percent_complete = interation * 1 / samples_number

                    my_bar.progress(percent_complete, text=progress_text)


        else:
            st.warning('Não há dispositivos conectados.')
