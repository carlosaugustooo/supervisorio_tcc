import time
import serial
from serial.tools.list_ports import comports
import streamlit as st


def get_ports():
    portlist = comports()
    return portlist


def findArduino(portsFound: list):
    commPort = ''

    for port in portsFound:
        if 'Arduino' in str(port):
            commPort = port.device
    return commPort


def connectSerial(commport):

    portlist = get_ports()
    commPort = findArduino(portlist)
    arduinoData = serial.Serial(port=commPort, baudrate=250000, timeout=1)
    time.sleep(6)
    return arduinoData


def connectSerialManual(commPort,baudrate):

    arduinoData = serial.Serial(port=commPort, baudrate=baudrate, timeout=1)
    time.sleep(2)
    return arduinoData


def disconnectSerial(arduinoData):
    if arduinoData and arduinoData.is_open:
        arduinoData.close()
    # arduinoData.__del__() # Avoid manual __del__ calls


def sendToArduino(arduinoData, textToSend):
    try:
        if arduinoData and arduinoData.is_open:
            if '\r' not in textToSend:
                textToSend += '\r'
            arduinoData.write(textToSend.encode())
            arduinoData.flush()
        else:
            # Port is closed or invalid
            pass 
    except (serial.SerialException, PermissionError) as e:
        # Log error or just pass to avoid crashing the whole app
        print(f"Serial Write Error: {e}")
        pass


def readFromArduino(arduinoData):
    try:
        if arduinoData and arduinoData.is_open:
            if arduinoData.in_waiting > 0: # Check if data is available
                line = arduinoData.readline().decode('utf-8', errors='ignore').strip()
                if line:
                    return float(line)
        return None
    except (serial.SerialException, ValueError, PermissionError):
        return None


def serialPortValidationToConnect(port_option,baudrate_connection):
    if not port_option:
        return st.error('Não há porta serial para conectar')

    if 'arduinoData' not in st.session_state.connected:
        with st.spinner('Processing...'):
            try:
                arduinoData = connectSerialManual(port_option,baudrate_connection)
                st.session_state.connected['arduinoData'] = arduinoData
                st.success("Conectado!")
            except Exception as e:
                st.error(f"Erro ao conectar: {e}")
    else:
        st.write('O arduino já está conectado.')


def serialPortValidationToDisconnect():
    if 'arduinoData' in st.session_state.connected:
        arduinoData = st.session_state.connected['arduinoData']
        with st.spinner('Processing...'):
            time.sleep(2)
            disconnectSerial(arduinoData)
            del st.session_state.connected['arduinoData'] # Properly remove from session state
            # st.session_state.connected = {} # Don't wipe everything
        st.success("Desconectado!")
    else:
        st.warning('O arduino já está desconectado.')