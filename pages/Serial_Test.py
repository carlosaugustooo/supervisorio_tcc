from connections import *
from components import *
from session_state import *
import streamlit as st
from datetime import datetime
from formatterInputs import *
from controllers_process.validations_functions import plot_chart_validation

st.set_page_config(
    page_title="Ex-stream-ly Cool App",
    page_icon="🧊",
    layout="wide",
    initial_sidebar_state="expanded",

)
loadSessionStates()


st.title('LABVCON - Laboratório Virtual de Controle')
st.header('Teste da comunicação Serial', divider='rainbow')


# SideBar
with st.sidebar:
    sidebarMenu()
    st.session_state

if get_session_variable('process_output_sensor'):       
    open_loop_process_output = datetime_obj_to_elapsed_time('process_output_sensor')
    ol_output_dataframe = dictionary_to_pandasDataframe(open_loop_process_output,'Open Loop')
        
    plot_chart_validation(ol_output_dataframe, x = 'Time (s)', y = 'Open Loop',height=500)



col1, col2 = st.columns(2)

with col1:
    
    datetimeList = list(get_session_variable('process_output_sensor').keys())

    # Define the format of your date string
    date_format = "%Y-%m-%d %H:%M:%S.%f"

    # Parse the string into a datetime object
    date_object = [datetime.strptime(datetimeElement, date_format)
                   for datetimeElement in datetimeList]

    time_interval = [(date_object[i] - date_object[i-1]).total_seconds()
                     for i in range(1, len(date_object))]

    st.line_chart(time_interval)

with col2:

    if time_interval:
        'Média do intervalo'
        mean_value = sum(time_interval) / len(time_interval)
        mean_value
        'Média do erro'
        mean_error = sum(abs(x - mean_value)
                         for x in time_interval) / len(time_interval)
        mean_error

    'Tempo da simulação'
    if date_object:
        date_object[-1] - date_object[0]


