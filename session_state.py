import streamlit as st

def loadSessionStates():
    """Inicializa as variáveis de estado da sessão se ainda não existirem."""
    if 'connected' not in st.session_state:
        st.session_state.connected = {}

    if 'sensor' not in st.session_state:
        st.session_state.sensor = {}

    if 'controller_parameters' not in st.session_state:
        st.session_state.controller_parameters = {}

    # Dicionário com valores padrões para evitar KeyErrors na inicialização
    defaults = {
        'sampling_time': 0.1,
        'samples_number': 100,
        'control_signal_1': {},
        'control_signal_2': {},
        'reference_input': {},
        'saturation_max_value': 100.0,
        'saturation_min_value': 0.0,
        'process_output_sensor': {},
        'iae_metric': 0.0,
        'tvc_1_metric': 0.0,
        'tvc_2_metric': 0.0,
        'simulation_time': 60.0,
        # Variáveis novas adicionadas para evitar erros no IMC e Relatórios
        'imc_calculated_params': {}, 
        'IAE_value': 0.0,
        'TVC_value': 0.0
    }

    # Garante que todas as chaves padrão existam no controller_parameters
    for key, value in defaults.items():
        if key not in st.session_state.controller_parameters:
            st.session_state.controller_parameters[key] = value

def get_session_variable(variable: str):
    """
    Retorna o valor de uma variável salva em st.session_state.controller_parameters.
    Usa .get() para retornar None caso a variável não exista, evitando erro de KeyError.
    """
    if 'controller_parameters' in st.session_state:
        return st.session_state.controller_parameters.get(variable)
    return None

def set_session_controller_parameter(controller_parameter: str, new_data) -> None:
    """
    Define ou atualiza um valor dentro de st.session_state.controller_parameters.
    """
    if 'controller_parameters' not in st.session_state:
        st.session_state.controller_parameters = {}
    
    st.session_state.controller_parameters[controller_parameter] = new_data
    
    # Debug opcional: salva erros na raiz para fácil acesso se necessário
    if controller_parameter == 'debug_error':
        st.session_state['debug_error'] = new_data

def set_session_variable(key: str, value):
    """
    Define uma variável diretamente na raiz do st.session_state.
    """
    st.session_state[key] = value

def clear_session_variable(key: str):
    """
    Remove uma variável do st.session_state se ela existir.
    """
    if key in st.session_state:
        del st.session_state[key]