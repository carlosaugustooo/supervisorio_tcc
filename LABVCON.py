from controladores_views.controller_imports import *

# Carrega os estados de sessão (corrigido em session_state.py)
loadSessionStates()

st.set_page_config(
    page_title="LABVCON",
    page_icon="🧊",
    layout="wide",
    initial_sidebar_state="expanded",
)
st.markdown(" <style> div[class^='block-container st-emotion-cache-z5fcl4 ea3mdgi4'] { padding: 1rem 3rem 10rem; } </style> ", unsafe_allow_html=True)

add_logo("images/app_logo2.png", height=150)

st.title('LABVCON - Laboratório Virtual de Controle')
selectMethod = option_menu(
    menu_title=None,
    # 1. ADICIONADO 'RST'
    options=['IMC', 'GMV', 'GPC', 'RST'],
    orientation='horizontal',
    # 2. ADICIONADO ÍCONE 'terminal'
    icons=['ui-radios-grid',
           'app', 'command', 'terminal'],
)

##########################################################################

# SideBar
with st.sidebar:
    mainSidebarMenu()
    with st.expander("Session States:"):
        st.session_state

##########################################################################

case_functions = {
    "IMC": imc_Controller_Interface,
    "GMV": gmv_Controller,
    "GPC": gpc_Controller,
    # 3. ADICIONADO MAPEAMENTO PARA A NOVA INTERFACE RST
    "RST": rst_controller_Interface,
}

# Executa a função da view selecionada
case_functions[selectMethod]()