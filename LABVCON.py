from controladores_views.controller_imports import (
    gmv_Controller_Interface,
    gpc_Controller_Interface,
    imc_Controller_Interface,
    rst_Controller_Interface,
    
    # Helpers e Bibliotecas
    st,
    loadSessionStates,
    add_logo,
    option_menu,
    mainSidebarMenu
)

# Carrega os estados de sessão
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
    options=['IMC', 'GMV', 'GPC', 'RST'],
    orientation='horizontal',
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
    'GMV': gmv_Controller_Interface,
    'GPC': gpc_Controller_Interface,
    'IMC': imc_Controller_Interface,
    'RST': rst_Controller_Interface
}


# Executa a função da view selecionada
case_functions[selectMethod]()