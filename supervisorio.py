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
    page_title="Supervisorio",
    page_icon="🧊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Ajuste de padding do container principal
st.markdown(" <style> div[class^='block-container st-emotion-cache-z5fcl4 ea3mdgi4'] { padding: 1rem 3rem 10rem; } </style> ", unsafe_allow_html=True)

# --- NOVO CABEÇALHO ---

# Cria colunas para o título e o logo da direita
# A proporção [0.8, 0.2] dá 80% do espaço para o título e 20% para o logo
col_title, col_gerae_logo = st.columns([0.8, 0.2], gap="small")

with col_title:
    # Título aumentado e personalizado usando HTML/CSS
    # 'font-size: 3.5rem' define o tamanho grande. Ajuste se necessário (ex: 3em, 4em).
    # 'padding-top: 20px' ajuda a alinhar verticalmente com a imagem ao lado.
    st.markdown(
        """
        <h1 style='font-size: 3.5rem; margin: 0; padding-top: 15px; line-height: 1.2;'>
            Supervisório Virtual de Controle
        </h1>
        """,
        unsafe_allow_html=True
    )

with col_gerae_logo:
    # Imagem do GERAE como logo no canto direito
    # Ela ocupará a largura da coluna estreita (0.2), ficando com tamanho de logo.
    st.image("images/Gerae.jpg", width="stretch")

# ----------------------
add_logo("images/Logo ifpa.png", height=300)
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