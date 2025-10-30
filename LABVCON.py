from controladores_views.controller_imports import *

loadSessionStates()

st.set_page_config(
    page_title="LABVCON",
    page_icon="🧊",
    layout="wide",
    initial_sidebar_state="expanded",


)
st.markdown(" <style> div[class^='block-container st-emotion-cache-z5fcl4 ea3mdgi4'] { padding: 1rem 3rem 10rem; } </style> ", unsafe_allow_html=True)

# title_alignment = """
# <style>
# .css-z5fcl4 {3
#   padding:2rem
# }
# </style>
# """
# st.markdown(title_alignment, unsafe_allow_html=True)

add_logo("images/app_logo2.png", height=150)

st.title('LABVCON - Laboratório Virtual de Controle')
selectMethod = option_menu(
    menu_title=None,
    options=['IMC', 'GMV', 'GPC'],
    orientation='horizontal',
    icons=['ui-radios-grid',
           'app', 'command'],

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
}

case_functions[selectMethod]()

##########################################################################
