import streamlit as st
import pandas as pd
from streamlit_option_menu import option_menu
from datetime import datetime, timedelta
from streamlit_extras.app_logo import add_logo

# --- Importações Explícitas das Views ---
# Isso garante que as funções sejam encontradas corretamente
from controladores_views.gmv_view import gmv_Controller_Interface
from controladores_views.gpc_view import gpc_Controller_Interface
from controladores_views.imc_view import imc_Controller_Interface
from controladores_views.rst_view import rst_Controller_Interface

# --- Importações de Helpers ---
from connections import *
from mainSideBar import mainSidebarMenu
from session_state import loadSessionStates

# Define o que será exportado para o LABVCON.py
__all__ = [
    # Views dos Controladores
    'gmv_Controller_Interface',
    'gpc_Controller_Interface',
    'imc_Controller_Interface',
    'rst_Controller_Interface',

    # Helpers do Sistema
    'loadSessionStates',
    'mainSidebarMenu',

    # Bibliotecas Comuns
    'st',
    'option_menu',
    'add_logo',
    'pd',
    'datetime',
    'timedelta'
]