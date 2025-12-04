import streamlit as st
import pandas as pd
import altair as alt
from formatterInputs import *
from connections import *
from session_state import *
from controllers_process.validations_functions import *
from controllers_process.rst_controller_process import rstControlProcessIncrementalSISO, rstControlProcessAdaptiveSISO

def calculate_time_limit():
    try:
        sim_time = get_session_variable('simulation_time')
        return sim_time if sim_time is not None else 60.0
    except:
        return 60.0

def rst_Controller_Interface():
    # Cabeçalho padrão (Será escondido na impressão)
    st.header('Regulador RST (Alocação de Polos)')

    # --- 1. Recuperação de Dados ---
    val_iae = get_session_variable('iae_metric')
    val_tvc = get_session_variable('tvc_1_metric')

    if val_iae is None: val_iae = 0.0
    if val_tvc is None: val_tvc = 0.0

    # --- 2. Checkbox para Modo de Impressão ---
    modo_relatorio = st.checkbox("🖨️ Modo Impressão (PDF para TCC)", value=False, key='print_mode_rst')

    if modo_relatorio:
        # ===================================================
        #           MODO RELATÓRIO (IMPRESSÃO PDF)
        # ===================================================
        st.markdown("""
            <style>
                @media print {
                    /* 1. Configuração da Página: Paisagem A4 */
                    @page { 
                        size: A4 landscape; 
                        margin: 5mm; 
                    }
                    
                    /* 2. LIMPEZA TOTAL (Esconde Interface) */
                    [data-testid="stSidebar"], header, footer, .stDeployButton, [data-testid="stToolbar"] {display: none !important;}
                    .stButton, .stCheckbox, .stSelectbox, .stSlider, .stRadio, .stNumberInput, .stTextInput {display: none !important;}
                    .stTabs, [data-baseweb="tab-list"], iframe, div[data-testid="stIFrame"] {display: none !important;}
                    
                    /* Esconde Títulos e Logos do App */
                    h1, h2, [data-testid="stImage"], img { display: none !important; }
                    
                    /* 3. LAYOUT COMPACTO */
                    .block-container {
                        padding: 0 !important;
                        margin: 0 !important;
                        max-width: 100% !important;
                        width: 100% !important;
                    }
                    .stApp {background-color: white !important;}
                    
                    /* Remove espaçamentos verticais do Streamlit */
                    div[data-testid="stVerticalBlock"] { gap: 0.2rem !important; }
                    
                    /* 4. GARANTIA DE NÃO-QUEBRA E RESOLUÇÃO */
                    .element-container {
                        break-inside: avoid !important;
                        page-break-inside: avoid !important;
                    }
                    
                    canvas {
                        max-width: 100% !important;
                        width: 100% !important;
                        height: auto !important;
                        display: block !important;
                    }
                    
                    /* Garante que títulos h3 fiquem próximos aos gráficos */
                    h3 { margin-bottom: 0px !important; margin-top: 10px !important; font-size: 16px !important;}
                }
            </style>
        """, unsafe_allow_html=True)

        # --- GRÁFICOS DO RELATÓRIO ---
        if get_session_variable('process_output_sensor'):
            st.markdown("### Saída do Processo (Nível)")
            
            df_out = dataframeToPlot('process_output_sensor', 'Process Output', 'reference_input')
            
            # Gráfico 1: Otimizado para impressão
            chart1 = alt.Chart(df_out).mark_line().encode(
                x=alt.X('Time (s)', title='Tempo (s)'),
                y=alt.Y('Process Output', title='Nível (cm)'),
                color=alt.value('#1f77b4')
            ).properties(
                width=1050, 
                height=300 
            )

            if 'Reference' in df_out.columns:
                line_ref = alt.Chart(df_out).mark_line(strokeDash=[5,5], color='red').encode(
                    x='Time (s)', y='Reference'
                )
                chart1 = chart1 + line_ref
            
            st.altair_chart(chart1, use_container_width=False)

            st.markdown("<div style='height: 5px;'></div>", unsafe_allow_html=True)

            if get_session_variable('control_signal_1'):
                st.markdown("### Sinal de Controle (Tensão)")
                control_signal_with_elapsed_time = datetime_obj_to_elapsed_time('control_signal_1')
                df_ctrl = dictionary_to_pandasDataframe(control_signal_with_elapsed_time, 'Control Signal 1')
                
                # Gráfico 2
                chart2 = alt.Chart(df_ctrl).mark_line().encode(
                    x=alt.X('Time (s)', title='Tempo (s)'),
                    y=alt.Y('Control Signal 1', title='Tensão (V)'),
                    color=alt.value('#2ca02c')
                ).properties(
                    width=1050, 
                    height=250
                )
                
                st.altair_chart(chart2, use_container_width=False)

    else:
        # ===================================================
        #           MODO NORMAL (PAINEL DE OPERAÇÃO)
        # ===================================================
        graphics_col, config_col = st.columns([0.7, 0.3])

        # --- Coluna da Direita: Configurações ---
        with config_col:
            st.write('### Configurações')
            # Tabs mantidas conforme seu código original
            inc_tab, adp_tab = st.tabs(["RST Incremental", "RST Adaptativo"])       
            
            with inc_tab:
                rst_incremental_siso_tab_form()           
            with adp_tab:
                rst_adaptive_siso_tab_form()

        # --- Coluna da Esquerda: Gráficos e Índices ---
        with graphics_col:
            y_max = get_session_variable('saturation_max_value')
            y_min = get_session_variable('saturation_min_value')
            
            has_data = get_session_variable('process_output_sensor')

            if has_data:
                # 1. Gráficos
                df_out = dataframeToPlot('process_output_sensor', 'Process Output', 'reference_input')
                st.subheader('Saída do Processo (Nível)')
                altair_plot_chart_validation(df_out, y_max=35.0, y_min=0.0,
                                             x_column='Time (s)', y_column=['Reference', 'Process Output'])    
            
                if get_session_variable('control_signal_1'):
                    st.subheader('Sinal de Controle (Tensão)')
                    control_signal_with_elapsed_time = datetime_obj_to_elapsed_time('control_signal_1')
                    df_ctrl = dictionary_to_pandasDataframe(control_signal_with_elapsed_time, 'Control Signal 1')
                    altair_plot_chart_validation(df_ctrl, control=True, y_max=y_max, y_min=y_min,
                                                 x_column='Time (s)', y_column='Control Signal 1', height=250)

                # 2. Índices de Desempenho
                st.divider()
                st.write('### Índices de Desempenho')
                c1, c2 = st.columns(2)
                with c1: st.metric("IAE (Erro Absoluto)", f"{val_iae:.4f}")
                with c2: st.metric("TVC (Variação Controle)", f"{val_tvc:.4f}")
                
                # 3. TABELA DE SINTONIA CALCULADA (Lógica Robusta)
                rst_params = {}
                
                # Busca parâmetros salvos na sessão (similar ao GPC)
                all_params = get_session_variable('controller_parameters')
                if all_params and isinstance(all_params, dict) and 'rst_calculated_params' in all_params:
                    rst_params = all_params['rst_calculated_params']
                elif 'controller_parameters' in st.session_state and 'rst_calculated_params' in st.session_state['controller_parameters']:
                    rst_params = st.session_state['controller_parameters']['rst_calculated_params']

                if rst_params:
                    st.divider()
                    st.write("### Sintonia Calculada (PID Equivalente)")
                    
                    # Ordem de prioridade para exibição
                    keys_order = ['Kc', 'Ki', 'Kd', 'Tau_MF', 'T0']
                    data_display = []
                    
                    for k in keys_order:
                        if k in rst_params:
                            val = rst_params[k]
                            if isinstance(val, float): val_str = f"{val:.4f}"
                            else: val_str = str(val)
                            
                            label_map = {
                                'Kc': 'Ganho Proporcional (Kc)',
                                'Ki': 'Ganho Integral (Ki)',
                                'Kd': 'Ganho Derivativo (Kd)',
                                'Tau_MF': 'Tau Malha Fechada (s)',
                                'T0': 'Rastreamento (T0)'
                            }
                            data_display.append({"Parâmetro": label_map.get(k, k), "Valor": val_str})
                    
                    if data_display:
                        st.table(pd.DataFrame(data_display))
                    
            else:
                st.info("Aguardando execução da simulação para exibir resultados.")


def rst_incremental_siso_tab_form():
    # 1. Modelo
    st.markdown("#### 1. Modelo da Planta")
    tf_type = st.radio('Domínio:', ['Continuo', 'Discreto'], horizontal=True, key='rst_inc_tf_type')
    num = st.text_input('Numerador:', key='rst_inc_num', help='Ex: 5.43', placeholder='5.43')
    den = st.text_input('Denominador:', key='rst_inc_den', help='Ex: 123, 1', placeholder='123, 1')
    
    # 2. Sintonia
    st.markdown("#### 2. Sintonia")
    tau_mf = st.number_input('Tau Malha Fechada (s):', value=5.0, min_value=0.1, step=0.5, key='rst_inc_tau')
    
    # 3. Estrutura
    st.markdown("#### 3. Estrutura")
    pid_struct = st.selectbox('Tipo:', ('RST Incremental Puro', 'I + PD', 'PI + D', 'PID Ideal', 'PID Paralelo'), key='rst_inc_struct')

    # 4. Referência
    st.markdown("#### 4. Referência")
    ref_type = st.radio('Modo:', ['Única', 'Múltiplas'], horizontal=True, key='rst_inc_ref_mode')
    
    if ref_type == 'Única':
        ref1 = st.number_input('Set-point (cm):', value=15.0, key='rst_inc_ref1')
        ref2, ref3 = ref1, ref1
        t2, t3 = 1.0, 1.0
    else:
        c1, c2, c3 = st.columns(3)
        with c1: ref1 = st.number_input('Ref 1:', value=10.0, key='rst_inc_mr1')
        with c2: ref2 = st.number_input('Ref 2:', value=20.0, key='rst_inc_mr2')
        with c3: ref3 = st.number_input('Ref 3:', value=15.0, key='rst_inc_mr3')
        
        limit = calculate_time_limit()
        c4, c5 = st.columns(2)
        with c4: t2 = st.number_input('Troca 1 (s):', value=limit*0.33, key='rst_inc_t2')
        with c5: t3 = st.number_input('Troca 2 (s):', value=limit*0.66, key='rst_inc_t3')

    # Botão
    st.markdown("---")
    if st.button('Iniciar Incremental', type='primary', use_container_width=True):
        if not num or not den: return st.error("Defina o modelo.")
        rstControlProcessIncrementalSISO(tf_type, num, den, tau_mf, pid_struct, ref1, ref2, ref3, t2, t3)
        st.rerun()

def rst_adaptive_siso_tab_form():
    # 1. Estimador (Modelo Inicial)
    st.markdown("#### 1. Modelo Inicial (Estimador)")
    st.caption("Define os parâmetros iniciais (a1, b0) do RLS.")
    
    # PADRONIZADO: Entradas iguais ao modo incremental
    tf_type = st.radio('Domínio:', ['Continuo', 'Discreto'], horizontal=True, key='rst_adp_tf_type')
    num = st.text_input('Numerador:', key='rst_adp_num', help='Ex: 5.43', placeholder='5.43')
    den = st.text_input('Denominador:', key='rst_adp_den', help='Ex: 123, 1', placeholder='123, 1')
    
    p0_val = st.number_input('P(0) Covariância:', value=1000.0, key='rst_adp_p0')

    # 2. Sintonia
    st.markdown("#### 2. Sintonia")
    tau_mf = st.number_input('Tau Malha Fechada (s):', value=5.0, min_value=0.1, key='rst_adp_tau')

    # 3. Estrutura
    st.markdown("#### 3. Estrutura")
    pid_struct = st.selectbox('Tipo:', ('RST Incremental Puro', 'I + PD', 'PI + D', 'PID Ideal', 'PID Paralelo'), key='rst_adp_struct')

    # 4. Referência
    st.markdown("#### 4. Referência")
    ref_type = st.radio('Modo:', ['Única', 'Múltiplas'], horizontal=True, key='rst_adp_ref_mode')
    
    if ref_type == 'Única':
        ref1 = st.number_input('Set-point (cm):', value=15.0, key='rst_adp_ref1')
        ref2, ref3 = ref1, ref1
        t2, t3 = 1.0, 1.0
    else:
        c1, c2, c3 = st.columns(3)
        with c1: ref1 = st.number_input('Ref 1:', value=10.0, key='rst_adp_mr1')
        with c2: ref2 = st.number_input('Ref 2:', value=20.0, key='rst_adp_mr2')
        with c3: ref3 = st.number_input('Ref 3:', value=15.0, key='rst_adp_mr3')
        limit = calculate_time_limit()
        c4, c5 = st.columns(2)
        with c4: t2 = st.number_input('Troca 1 (s):', value=limit*0.33, key='rst_adp_t2')
        with c5: t3 = st.number_input('Troca 2 (s):', value=limit*0.66, key='rst_adp_t3')

    # Botão
    st.markdown("---")
    if st.button('Iniciar Adaptativo', type='primary', use_container_width=True):
        if not num or not den: return st.error("Defina o modelo inicial.")
        rstControlProcessAdaptiveSISO(tf_type, num, den, tau_mf, pid_struct, p0_val, ref1, ref2, ref3, t2, t3)
        st.rerun()