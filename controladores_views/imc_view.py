import streamlit as st
import pandas as pd
import altair as alt
from formatterInputs import *
from connections import *
from session_state import *
from controllers_process.validations_functions import *
from controllers_process.imc_controller_process import imcControlProcessSISO

def calculate_time_limit():
    try:
        sim_time = get_session_variable('simulation_time')
        return sim_time if sim_time is not None else 60.0
    except:
        return 60.0

def imc_Controller_Interface():
    # Cabeçalho padrão (Será escondido na impressão)
    st.header('Controle por Modelo Interno (IMC)')

    # --- 1. Recuperação de Dados ---
    calculated_params = get_session_variable('imc_calculated_params')
    val_iae = get_session_variable('iae_metric')
    val_tvc = get_session_variable('tvc_1_metric')

    if val_iae is None: val_iae = 0.0
    if val_tvc is None: val_tvc = 0.0

    # --- 2. Checkbox para Modo de Impressão ---
    modo_relatorio = st.checkbox("🖨️ Modo Impressão (PDF para TCC)", value=False)

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
                        margin: 5mm; /* Margens mínimas para aproveitar espaço */
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
                    /* Impede que o gráfico seja cortado ao meio */
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
            
            # Gráfico 1: Largura 1050 (seguro para A4) e Altura 300 (para caber os dois)
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

            # Pequeno espaço, mas não muito para não estourar a página
            st.markdown("<div style='height: 5px;'></div>", unsafe_allow_html=True)

            if get_session_variable('control_signal_1'):
                st.markdown("### Sinal de Controle (Tensão)")
                control_signal_with_elapsed_time = datetime_obj_to_elapsed_time('control_signal_1')
                df_ctrl = dictionary_to_pandasDataframe(control_signal_with_elapsed_time, 'Control Signal 1')
                
                # Gráfico 2: Altura reduzida para 250 para garantir o fit na página
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

        with config_col:
            st.write('### Configurações')
            imc_siso_tab_form()
            
            if calculated_params:
                st.markdown("---")
                st.markdown("#### Sintonia Calculada")
                df_params = pd.DataFrame([calculated_params]).T.reset_index()
                df_params.columns = ['Parâmetro', 'Valor']
                st.dataframe(df_params, hide_index=True, use_container_width=True)

        with graphics_col:
            y_max = get_session_variable('saturation_max_value')
            y_min = get_session_variable('saturation_min_value')

            if get_session_variable('process_output_sensor'):
                df_out = dataframeToPlot('process_output_sensor', 'Process Output', 'reference_input')
                st.subheader('Saída do Processo (Nível em cm)')
                altair_plot_chart_validation(df_out, y_max=35.0, y_min=0.0,
                                             x_column='Time (s)', y_column=['Reference', 'Process Output'])    
            
            if get_session_variable('control_signal_1'):
                st.subheader('Sinal de Controle (Tensão em Volts)')
                control_signal_with_elapsed_time = datetime_obj_to_elapsed_time('control_signal_1')
                df_ctrl = dictionary_to_pandasDataframe(control_signal_with_elapsed_time, 'Control Signal 1')
                altair_plot_chart_validation(df_ctrl, control=True, y_max=y_max, y_min=y_min,
                                             x_column='Time (s)', y_column='Control Signal 1', height=250)

            st.write('### Índices de Desempenho')
            c1, c2 = st.columns(2)
            with c1: st.metric("IAE (Erro Absoluto)", f"{val_iae:.4f}")
            with c2: st.metric("TVC (Variação Controle)", f"{val_tvc:.4f}")

def imc_siso_tab_form():
    # 1. Modelo
    st.markdown("#### 1. Modelo da Planta")
    tf_type = st.radio('Domínio:', ['Continuo', 'Discreto'], horizontal=True, key='imc_tf_type')
    num = st.text_input('Numerador:', key='imc_num', placeholder='5.43')
    den = st.text_input('Denominador:', key='imc_den', placeholder='123, 1')

    # 2. Sintonia
    st.markdown("#### 2. Sintonia")
    tau_mf = st.number_input('Tau Malha Fechada (s):', value=5.0, min_value=0.1, step=0.5, key='imc_tau')

    # 3. Estrutura
    st.markdown("#### 3. Estrutura")
    struct = st.selectbox('Tipo:', ('IMC Padrão', 'I + PD', 'PI + D', 'PID Ideal', 'PID Paralelo'), key='imc_struct')

    # 4. Referência
    st.markdown("#### 4. Referência")
    ref_type = st.radio('Modo:', ['Única', 'Múltiplas'], horizontal=True, key='imc_ref_mode')
    
    if ref_type == 'Única':
        ref1 = st.number_input('Set-point (cm):', value=15.0, key='imc_ref1')
        ref2, ref3 = ref1, ref1
        t2, t3 = 1.0, 1.0
    else:
        c1, c2, c3 = st.columns(3)
        with c1: ref1 = st.number_input('Ref 1:', value=10.0, key='imc_mr1')
        with c2: ref2 = st.number_input('Ref 2:', value=20.0, key='imc_mr2')
        with c3: ref3 = st.number_input('Ref 3:', value=15.0, key='imc_mr3')
        limit = calculate_time_limit()
        c4, c5 = st.columns(2)
        with c4: t2 = st.number_input('Troca 1 (s):', value=limit*0.33, key='imc_t2')
        with c5: t3 = st.number_input('Troca 2 (s):', value=limit*0.66, key='imc_t3')

    # Botão
    if st.button('Iniciar IMC', type='primary'):
        if not num or not den: return st.error("Defina o modelo.")
        imcControlProcessSISO(tf_type, num, den, tau_mf, struct, ref1, ref2, ref3, t2, t3)
        st.rerun()