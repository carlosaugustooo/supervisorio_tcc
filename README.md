# Supervisório de Planta de Nível

## Descrição

Este projeto consiste em uma interface de supervisão e controle para protótipos de sistemas dinâmicos implementados em Arduino em tempo real. O sistema permite a implementação e análise de controladores avançados como **RST**, **IMC** (Internal Model Control), **GMV** (Generalized Minimum Variance) e **GPC** (Generalized Predictive Control) em sistemas SISO (Single Input Single Output), permitindo tbm a sintonia dos controladores aos modelos PID ideal, PID paralelo, I+PD e PI+D.

O usuário insere o modelo do sistema e o tempo de amostragem, implementa os controladores e obtém gráficos de resposta em tempo real, sinal de controle e índices de métricas como **IAE** e **TVC** para comparação de desempenho.

## Instalação

Para instalar e começar a usar o supervisório, siga estas etapas:

1. **Clone este repositório (ou baixe os arquivos):**
   ```bash
   git clone https://github.com/plantadenivel-supervisorio/supervisorio_tcc
   ```
2. **Crie um ambiente virtual (opcional, mas recomendado):**
    ```bash
    python -m venv venv
    ```
3. **Ative o ambiente virtual:**
    - No Windows:
     ```bash
    venv\Scripts\activate
    ```
    - No Linux/Mac:
     ```bash
    source venv/bin/activate
    ```

4. **Instale as dependências do projeto:**
    ```bash
    pip install -r requirements.txt
    ```

5. **Execute o servidor Streamlit do projeto:**
    ```bash
    streamlit run supervisorio.py
    ```
