import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import skfuzzy as fuzz
import google.generativeai as genai
import json
import re

# -----------------------------------------------
# FUNÇÕES FUZZY MANUAIS
# -----------------------------------------------


def trimf(x, params):
    a, b, c = params
    y = np.zeros_like(x)
    mask1 = (a < x) & (x < b)
    y[mask1] = (x[mask1] - a) / (b - a)
    y[x == b] = 1.0
    mask2 = (b < x) & (x < c)
    y[mask2] = (c - x[mask2]) / (c - b)
    return y


def trapmf(x, params):
    a, b, c, d = params
    y = np.zeros_like(x)
    y[(a < x) & (x < b)] = (x[(a < x) & (x < b)] - a) / (b - a)
    y[(b <= x) & (x <= c)] = 1
    y[(c < x) & (x < d)] = (d - x[(c < x) & (x < d)]) / (d - c)
    return y


def gaussmf(x, params):
    sigma, mean = params
    return np.exp(-((x - mean) ** 2) / (2 * sigma ** 2))


def interp_membership(x, y, value):
    return np.interp(value, x, y)


def extrair_json(texto):
    try:
        texto = re.sub(r"```json", "", texto)
        texto = re.sub(r"```", "", texto)

        match = re.search(r"\{.*\}", texto, re.DOTALL)
        if match:
            bloco = match.group(0)
            return json.loads(bloco)

    except Exception as e:
        return None

    return None


def validate_mf_params(tipo, params):
    """
    Valida params (lista) para cada tipo:
    - trimf: espera 3 params
    - trapmf: espera 4 params
    - gaussmf: espera 2 params (sigma, mean) ou (mean, sigma) dependendo do seu design
    Retorna (True, "") se válido, caso contrário (False, mensagem_erro)
    """
    if not isinstance(params, (list, tuple)):
        return False, "Parâmetros devem estar em forma de lista separados por vírgula."

    try:
        params_f = [float(p) for p in params]
    except Exception:
        return False, "Todos os parâmetros precisam ser numéricos."

    if tipo == "trimf" and len(params_f) == 3:
        a, b, c = params_f
        if not (a <= b <= c):
            return False, "Para trimf precisa valer: a <= b <= c."
        return True, ""
    if tipo == "trapmf" and len(params_f) == 4:
        a, b, c, d = params_f
        if not (a <= b <= c <= d):
            return False, "Para trapmf precisa valer: a <= b <= c <= d."
        return True, ""
    if tipo == "gaussmf" and len(params_f) == 2:
        sigma, mean = params_f
        if sigma <= 0:
            return False, "Para gaussmf o sigma deve ser > 0."
        return True, ""
    return False, f"Parâmetros incompatíveis para tipo '{tipo}'."


def normalize_fuzzy_json(dados):
    """
    Converte chaves que deveriam ser dicionários mas vieram como listas.
    Corrige automaticamente formatos problemáticos devolvidos pelo Gemini.
    """
    fix_keys = ["entradas", "saidas"]

    for k in fix_keys:
        if k in dados:
            if isinstance(dados[k], list):
                novo = {}
                for item in dados[k]:
                    if isinstance(item, dict):
                        subkeys = list(item.keys())
                        if len(subkeys) == 1:
                            nome = subkeys[0]
                            novo[nome] = item[nome]
                dados[k] = novo

    for section in fix_keys:
        if section in dados and isinstance(dados[section], dict):
            for var, info in dados[section].items():
                if "conjuntos" in info and isinstance(info["conjuntos"], list):
                    novo = {}
                    for item in info["conjuntos"]:
                        if isinstance(item, dict):
                            subkeys = list(item.keys())
                            if len(subkeys) == 1:
                                conj = subkeys[0]
                                novo[conj] = item[conj]
                    info["conjuntos"] = novo

    return dados


def normalize_antecedentes(regras):
    """
    Normaliza qualquer formato de antecedentes em:
    [("variavel", "conjunto"), ...]
    """
    regras_norm = []

    for r in regras:
        ants = r.get("antecedentes", [])
        novos_ants = []

        if isinstance(ants, dict):
            for k, v in ants.items():
                novos_ants.append((k, v))
        else:
            for item in ants:
                if isinstance(item, list) and len(item) == 2:
                    novos_ants.append((item[0], item[1]))

                elif isinstance(item, dict):
                    keys = list(item.keys())

                    if len(keys) == 1:
                        k = keys[0]
                        novos_ants.append((k, item[k]))

                    elif "var" in item and "conj" in item:
                        novos_ants.append((item["var"], item["conj"]))

        regras_norm.append({
            "antecedentes": novos_ants,
            "consequente": tuple(r["consequente"]),
            "logica": r["logica"]
        })

    return regras_norm


def normalize_regras(regras_raw):
    regras_norm = []

    for r in regras_raw:
        if not isinstance(r, dict):
            continue

        antecedentes = r.get("antecedentes", [])
        consequente = r.get("consequente", None)
        logica = r.get("logica", "AND")

        if not consequente or len(consequente) != 2:
            continue

        novos_ants = []

        if isinstance(antecedentes, dict):
            for k, v in antecedentes.items():
                novos_ants.append((k, v))

        elif isinstance(antecedentes, list):
            for item in antecedentes:

                if isinstance(item, list) and len(item) == 2:
                    novos_ants.append((item[0], item[1]))

                elif isinstance(item, dict):
                    if "var" in item and "conj" in item:
                        novos_ants.append((item["var"], item["conj"]))
                    else:
                        keys = list(item.keys())
                        if len(keys) == 1:
                            novos_ants.append((keys[0], item[keys[0]]))

        if not novos_ants:
            continue

        regras_norm.append({
            "antecedentes": novos_ants,
            "consequente": (consequente[0], consequente[1]),
            "logica": logica
        })

    return regras_norm


# -----------------------------------------------
# CONFIG GERAL
# -----------------------------------------------
st.set_page_config(page_title="Tutor Fuzzy Interativo", layout="wide")

genai.configure(api_key="AIzaSyAUTHz9CRle9t-slGUzmQ5WSbmaZtRFboU")
modelo = genai.GenerativeModel("models/gemini-2.5-flash")
st.sidebar.title("Navegação")
pagina = st.sidebar.selectbox("Escolha uma página:", [
    "Introdução / Visualizador",
    "Chatbot Fuzzy",
    "Exemplo Controle Fuzzy",
    "Editor Fuzzy",
    "Simulador Fuzzy",
    "Gerador Automático de Exemplos"
])

# ===========================================================
#  PÁGINA 1 — VISUALIZADOR
# ===========================================================
if pagina == "Introdução / Visualizador":
    st.title("Introdução / Visualizador de Funções de Pertinência")

    st.markdown("""
    ### O que é um sistema fuzzy?
    Sistemas fuzzy usam *conjuntos nebulosos* para representar termos linguísticos
    (ex.: "quente", "frio") por meio de **funções de pertinência**.
    Em vez de decisões binárias, o fuzzy permite transições suaves e regras linguísticas.
    """)

    st.write("""
    **Como usar este visualizador:**  
    1. Escolha o tipo de função (triangular, trapezoidal, gaussiana).  
    2. Ajuste os parâmetros com os sliders.  
    3. Observe a pertinência para um ponto e veja como os conjuntos se sobrepõem.
    """)

    x_demo = np.linspace(0, 40, 400)
    demo_a = trimf(x_demo, [0, 5, 15])
    demo_b = trimf(x_demo, [10, 20, 30])
    demo_c = trimf(x_demo, [25, 35, 40])
    fig1, ax1 = plt.subplots(figsize=(7, 2.5))
    ax1.plot(x_demo, demo_a, label="Conjunto A (triangular)")
    ax1.plot(x_demo, demo_b, label="Conjunto B (triangular)")
    ax1.plot(x_demo, demo_c, label="Conjunto C (triangular)")
    ax1.fill_between(x_demo, np.maximum(demo_a, np.maximum(
        demo_b, demo_c)), alpha=0.1, label="Sobreposição (exemplo)")
    ax1.set_title("Exemplo: Sobreposição de conjuntos fuzzy")
    ax1.set_xlabel("Universo")
    ax1.set_ylabel("Pertinência")
    ax1.legend()
    st.pyplot(fig1)

    x_cmp = np.linspace(-10, 10, 400)
    t_tri = trimf(x_cmp, [-6, -2, 2])
    t_trap = trapmf(x_cmp, [-8, -4, 0, 4])
    t_gauss = gaussmf(x_cmp, [2.0, 0.0])
    fig2, ax2 = plt.subplots(figsize=(7, 2.5))
    ax2.plot(x_cmp, t_tri, label="Triangular", linestyle='-')
    ax2.plot(x_cmp, t_trap, label="Trapezoidal", linestyle='--')
    ax2.plot(x_cmp, t_gauss, label="Gaussiana", linestyle=':')
    ax2.set_title("Comparação: Triangular / Trapezoidal / Gaussiana")
    ax2.set_xlabel("Universo")
    ax2.set_ylabel("Pertinência")
    ax2.legend()
    st.pyplot(fig2)

    tipo = st.selectbox(
        "Selecione o tipo de função de pertinência",
        ["Triangular", "Trapezoidal", "Gaussiana"]
    )

    st.write("### Parâmetros da Função")
    if tipo == "Triangular":
        a = st.slider("a", -10.0, 10.0, -5.0)
        b = st.slider("b (pico)", -10.0, 10.0, 0.0)
        c = st.slider("c", -10.0, 10.0, 5.0)
    elif tipo == "Trapezoidal":
        a = st.slider("a", -10.0, 10.0, -6.0)
        b = st.slider("b (início do topo)", -10.0, 10.0, -2.0)
        c = st.slider("c (fim do topo)", -10.0, 10.0, 2.0)
        d = st.slider("d", -10.0, 10.0, 6.0)
    else:
        mean = st.slider("Média μ", -10.0, 10.0, 0.0)
        sigma = st.slider("Desvio σ", 0.1, 10.0, 2.0)

    x = np.linspace(-10, 10, 400)
    if tipo == "Triangular":
        y = trimf(x, [a, b, c])
    elif tipo == "Trapezoidal":
        y = trapmf(x, [a, b, c, d])
    else:
        y = gaussmf(x, [sigma, mean])

    fig, ax = plt.subplots(figsize=(7, 3))
    ax.plot(x, y, label=f"{tipo}")
    ax.set_ylim(-0.1, 1.1)
    ax.set_xlabel("Entrada")
    ax.set_ylabel("Pertinência")
    ax.set_title(f"Função de pertinência: {tipo}")
    ax.legend()
    st.pyplot(fig)

    valor = st.slider("Valor para pertinência", -10.0, 10.0, 0.0)
    mu_val = interp_membership(x, y, valor)
    st.write(f"Pertinência: **{mu_val:.4f}**")

    with st.expander("Explicação da função"):
        if tipo == "Triangular":
            st.write("""
            A função **triangular** é definida pelos pontos (a, b, c).  
            Ela sobe linearmente até o pico em **b**, e depois desce linearmente até **c**.  
            Representa conceitos como *morno*, *velocidade média*, *altura mediana*, etc.
            """)
        elif tipo == "Trapezoidal":
            st.write("""
            A função **trapezoidal** possui um topo plano entre **b** e **c**,  
            permitindo representar conceitos que permanecem totalmente verdadeiros por uma faixa de valores.  
            Ex.: *temperatura confortável*, *peso ideal* etc.
            """)
        else:
            st.write("""
            A função **gaussiana** é suave e baseada na distribuição normal,  
            sendo ótima para representar conceitos naturais e contínuos.  
            Ex.: *conforto térmico*, *luminosidade adequada* etc.
            """)

# ===========================================================
#  PÁGINA 2 — CHATBOT FUZZY
# ===========================================================
elif pagina == "Chatbot Fuzzy":
    st.title("Chatbot Fuzzy")

    if "historico" not in st.session_state:
        st.session_state.historico = []

    pergunta = st.text_input("Pergunte algo:")

    col1, col2 = st.columns(2)
    if col1.button("Enviar"):
        if pergunta:
            resp = modelo.generate_content(pergunta)
            st.session_state.historico.append(("Você", pergunta))
            st.session_state.historico.append(("Chatbot", resp.text))

    if col2.button("Limpar histórico"):
        st.session_state.historico = []

    st.write("### Conversa")
    for autor, msg in st.session_state.historico:
        st.markdown(f"**{autor}:** {msg}")


# ===========================================================
#  PÁGINA 3 — MINI CONTROLE FUZZY
# ===========================================================
elif pagina == "Exemplo Controle Fuzzy":
    st.title("Sistema Fuzzy — Controle Automático de Ventilador")

    st.write("""
    Este é um **sistema fuzzy real**, modelado com a lógica de controle de ventiladores:

    **Entrada:** Temperatura ambiente  
    **Saída:** Potência do ventilador  
    """)

    # ---- Domínios ----
    x_temp = np.linspace(0, 40, 400)
    x_power = np.linspace(0, 100, 400)

    # ---- Funções fuzzy da entrada (temperatura) ----
    temp_fria = trimf(x_temp, [0, 0, 15])
    temp_amena = trimf(x_temp, [10, 20, 30])
    temp_quente = trimf(x_temp, [25, 40, 40])

    # ---- Funções fuzzy da saída (potência) ----
    pot_baixa = trimf(x_power, [0, 0, 50])
    pot_media = trimf(x_power, [25, 50, 75])
    pot_alta = trimf(x_power, [50, 100, 100])

    st.subheader("Funções de Pertinência da Entrada (Temperatura)")
    fig1, ax1 = plt.subplots()
    ax1.plot(x_temp, temp_fria, label="Fria", linestyle='--')
    ax1.plot(x_temp, temp_amena, label="Amena", linestyle='--')
    ax1.plot(x_temp, temp_quente, label="Quente", linestyle='--')
    ax1.set_title("Temperatura — Conjuntos Fuzzy")
    ax1.set_xlabel("Temperatura (°C)")
    ax1.set_ylabel("Pertinência")
    ax1.legend()
    st.pyplot(fig1)

    st.subheader("Funções de Pertinência da Saída (Potência do Ventilador)")
    fig2, ax2 = plt.subplots()
    ax2.plot(x_power, pot_baixa, label="Baixa", linestyle='--')
    ax2.plot(x_power, pot_media, label="Média", linestyle='--')
    ax2.plot(x_power, pot_alta,  label="Alta", linestyle='--')
    ax2.set_title("Potência — Conjuntos Fuzzy")
    ax2.set_xlabel("Potência (%)")
    ax2.set_ylabel("Pertinência")
    ax2.legend()
    st.pyplot(fig2)

    # Valor de entrada do usuário
    temp_val = st.slider("Temperatura atual (°C)", 0.0, 40.0, 20.0)

    # ---- Fuzzificação ----
    mu_fria = interp_membership(x_temp, temp_fria, temp_val)
    mu_amena = interp_membership(x_temp, temp_amena, temp_val)
    mu_quente = interp_membership(x_temp, temp_quente, temp_val)

    st.write(f"""
    ### Fuzzificação
    - Grau de pertença a **Fria**: `{mu_fria:.3f}`
    - Grau de pertença a **Amena**: `{mu_amena:.3f}`
    - Grau de pertença a **Quente**: `{mu_quente:.3f}`
    """)

    # ---- Regras ----
    regra1 = np.fmin(mu_fria, pot_baixa)
    regra2 = np.fmin(mu_amena, pot_media)
    regra3 = np.fmin(mu_quente, pot_alta)

    # ---- Agregação ----
    agregada = np.fmax(regra1, np.fmax(regra2, regra3))

    # ---- Defuzzificação ----
    potencia = np.sum(agregada * x_power) / np.sum(agregada)

    st.write(f"## 🔥 Potência recomendada: **{potencia:.2f}%**")

    st.subheader("Agregação e Centroide")
    fig3, ax3 = plt.subplots()
    ax3.plot(x_power, pot_baixa, label="Baixa", linestyle='--')
    ax3.plot(x_power, pot_media, label="Média", linestyle='--')
    ax3.plot(x_power, pot_alta, label="Alta", linestyle='--')
    ax3.fill_between(x_power, agregada, alpha=0.4,
                     color="orange", label="Agregação")
    ax3.axvline(potencia, color="red", linestyle=":",
                label=f"Centroide = {potencia:.2f}")
    ax3.set_xlabel("Potência (%)")
    ax3.set_ylabel("Pertinência")
    ax3.legend()
    st.pyplot(fig3)

    st.subheader("📘 Como o sistema funciona")
    st.write(f"""
    O controle fuzzy funciona analisando a temperatura:

    - A {temp_val}°C, a temperatura pertence **{mu_fria:.2f}** ao conjunto *Fria*
    - Pertence **{mu_amena:.2f}** ao conjunto *Amena*
    - Pertence **{mu_quente:.2f}** ao conjunto *Quente*

    As regras aplicadas são:

    1. **SE temperatura é fria → potência baixa**
    2. **SE temperatura é amena → potência média**
    3. **SE temperatura é quente → potência alta**

    A combinação dos graus de cada regra gera uma área agregada,  
    e o centro dessa área resulta na potência final de **{potencia:.2f}%**.

    Quanto mais quente o ambiente, maior será a ativação das regras que aumentam a potência,  
    o que leva a um valor final mais alto.
    """)

    with st.expander("Como este controlador toma decisões (detalhado)"):
        st.write("""
        1. **Fuzzificação:** convertemos a temperatura em graus de pertinência nos conjuntos *fria*, *amena*, *quente*.  
        2. **Regras:** cada regra aciona um conjunto de saída (ex.: "se temperatura é quente → potência é alta").  
        3. **Agregação:** unimos as saídas cortadas pela força de cada regra (operação max / min - Mamdani).  
        4. **Defuzzificação (centroide):** calculamos um valor numérico final (linha vermelha no gráfico).
        """)
        st.write("Alterar a temperatura muda os graus de pertinência — isso desloca quais regras têm maior força, mudando o agregado e, por consequência, o centroide (valor final).")


# ===========================================================
#  PÁGINA 4 — EDITOR DE SISTEMA FUZZY
# ===========================================================
elif pagina == "Editor Fuzzy":
    st.title("Editor de Sistema Fuzzy (Genérico)")

    # Inicializa o sistema se não existir
    if "sistema_fuzzy" not in st.session_state:
        st.session_state.sistema_fuzzy = {
            "entradas": {},
            "saidas": {},
            "conjuntos": {},
            "regras": []
        }

    sistema = st.session_state.sistema_fuzzy

    st.write("### ⚙️ Nesta aba você configura o sistema fuzzy. A simulação será exibida na aba **Simulador Fuzzy**.")

    # ----------------------------------------------------------
    # 1. ENTRADAS
    # ----------------------------------------------------------
    st.header("🟩 Variáveis de Entrada")

    nome_in = st.text_input("Nome da nova variável de entrada:")
    if st.button("Adicionar entrada"):
        if nome_in and nome_in not in sistema["entradas"]:
            sistema["entradas"][nome_in] = {
                "universo": [0.0, 100.0],
                "conjuntos": {}
            }
        else:
            st.error("Entrada já existe ou nome inválido.")

    for nome, var in list(sistema["entradas"].items()):
        st.subheader(f"Entrada: **{nome}**")

        umin, umax = st.slider(
            f"Universo de {nome}",
            0.0,
            200.0,
            tuple(map(float, var["universo"])),
            0.1,
            key=f"univ_{nome}"
        )
        var["universo"] = [umin, umax]

        if st.button(f"Excluir entrada {nome}", key=f"del_in_{nome}"):
            del sistema["entradas"][nome]
            st.experimental_rerun()

        st.write("#### ➕ Adicionar conjunto fuzzy")

        tipo = st.selectbox(
            f"Tipo de conjunto para {nome}",
            ["trimf", "trapmf", "gaussmf"],
            key=f"tipo_in_{nome}"
        )

        msg_param = {
            "trimf": "Parâmetros (a, b, c). Exemplo: 0, 10, 20",
            "trapmf": "Parâmetros (a, b, c, d). Exemplo: 0, 5, 15, 20",
            "gaussmf": "Parâmetros (σ, média). Exemplo: 5, 10"
        }

        nome_conj = st.text_input(
            f"Nome do conjunto fuzzy para {nome}",
            key=f"nome_conj_in_{nome}"
        )
        parametros = st.text_input(
            msg_param[tipo],
            key=f"param_in_{nome}"
        )

        if st.button(f"Adicionar conjunto a {nome}", key=f"addconj_in_{nome}"):
            raw = parametros.replace(" ", "").split(",")
            ok, msg = validate_mf_params(tipo, raw)
            if not ok:
                st.error("Parâmetros inválidos: " + msg)
            else:
                lista = [float(v) for v in raw]
                var["conjuntos"][nome_conj] = {"tipo": tipo, "params": lista}
                st.success(f"Conjunto '{nome_conj}' adicionado.")

        if var["conjuntos"]:
            st.write("#### Conjuntos existentes:")
            for conj_nome, conj_info in list(var["conjuntos"].items()):
                colA, colB = st.columns([3, 1])
                colA.write(f"• **{conj_nome}** — {conj_info}")
                if colB.button("Excluir", key=f"delconj_in_{nome}_{conj_nome}"):
                    del var["conjuntos"][conj_nome]
                    st.experimental_rerun()

    # ----------------------------------------------------------
    # 2. SAÍDAS
    # ----------------------------------------------------------
    st.header("🟦 Variáveis de Saída")

    nome_out = st.text_input("Nome da nova variável de saída:")
    if st.button("Adicionar saída"):
        if nome_out and nome_out not in sistema["saidas"]:
            sistema["saidas"][nome_out] = {
                "universo": [0.0, 100.0],
                "conjuntos": {}
            }
        else:
            st.error("Saída já existe ou nome inválido.")

    for nome, var in list(sistema["saidas"].items()):
        st.subheader(f"Saída: **{nome}**")

        umin, umax = st.slider(
            f"Universo de {nome}",
            0.0,
            200.0,
            tuple(map(float, var["universo"])),
            0.1,
            key=f"univ_out_{nome}"
        )
        var["universo"] = [umin, umax]

        if st.button(f"Excluir saída {nome}", key=f"del_out_{nome}"):
            del sistema["saidas"][nome]
            st.experimental_rerun()

        tipo = st.selectbox(
            f"Tipo de conjunto para {nome}",
            ["trimf", "trapmf", "gaussmf"],
            key=f"tipo_out_{nome}"
        )

        msg_param = {
            "trimf": "Parâmetros (a, b, c). Exemplo: 0, 50, 100",
            "trapmf": "Parâmetros (a, b, c, d). Exemplo: 0, 20, 80, 100",
            "gaussmf": "Parâmetros (σ, média). Exemplo: 10, 50"
        }

        nome_conj = st.text_input(
            f"Nome do conjunto fuzzy para {nome}",
            key=f"nome_conj_out_{nome}"
        )
        parametros = st.text_input(
            msg_param[tipo],
            key=f"param_out_{nome}"
        )

        if st.button(f"Adicionar conjunto a saída {nome}", key=f"addconj_out_{nome}"):
            raw = parametros.replace(" ", "").split(",")
            ok, msg = validate_mf_params(tipo, raw)
            if not ok:
                st.error("Parâmetros inválidos: " + msg)
            else:
                lista = [float(v) for v in raw]
                var["conjuntos"][nome_conj] = {"tipo": tipo, "params": lista}
                st.success(f"Conjunto '{nome_conj}' adicionado.")

        if var["conjuntos"]:
            st.write("#### Conjuntos existentes:")
            for conj_nome, conj_info in list(var["conjuntos"].items()):
                c1, c2 = st.columns([3, 1])
                c1.write(f"• **{conj_nome}** — {conj_info}")
                if c2.button("Excluir", key=f"delconj_out_{nome}_{conj_nome}"):
                    del var["conjuntos"][conj_nome]
                    st.experimental_rerun()

    # ----------------------------------------------------------
    # 3. REGRAS
    # ----------------------------------------------------------
    st.header("🟧 Regras Fuzzy")

    if sistema["entradas"] and sistema["saidas"]:

        entrada_sel = st.selectbox(
            "Variável de entrada:", list(sistema["entradas"].keys()))
        conj_in_sel = st.selectbox(
            "Conjunto da entrada:",
            list(sistema["entradas"][entrada_sel]["conjuntos"].keys())
        )

        saida_sel = st.selectbox("Variável de saída:",
                                 list(sistema["saidas"].keys()))
        conj_out_sel = st.selectbox(
            "Conjunto da saída:",
            list(sistema["saidas"][saida_sel]["conjuntos"].keys())
        )

        logica = st.selectbox("Operador lógico:", ["AND", "OR"])

        if st.button("Adicionar regra"):
            sistema["regras"].append({
                "antecedentes": [(entrada_sel, conj_in_sel)],
                "consequente": (saida_sel, conj_out_sel),
                "logica": logica
            })

    if sistema["regras"]:
        st.write("### 📜 Regras existentes:")
        for i, regra in enumerate(sistema["regras"]):
            col1, col2 = st.columns([4, 1])
            col1.write(f"• **Regra {i+1}** → {regra}")
            if col2.button("Excluir", key=f"delreg_{i}"):
                del sistema["regras"][i]
                st.experimental_rerun()

    st.subheader("Configurações do Sistema Fuzzy")

    operador_and_legivel = st.selectbox(
    "Operador AND",
        ["min (pega o menor grau entre os antecedentes)",
        "prod (multiplica os graus de pertinência)"]
    )

    operador_and = "min" if operador_and_legivel.startswith("min") else "prod"

    operador_or = st.selectbox(
        "Operador OR",
        ["max (pega o maior grau entre os antecedentes)",
         "prob_sum (soma probabilística: a + b − a·b)"]
    )

    st.session_state["fuzzy_ops"] = {
        "and": operador_and,
        "or": operador_or,
    }

    metodo_agregacao = st.selectbox(
        "Método de agregação das regras",
        ["max (seleciona o maior valor entre as regras)",
         "sum_clipped (soma limitada, acumulando contribuições até 1)"]
    )

    st.session_state["fuzzy_agregacao"] = metodo_agregacao

    metodo_defuzz = st.selectbox(
        "Método de defuzzificação",
        ["centroid (centro de área — o mais equilibrado)", "mom (média dos máximos)",
         "lom (maior valor entre os máximos)", "som (menor valor entre os máximos)"]
    )

    st.session_state["fuzzy_defuzz"] = metodo_defuzz

    # ----------------------------------------------------------
    # BOTÃO ATUALIZAR
    # ----------------------------------------------------------
    if st.button("💾 Atualizar sistema fuzzy"):
        st.success(
            "Sistema atualizado! Veja o resultado na aba **Simulador Fuzzy**.")


# ===========================================================
#  PÁGINA 5 — SIMULADOR FUZZY GENÉRICO
# ===========================================================
elif pagina == "Simulador Fuzzy":
    st.title("Simulador Fuzzy (Genérico)")

    if "sistema_fuzzy" not in st.session_state:
        st.error("Nenhum sistema fuzzy definido no editor.")
        st.stop()

    ops = st.session_state.get("fuzzy_ops", {"and": "min", "or": "max"})
    ag = st.session_state.get("fuzzy_agregacao", "max")
    df = st.session_state.get("fuzzy_defuzz", "centroid")

    sistema = st.session_state.sistema_fuzzy

    st.header("Entradas do sistema")

    valores = {}
    for nome, var in sistema["entradas"].items():
        umin, umax = var["universo"]
        valores[nome] = st.slider(
            f"Valor para {nome}", umin, umax, (umin + umax) / 2
        )

    def aplicar_and(a, b):
        if ops["and"] == "min":
            return min(a, b)
        elif ops["and"] == "prod":
            return a * b
        return min(a, b)

    def aplicar_or(a, b):
        if ops["or"] == "max":
            return max(a, b)
        elif ops["or"] == "prob_or":
            return a + b - a * b
        return max(a, b)
    
    def defuzzificar(xs, y, metodo):
        metodo = metodo.lower().strip()

        if metodo.startswith("centroid"):
            if np.sum(y) == 0:
                return 0
            return np.sum(xs * y) / np.sum(y)

        elif metodo.startswith("mom"):  # mean of maxima
            m = y.max()
            pts = xs[y == m]
            return np.mean(pts) if len(pts) > 0 else 0

        elif metodo.startswith("lom"):  # largest of maxima
            m = y.max()
            pts = xs[y == m]
            return np.max(pts) if len(pts) > 0 else 0

        elif metodo.startswith("som"):  # smallest of maxima
            m = y.max()
            pts = xs[y == m]
            return np.min(pts) if len(pts) > 0 else 0

        return 0
    def calcular_saida(sistema, valores):

        metodo = df.lower().strip()
        if metodo.startswith("centroid"):
            metodo = "centroid"
        elif metodo.startswith("mom"):
            metodo = "mom"
        elif metodo.startswith("lom"):
            metodo = "lom"
        elif metodo.startswith("som"):
            metodo = "som"
        else:
            metodo = "centroid"  

        ag_norm = ag.lower().strip()
        if ag_norm.startswith("max"):
            ag_op = "max"
        else:
            ag_op = "sum"

        agregadas = {}

        for saida in sistema["saidas"].keys():
            umin, umax = sistema["saidas"][saida]["universo"]
            agregadas[saida] = np.zeros(400)

        regras_at = []

        # ---------------------------------------------------
        # PROCESSAMENTO DAS REGRAS
        # ---------------------------------------------------
        for regra in sistema["regras"]:

            vals = []
            for (var, conj) in regra["antecedentes"]:

                entrada = sistema["entradas"][var]
                tipo = entrada["conjuntos"][conj]["tipo"]
                params = entrada["conjuntos"][conj]["params"]

                umin, umax = entrada["universo"]
                x = np.linspace(umin, umax, 400)

                if tipo == "trimf":
                    y = trimf(x, params)
                elif tipo == "trapmf":
                    y = trapmf(x, params)
                else:
                    y = gaussmf(x, params)

                vals.append(interp_membership(x, y, valores[var]))

            if len(vals) == 1:
                força = vals[0]
            else:
                força = vals[0]
                for v in vals[1:]:
                    if regra["logica"] == "AND":
                        força = aplicar_and(força, v)
                    else:
                        força = aplicar_or(força, v)

            regras_at.append((regra, força))

            saida, conj_s = regra["consequente"]
            tipo_s = sistema["saidas"][saida]["conjuntos"][conj_s]["tipo"]
            params_s = sistema["saidas"][saida]["conjuntos"][conj_s]["params"]

            umin, umax = sistema["saidas"][saida]["universo"]
            xs = np.linspace(umin, umax, 400)

            if tipo_s == "trimf":
                ys = trimf(xs, params_s)
            elif tipo_s == "trapmf":
                ys = trapmf(xs, params_s)
            else:
                ys = gaussmf(xs, params_s)

            # ---- 4. Agregação ----
            if ag_op == "max":
                agregadas[saida] = np.fmax(agregadas[saida], np.fmin(força, ys))
            else:  # Soma Limitada
                agregadas[saida] = np.minimum(1, agregadas[saida] + np.fmin(força, ys))

        # ---------------------------------------------------
        # DEFUZZIFICAÇÃO
        # ---------------------------------------------------
        resultados = {}
        for saida, yagg in agregadas.items():

            xs = np.linspace(
                sistema["saidas"][saida]["universo"][0],
                sistema["saidas"][saida]["universo"][1],
                400
            )

            centroide = defuzzificar(xs, yagg, metodo)

            resultados[saida] = (centroide, xs, yagg)

        return resultados, regras_at
    
    resultados, regras_at = calcular_saida(sistema, valores)

    st.header("Resultados")
    for saida, (centroide, xs, yagg) in resultados.items():
        st.write(f"**Saída {saida}: {centroide:.3f}**")

        fig, ax = plt.subplots()
        

        ax.plot(xs, yagg, label="Função agregada")

        ax.axvline(centroide, color='red', linestyle='--',
                   label=f"{df.lower().strip()} = {centroide:.2f}")

        ax.set_xlabel("Universo da saída")
        ax.set_ylabel("Pertinência")
        ax.set_title(f"Resultado fuzzy para saída: {saida}")
        ax.legend()

        st.pyplot(fig)

    st.header("Explicação com Gemini")
    if st.button("Gerar explicação"):
        prompt = f"""
        Explique o comportamento do sistema fuzzy.
        Entradas: {valores}
        Regras acionadas: {[ (r[0], round(r[1],3)) for r in regras_at ]}
        Resultados: {[ (s, round(v[0],3)) for s,v in resultados.items() ]}
        """

        resp = modelo.generate_content(prompt)
        st.write(resp.text)

# ===========================================================
#  PÁGINA 6 — GERADOR AUTOMÁTICO (SUBSTITUIR)
# ===========================================================
elif pagina == "Gerador Automático de Exemplos":
    import google.generativeai as genai
    import json

    st.title("Gerador Automático de Exemplos Fuzzy 🌱🤖")
    st.write("Digite um tema e o Gemini criará um exemplo completo. O sistema validará o JSON antes de mostrar resultados.")

    if "gerador_json" not in st.session_state:
        st.session_state.gerador_json = None

    tema = st.text_input(
        "Tema do exemplo (ex: irrigação, climatização, trânsito):")

    if st.button("Gerar Exemplo Fuzzy"):
        if not tema.strip():
            st.warning("Digite um tema primeiro!")
            st.stop()

        prompt = f"""
        Gere APENAS um JSON VÁLIDO seguindo estritamente este formato:

        {{
        "entradas": {{
            "variavel": {{
                "universo": [min, max],
                "conjuntos": {{
                    "nome": {{"tipo": "trimf"|"trapmf"|"gaussmf", "params": [números]}}
                }}
            }}
        }},
        "saidas": {{
            "variavel": {{
                "universo": [min, max],
                "conjuntos": {{
                    "nome": {{"tipo": "trimf"|"trapmf"|"gaussmf", "params": [números]}}
                }}
            }}
        }},
        "regras": [
            {{
                "antecedentes": [["entrada", "conjunto"], ["entrada2", "conjunto2"]],
                "consequente": ["saida", "conjunto"],
                "logica": "AND"
            }}
        ],
        "explicacao": "texto explicativo"
        }}

        REGRAS E RESTRIÇÕES IMPORTANTES:
        - NÃO gere chaves numéricas (ex: "0": {{...}}).
        - NÃO gere listas de pares (x,y).
        - NÃO gere membership functions como listas de pontos.
        - NÃO gere arrays com índice (ex: 0:, 1: ...).
        - Gere apenas trimf, trapmf ou gaussmf.
        - NÃO escreva NADA fora do JSON.
        - Gere no máximo 2–3 entradas e 1–2 saídas.

        Gere um sistema fuzzy com tema: "{tema}"
        """
        try:
            resposta = modelo.generate_content(prompt)
            bruto = resposta.text.strip()
        except Exception as e:
            st.error("Erro ao contactar o Gemini: " + str(e))
            st.stop()

        dados = None
        try:
            dados = json.loads(bruto)
        except Exception:
            dados = extrair_json(bruto)

        if not dados:
            st.error("❌ Não foi possível extrair JSON válido do Gemini.")
            dados = normalize_fuzzy_json(dados)
            dados["regras"] = normalize_regras(dados.get("regras", []))
            st.info("Resposta recebida:")
            st.write(bruto[:1000] + ("..." if len(bruto) > 1000 else ""))
            st.stop()

        missing = []
        dados = normalize_fuzzy_json(dados)
        for chave in ("entradas", "saidas", "regras"):
            if chave not in dados:
                missing.append(chave)
        if missing:
            st.error("JSON inválido: faltam as chaves: " + ", ".join(missing))
            st.info("Resposta recebida:")
            st.write(bruto[:1000] + ("..." if len(bruto) > 1000 else ""))
            st.stop()

        try:
            for nome, info in dados["entradas"].items():
                if "universo" not in info or "conjuntos" not in info:
                    raise ValueError(
                        f"Entrada '{nome}' sem universo ou conjuntos.")
                # validar tipos e params
                for conj, cinfo in info["conjuntos"].items():
                    ok, msg = validate_mf_params(
                        cinfo["tipo"], cinfo["params"])
                    if not ok:
                        raise ValueError(
                            f"Conjunto '{conj}' em '{nome}' inválido: {msg}")
            for nome, info in dados["saidas"].items():
                if "universo" not in info or "conjuntos" not in info:
                    raise ValueError(
                        f"Saída '{nome}' sem universo ou conjuntos.")
                for conj, cinfo in info["conjuntos"].items():
                    ok, msg = validate_mf_params(
                        cinfo["tipo"], cinfo["params"])
                    if not ok:
                        raise ValueError(
                            f"Conjunto '{conj}' em saída '{nome}' inválido: {msg}")
        except Exception as e:
            st.error("JSON inválido: " + str(e))
            st.stop()

        st.session_state.gerador_json = dados
        st.success(
            "Exemplo fuzzy gerado e validado com sucesso! Role para ver detalhes.")

    if st.session_state.gerador_json is None:
        st.stop()

    dados = st.session_state.gerador_json

    st.subheader("📘 Explicação do Sistema")
    st.write(dados.get("explicacao", "Sem explicação."))

    st.subheader("🔍 Variáveis Fuzzy Criadas")
    st.write("### Entradas")
    for nome, info in dados["entradas"].items():
        st.markdown(f"**• {nome}** — universo {info['universo']}")
        for conj, ci in info["conjuntos"].items():
            st.write(f" - {conj}: {ci.get('tipo')} {ci.get('params')}")

    st.write("### Saídas")
    for nome, info in dados["saidas"].items():
        st.markdown(f"**• {nome}** — universo {info['universo']}")
        for conj, ci in info["conjuntos"].items():
            st.write(f" - {conj}: {ci.get('tipo')} {ci.get('params')}")

    st.write("### Regras")
    for r in dados["regras"]:

        if "antecedentes" not in r or not r["antecedentes"]:
            st.write("- Regra ignorada (antecedentes vazios ou inválidos)")
            continue

        ant_list = []
        for a in r["antecedentes"]:
            try:
                ant_list.append(f"{a[0]} é {a[1]}")
            except:
                ant_list.append("(formato inválido)")

        ant = ", ".join(ant_list)

        cons = f"{r['consequente'][0]} é {r['consequente'][1]}"
        st.write(f"- SE {ant} ENTÃO {cons} ({r.get('logica','AND')})")

    st.subheader("🎛 Simulação do Sistema Fuzzy")
    valores_usuario = {}
    for nome, info in dados["entradas"].items():
        umin, umax = info["universo"]
        valores_usuario[nome] = st.slider(
            f"{nome}:", float(umin), float(umax), float((umin + umax) / 2), key=f"g_{nome}"
        )

    if st.button("Simular Sistema"):
        dominios = {}
        agregados = {}
        for nome_saida, info_saida in dados["saidas"].items():
            umin, umax = info_saida["universo"]
            dominios[nome_saida] = np.linspace(umin, umax, 400)
            agregados[nome_saida] = np.zeros(400)

        for regra in dados["regras"]:
            cons_var, cons_set = regra["consequente"]
            graus = []
            for var, conj in regra["antecedentes"]:
                mf_info = dados["entradas"][var]["conjuntos"][conj]
                tipo = mf_info["tipo"]
                params = mf_info["params"]
                universo = np.linspace(dados["entradas"][var]["universo"][0],
                                       dados["entradas"][var]["universo"][1], 400)
                if tipo == "trimf":
                    y = trimf(universo, params)
                elif tipo == "trapmf":
                    y = trapmf(universo, params)
                else:
                    y = gaussmf(universo, params)
                graus.append(np.interp(valores_usuario[var], universo, y))
            firing = min(graus) if graus else 0.0

            tipo_out = dados["saidas"][cons_var]["conjuntos"][cons_set]["tipo"]
            params_out = dados["saidas"][cons_var]["conjuntos"][cons_set]["params"]
            dom = dominios[cons_var]
            y = (trimf(dom, params_out) if tipo_out == "trimf"
                 else trapmf(dom, params_out) if tipo_out == "trapmf"
                 else gaussmf(dom, params_out))
            agregados[cons_var] = np.fmax(
                agregados[cons_var], np.fmin(firing, y))

        for nome_saida, agg in agregados.items():
            dom = dominios[nome_saida]
            centroide = (np.sum(dom * agg) / np.sum(agg)
                         ) if np.sum(agg) != 0 else 0.0
            st.write(f"### Saída: **{nome_saida} = {centroide:.2f}**")
            fig, ax = plt.subplots(figsize=(8, 3))
            ax.plot(dom, agg, label="Agregado (resultado final)")
            ax.axvline(centroide, color='red', linestyle='--',
                       label=f"Centroide = {centroide:.2f}")
            ax.set_xlabel("Universo da saída")
            ax.set_ylabel("Pertinência")
            ax.legend()
            st.pyplot(fig)

    if st.button("Importar exemplo para o Editor"):
        st.session_state.sistema_fuzzy = {
            "entradas": dados["entradas"],
            "saidas": dados["saidas"],
            "regras": [
                {"antecedentes": [(a[0], a[1]) for a in r["antecedentes"]],
                 "consequente": (r["consequente"][0], r["consequente"][1]),
                 "logica": r.get("logica", "AND")}
                for r in dados["regras"]
            ]
        }
        st.success("Exemplo importado para o Editor Fuzzy.")
