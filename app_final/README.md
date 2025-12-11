🧠 Sistema Fuzzy – Editor, Simulador e Gerador de Exemplos

Aplicação completa em Python + Streamlit para criação, edição e simulação de sistemas fuzzy, incluindo:
✔ Editor gráfico (entradas, saídas, regras, universos, funções de pertinência)
✔ Simulador com defuzzificação e gráficos
✔ Gerador automático de exemplos
✔ Armazenamento de sistema usando st.session_state
✔ Lógica fuzzy configurável (AND/OR, agregação, defuzzificação)

📁 Estrutura do Projeto
app_final/
│── app.py                # Arquivo principal da aplicação
│── requirements.txt      # Dependências
│── README.md             # Este guia
└── .gitignore            # Arquivos ignorados pelo Git

🚀 Como Rodar o Projeto

Este projeto utiliza Python 3.9+ e Streamlit.
Funciona em Windows, Linux e macOS.

✔️ 1. Instalar o Python (se necessário)

Baixe em:
https://www.python.org/downloads/

Durante a instalação, marque:

☑ Add Python to PATH

✔️ 2. Clonar este repositório

No terminal:

git clone https://github.com/RichardLopesPereira/app_final.git
cd app_final


Ou baixe o ZIP pelo GitHub.

✔️ 3. Criar ambiente virtual (recomendado)
Windows:
python -m venv venv
venv\Scripts\activate

Linux/macOS:
python3 -m venv venv
source venv/bin/activate

✔️ 4. Instalar dependências
pip install -r requirements.txt


O arquivo contém:

streamlit
numpy
matplotlib

✔️ 5. Rodar a aplicação
streamlit run app.py

Isso abrirá automaticamente no navegador em:

http://localhost:8501

🧠 Como Usar a Aplicação

A interface possui três módulos principais:

1️⃣ Gerador Automático de Exemplos

Cria um sistema fuzzy totalmente funcional com:

Entradas e saídas

Universos

Conjuntos fuzzy (trimf, trapmf, gaussmf)

Regras

Parâmetros padrão

O sistema gerado já pode ser simulado imediatamente.

2️⃣ Editor Fuzzy

Permite modificar:

Entradas e saídas

Universos

Tipos de funções de pertinência

Parâmetros dos conjuntos

Regras fuzzy

Métodos AND, OR, agregação e defuzzificação

Clique em Atualizar Sistema Fuzzy para aplicar suas edições.

✔ O estado é salvo com st.session_state, então nada se perde ao navegar pela interface.

3️⃣ Simulador

Permite:

Escolher valores de entrada

Visualizar:

funções de pertinência

regras ativadas

agregação fuzzy

defuzzificação

valor final da saída

A defuzzificação disponível inclui:

Centroid

Mean of Maxima

Largest of Maxima

Smallest of Maxima

🛠️ Tecnologias Utilizadas

Python

Streamlit (interface)

NumPy (cálculos numéricos)

Matplotlib (gráficos)

Lógica fuzzy implementada manualmente:

trimf()

trapmf()

gaussmf()

interp_membership()

Operadores AND/OR configuráveis

Agregação de regras (max ou soma-limitada)

Defuzzificação customizada


📦 Distribuição / Execução em Outra Máquina

Para rodar este projeto em qualquer computador:

Copie a pasta inteira

Tenha Python instalado

Execute:

pip install -r requirements.txt
streamlit run app.py


Não é necessário instalar nada adicional além das dependências listadas.

🐛 Problemas Comuns
Problema	Solução
AttributeError: st.session_state...	Rode primeiro a página inicial do app ou inicialize o estado
Gráficos não aparecem	Verifique se o Matplotlib está instalado
Navegador não abre	Acesse manualmente: http://localhost:8501

Streamlit não encontrado	Execute: pip install streamlit

🤝 Contribuições

Sinta-se à vontade para enviar melhorias, abrir issues ou sugerir novas funcionalidades.

📄 Licença

Este projeto é distribuído sob a licença MIT — livre para uso e modificação.
