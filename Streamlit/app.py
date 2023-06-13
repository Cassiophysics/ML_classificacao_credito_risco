# Importação das Bibliotecas
import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Carregar o modelo
modelo_lr = joblib.load('modelo_lr.sav')

# Carregar o StandardScaler
scaler = joblib.load('standard_scaler.sav')

# Criar a interface do Streamlit
st.title('💳 Sistema de Aprovação de Empréstimos')
st.header('Insira os Dados')

# Opções para as colunas de escolha
opcoes_sexo = ['female', 'male']
opcoes_habitacao = ['free', 'own', 'rent']
opcoes_conta_poupanca = ['little', 'moderate', 'quite rich', 'rich']
opcoes_conta_corrente = ['little', 'moderate', 'rich']
opcoes_proposito = ['business', 'car', 'domestic appliances', 'education', 'furniture/equipment', 'radio/TV', 'repairs', 'vacation/others']
opcoes_emprego = ['0', '1', '2', '3']

# Obtendo os valores escolhidos nas caixas de seleção
opcao_sexo = st.selectbox('Sexo', opcoes_sexo)
opcao_habitacao = st.selectbox('Habitação', opcoes_habitacao)
opcao_conta_poupanca = st.selectbox('Conta Poupança', opcoes_conta_poupanca)
opcao_conta_corrente = st.selectbox('Conta Corrente', opcoes_conta_corrente)
opcao_proposito = st.selectbox('Propósito', opcoes_proposito)
opcao_emprego = st.selectbox('Emprego', opcoes_emprego)

# Dados numéricos
idade = st.slider('Idade', 19, 75)
credito = st.slider('Crédito', 250, 18424)
duracao = st.slider('Duração', 4, 72)

# Realizar a transformação dos dados de entrada
X = pd.DataFrame({
    'sexo': [opcao_sexo],
    'habitacao': [opcao_habitacao],
    'conta_poupanca': [opcao_conta_poupanca],
    'conta_corrente': [opcao_conta_corrente],
    'proposito': [opcao_proposito],
    'emprego': [opcao_emprego],
    'idade': [idade],
    'credito': [credito],
    'duracao': [duracao]
})

# Aplicar o StandardScaler nos dados numéricos
X[['idade', 'credito', 'duracao']] = scaler.transform(X[['idade', 'credito', 'duracao']])

# Fazer a previsão usando o modelo carregado
if st.button('Fazer Previsão'):
    resultado = modelo_lr.predict(X)
    st.header('Resultado da Previsão')
    previsao = "Empréstimo Aprovado!" if resultado == 0 else "Empréstimo Negado!"
    st.write(f'A previsão é: {previsao}')