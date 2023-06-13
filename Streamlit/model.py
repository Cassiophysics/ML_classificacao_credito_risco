# Importação das Bibliotecas
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LogisticRegression
from sklearn.compose import ColumnTransformer
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.pipeline import Pipeline
from imblearn.pipeline import Pipeline

import joblib

# Carregamento dos dados em arquivo csv
df = pd.read_csv('german_credit_data.csv', index_col=[0])

# Renomeação das colunas 
df.columns = ['idade', 'sexo', 'emprego', 'habitacao', 'conta_poupanca', 'conta_corrente', 'credito',
              'duracao', 'proposito', 'risco']

# Transformação da coluna emprego em tipo categórico
df['emprego'] = df['emprego'].astype('category')

# Redefinição do risco em 0 para bom e 1 para ruim
df['risco'] = np.where(df['risco']=='bad', 1, 0)

# Separação das variáveis preditoras da variável alvo
X = df.drop(columns='risco') 
Y = df['risco']

# Divisão do Dataset em treino e test
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.20, random_state=18)

# Pipeline somente para Valores Nulos
cols_nulos = ['conta_poupanca', 'conta_corrente']
preprocessor = ColumnTransformer(
    transformers=[
        ('imputer', 
            SimpleImputer(missing_values=np.nan, strategy='most_frequent'),
            cols_nulos),
])

preprocessor.fit(X_train)
X_train_new = preprocessor.transform(X_train)
X_test_new = preprocessor.transform(X_test)

for icol, col in enumerate(cols_nulos):
    X_train.loc[:, col] = X_train_new[:, icol]
    X_test.loc[:, col] = X_test_new[:, icol]

# Preprocessor
preprocessor1 = ColumnTransformer(
transformers=[
    ('OneHotEncoder', OneHotEncoder(handle_unknown='ignore'), ['sexo', 'habitacao', 'conta_poupanca', 'conta_corrente', 'proposito', 'emprego']),
    ('Padronização', StandardScaler(), ['idade', 'credito', 'duracao'])])


# Pipeline

lr = Pipeline(steps=[('preprocessor', preprocessor1), ('modelo', LogisticRegression(random_state=18, class_weight={0: 1, 1: 3.2222222222222223}))])
lr.fit(X_train, Y_train)
Y_pred = lr.predict(X_test)
print(confusion_matrix(Y_test, Y_pred))
print('\n')
print(classification_report(Y_test, Y_pred))

# Salvar o modelo
joblib.dump(lr, 'modelo_lr.sav')

# Criar o StandardScaler
scaler = StandardScaler()

# Aplicar o StandardScaler nos dados numéricos
X_numerical = X_train[['idade', 'credito', 'duracao']]
X_numerical_scaled = scaler.fit_transform(X_numerical)

# Salvar o StandardScaler
joblib.dump(scaler, 'standard_scaler.sav')

# Salvar o Modelo Final
#joblib.dump(xgbr2, "xgbr2_model.sav")








