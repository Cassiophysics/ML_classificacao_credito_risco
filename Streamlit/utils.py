# Função para excluir colunas
#def drop_original_columns(X):
#    return X.drop(['sexo', 'habitacao', 'conta_poupanca', 'conta_corrente', 'proposito', 'emprego'], axis=1)


import pandas as pd

#def drop_original_columns(X):
    # Remover as colunas originais do DataFrame X
#    X = pd.DataFrame(X)
#    X = X.drop(['sexo', 'habitacao', 'conta_poupanca', 'conta_corrente', 'proposito', 'emprego'], axis=1)
#    return X


#def drop_original_columns(X):
    # Remover as colunas originais do DataFrame X
#    X = X.drop(['sexo', 'habitacao', 'conta_poupanca', 'conta_corrente', 'proposito', 'emprego'], axis=1)
#    return X

#def drop_original_columns(X):
#    X.drop(['sexo', 'habitacao', 'conta_poupanca', 'conta_corrente', 'proposito', 'emprego'], axis=1, inplace=True)
#    return X

# Função para excluir colunas
def drop_original_columns(X):
    return X.drop(['sexo', 'habitacao', 'conta_poupanca', 'conta_corrente', 'proposito', 'emprego'], axis=1)
