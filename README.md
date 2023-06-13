# Machine Learning - Classificação do Risco de Crédito

![capa_ml_rlg](https://github.com/Cassiophysics/ML_classificacao_credito_risco/assets/108491443/9e460f0f-ae4b-467f-b722-ad6a06bf1b5f)

## Teste você mesmo o modelo: [💳 SISTEMA DE APROVAÇÃO DE EMPRÉSTIMOS](https://cassiophysics-ml-classificacao-credito-risc-streamlitapp-8jj1cb.streamlit.app/)

Este é um projeto de Machine Learning de aprendizado supervisionado que visa a elaboração de um modelo capaz de classificar quando um empréstimo foi bom e quando foi ruim.  Para tal finalidade, o conjunto de dados utilizado foi obtido a partir do site [KAGGLE](https://www.kaggle.com/datasets/uciml/german-credit), mas também pode ser encontrado em  [UCI](https://archive.ics.uci.edu/ml/datasets/statlog+(german+credit+data)). Este Dataset contém uma gama de atributos de clientes a fim de se prever o risco do empréstimo.

Neste projeto foi feito o uso de Pipelines para dar maior legibilidade ao código, facilitar a leitura do código, forçar a execução das transformações na ordem correta e tornar o script mais reproduzível. E as métricas definidas para a escolha do modelo mais adequado foram:

- Maior valor de recall
- Maior valor de f1-score
- A diferença entre falsos positivos e falsos negativos não ser acima de 50, de modo a manter um certo equilíbrio.

Assim, o raciocínio utilizado para este empreendimento foi sistematizado em:

**1. Análise Exploratória**

Nesta etapa foi utilizado com conjunto de técnicas como, por exemplo, gráficos e tabelas com o intuito de tentar compreender os dados e os seus padrões de uma maneira mais profunda. Esta é uma etapa importante no processo de análise de dados, pois permite a geração de insights valiosos sobre os dados e também verificar se os dados estão aptos para serem usados para responder o problema de negócio em questão. Neste projeto, através da análise exploratória feita foi possível constatar que o Dataset utilizado não contém valores duplicados, mas possui valores nulos e outliers, assim como diversas relações diferentes entre os atributos.

**2. Pré-processamento**

Dados sujos ou imprecisos podem levar a resultados incorretos, ou enganosos, por isso o pré-processamento dos dados é uma etapa importante, pois tem como intuito realizar procedimentos para garantir que os dados estejam prontos para a elaboração dos modelos de machine learning e que os resultados sejam confiáveis. À vista disso, fizemos o tratamento de valores nulos preenchendo pelo valor mais frequente no conjunto de dados, porque as colunas que continham valores ausentes eram categóricas, também realizamos a padronização dos dados convertendo para a mesma escala e a codificação das colunas categóricas.

**3. Tentativa de Tratamento do Desbalanceamento das Classes com Diferentes Pesos**

Foi testado diferentes valores de pesos para a classe de menor frequência, que no caso é a de empréstimos ruins, tendo em vista encontrar os melhores resultados conforme as métricas pré-definidas. Primeiro verificamos com Class Weight = 'balanced' nos algoritmos que possuem este parâmetro, em seguida foi utilizando GridSearchCV para encontrar o melhor peso, e por fim fizemos um loop testando diversos pesos diferentes.

**4. Tentativa de Tratamento do Desbalanceamento das Classes com SMOTE**

Ao ter conjuntos de dados com classes desbalanceadas pode-se levar a modelos de machine learning com desempenho ruim para a classe minoritária. Diante disso, foi utilizada a técnica SMOTE que propõe criar exemplos sintéticos da classe minoritária, no qual funciona selecionando exemplos da classe minoritária e encontra o k-ésimo vizinho mais próximo para cada um deles. Em seguida, ele cria um exemplo sintético no espaço de características, colocando-o a meio caminho entre o exemplo selecionado e seu vizinho mais próximo. Isso é repetido até que a classe minoritária tenha o mesmo tamanho da classe majoritária. Isto posto, é importante ter em mente que em alguns casos o SMOTE pode levar a uma perda de precisão e a um overfitting do modelo. No nosso caso, a título de exemplo, o tratamento de desbalanceamento testando diferentes pesos dispôs de melhores resultados comparado com o SMOTE.

**5. Otimização de Hiperparâmetros**

A otimização de hiperparâmetros é importante porque diferentes conjuntos de hiperparâmetros podem resultar em modelos com desempenhos significativamente melhores. Além disso, os hiperparâmetros podem afetar a velocidade do treinamento do modelo e a capacidade de generalização do modelo para novos dados. Para tal finalidade, foi utilizado GridSearchCV e RandomizedSearchCV para obter os hiperparâmetros mais adequados. Entretanto, como tivemos que definir uma única métrica previamente, os melhores hiperparâmetros encontrados consideram somente essa métrica, o que não leva aos melhores resultados segundo os critérios definidos neste projeto.

___

**Resultado Final:** Conforme os critérios que definimos, o modelo baseline LogisticRegression com peso 3.22 para a classe 1, apresentou o melhor resultado, com recall 0.80, f1-score 0.59, diferença entre falsos positivos e falsos negativos, igual a 50 e uma acurácia de 0.61. Ou seja, de todos os empréstimos presentes na amostra que realmente foram ruins (Falsos Negativos) nosso modelo conseguiu acertar 80%, mas também manteve um número razoável de empréstimos bons classificados erroneamente como ruins (Falsos Positivos). A escolha do modelo ideal deve ser feita conforme as métricas de negócio estabelecidas. Se o custo de perder um empréstimo ruim supera em muito o custo de cancelar vários empréstimos legítimos, ou seja, falsos positivos, talvez possamos escolher um peso que nos dê uma taxa de recall mais alta. Isso ocorre porque aumentamos nossa pontuação de recall de maus empréstimos à custa de mais casos legítimos mal classificados.


