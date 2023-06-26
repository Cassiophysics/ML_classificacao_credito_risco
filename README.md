# Machine Learning - Classificação do Risco de Crédito

![capa_ml_rlg](https://github.com/Cassiophysics/ML_classificacao_credito_risco/assets/108491443/4bb91092-6948-48e5-ba8d-945fbffeafd3)

## Teste você mesmo o modelo: [💳 SISTEMA DE APROVAÇÃO DE EMPRÉSTIMOS](https://cassiophysics-ml-classificacao-credito-risc-streamlitapp-8jj1cb.streamlit.app/)

Este é um projeto de Machine Learning de aprendizado supervisionado que visa a elaboração de um modelo capaz de classificar quando um empréstimo foi bom e quando foi ruim.  Para tal finalidade, o conjunto de dados utilizado foi obtido a partir do site [KAGGLE](https://www.kaggle.com/datasets/uciml/german-credit), mas também pode ser encontrado em  [UCI](https://archive.ics.uci.edu/ml/datasets/statlog+(german+credit+data)). Este Dataset contém uma gama de atributos de clientes a fim de se prever o risco do empréstimo.

## Motivação:

O setor de empréstimos desempenha um papel crucial na economia, fornecendo recursos financeiros para indivíduos e empresas realizarem seus objetivos e projetos. No entanto, é essencial que os empréstimos sejam concedidos de forma responsável, considerando os riscos envolvidos e minimizando o impacto de possíveis inadimplências.

Diante desse desafio, surge a necessidade de desenvolver abordagens inovadoras para auxiliar instituições financeiras, bancos e empresas de crédito na avaliação de solicitações de empréstimos. É nesse contexto que se insere o projeto de classificação de empréstimos que desenvolvi.

O objetivo desse projeto é utilizar técnicas de machine learning para criar um modelo capaz de classificar se um empréstimo é considerado bom ou ruim com base em informações e dados disponíveis. A classificação correta dos empréstimos permite que as instituições financeiras tomem decisões mais informadas, minimizem riscos e melhorem a eficiência dos processos de concessão de crédito.

Para alcançar esse objetivo, foram explorados algoritmos de aprendizado de máquina, técnicas de pré-processamento de dados e validação de modelos. A construção desse modelo envolveu etapas cruciais, como a coleta e preparação dos dados relevantes e a escolha do algoritmo de classificação mais adequado.

Ao desenvolver esse projeto, busquei não apenas aprimorar minhas habilidades em modelagem de machine learning, mas também contribuir para o avanço da área de crédito e finanças. O modelo criado possui o potencial de otimizar o processo de tomada de decisão, identificando padrões e características relevantes para a classificação de empréstimos, proporcionando uma avaliação mais precisa e confiável.

Ao disponibilizar esse projeto no GitHub, espero compartilhar conhecimento e incentivar a colaboração com outros profissionais e entusiastas da área. Além disso, acredito que essa iniciativa pode contribuir para a transparência e aprimoramento dos processos de concessão de crédito, beneficiando tanto as instituições financeiras quanto os clientes em busca de empréstimos.

Estou entusiasmado com os resultados alcançados neste projeto e confiante de que ele pode ter um impacto positivo no setor financeiro. Convido você a explorar o código, os dados e os resultados apresentados neste repositório e a se juntar a mim nessa jornada em busca de soluções inovadoras para o mercado de empréstimos.

Neste projeto foi feito o uso de Pipelines para dar maior legibilidade ao código, facilitar a leitura do código, forçar a execução das transformações na ordem correta e tornar o script mais reproduzível. E as métricas definidas para a escolha do modelo mais adequado foram:

- Maior valor de recall
- Maior valor de f1-score
- A diferença entre falsos positivos e falsos negativos não ser acima de 50, de modo a manter um certo equilíbrio.

Assim, o raciocínio utilizado para este empreendimento foi sistematizado em:

## **1. Análise Exploratória**

Nesta etapa foi utilizado com conjunto de técnicas como, por exemplo, gráficos e tabelas com o intuito de tentar compreender os dados e os seus padrões de uma maneira mais profunda. Esta é uma etapa importante no processo de análise de dados, pois permite a geração de insights valiosos sobre os dados e também verificar se os dados estão aptos para serem usados para responder o problema de negócio em questão. Neste projeto, através da análise exploratória feita foi possível constatar que o Dataset utilizado não contém valores duplicados, mas possui valores nulos e outliers, assim como diversas relações diferentes entre os atributos.

## **2. Pré-processamento**

Dados sujos ou imprecisos podem levar a resultados incorretos, ou enganosos, por isso o pré-processamento dos dados é uma etapa importante, pois tem como intuito realizar procedimentos para garantir que os dados estejam prontos para a elaboração dos modelos de machine learning e que os resultados sejam confiáveis. À vista disso, fizemos o tratamento de valores nulos preenchendo pelo valor mais frequente no conjunto de dados, porque as colunas que continham valores ausentes eram categóricas, também realizamos a padronização dos dados convertendo para a mesma escala e a codificação das colunas categóricas.

## **3. Tentativa de Tratamento do Desbalanceamento das Classes com Diferentes Pesos**

Foi testado diferentes valores de pesos para a classe de menor frequência, que no caso é a de empréstimos ruins, tendo em vista encontrar os melhores resultados conforme as métricas pré-definidas. Primeiro verificamos com Class Weight = 'balanced' nos algoritmos que possuem este parâmetro, em seguida foi utilizando GridSearchCV para encontrar o melhor peso, e por fim fizemos um loop testando diversos pesos diferentes.

## **4. Tentativa de Tratamento do Desbalanceamento das Classes com SMOTE**

Ao ter conjuntos de dados com classes desbalanceadas pode-se levar a modelos de machine learning com desempenho ruim para a classe minoritária. Diante disso, foi utilizada a técnica SMOTE que propõe criar exemplos sintéticos da classe minoritária, no qual funciona selecionando exemplos da classe minoritária e encontra o k-ésimo vizinho mais próximo para cada um deles. Em seguida, ele cria um exemplo sintético no espaço de características, colocando-o a meio caminho entre o exemplo selecionado e seu vizinho mais próximo. Isso é repetido até que a classe minoritária tenha o mesmo tamanho da classe majoritária. Isto posto, é importante ter em mente que em alguns casos o SMOTE pode levar a uma perda de precisão e a um overfitting do modelo. No nosso caso, a título de exemplo, o tratamento de desbalanceamento testando diferentes pesos dispôs de melhores resultados comparado com o SMOTE.

## **5. Otimização de Hiperparâmetros**

A otimização de hiperparâmetros é importante porque diferentes conjuntos de hiperparâmetros podem resultar em modelos com desempenhos significativamente melhores. Além disso, os hiperparâmetros podem afetar a velocidade do treinamento do modelo e a capacidade de generalização do modelo para novos dados. Para tal finalidade, foi utilizado GridSearchCV e RandomizedSearchCV para obter os hiperparâmetros mais adequados. Entretanto, como tivemos que definir uma única métrica previamente, os melhores hiperparâmetros encontrados consideram somente essa métrica, o que não leva aos melhores resultados segundo os critérios definidos neste projeto.

## Resultado Final do Modelo:

Conforme os critérios que definimos, o modelo baseline LogisticRegression com peso 3.22 para a classe 1, apresentou o melhor resultado, com recall 0.80, f1-score 0.59, diferença entre falsos positivos e falsos negativos, igual a 50 e uma acurácia de 0.61. Ou seja, de todos os empréstimos presentes na amostra que realmente foram ruins (Falsos Negativos) nosso modelo conseguiu acertar 80%, mas também manteve um número razoável de empréstimos bons classificados erroneamente como ruins (Falsos Positivos). A escolha do modelo ideal deve ser feita conforme as métricas de negócio estabelecidas. Se o custo de perder um empréstimo ruim supera em muito o custo de cancelar vários empréstimos legítimos, ou seja, falsos positivos, talvez possamos escolher um peso que nos dê uma taxa de recall mais alta. Isso ocorre porque aumentamos nossa pontuação de recall de maus empréstimos à custa de mais casos legítimos mal classificados.

## Impacto nos negócios:

**Melhoria da tomada de decisões:** O modelo de classificação empréstimo bom/ruim oferece uma ferramenta confiável para avaliar a viabilidade de empréstimos, reduzindo o risco de concessões a clientes com maior probabilidade de inadimplência.

**Redução de riscos e perdas financeiras:** O modelo identifica empréstimos de alto risco antecipadamente, permitindo medidas preventivas, como limites de crédito mais baixos ou recusa do empréstimo, reduzindo as perdas financeiras.

**Aumento da eficiência operacional:** O processo automatizado acelera a avaliação de crédito, economizando tempo e recursos, direcionando a equipe para atividades estratégicas.

**Melhoria da experiência do cliente:** Com uma avaliação precisa, a empresa pode oferecer melhores condições de empréstimo a clientes confiáveis, melhorando sua experiência e fortalecendo o relacionamento.

**Aprimoramento da gestão de riscos:** O modelo fornece insights sobre fatores que influenciam a qualidade do empréstimo, melhorando as estratégias de gerenciamento de riscos e a saúde financeira.

**Vantagem competitiva:** A implementação do modelo diferencia a empresa, atraindo mais clientes e fortalecendo sua reputação no mercado.

## Identificação de melhorias para o modelo:

**Aumentar o tamanho e a qualidade do conjunto de dados:** Coletar mais dados de alta qualidade para enriquecer a variabilidade e representatividade das amostras.

**Feature Engineering:** Analisar de forma mais detalhada as características existentes e identificar se há oportunidades para criar novas variáveis ou transformar as existentes de maneira mais informativa. Isso pode envolver a combinação de variáveis, criação de variáveis ​​interativas, extração de características relevantes ou até mesmo a utilização de técnicas avançadas como redução de dimensionalidade.

**Regularização:** Considerar a aplicação de técnicas de regularização, como a penalização L1 ou L2, para evitar o overfitting e melhorar a generalização do modelo. Isso pode ajudar a controlar a complexidade do modelo e reduzir a sensibilidade a outliers ou ruídos nos dados.

**Análise de Resíduos:** Realizar uma análise detalhada dos resíduos do modelo para identificar possíveis padrões não capturados ou problemas de modelagem. A análise de resíduos pode ajudar a identificar áreas de melhoria, como a inclusão de variáveis ​​relevantes ou a aplicação de transformações adicionais.

**Monitoramento e Retreinamento:** Estabelecer um processo de monitoramento contínuo do desempenho do seu modelo em produção. Se possível, coletar novos dados e reavaliar regularmente o desempenho do modelo. Isso permitirá identificar mudanças nos padrões ou no comportamento dos clientes ao longo do tempo e garantir que o modelo permaneça atualizado e relevante.




