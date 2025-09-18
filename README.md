# How-To-train-and-evaluate-a-classification-model-using-scikit-learn.
Este **How to** é um guia  para iniciantes que desejam aprender a treinar, prever e avaliar modelos de Machine Learning usando Python e a biblioteca Scikit-learn.

Ele cobre os dois principais tipos de aprendizado supervisionado: Classificação (categorias) e Regressão (valores contínuos), mostrando passo a passo como preparar os dados, treinar os modelos, fazer previsões e avaliar a performance.

## Sobre o Projeto:
O objetivo deste projeto é 

- Demonstrar como preparar dados para Machine Learning.
- Treinar modelos de classificação e regressão.
- Fazer previsões em novos dados.
- Avaliar a performance do modelo usando métricas específicas.
- Interpretar resultados e comparar algoritmos.

> Como exemplo prático, usamos o clássico dataset Iris, que classifica flores em três espécies com base em suas características físicas.
> 
## ⚙️ Estrutura do Projeto
### Pré-requisitos:
Para executar os exemplos, você precisará de:
- Python 3.8+

- Bibliotecas Python:
    - scikit-learn
>  **Instalação rápida :**
> pip install scikit-learn

> Bibliotecas : numpy, pandas e matplotlib são recomendadas, mas não  fundamentais.


## O que é o Skitit- learn?
O Scikit-learn (ou apenas sklearn) é uma biblioteca de aprendizado de máquina em Python.

Ela foi construída sobre outras bibliotecas muito usadas em ciência de dados, como NumPy, SciPy e matplotlib, e se tornou uma das ferramentas mais populares para treinar, testar e avaliar modelos de Machine Learning de forma simples e eficiente.

### Para que serve?

Com o Scikit-learn, você pode:

- Treinar modelos de Machine Learning → como regressão, classificação, clusterização, etc.
- Pré-processar dados → normalizar, padronizar, dividir entre treino e teste.
- Avaliar modelos → métricas como acurácia, precisão, recall, F1-score.
- Fazer seleção de atributos → escolher variáveis mais importantes.
- Fazer validação cruzada → avaliar se o modelo generaliza bem.
  
> 💡 Dica: O Scikit-learn segue uma lógica simples e padronizada: fit() → treinar, predict() → prever, score() → avaliar.

### Exemplos de algoritmos que o Scikit-learn oferece

- Classificação → Árvores de decisão, Regressão logística, SVM, Naive Bayes, KNN.
- Regressão → Regressão linear, Regressão ridge/lasso.
- Clusterização → K-Means, DBSCAN, Agglomerative Clustering.
- Redução de dimensionalidade → PCA (Análise de Componentes Principais).

### Por que é importante?

Ele é considerado a "porta de entrada" para quem está aprendendo ciência de dados, porque:

- Tem interface padronizada (todos os modelos seguem a lógica fit → treinar, predict → prever).
- É bem documentado e com muitos exemplos.
- Funciona muito bem em datasets pequenos e médios.

> 👉 Resumindo: o Scikit-learn é como uma “caixa de ferramentas” completa para testar rapidamente ideias em Machine Learning.

### 📌 Próximos passos no projeto
1. Carregar e explorar os dados (ex.: Iris para classificação, Diabetes para regressão).

2. Pré-processar os dados (normalização, tratamento de valores faltantes, divisão treino/teste).

3. Treinar modelos de classificação (ex.: KNN, Logistic Regression, Decision Tree).

4. Treinar modelos de regressão (ex.: Linear Regression, Ridge, Lasso).

5. Fazer previsões em novos dados.

6. Avaliar desempenho do modelo usando métricas específicas (acurácia, R², RMSE etc.).

