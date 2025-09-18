# 📘 Regressão com Scikit-learn

Este guia apresenta uma introdução completa a modelos de regressão, explicando conceitos, aplicações, pré-processamento, avaliação de performance e implementação prática em Python com scikit-learn

## 📌 O que é Regressão?
Regressão é uma técnica de aprendizado supervisionado utilizada quando a variável alvo (y) é contínua (valores numéricos).

> 👉 Exemplo na área da saúde: prever o nível de glicose no sangue de um paciente a partir de variáveis como idade, peso e hábitos de vida.

### Diferença para Classificação

- Classificação → prevê categorias (ex.: "tem diabetes" ou "não tem diabetes").

- Regressão → prevê valores contínuos (ex.: "nível de glicose = 135 mg/dL").

## 🏥 Exemplos de problemas resolvidos com Regressão

- Previsão do nível de colesterol com base em dieta e exercícios.

- Estimativa da pressão arterial a partir de idade e IMC.

- Cálculo do tempo de internação hospitalar.

- Progressão de tumores em mm³ ao longo do tempo.

## 🔹 Algoritmos de Regressão no Scikit-learn
### 1. Regressão Linear (LinearRegression)

- 📖 O modelo mais simples: assume que a relação entre variáveis é linear (reta ou hiperplano).

- ✅ Bom para dados simples e quando queremos interpretabilidade (coeficientes mostram a importância de cada variável).

- 🏥 Exemplo em saúde: prever pressão arterial a partir de idade e IMC.

### 2. Ridge Regression (Ridge)

- 📖 Variante da regressão linear com regularização L2 (penaliza coeficientes grandes).

- ✅ Ajuda a evitar overfitting quando temos muitas variáveis.

- 🏥 Exemplo: prever níveis de colesterol a partir de exames laboratoriais diversos.

### 3. Lasso Regression (Lasso)

- 📖 Variante da linear com regularização L1 (pode zerar coeficientes irrelevantes → seleção de variáveis automática).

- ✅ Bom para datasets com muitas variáveis irrelevantes.

- 🏥 Exemplo: identificar quais exames são realmente relevantes para prever glicose.

### 4. Elastic Net (ElasticNet)

- 📖 Combina L1 (Lasso) e L2 (Ridge).

- ✅ Mais flexível: pode ao mesmo tempo selecionar variáveis e evitar overfitting.

- 🏥 Exemplo: prever risco de diabetes combinando exames laboratoriais e fatores de estilo de vida.

### 5. Regressão Polinomial (PolynomialFeatures + LinearRegression)

- 📖 Expande variáveis para potências maiores (x², x³…) e aplica regressão linear.

- ✅ Útil quando a relação não é uma reta, mas uma curva.

- 🏥 Exemplo: crescimento de um tumor que segue curva quadrática, não linear.

### 6. Árvore de Regressão (DecisionTreeRegressor)

- 📖 Divide os dados em regras de decisão (se idade > 50 e IMC > 30 → risco maior).

- ✅ Fácil de interpretar, mas pode superajustar (overfitting).

- 🏥 Exemplo: prever tempo de internação hospitalar baseado em condições pré-existentes.

### 7. Random Forest Regressor (RandomForestRegressor)

- 📖 Conjunto de várias árvores (floresta).

- ✅ Mais robusto que uma árvore só, generaliza melhor.

- 🏥 Exemplo: prever tempo de recuperação pós-cirurgia a partir de vários fatores.

### 8. Gradient Boosting Regressor (GradientBoostingRegressor)

- 📖 Cria árvores de forma sequencial, cada uma corrigindo os erros da anterior.

- ✅ Excelente desempenho em dados tabulares, muitas vezes melhor que Random Forest.

- 🏥 Exemplo: prever níveis de glicose ao longo do tempo com base em histórico clínico.

### 9. AdaBoost Regressor (AdaBoostRegressor)

- 📖 Similar ao Gradient Boosting, mas dá mais peso a exemplos onde o modelo anterior errou.

- ✅ Bom para dados com ruído moderado.

- 🏥 Exemplo: prever pressão arterial em pacientes onde alguns dados são inconsistentes.

### 10. HistGradientBoosting Regressor (HistGradientBoostingRegressor)

- 📖 Versão otimizada do Gradient Boosting, muito rápida e eficiente para grandes datasets.

- ✅ Escalável para milhões de amostras.

- 🏥 Exemplo: prever custos hospitalares em grandes bases de dados de saúde pública.

### 11. K-Nearest Neighbors Regressor (KNeighborsRegressor)

- 📖 Usa a média dos k vizinhos mais próximos para prever.

- ✅ Simples, não assume forma da função.

- ❌ Fica lento com muitos dados.

- 🏥 Exemplo: prever nível de glicose de um paciente comparando com pacientes similares.

### 12. Support Vector Regressor (SVR)

- 📖 Usa máquinas de vetores de suporte para prever, podendo usar funções de kernel (linear, polinomial, radial).

- ✅ Bom para dados complexos e não lineares.

- ❌ Pode ser lento em datasets grandes.

- 🏥 Exemplo: prever progressão da diabetes em pacientes com muitos fatores interdependentes.

### 13. Bayesian Ridge Regression (BayesianRidge)

- 📖 Regressão linear com abordagem bayesiana: fornece incerteza das previsões.

- ✅ Útil quando queremos intervalos de confiança.

- 🏥 Exemplo: prever níveis de colesterol e dar uma margem de incerteza (ex.: 180 ± 15).

### 14. Huber Regressor (HuberRegressor)

- 📖 Combina vantagens da regressão linear e robustez contra outliers.

- ✅ Melhor que linear quando há valores extremos nos dados.

- 🏥 Exemplo: prever pressão arterial ignorando medidas erradas de alguns pacientes.

### 15. Theil-Sen Regressor (TheilSenRegressor)

- 📖 Método robusto que usa medianas em vez de médias, resistente a outliers.

- ✅ Bom para pequenas bases de dados com ruídos.

- 🏥 Exemplo: prever peso fetal em um estudo pequeno, com algumas medidas imprecisas.

### 16. RANSAC Regressor (RANSACRegressor)

- 📖 Ajusta o modelo várias vezes, descartando pontos que parecem outliers.

- ✅ Excelente para dados com muitos erros de medição.

- 🏥 Exemplo: prever taxa de crescimento tumoral, ignorando exames claramente errados.

| Modelo                          | Descrição                           | Vantagens                         | Desvantagens                    | Exemplo em Saúde |
|---------------------------------|-------------------------------------|-----------------------------------|----------------------------------|------------------|
| LinearRegression                | Modelo linear simples               | Interpretação fácil                | Não captura relações complexas   | Pressão arterial vs idade |
| Ridge                           | Linear com regularização L2         | Evita overfitting                  | Difícil interpretar coeficientes | Colesterol vs exames |
| Lasso                           | Linear com regularização L1         | Seleciona variáveis automaticamente| Pode zerar variáveis importantes | Seleção de exames para prever glicose |
| ElasticNet                      | Combina Lasso e Ridge               | Flexível e equilibrado             | Mais parâmetros para ajustar     | Risco de diabetes |
| Polynomial Regression           | Linear com termos polinomiais       | Captura curvas                     | Pode overfittar                  | Crescimento tumoral |
| DecisionTreeRegressor           | Baseado em regras                   | Fácil de interpretar               | Overfitting                      | Tempo de internação |
| RandomForestRegressor           | Conjunto de árvores                 | Robusto, bom desempenho            | Menos interpretável              | Recuperação pós-cirurgia |
| GradientBoostingRegressor       | Árvores sequenciais                 | Alta performance                   | Mais lento que Random Forest     | Progressão da diabetes |
| AdaBoostRegressor               | Árvores com reponderação            | Lida bem com ruído                 | Pode falhar com outliers extremos| Pressão arterial |
| HistGradientBoostingRegressor   | Boosting otimizado                  | Muito rápido e escalável           | Menos interpretável              | Custos hospitalares |
| KNeighborsRegressor             | Média dos vizinhos                  | Simples, não assume função         | Lento em grandes bases           | Glicose de pacientes similares |
| SVR                             | Regressão por vetores de suporte    | Bom para dados não lineares        | Lento em bases grandes           | Progressão da diabetes |
| BayesianRidge                   | Linear Bayesiana                    | Dá intervalo de incerteza          | Mais complexo de interpretar     | Colesterol ± incerteza |
| HuberRegressor                  | Linear robusta                      | Resiste a outliers                 | Menos precisa sem outliers       | Pressão arterial com medidas erradas |
| TheilSenRegressor               | Linear robusta com medianas         | Resistente a ruídos                | Lento em bases grandes           | Peso fetal em bases pequenas |
| RANSACRegressor                 | Ajuste robusto a outliers           | Excelente com erros de medição     | Pode descartar bons pontos       | Crescimento tumoral com erros |


## 📘 Dataset Diabetes (Scikit-learn)
O dataset Diabetes é um conjunto de dados real usado para estudos de regressão. Ele contém informações de 442 pacientes, cada um descrito por 10 variáveis clínicas (idade, sexo, IMC, pressão arterial, exames de sangue etc.).

O que queremos prever é um número contínuo:
    ➡️ A progressão da doença após 1 ano.

> 👉 Ou seja, dado um paciente, tentamos prever o quanto a diabetes pode evoluir em 1 ano.
### Estrutura dos dados
  - X (features) → matriz com 442 linhas (pacientes) × 10 colunas (atributos).

  - y (target) → vetor com 442 valores, representando a progressão da doença.
    
### 💻 Carregando o dataset:
```
    from sklearn.datasets import load_diabetes
    diabetes = load_diabetes()

    # X = dados clínicos (features), y = progressão da doença (target)
    X = diabetes.data
    y = diabetes.target

    print("Formato de X:", X.shape)  # (442, 10)
    print("Formato de y:", y.shape)  # (442,)
```

## 🧹 Pré-processamento dos Dados
Antes de treinar, os dados precisam ser preparados:
> O scikit learn já traz os dados do dataset `diabetes` pré-processado, contudo  é recomendado reconferir, além de que, para outros datasets, esses passos serão de suma importância.

> Para mais informações sobre pré- processamento de dados volte a [Pré Processamento](https://github.com/GabryelleDart/How-To-train-and-evaluate-a-classification-model-using-scikit-learn/tree/main/Pr%C3%A9-Processamento) .

1. Inspeção inicial

    - Identificar tipos de dados, valores faltantes, outliers.

    - Ferramentas: df.info(), df.describe().

2. Tratar valores faltantes

    - Remover linhas/colunas incompletas.

    - Preencher com média/mediana (SimpleImputer).

3. Transformar variáveis categóricas

    - Ex.: “fuma = sim/não” → converter para 0 e 1.

    - Usar OneHotEncoder ou pd.get_dummies.

4. Escalonar variáveis

    - Padrão comum: StandardScaler ou MinMaxScaler.

5. Separar dados de treino e teste
```
    from sklearn.model_selection import train_test_split

    X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
# 80% dos pacientes → treino
# 20% dos pacientes → teste
```
### 📏 Treinando um modelo de regressão linear com Diabetes dataset
Aqui vamos usar Regressão Linear.
Ele tenta encontrar uma linha de tendência que melhor explica a relação entre as características dos pacientes e a gravidade da diabetes.
```
  from sklearn.linear_model import LinearRegression

  model = LinearRegression()

```
Agora o modelo irá aprender a relação entre as características dos pacientes (idade, pressão etc.) e a gravidade da doença.
```
  model.fit(X_train, y_train)

```
### 📏 Previsões
Com o modelo treinado, agora damos os dados de pacientes do teste (que ele nunca viu) e pedimos:
“Qual seria o valor da gravidade para este paciente?”
```
 y_pred = model.predict(X_test)

```
### 📈 Avaliação da Performance
Quando treinamos um modelo de machine learning (classificação ou regressão), não basta só treinar: precisamos saber se ele realmente funciona. É aqui que entra a avaliação da performance.

> Não adianta só prever, temos que medir o quão próximo o modelo ficou da realidade.

#### 🔹 O que significa avaliar a performance?
Avaliar a performance significa medir quão bem o modelo consegue fazer previsões corretas em dados que ele não viu durante o treino.

Exemplo :

- O modelo tenta prever o nível de glicose de pacientes.
- Se o modelo prever 140 mg/dL e o valor real for 145 mg/dL, o erro é 5 mg/dL.
- Avaliar o desempenho consiste em resumir esses erros para todos os pacientes e entender se o modelo é confiável.
  
> ⚠️ Lembre-se: treinar o modelo e avaliar nos mesmos dados é enganoso. O modelo pode simplesmente “decorar” os exemplos.

#### 🔹Por que precisamos avaliar?
Avaliar é essencial porque:

1. Medimos a qualidade do modelo → não adianta ter um modelo se ele não funciona na prática.

2. Evitamos overfitting → o modelo pode aprender “truques” dos dados de treino que não funcionam em novos pacientes.

3. Comparamos modelos → saber qual algoritmo é mais adequado para o problema.
4. Ajudamos na tomada de decisão → médicos, gestores ou sistemas precisam confiar nas previsões.

Exemplo:
> Um modelo que prevê diabetes errado 80% das vezes é inútil na prática clínica, mesmo que tenha sido treinado com muitos dados.

#### 🔹Métricas comuns para regressão:

- MAE (Mean Absolute Error) → erro médio em unidades reais.

- MSE (Mean Squared Error) → penaliza mais erros grandes.

- RMSE (Root Mean Squared Error) → interpretação fácil (mesma unidade do alvo).

- R² (Coeficiente de determinação) → quanto da variação do alvo o modelo explica (0 a 1).

```
  from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
  import numpy as np
  
  mae = mean_absolute_error(y_test, y_pred)
  rmse = np.sqrt(mean_squared_error(y_test, y_pred))
  r2 = r2_score(y_test, y_pred)
  
  print("MAE:", mae)
  print("RMSE:", rmse)
  print("R²:", r2)

```
- Se o R² for alto (perto de 1), significa que o modelo explica bem a gravidade da diabetes.

- Se for baixo (perto de 0), significa que as informações que temos não são suficientes para prever bem.

- Os erros (MAE, RMSE) mostram o quanto o modelo erra em média.
