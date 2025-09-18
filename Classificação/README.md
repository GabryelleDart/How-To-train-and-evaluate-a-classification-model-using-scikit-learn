# 📘 Classificação com Scikit-learn
Este **How to** apresenta uma introdução completa a modelos de classificação, explicando conceitos, aplicações, pré-processamento, avaliação de performance e implementação prática em Python com scikit-learn.
## 📌 O que é Classificação?
Classificação é uma técnica de aprendizado supervisionado utilizada quando a variável alvo (y) é categórica (categorias ou classes).
> 👉 Exemplo na área da saúde: diagnosticar se um paciente tem diabetes ou não tem diabetes a partir de variáveis como idade, peso e histórico médico.
### ⚡ Diferença entre Classificação e Regressão
- Classificação → prevê categorias (ex.: "Iris-setosa", "Iris-versicolor", "Iris-virginica").
- Regressão → prevê valores contínuos (ex.: "nível de glicose = 135 mg/dL").
### 🌸 Exemplos de problemas resolvidos com Classificação
- Diagnóstico de doenças (diabetes, câncer).

- Classificação de imagens médicas (tumor benigno ou maligno).

- Previsão de risco de readmissão hospitalar (alto, médio, baixo).

 - Identificação de espécies de plantas ou animais.
## 🔹 Principais Algoritmos de Classificação no Scikit-learn
### 1. Regressão Logística (LogisticRegression)

- 📖 Apesar do nome, é classificador. Modela a probabilidade de pertencer a uma classe.

- ✅ Bom para problemas lineares e interpretáveis.

- 🏥 Exemplo: prever se um paciente tem diabetes baseado em exames laboratoriais.

### 2. K-Nearest Neighbors (KNeighborsClassifier)

- 📖 Classifica novos pontos com base nas classes dos k vizinhos mais próximos.

- ✅ Simples, eficiente para datasets pequenos.

- 🏥 Exemplo: classificar tipo de célula comparando com células conhecidas.

### 3. Support Vector Classifier (SVC)

- 📖 Encontra o hiperplano que separa melhor as classes.

- ✅ Excelente para problemas lineares e não lineares usando kernel.

- 🏥 Exemplo: identificar pacientes de alto risco com base em múltiplos fatores.

### 4. Decision Tree Classifier (DecisionTreeClassifier)

- 📖 Divide os dados usando regras de decisão em árvore.

- ✅ Fácil de interpretar e visualizar.

- 🏥 Exemplo: decidir se um paciente deve receber um tratamento específico.

### 5. Random Forest Classifier (RandomForestClassifier)

- 📖 Conjunto de árvores de decisão que votam na classe final.

- ✅ Mais robusto que uma árvore individual.

- 🏥 Exemplo: prever diagnóstico de doença baseado em múltiplos exames.

### 6. Gradient Boosting Classifier (GradientBoostingClassifier)

- 📖 Cria árvores sequenciais, cada uma corrigindo os erros da anterior.

- ✅ Alta performance em dados tabulares.

- 🏥 Exemplo: classificar risco de complicações hospitalares.

### 7. AdaBoost Classifier (AdaBoostClassifier)

- 📖 Dá mais peso a exemplos onde o modelo anterior errou.

- ✅ Bom para lidar com ruído moderado.

- 🏥 Exemplo: prever readmissão hospitalar em casos inconsistentes.

### 8. Naive Bayes (GaussianNB)

- 📖 Baseado no teorema de Bayes, assume independência entre atributos.

- ✅ Simples, rápido e funciona bem com pequenas amostras.

- 🏥 Exemplo: classificar pacientes com base em sintomas.

| Modelo                     | Descrição                | Vantagens                | Desvantagens                           | Exemplo em Saúde          |
|-----------------------------|--------------------------|--------------------------|----------------------------------------|--------------------------|
| LogisticRegression         | Probabilidade de classe  | Interpretável            | Linear, não captura relações complexas | Diagnóstico de diabetes   |
| KNeighborsClassifier       | Vizinhos mais próximos   | Simples, não paramétrico | Lento com muitos dados                 | Classificação de células  |
| SVC                        | Hiperplano separador     | Não linear, robusto      | Pode ser lento em grandes bases        | Risco de complicações     |
| DecisionTreeClassifier     | Regras de decisão        | Fácil de interpretar     | Overfitting                            | Decisão de tratamento     |
| RandomForestClassifier     | Floresta de árvores      | Robusto, generaliza bem  | Menos interpretável                    | Diagnóstico complexo      |
| GradientBoostingClassifier | Árvores sequenciais      | Alta performance         | Mais lento que Random Forest           | Risco hospitalar          |
| AdaBoostClassifier         | Reponderação de exemplos | Lida bem com ruído       | Sensível a outliers extremos           | Readmissão hospitalar     |
| GaussianNB                 | Probabilidade com Bayes  | Simples e rápido         | Supõe independência                    | Classificação de sintomas |


## 🌸 Dataset Iris (Scikit-learn)
O dataset Iris é um clássico de classificação. Contém 150 flores de 3 espécies:

- **Iris-setosa**

- **Iris-versicolor**

- **Iris-virginica**

Cada flor possui 4 características:

- Sepal Length (cm)

- Sepal Width (cm)

- Petal Length (cm)

- Petal Width (cm)

> Objetivo: prever a espécie da flor com base nas medidas das pétalas e sépalas.

### 💻 Carregando o dataset
```
from sklearn.datasets import load_iris
iris = load_iris()

X = iris.data      # Features (4 colunas)
y = iris.target    # Classes (0=setosa, 1=versicolor, 2=virginica)

print("Formato de X:", X.shape)  # (150, 4)
print("Formato de y:", y.shape)  # (150,)
```
### 🧹 Pré-processamento dos Dados
Antes de treinar, os dados precisam ser preparados:
> O scikit learn já traz os dados do dataset `iris` pré-processado, contudo  é recomendado reconferir, além de que, para outros datasets, esses passos serão de suma importância.

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
### 📏 Treinando um modelo de classificação
O KNN (K-Nearest Neighbors) é um dos algoritmos mais simples de classificação. Ele funciona de maneira muito intuitiva:
> Imagine que você está em um jardim cheio de flores de diferentes espécies. Uma nova flor aparece e você quer descobrir a qual espécie ela pertence.
> O KNN olha para as K flores mais próximas dela (os vizinhos mais próximos) e vota qual espécie é mais comum entre esses vizinhos. A espécie mais frequente será a previsão do modelo.
```
    from sklearn.neighbors import KNeighborsClassifier
    
    
    knn = KNeighborsClassifier(n_neighbors=3)

```
> 🖐 **n_neighbors=3** significa que o modelo vai olhar para as **3 flores mais próximas** e escolher a espécie que aparecer mais vezes.

Treinar o modelo significa ensinar o KNN usando os dados de treino. Ele vai "memorizar" as posições das flores no espaço das características para que possa comparar com novas flores depois.
```
    knn.fit(X_train, y_train)

```
> ✅ Aqui, não há magia de cálculos complexos: o KNN apenas memoriza os exemplos de treino e suas classes.

> 📌 Diferente de regressão linear, ele não tenta desenhar uma linha, ele se baseia na proximidade.

### 📏 Previsões
Agora podemos dar flores do conjunto de teste e perguntar ao KNN: “Qual espécie você acha que essa flor é?”
```
    y_pred = knn.predict(X_test)
    print("Previsões do KNN:", y_pred)

```
> 💡 Cada número na lista y_pred representa a classe prevista:

> 0 → Iris-setosa, 1 → Iris-versicolor, 2 → Iris-virginica

### 📈 Avaliação da Performance
Para classificação, usamos métricas diferentes da regressão:
- **Acurácia** → % de previsões corretas
- **Matriz de Confusão** → mostra acertos e erros por classe
- **Precision, Recall e F1-score** → medidas detalhadas por classe
```
    from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
    
    accuracy = accuracy_score(y_test, y_pred)
    print("Acurácia:", accuracy)
    
    cm = confusion_matrix(y_test, y_pred)
    print("Matriz de Confusão:\n", cm)
    
    report = classification_report(y_test, y_pred, target_names=iris.target_names)
    print("Relatório de Classificação:\n", report)

```
> 🔹 Interpretando a matriz de confusão:
> - Cada linha representa a classe real.
> - Cada coluna representa a classe prevista.
> - O ideal é que todos os números estejam na diagonal principal (acertos).
      
> 🔹 Exemplo de interpretação:
> - Se a matriz mostra que algumas versicolor foram classificadas como virginica, isso indica que o modelo confundiu essas duas espécies.
> - A acurácia geral mostra quanto ele acerta em porcentagem.

### 📝 Dicas Importantes para Classificação
- Sempre verifique o balanceamento das classes; classes desbalanceadas podem prejudicar o modelo.
- Use cross-validation para avaliar robustez do modelo.
- Experimente diferentes algoritmos e compare acurácia e F1-score.
- Para problemas reais, trate valores faltantes, outliers e escalonamento cuidadosamente.
- Para modelos complexos (Random Forest, Boosting), use feature importance para interpretar os fatores mais relevantes.
