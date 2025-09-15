# How-To-train-and-evaluate-a-classification-model-using-scikit-learn.

## Sobre o Projeto:
## ⚙️ Estrutura do Projeto
### Pré-requisitos:
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

## O que são Modelos de Classificação?
Um modelo de classificação é um tipo de algoritmo de Machine Learning que tem como objetivo prever uma categoria (classe) a partir de dados de entrada.

👉 Exemplos do dia a dia:

- Prever se um e-mail é spam ou não spam.
- Diagnosticar se um paciente tem doença X ou não.
- Reconhecer uma imagem como gato ou cachorro.

> Ou seja, ao invés de prever um número (como na regressão), a classificação lida com rótulos/categorias.

### Para que servem ?
Os modelos de classificação servem para tomar decisões automáticas baseadas em dados, atribuindo categorias a novos exemplos.

Em outras palavras: eles ajudam a responder perguntas do tipo *“isso é A ou B?” ou “isso pertence a qual grupo?”*.

📌 Exemplos práticos
- Diagnóstico de doenças → prever se um paciente tem diabetes (sim/não) com base em exames de sangue.
- Detecção de câncer → classificar uma imagem de raio-X ou mamografia em câncer maligno ou benigno.
- Covid-19 → identificar, a partir de sintomas e exames, se o paciente está infectado ou não infectado.
- Doenças cardíacas → prever risco de infarto (alto risco, médio risco, baixo risco).
- Exames laboratoriais → classificar resultados em normal ou alterado.
- Saúde mental → analisar questionários e classificar se a pessoa apresenta sinais de depressão ou não.
- Triagem hospitalar → categorizar pacientes em emergência, urgência ou não urgente.

---
&nbsp;<br>
&nbsp;<br>
&nbsp;<br>
&nbsp;<br>


## 1° task: Preparar e pré-processar um dataset pequeno para experimentação
> A tarefa consiste em pegar um conjunto de dados pequeno e deixá-lo "arrumado" para que o computador consiga aprender com ele.

O pré- processamento consistiria em:

- **Organizar os dados** → tirar duplicados, completar onde falta informação ou decidir jogar fora dados incompletos.


- **Traduzir informações em números** → o computador entende melhor números do que palavras.
    - Ex: transformar “Masculino/Feminino” em `0` e `1`.
- **Colocar tudo na mesma escala** → se uma coluna tem valores como “idade = 25” e outra “salário = 8000”, o modelo pode dar mais importância pro salário só porque é um número maior. A normalização serve pra “equilibrar” isso.
- **Separar treino e teste** → é como estudar para uma prova: você aprende com um pedaço do material (treino) e depois se testa com outro pedaço (teste) para ver se realmente aprendeu.

Então, pré-processar um dataset pequeno serve para limpar, organizar e transformar os dados em um formato que o modelo de inteligência artificial consiga entender e aprender de verdade.
Assim, quando você for treinar o modelo (tipo uma árvore de decisão, uma regressão ou uma rede simples), ele não vai se confundir com valores faltando, categorias em texto ou números em escalas muito diferentes.
### Pré- requisitos para essa tarefa:
- [x]  Python instalado

```jsx
python --version

```

- [x]  Bibliotecas :
- `numpy` (cálculos numéricos)
- `pandas` (manipulação de dados)
- `matplotlib` / `seaborn` (visualização)
- `scikit-learn` (modelos de machine learning)

```jsx
python -m pip show scikit-learn
//caso nao apareça baixe usando 
conda install scikit-learn
//ou 
pip install scikit-learn

```

- [x]  Editor para rodar os Códigos ( Jupiter Notebook)
- [x]  Conhecimentos básicos de Python (mínimo necessário)
### Realizando a tarefa:
#### ✅ Forma 1 — Usando um dataset nativo do Scikit-learn

O Scikit-learn já traz datasets pequenos para prática (Iris, Breast Cancer, Digits, Wine...).

No arquivo   tem a demostração da preparação e pré-processamento do Dataset Iris `(conjunto de dados que  consiste em 50 amostras de cada uma das três espécies de Iris ( Iris setosa, Iris virginica e Iris versicolor)`.
<img width="1897" height="902" alt="image" src="https://github.com/user-attachments/assets/23768f75-547a-45ea-b4b6-7425c96beca3" />

#### ✅ Forma 2 — Usando um dataset de CSV (ex.: baixado da internet)
> ##### 🔹  Onde conseguir datasets pequenos
> Aqui alguns lugares ótimos:
> 1. **Nativos no Scikit-learn** → Iris, Wine, Breast Cancer, Digits.
> 2. **Seaborn** → vem com vários datasets prontos (`sns.load_dataset("tips")`).
> 3. **Kaggle** → plataforma com milhares de datasets (precisa conta gratuita). `https://www.kaggle.com/datasets`
> 4. **UCI Machine Learning Repository** → datasets clássicos. ` https://archive.ics.uci.edu/ml/index.php`
> 5. **GitHub** → vários repositórios com datasets em CSV.

#### ✅ Forma 3 — Criando um dataset aleatório (útil para teste rápido)
Você pode gerar dados fictícios para treinar um modelo.

---
&nbsp;<br>
&nbsp;<br>
&nbsp;<br>
&nbsp;<br> 

## 2° task: Avaliar rapidamente o desempenho do modelo treinado
### 🔹 O que significa avaliar o desempenho?
### 🔹Por que precisamos avaliar?
### 🔹 Para que serve?
### 🔹 Formas de avaliar modelos de classificação:
### 🔹 O que define se foi bem avaliado ou não?
