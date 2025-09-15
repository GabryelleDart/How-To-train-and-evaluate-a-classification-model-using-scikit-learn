# How-To-train-and-evaluate-a-classification-model-using-scikit-learn.

## Sobre o Projeto:
## ⚙️ Estrutura do Projeto
### Pré-requisitos:
## O que é o Skitit- learn?
## O que são Modelos de Classificação?
### Para que servem ?
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
