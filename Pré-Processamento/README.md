# Preparar e pré-processar um dataset pequeno para experimentação
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
#### 🚦 Passo a passo:
1. Carregar o dataset (ex.: Iris, CSV etc.).
```
iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df['target'] = iris.target
```
2. Inspecionar os dados (ver tamanho, tipos, valores faltantes, classes).
```
# tamanho (linhas, colunas)
print("Shape:", df.shape)

# primeiras linhas
print(df.head())

# tipos de dados
print(df.info())

# valores faltantes
print("Valores faltantes por coluna:\n", df.isnull().sum())

# distribuição do target
print("Classes disponíveis:", df['target'].unique())
print("Contagem por classe:\n", df['target'].value_counts())
```
3. Limpar/transformar os dados (tratar valores nulos, remover colunas inúteis, ajustar formatos).
```
# exemplo: remover colunas inúteis (não necessário no Iris)
# df = df.drop(columns=['coluna_irrelevante'])

# exemplo: tratar valores nulos
df = df.fillna(df.mean(numeric_only=True))  # preenche nulos com média
```
4. Dividir em features (X) e target (y) (o que entra no modelo e o que queremos prever).
```
# X = dados de entrada (features)
X = df.drop('target', axis=1)

# y = o que queremos prever (rótulo / classe)
y = df['target']

print("Features (X):")
print(X.head())
print("\nTarget (y):")
print(y.head())
```
5. Separar treino e teste (train_test_split) — muito importante para garantir que o modelo seja avaliado de forma justa.
```
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print("Tamanho treino:", X_train.shape)
print("Tamanho teste:", X_test.shape)
```
6. Aplicar transformações:
- 6.1 Imputação de valores faltantes
```
from sklearn.impute import SimpleImputer

imputer = SimpleImputer(strategy='mean')
X_train_imputed = imputer.fit_transform(X_train)
X_test_imputed = imputer.transform(X_test)

```
- 6.2 Normalização / Padronização
```
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_imputed)
X_test_scaled = scaler.transform(X_test_imputed)
```
- 6.3 Codificação de variáveis categóricas (se existissem)
```
from sklearn.preprocessing import OneHotEncoder

encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
categorical_data = [['vermelho'], ['azul'], ['verde']]
encoded = encoder.fit_transform(categorical_data)

print(encoded)  # vira números binários (one-hot)

```

#### ✅ Forma 1 — Usando um dataset nativo do Scikit-learn

O Scikit-learn já traz datasets pequenos para prática (Iris, Breast Cancer, Digits, Wine...).

No arquivo   tem a demostração da preparação e pré-processamento do Dataset Iris `(conjunto de dados que  consiste em 50 amostras de cada uma das três espécies de Iris ( Iris setosa, Iris virginica e Iris versicolor)`.
<img width="1898" height="823" alt="image" src="https://github.com/user-attachments/assets/a0407a65-35bf-4bdf-a7a7-0e9a508dd772" />


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
