# A Pesquisa Mensal de Serviços (PMS)

## 1. Pacotes 
---
- **Pacotes para o python**
```python
import requests
import pandas as pd 
import locale
import datetime 
from statsmodels.tsa.stattools import adfuller,acf,pacf
from statsmodels.tsa.arima.model import ARIMA
import numpy as np 
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import *
from scipy import stats
```
- **request** $\rightarrow$ Permite que você envie solicitações HTTP / 1.1 com extrema facilidade. 
- **pandas** $\rightarrow$ É uma biblioteca de software criada para a linguagem Python para manipulação e análise de dados
- **locale** $\rightarrow$ O módulo de localidade abre o acesso ao banco de dados de localidade POSIX e à funcionalidade.
- **datetime** $\rightarrow$ O módulo datetime fornece classes para manipulação de datas e horas.
- **statsmodel** $\rightarrow$ É um pacote Python que permite aos usuários explorar dados, estimar modelos estatísticos e executar testes estatísticos
- **numpy** $\rightarrow$ É um pacote para a linguagem Python que suporta arrays e matrizes multidimensionais, possuindo uma larga coleção de funções matemáticas para trabalhar com estas estruturas
- **tensorflow** $\rightarrow$ É uma biblioteca de código aberto para aprendizado de máquina aplicável a uma ampla variedade de tarefas 
- **sklearn** $\rightarrow$ É uma biblioteca de aprendizado de máquina de código aberto para a linguagem de programação Python
- **scipy** $\rightarrow$ É uma biblioteca Open Source em linguagem Python que foi feita para matemáticos, cientistas e engenheiros. Também tem o nome de uma popular conferência de programação científica com Python. A sua biblioteca central é NumPy que fornece uma manipulação conveniente e rápida de um array N-dimensional.

---
- **Pacotes para o R**
```R
library(jsonlite)
library(dplyr)
library(urca)
library(stats)
library(forecast)
```
- **Jsonlite** $\rightarrow$ Um analisador e gerador JSON razoavelmente rápido, otimizado para dados estatísticos e para a web. 
- **dplyr** $\rightarrow$ Um dos pacotes principais do tidyverse na linguagem de programação R, dplyr é principalmente um conjunto de funções projetado para permitir a manipulação de dataframe de uma forma intuitiva e amigável.
- **urca** $\rightarrow$ Unit root and cointegration tests encountered in applied econometric analysis are implemented.
- **forecast** $\rightarrow$ Métodos e ferramentas para exibir e analisar previsões de séries temporais univariadas, incluindo suavização exponencial por meio de modelos de espaço de estado e modelagem ARIMA automática.
## 2. Coletando os dados 

### 2.1 Acessando os dados 

Para acesso dos dados Pesquisa Mensal de Serviços (PMS), existem diversas formas, como o download arquivos em formato excel ou arquivos de CSV. Porem, mesmo a simplicidade da utilização de arquivos em formatos mais tradicionais, quando aplicado em trabalhos com grande volume de dados, existe uma chance da criação de residuos futuros. 

Portanto, para evitar a geração de residuos no disco rigido, os dados serão obtidos diretamente da API disponibilizada pelo SIDRA/IBGE. 

Para o acesso dos dados da Pesquisa Mensal de Serviços através da API do SIDRA/IBGE, podem ser utilizados os seguintes codigos nas linguagens python e R:

- **Python**
```python 
# Acesando o indice de serviço com a api do sidra/IBGE
url = requests.get('https://apisidra.ibge.gov.br/values/t/6443/n1/all/v/8676/p/all/c11046/40311/c12355/107071/d/v8676%201')

# Criando arquivo json com o indice 
json = url.json()

# Criando um Dataframe 
data = pd.json_normalize(json)
```
- **R**
```R
# Acesando o indice de serviço com a api do sidra/IBGE
dados = fromJSON('https://apisidra.ibge.gov.br/values/t/6443/n1/all/v/8676/p/all/c11046/40311/c12355/107071/d/v8676%201')
```
Portanto a estrutura dos dados é a seguinte
|    | NC                         | NN                | MC                         | MN                | V     | D1C             | D1N    | D2C               | D2N                                   | D3C          | D3N            | D4C                      | D4N                         | D5C                             | D5N                    |
|---:|:---------------------------|:------------------|:---------------------------|:------------------|:------|:----------------|:-------|:------------------|:--------------------------------------|:-------------|:---------------|:-------------------------|:----------------------------|:--------------------------------|:-----------------------|
|  0 | Nível Territorial (Código) | Nível Territorial | Unidade de Medida (Código) | Unidade de Medida | Valor | Brasil (Código) | Brasil | Variável (Código) | Variável                              | Mês (Código) | Mês            | Tipos de índice (Código) | Tipos de índice             | Atividades de serviços (Código) | Atividades de serviços |
|  1 | 1                          | Brasil            | 30                         | Número-índice     | 71.3  | 1               | Brasil | 8676              | Índice de receita nominal de serviços | 201101       | janeiro 2011   | 40311                    | Índice base fixa (2014=100) | 107071                          | Total                  |
|  2 | 1                          | Brasil            | 30                         | Número-índice     | 71.3  | 1               | Brasil | 8676              | Índice de receita nominal de serviços | 201102       | fevereiro 2011 | 40311                    | Índice base fixa (2014=100) | 107071                          | Total                  |
|  3 | 1                          | Brasil            | 30                         | Número-índice     | 76.2  | 1               | Brasil | 8676              | Índice de receita nominal de serviços | 201103       | março 2011     | 40311                    | Índice base fixa (2014=100) | 107071                          | Total                  |
|  4 | 1                          | Brasil            | 30                         | Número-índice     | 75.5  | 1               | Brasil | 8676              | Índice de receita nominal de serviços | 201104       | abril 2011     | 40311                    | Índice base fixa (2014=100) | 107071                          | Total                  |

## 3. Limpando os dados

A maioria dos algoritimos para previsão não podem funcioonar com as residuos, dados com má formatação, caracteristicas faltantes, entre outros problemas existentes nos dados. Portanto nas proximas subseções serão desenvolvida algumas funções e rotinas para a correção de possiveis problemas dos nossos dados


### 3.1 Renomeando as colunas 
Como visto nos dados acima, existe duas inconsistências na nomeação das colunas, a primeira é a utilização de alguns códigos de referência no lugar de colunas, e a nomeação das colunas está na primeira linha dos dataframe. Portanto, será necessaria a renomeação das colunas, e em seguida a remoção da primeira linha do Dataframe.

- **Python** 

```python
# Renomeando os nomes das colunas
data=data.rename(columns=data.iloc[0]).drop([0])
```

- **R**
```R
# Renomeando as colunas
names(dados)<-dados[1,]
# Removendo as linhas contendo nomes das colunas 
dados <- dados[-1,]
```

Portanto a estrutura dos dados é a seguinte
|  0 | Nível Territorial (Código) | Nível Territorial | Unidade de Medida (Código) | Unidade de Medida | Valor | Brasil (Código) | Brasil | Variável (Código) | Variável                              | Mês (Código) | Mês            | Tipos de índice (Código) | Tipos de índice             | Atividades de serviços (Código) | Atividades de serviços |
|---:|:---------------------------|:------------------|:---------------------------|:------------------|:------|:----------------|:-------|:------------------|:--------------------------------------|:-------------|:---------------|:-------------------------|:----------------------------|:--------------------------------|:-----------------------|
|  1 | 1                          | Brasil            | 30                         | Número-índice     | 71.3  | 1               | Brasil | 8676              | Índice de receita nominal de serviços | 201101       | janeiro 2011   | 40311                    | Índice base fixa (2014=100) | 107071                          | Total                  |

### 3.2 Filtrando as colunas
Agora podemos selecionar linhas de DataFrame que contenha ou não o uma coluna especifica. Essa operação auxilia na otimização da memoria RAM do computador e facilita na manipulação de operações com o DataFrame. 

Portanto as colunas utilizadas serão as seguintes:

- **Nível Territorial** $\rightarrow$ Variavel que indica a unidade geografica trabalhada 
- **Valor** $\rightarrow$ Variavel referênte ao valor do índice de serviços 
- **Mês** $\rightarrow$ Variavel referênte aos meses e anos  
- **Atividades de serviços** $\rightarrow$ referênte a atividade do serviço 

Portanto, o código para a filtragem dos dados é o seguinte 

- **Python**
```python
# Limpando as colunas 
data = data.filter(items=['Nível Territorial','Valor','Mês','Atividades de serviços'],axis=1)
```
- **R**
```R
# Limpando as colunas 
dados <- dados[,c('Nível Territorial','Valor','Mês','Atividades de serviços')]
```

### 3.3 Ajustando as unidades de tempo 

Para a criação de series temporais consistentes, é necessário o estabelecimento de datas em um formato bem estruturado. Para a formatação dos dados, será aplicada os seguinte processo 

- **Python** 
```python 
# Corrigindo a localização do computador
locale.setlocale(locale.LC_ALL, 'pt_BR.utf8')

# Convertendo a coluna 'Mês' para formato de datetime 
date = []
for mes in data['Mês']:
    date.append(datetime.datetime.strptime(mes,'%B %Y').strftime('%m.%y'))
data['Mês']=date 
```

### 3.4 Criando uma serie temporal 

Como é defininido no manual do software estatistico Minitab (2019), Uma série temporal é uma sequência de observações em intervalos de tempo regularmente espaçados. Logo para as futuras operações com os dados temporais, devemos converter a coluna que contem os valores do indice em uma serie temporal, como apresentado a seguir 

- **Python** 
```python 
# Criando uma serie temporal 
serie = pd.Series(data['Valor'].astype('float64').to_list(),index=data['Mês'])
```
- **R**

```R
# Criando uma serie temporal
serie = ts(data=as.numeric(dados$Valor),start=c(2011,1), end=c(2021,5),frequency=12)
```

![](https://i.imgur.com/fkF7DVb.png)

## 4 Modelando as series temporais 
### 4.1 Testes
#### 4.1.1 Teste de Estacionaridade 

Como aponta Freitas (2012), existem diversos testes de raiz unitária, entre eles 

- Augmented Dickey-Fuller
- Phillips-Perron
- Kwiatkowski-Phillips-Schmidt-Shin 

Na maioria dos testes a hipótese nula é de que a série tenha raiz unitária, e portanto não seja estacionária, logo: 

$$
\begin{aligned}
\begin{split}
H_0: & \textrm{tem raiz unitária (não é estacionária)}\\
H_1: & \textrm{não tem raiz unitária (é estacionária)}
\end{split}
\end{aligned}
$$

Como apresentado em Enders (2014), Dickey e Fuller (1979) consideram três equações de regressões diferentes que podem ser usadas para testar a presença de uma raiz unitária

$$
\begin{aligned}
\begin{split}
\Delta y_t = \gamma y_{t -1} +\epsilon_t z  \;\;\;\;\; (1)\\
\Delta y_t = a_0 + \gamma y_t + \epsilon_t \;\;\;\; (2)\\
\Delta y_t = a_0 + \gamma y_t + +a_2t +\epsilon_t \; \; \;(3) 
\end{split}
\end{aligned}
$$

- (1) $\rightarrow$ Modelo de passeio aleatório puro
- (2) $\rightarrow$ Modelo de passeio aleatório com *drift*
- (3) $\rightarrow$ Modelo de passeio aleatório com *drift* e componente de tendência 

Portanto podemos testar os diferentes modelos da seguinte maneira: 

- **Python**
```python
# Test Augmented Dickey-Fuller - com constante e com tendência 
adf_1 = {'adf':adfuller(serie,regression="ct")[0],'pvalue':adfuller(serie,regression="ct")[1],'Nº de lags': adfuller(serie,regression="ct")[2],'Critical_value':adfuller(serie, regression="ct")[4]['5%']}

# Test Augmented Dickey-Fuller - Com constante
adf_2 = {'adf':adfuller(serie)[0],'pvalue':adfuller(serie)[1],'Nº de lags': adfuller(serie)[2],'Critical_value':adfuller(serie)[4]['5%']}

# Test Augmented Dickey-Fuller - Sem constante e Sem tendência 
adf_3 = {'adf':adfuller(serie,regression="nc")[0],'pvalue':adfuller(serie,regression="nc")[1],'Nº de lags': adfuller(serie,regression="nc")[2],'Critical_value':adfuller(serie, regression="nc")[4]['5%']}
```
- **R**
```R
# Test Augmented -Fuller - Com constante e com tendência 
adf_1 = ur.df(as.numeric(serie),type="trend")

# Test Augmented Dickey-Fuller - Com constante
adf_2 = ur.df(as.numeric(serie),type="drift")

# Test Augmented Dickey-Fuller - Com constante
adf_3 = ur.df(as.numeric(serie),type="none")
```
|    |      adf |   pvalue |   Nº de lags |   Critical_value |
|---:|---------:|---------:|-------------:|-----------------:|
|  Com caonstante e Tendência | -2.94387 | 0.148457 |           12 |         -3.45044 |
|  Com constante e sem tendência | -2.09686 | 0.245778 |           13 |         -2.88795 |
|  Sem constante e sem tendência |  1.25676 | 0.946364 |           12 |         -1.94364 |'

#### 4.1.2 Teste de Sazonalidade

Como define Brownlee (2016), uma variação temporal é um ciclo que se repete ao longo do tempo, com uma frequencia que pode ser mensal ou anual. Esse ciclo de repetição pode obscurecer a sua modelagem, e por sua vez, pode fornecer problemas para a construção de modelos preditivos

Uma alternativa para detecção de sazonalidade na serie temporal é a aplicação do teste de Friedman (1937), na qual é uma alternativa à analise de variância quando não é possivel assumir que os dados provem de uma população com distribuição normal. No contexto de series temporais, o Teste de Friedman divide a érie em blocos de períodos e calcula o posto de cada observção em cada bloco. Logo, se houver sazonalidade, há evidências de que pelo menos em dois meses existe diferença significativa em suas médias ao longo dos anos. 

Logo a hipótese desse teste é: 

$$
\begin{aligned}
\begin{split}
H_0: & T_1 = T_2 = T_3= ...= T_k \\
H_1: & T_1 \neq T_2 \neq T_3\neq ...\neq T_k
\end{split}
\end{aligned}
$$

Portanto podemos aplicar o teste de friedman da seguinte forma:

- **Python**
```python 
# Função de destrinchamento das series
def Sazonality(serie,type):
    if type == "trimestre":
        size = list(range(0,len(serie),3))
        data=[]
        for i in range(0,len(size)-1):
            data.append(serie[size[i]:size[i+1]])

    elif type == "quadrimestre":
        size = list(range(0,len(serie),4))
        data=[]
        for i in range(0,len(size)-1):
            data.append(serie[size[i]:size[i+1]])
        
    elif type == "semestre":
        size = list(range(0,len(serie),6))
        data=[]
        for i in range(0,len(size)-1):
            data.append(serie[size[i]:size[i+1]])

    elif type == "anual":
        size = list(range(0,len(serie),12))
        data=[]
        for i in range(0,len(size)-1):
            data.append(serie[size[i]:size[i+1]])

    return data
```

```python
# Sazonalidade - Quadrimestre
data=Sazonality(serie,"quadrimestre")
saz_quart=stats.friedmanchisquare(data[0],data[1],data[2],data[3],data[4],data[5],data[6])

# Sazonalidade - Trimestral
data=Sazonality(serie,"trimestre")
saz_trim = stats.friedmanchisquare(data[1],data[2],data[3],data[4],data[5],data[6],data[7])

# Sazonalidade - Semestral
data=Sazonality(serie,"semestre")
saz_semes= stats.friedmanchisquare(data[7],data[6],data[5],data[4],data[3],data[2],data[1])

# Sazonalidade - Anual
data=Sazonality(serie,"anual")
saz_anual= stats.friedmanchisquare(data[1],data[2],data[3],data[4],data[5],data[6],data[7])
```


| Sazonalidade | Valor | p-valor |
| -------- | -------- | -------- |
| Trimestre    | 16.14285   |  0.01300 |
| Quadrimestre    | 22.17857    | 0.00112   |
| Semestre    |33.35714  |8.95e-06   |
| Anual   | 65.64285 | 3.188e-12   |


### 4.1.3 Correlograma 
Na análise de dados, um correlograma é um gráfico de estatísticas de correlação. O correlograma é uma ferramenta comumente usada para verificar a aleatoriedade em um conjunto de dados. Se aleatórias, as autocorrelações devem ser próximas a zero para todas as separações de intervalo de tempo. Se não for aleatório, então uma ou mais das autocorrelações serão significativamente diferentes de zero

- **Autocorrelação** $\rightarrow$ Como apresenta Holmes, Scheuerell e Ward (2021), A função de autocorrelação é a correlação de uma variável consigo mesma em diferentes intervalos de tempo.
- **Autocorrelação Parcial**  $\rightarrow$ Tambem como apresenta Holmes, Scheuerell e Ward (2021), A função de autocorrelação parcial (PACF) mede a correlação linear de uma série $\{x_t \}$ e uma versão defasada de si mesma $\{ x_t + k \}$ com a dependência linear de $\{ x_{t − 1}, x_{t − 2},…, x_{t− (k − 1 )} \}$ removido
 
- **Python**
```python 
# Autocorrelação
autocorrelação = acf(serie)

# Autocorrelação Parcial 
autocorrelação_parcial = pacf(serie)
```

- **R**
```R
# 3.3.1 Autocorrelação 
autocorrelação <- acf(as.numeric(serie),lag.max=36,plot=FALSE)

# 3.3.2 Autocorrelação Parcial 
autocorrelação_parcial <- pacf(as.numeric(serie),lag.max=36,plot=FALSE)
```
- **Função de autocorrelação**
![](https://i.imgur.com/eK9lUuY.png)

- **Função de autocorrelação parcial**
![](https://i.imgur.com/U0vv9m9.png)

### 4.2 Diferenciando as series
Como afirma Enders (2014), uma série conter uma tendência estocástica não reverterá para um nível de longo prazo.A tendência presente em uma serie temporal pode ter componentes determinísticos e estocásticos. Portanto um dos métodos usuais para eliminar a tendência é a diferenciação da serie temporal. 

Portanto para a diferenciação da serie pode-se utilizar os seguintes códigos : 

```python
# Criando a serie diferenciada 
serie_diff = serie.diff().dropna()
```

```R
# Criando a serie diferenciada 
serie_diff = diff(as.numeric(serie))
```

![](https://i.imgur.com/QEacPQK.png)


#### 4.2.1 Teste de estacionariedade 

```python 
#  Test Augmented -Fuller - Com constante e com tendência 
adf_1 = {'adf':adfuller(serie_diff,regression="ct")[0],'pvalue':adfuller(serie_diff,regression="ct")[1],'Nº de lags': adfuller(serie_diff,regression="ct")[2],'Critical_value':adfuller(serie_diff, regression="ct")[4]['5%']}

# Test Augmented Dickey-Fuller - Com constante
adf_2 = {'adf':adfuller(serie_diff)[0],'pvalue':adfuller(serie_diff)[1],'Nº de lags': adfuller(serie_diff)[2],'Critical_value':adfuller(serie_diff)[4]['5%']}

# Test Augmented Dickey-Fuller - Sem constante e Sem tendência 
adf_3 = {'adf':adfuller(serie_diff,regression="nc")[0],'pvalue':adfuller(serie_diff,regression="nc")[1],'Nº de lags': adfuller(serie_diff,regression="nc")[2],'Critical_value':adfuller(serie_diff, regression="nc")[4]['5%']}
```

```R
# Test Augmented Dickey-Fuller - Sem constante e sem tendência 
adf_1 = ur.df(serie_diff,type="trend")

# Test Augmented -Fuller - Com constante e com tendência 
adf_2 = ur.df(serie_diff,type="drift")

# Test Augmented Dickey-Fuller - Sem constante e Sem tendência 
adf_3 = ur.df(serie_diff,type="none")

```

|    |      adf |    pvalue |   Nº de lags |   Critical_value |
|---:|---------:|----------:|-------------:|-----------------:|
|  Com constante e tendência | -2.2136  | 0.482198  |           11 |         -3.45044 |
|  Com constante e sem tendência | -2.46914 | 0.123164  |           11 |         -2.88771 |
|  Sem tendência e sem constante| -2.06031 | 0.0377089 |           11 |         -1.94364 |

```python
# Autocorrelação
autocorrelação = acf(serie_diff)

# Autocorrelação Parcial 
autocorrelação_parcial = pacf(serie_diff)
```


#### 4.2.2 Correlograma 

```R
# Autocorrelação
autocorrelação = acf(serie_diff)

# Autocorrelação Parcial 
autocorrelação_parcial = pacf(serie_diff)
```
- **Função de autocorrelação**
![](https://i.imgur.com/fEVaXqa.png)

- **Função de autocorrelação parcial**
![](https://i.imgur.com/SrYTLTB.png)

## 5. Modelando a Serie Temporal 
Os modelos ARMA (p, q) têm uma história rica na literatura de séries temporais, mas não são tão comuns na ecologia como os modelos AR (p) simples, como discute Holmes, Scheuerell e Ward (2021). 
POdemos escrever os modelos ARMA(p,q) como uma combinação de modelos AR(p) e modelos MA(q) models, que é representada por 

$$
\begin{aligned}
\begin{split}
x_t = \phi_1x_{t − 1} + \phi_2 x_{t−2}+⋯+\phi_p x_{t−p}+ \epsilon_t+\theta \epsilon_{t−1}+\theta_2 \epsilon_{t−2}+⋯+\theta_q x_{t−q}
\end{split}
\end{aligned}
$$

### 5.1 Descobrindo o melhor modelo - ARIMA
A seleção de um modelo que se ajuste sofre com *trade off* entre o ajuste de dados e a perda de de graus de liberdades na estimação. Para o auxilio da determinação do melhor modelagem dos dados, existem vários modelos critérios de seleção que compensam uma redução na soma dos quadrados dos resíduos por um modelo mais parcimonioso. Segundo Enders (2014), os dois critérios de seleção de modelos mais comumente usados são o Akaike Information Criterion (AIC) e o Bayesian information criterion (BIC).

- **Akaike Information Criterion (AIC)** $\rightarrow$ Foi formulado pelo estatístico japonês Hirotugu Akaike, sendo uma métrica que mensura a qualidade de um modelo estatístico visando também a sua simplicidade. Fornece, portanto, uma métrica para comparação e seleção de modelos, em que menores valores de AIC representam uma maior qualidade e simplicidade, segundo este critério. 
$$
\begin{aligned}
\begin{split}
AIC = -\dfrac{2 ln(L)}{T} + \dfrac{2N}{T}
\end{split}
\end{aligned}
$$

- **Bayesian information criterion (BIC)** $\rightarrow$ É um critério para seleção de modelo entre um conjunto finito de modelos; o modelo com o BIC mais baixo é o preferido. Baseia-se, em parte, na função de verossimilhança e está intimamente relacionado ao critério de informação de Akaike (AIC).
$$
\begin{aligned}
\begin{split}
BIC = -\dfrac{2 ln(L)}{T} + n \dfrac{ln(T)}{T}
\end{split}
\end{aligned}
$$

Portanto para encontrar o melhor modelo pode-se utilizar os seguintes códigos : 

- **Python**
```python
# Função de detecção de melhores modelos 
def best_arima(serie,order,criterion):
    Modelo,Bic,Aic,Aicc = [],[],[],[]
    for ar in range(order[0]+1):
        for i in range(order[1]+1):
            for ma in range(order[2]+1):
                    print(serie)
                    arima = ARIMA(serie,order=(ar,i,ma))
                    model = arima.fit()
                    print(model.summary())
                    Modelo.append(str(ar)+","+str(i)+","+str(ma))
                    Bic.append(model.bic)
                    Aic.append(model.aic)
                    Aicc.append(model.aicc)
    dic = {'Modelo':Modelo,'Bic':Bic,'Aic':Aic,'Aicc':Aicc}
    data = pd.DataFrame(dic)
    return data.sort_values(by=[criterion],ascending=True)

```

```python
# Melhor modelo bic
modelo_bic = best_arima(serie,order=[4,1,4],criterion='Bic')
arima_bic = ARIMA(serie,order=(3,1,3))
model_bic = arima_bic.fit()
model_bic.summary()

# Melhor modelo aic
modelo_aic = best_arima(serie,order=[4,1,4],criterion='Aic')
arima_aic = ARIMA(serie,order=(3,1,3))
model_aic = arima_aic.fit()
model_aic.summary()

# Melhor modelo aicc
modelo_aicc = best_arima(serie,order=[4,1,4],criterion='Aicc')
arima_aicc = ARIMA(serie,order=[3,1,3])
model_aicc = arima_aicc.fit()
model_aicc.summary()
```

- **R**
```R
# Melhor modelo bic
modelo_bic=auto.arima(as.numeric(serie),max.p=4,max.q=4,max.d=1,ic=c("bic"))
summary(modelo_bic)

# Melhor modelo aic 
modelo_aic=auto.arima(as.numeric(serie),max.p=4,max.q=4,max.d=1,ic=c("aic"))
summary(modelo_aic)

# Melhor modelo aicc 
modelo_aicc=auto.arima(as.numeric(serie),max.p=4,max.q=4,max.d=1,ic=c("aicc"))
summary(modelo_aicc)
```
- **Modelo Aicc**

| Modelo   |    Bic |     Aic |    Aicc |
|:---------|-------:|--------:|--------:|
| 3,1,3    | 750.64 | 730.898 | 731.863 |

-  **Modelo Aic** 

| Modelo   |    Bic |     Aic |    Aicc |
|:---------|-------:|--------:|--------:|
| 3,1,3    | 750.64 | 730.898 | 731.863 |

- **Modelo Bic**

| Modelo   |    Bic |     Aic |    Aicc |
|:---------|-------:|--------:|--------:|
| 3,1,3    | 750.64 | 730.898 | 731.863 |


### 5.2 Descobrindo o melhor modelo - SARIMA
Como define Graves (2020), os modelos SARIMA são modelos ARIMA com um componente sazonal. Pela fórmula SARIMA (p, d, q) x (P, D, Q, s).
Portanto podemos estimar o modelo sarima da seguinte forma

```python
# Função de detecção de melhores modelos 
def best_arima(serie,order,criterion):
    Modelo,Bic,Aic,Aicc = [],[],[],[]
    for ar in range(order[0]+1):
        for i in range(order[1]+1):
            for ma in range(order[2]+1):
                for sar in range(order[3]+1):
                    for si in range(order[4]+1):
                        for sma in range(order[5]+1):
                            for saz in [3,4,6,12]:
                                try:
                                    arima = ARIMA(serie,order=(ar,i,ma),seasonal_order=(sar,si,sma,saz))
                                    model = arima.fit()
                                    Modelo.append("("+str(ar)+","+str(i)+","+str(ma)+")"+"("+str(sar)+","+str(si)+","+str(sma)+","+str(saz)+")")
                                    Bic.append(model.bic)
                                    Aic.append(model.aic)
                                    Aicc.append(model.aicc)
                                except:
                                    pass
    dic = {'Modelo':Modelo,'Bic':Bic,'Aic':Aic,'Aicc':Aicc}
    data = pd.DataFrame(dic)
    return data.sort_values(by=[criterion],ascending=True)
```
```python 
# Melhor modelo bic
modelo_bic = best_arima(serie,order=[1,1,1,4,1,4,12],criterion='Bic')
arima_bic = ARIMA(serie,order=(3,1,3))
model_bic = arima_bic.fit()
model_bic.summary()

# Melhor modelo aic
modelo_bic = best_arima(serie,order=[1,1,1,4,1,4,12],criterion='Aic')
arima_aic = ARIMA(serie,order=(3,1,3))
model_aic = arima_aic.fit()
model_aic.summary()

# Melhor modelo aicc
modelo_bic = best_arima(serie,order=[1,1,1,4,1,4,12],criterion='aicc')
arima_aicc = ARIMA(serie,order=[3,1,3])
model_aicc = arima_aicc.fit()
model_aicc.summary()
```

### 5.3 Descobrindo o melhor modelo - Naive Bayes 
O algoritmo “Naive Bayes” é um classificador probabilístico baseado no “Teorema de Bayes”, o qual foi criado por Thomas Bayes (1701 - 1761) para tentar provar a existência de Deus. 

Portanto como define Becker (2019), O algoritmo de Naive Bayes consiste em encontrar uma probabilidade a posteriori (possuir a doença, dado que recebeu um resultado positivo), multiplicando a probabilidade a priori (possuir a doença) pela probabilidade de “receber um resultado positivo, dado que tem a doença”.

Para modelagem
```python
# Definindo as amostras de treino e teste
serie_train = serie_diff[0:int(len(serie_diff)*0.8)]
serie_test = serie_diff[int(len(serie_diff)*0.8):]

# Definindo o intervalo de armazenamento de dados 
n_bins = 10
bins = np.linspace(serie_train.min(), serie_train.max(), n_bins)
binned = np.digitize(serie_train, bins)

# Criando uma serie de dados armazenados
serie_train1 = pd.Series(binned, index = serie_train.index)

# Media de realizações da serie
bin_means = {}
for binn in range(1,n_bins+1):
    bin_means[binn] = serie_train[binned == binn].mean()

# Dataframe com defasagens
lagged_list = []
for s in range(13):
    lagged_list.append(serie_train1.shift(s))
lagged_frame = pd.concat(lagged_list,1).dropna()

# Criando as amostras de treino
train_X = lagged_frame.iloc[:,1:]
train_y = lagged_frame.iloc[:,0]

# Modelando a Serie temporal 
# Setando o modelo Naive Bayes Gaussiano
model = GaussianNB()

# Ajustando o modelo com as amostras de treino 
model.fit(train_X, train_y)
```
## 6. Modelando a Serie Temporal 
### 6.1 Econtrando os melhores modelos preditivos - ARIMA
```python
def best_forecast(serie,order):
    Modelo, Rmse =[],[]
    for ar in range(order[0]+1):
        for i in range(order[1]+1):
            for ma in range(order[2]+1):
                arima = ARIMA(serie,order=(ar,i,ma))
                model = arima.fit()
                Modelo.append(str(ar)+","+str(i)+","+str(ma))
                Rmse.append(np.sqrt(model.mse))
    dic = {'Modelo':Modelo,'Rmse':Rmse}
    data = pd.DataFrame(dic)
    return data.sort_values(by=['Rmse'],ascending=True)            
```

#### 6.1.2 Gerando previsões 
- **Python**
```python
# Gerando o melhor modelo arma para previsão 
modelo_rmse = best_forecast(serie,[4,1,4])
arima_rmse = ARIMA(serie,order=[3,0,4])
model_rmse = arima_rmse.fit()
```

```python
# Previsão: Forecast 
forecast_rmse = model_rmse.forecast(steps=7)
# Previsão: Predict 
predict_rmse = model_rmse.predict(start=125,end=131) 
```

```R
# Econtrando os melhores modelos preditivos
best_forecast<-function(serie,order){
    modelo =c()
    Rmse = c()
    for(ar in seq(0,order[1]+1)){
        for(i in seq(0,order[2]+1)){
            for(ma in seq(0,order[3]+1)){
                arima = arima(serie,order=c(ar,i,ma))
                modelo<-append(modelo,paste(as.character(ar),as.character(i),as.character(ma)))
                Rmse<-append(Rmse,accuracy(arima)[2])
                }
            }   
        }
    data<-data.frame(modelo,Rmse)
    return(data[order(Rmse),])
}
```
* **R**
```R
# Gerando o melhor modelo modelo arma para previsão 
modelo_rmse<-best_forecast(serie,c(3,1,3))
modelo_rmse<-arima(serie,order=c(4,1,4))

# Previsão: Forecast 
forecast_rmse = forecast(modelo_rmse,h=7)

# Previsão: Predict 
predict_rmse = predict(modelo_rmse,n.ahead=7)
```
- **Modelos com menores erros médios de previsão**


| Modelo   |    Rmse |
|:---------|--------:|
| 3,0,4    | 4.87892 |
| 4,0,3    | 4.93757 |
| 3,0,3    | 4.98122 |
| 2,0,4    | 5.14248 |
| 1,0,4    | 5.14998 |

- **Previsão**
![](https://i.imgur.com/MDZs58h.png)



### 6.2 Econtrando os melhores modelos preditivos - SARIMA
```python
# Econtrando os melhores modelos preditivos 
def best_forecast(serie,order):
    Modelo, Rmse =[],[]
    for ar in range(order[0]+1):
        for i in range(order[1]+1):
            for ma in range(order[2]+1):
                for sar in range(order[3]+1):
                    for si in range(order[4]+1):
                        for sma in range(order[5]+1):
                            for saz in range(2,order[6]+1):
                                arima = ARIMA(serie,order=(ar,i,ma),seasonal_order=(sar,si,sma,saz))
                                model = arima.fit()
                                Modelo.append("("+str(ar)+","+str(i)+","+str(ma)+")"+"("+str(sar)+","+str(si)+","+str(sma)+","+str(saz)+")")
                                Rmse.append(np.sqrt(model.mse))
    dic = {'Modelo':Modelo,'Rmse':Rmse}
    data = pd.DataFrame(dic)
    return data.sort_values(by=['Rmse'],ascending=True)            
```

```python
# Gerando previsões  
modelo_rmse = best_forecast(serie,[4,1,4])
arima_rmse = ARIMA(serie,order=[3,0,4])
model_rmse = arima_rmse.fit()

# Previsão: Forecast 
forecast_rmse = model_rmse.forecast(steps=7)

# Previsão: Predict 
predict_rmse = model_rmse.predict(start=125,end=131) 
```

### 6.3 Naive bayes
```python

# Prevendo a serie 
prediction_frame = pd.DataFrame(np.nan, index = serie_test.index, columns = range(train_X.shape[1]))
predictions = pd.Series(index = serie_test.index)
prediction_frame.iloc[0,1:] = tr.iloc[-1,:-1].values
prediction_frame.iloc[0,0] = train_y.iloc[-1]

# Media das amostras de treino
def get_mean_from_class(prediction):
    return(bin_means[prediction[0]])

# Criando previsões
for i in range(len(serie_test)):
    pred = model.predict(prediction_frame.iloc[i,:].values.reshape(1,-1))
    pred_num = get_mean_from_class(pred.reshape(-1))
    predictions.iloc[i] = pred_num
    try:
        prediction_frame.iloc[i+1,1:] = prediction_frame.iloc[i,:-1].values
        prediction_frame.iloc[i+1,0] = pred[0]
    except:
        pass



trend_test = np.arange(len(train),len(train)+len(serie_test)).reshape(-1,1)
final_prediction = predictions.cumsum()* ((trend_test+1)**(1/2)).reshape(-1)+train.iloc[-1]
rmse = np.sqrt(np.mean((serie_test-final_prediction)**2))

```
![](https://i.imgur.com/J0GSY3b.png)


### 6.4 Comparação dos modelos 



| Posição | Modelo | Rmse |
| -------- | -------- | -------- |
| 1     |  Naive Bayes    | 124.31573804988209  |
| 2     | ARIMA    | 4.87892   |


--- 

# Referencias 

- [x] O QUE é uma série temporal?. In: O que é uma série temporal?. [S. l.], 18 jun. 2019. Disponível em: https://support.minitab.com/pt-br/minitab/18/help-and-how-to/modeling-statistics/time-series/supporting-topics/basics/what-is-a-time-series/. Acesso em: 25 jul. 2021.
- [x] FREITAS, Wilson. Testes de raiz unitária: Avaliando estacionariedade em séries temporais financeiras. In: Testes de raiz unitária: Avaliando estacionariedade em séries temporais financeiras. [S. l.], 15 fev. 2012. Disponível em: http://wilsonfreitas.github.io/estrategias-de-trading/pdf/unit-root-tests.pdf. Acesso em: 25 jul. 2021.

- [x] BROWNLEE , Jason. How to Identify and Remove Seasonality from Time Series Data with Python. In: BROWNLEE , Jason. How to Identify and Remove Seasonality from Time Series Data with Python. [S. l.], 23 dez. 2016. Disponível em: http://wilsonfreitas.github.io/estrategias-de-trading/pdf/unit-root-tests.pdf. Acesso em: 25 jul. 2021.

- [x] SÉRIES Temporais Tendência e sazonalidade. [S. l.], 1 ago. 2015. Disponível em: https://docs.ufpr.br/~lucambio/CE017/2S2015/Series02.pdf. Acesso em: 25 jul. 2021.

- [x] AJUSTE SAZONAL DA PRODUÇÃO INDUSTRIAL EM GOIÁS: UMA ANÁLISE DA INDÚSTRIA DE TRANSFORMAÇÃO E SUAS DESAGREGAÇÕES.. In: RODRIGUES LIMA, ALEX FELIPE et al. AJUSTE SAZONAL DA PRODUÇÃO INDUSTRIAL EM GOIÁS: UMA ANÁLISE DA INDÚSTRIA DE TRANSFORMAÇÃO E SUAS DESAGREGAÇÕES.. [S. l.: s. n.], 2016.

- [x] Holmes, E. E., M. D. Scheuerell, and E. J. Ward. Applied time series analysis for fisheries and environmental data. NOAA Fisheries, Northwest Fisheries Science Center, 2725 Montlake Blvd E., Seattle, WA 98112. Contacts eeholmes@uw.edu, warde@uw.edu, and scheuerl@uw.edu 

- [x] GRAVES , Andrew. Time Series Forecasting with a SARIMA Model: Predicting daily electricity loads for a building on the UC Berkeley campus. In: Time Series Forecasting with a SARIMA Model. [S. l.], 2021. Disponível em: https://towardsdatascience.com/time-series-forecasting-with-a-sarima-model-db051b7ae459. Acesso em: 25 jul. 2021.

- [x]  Enders, W. (2014) Applied Econometric Time Series. 4th Edition. John Wiley, New York. 

- [x]  Hamilton, J. D. (1994). Time Series Analysis. Princeton University Press. ISBN: 0691042896 