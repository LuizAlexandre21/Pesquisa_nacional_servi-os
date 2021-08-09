# Pacotes
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

# 1. Coletando os dados 
# 1.1 Acesando o indice de serviço com a api do sidra/IBGE
url = requests.get('https://apisidra.ibge.gov.br/values/t/6443/n1/all/v/8676/p/all/c11046/40311/c12355/107071/d/v8676%201')

# 1.2 Criando arquivo json com o indice 
json = url.json()

# 1.3 Criando um Dataframe 
data = pd.json_normalize(json)

# 2. Limpeza e Ajuste dos dados 
# 2.1 Ajuste das colunas
# 2.1.1 Renomeando as colunas 
data=data.rename(columns=data.iloc[0]).drop([0])

# 2.1.2 Limpando as colunas 
data = data.filter(items=['Nível Territorial','Valor','Mês','Atividades de serviços'],axis=1)

# 2.2 Ajustando a unidade de tempo
# 2.2.1 Corrigindo a localização do computador
locale.setlocale(locale.LC_ALL, 'pt_BR.utf8')

# 2.2.2 Convertendo a coluna 'Mês' para formato de datetime 
date = []
for mes in data['Mês']:
    date.append(datetime.datetime.strptime(mes,'%B %Y').strftime('%m.%y'))
data['Mês']=date 

# 2.3 Criando uma serie temporal 
serie = pd.Series(data['Valor'].astype('float64').to_list(),index=data['Mês'])

# 3. Modelando a Serie Temporal 
# 3.1 Analisando o comportamento da serie temporal 
# 3.1.1 Teste de estacionariedade 
# 3.1.1.1 Test Augmented Dickey-Fuller - Sem constante e sem tendência 
adf_1 = {'adf':adfuller(serie,regression="ct")[0],'pvalue':adfuller(serie,regression="ct")[1],'Nº de lags': adfuller(serie,regression="ct")[2],'Critical_value':adfuller(serie, regression="ct")[4]['5%']}

# 3.1.1.2 Test Augmented Dickey-Fuller - Com constante
adf_2 = {'adf':adfuller(serie)[0],'pvalue':adfuller(serie)[1],'Nº de lags': adfuller(serie)[2],'Critical_value':adfuller(serie)[4]['5%']}

# 3.1.1.3 Test Augmented Dickey-Fuller - Sem constante e Sem tendência 
adf_3 = {'adf':adfuller(serie,regression="nc")[0],'pvalue':adfuller(serie,regression="nc")[1],'Nº de lags': adfuller(serie,regression="nc")[2],'Critical_value':adfuller(serie, regression="nc")[4]['5%']}

# 3.2.1 Teste de sazonalidade
# 3.2.1.1 Função de destrinchamento das series
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
    
# 3.2.2.1 Sazonalidade - Quadrimestre
data=Sazonality(serie,"quadrimestre")
saz_quart=stats.friedmanchisquare(data[0],data[1],data[2],data[3],data[4],data[5],data[6])

# 3.2.2.2 Sazonalidade - Trimestral
data=Sazonality(serie,"trimestre")
saz_trim = stats.friedmanchisquare(data[1],data[2],data[3],data[4],data[5],data[6],data[7])

# 3.2.2.3 Sazonalidade - Semestral
data=Sazonality(serie,"semestre")
saz_semes= stats.friedmanchisquare(data[7],data[6],data[5],data[4],data[3],data[2],data[1])

# 3.2.2.4 Sazonalidade - Anual
data=Sazonality(serie,"anual")
saz_anual= stats.friedmanchisquare(data[1],data[2],data[3],data[4],data[5],data[6],data[7])

# 3.3 Correlograma 
# 3.3.1 Autocorrelação
autocorrelação = acf(serie)

# 3.3.2 Autocorrelação Parcial 
autocorrelação_parcial = pacf(serie)

# 3.4 Diferenciando a serie 
# 3.4.1 Criando a serie diferenciada 
serie_diff = serie.diff().dropna()

# 3.4.2 Aplicando os testes 
# 3.4.2.1 Teste de estacionariedade 
# 3.4.2.1.2 Test Augmented -Fuller - Com constante e com tendência 

adf_1 = {'adf':adfuller(serie_diff,regression="ct")[0],'pvalue':adfuller(serie_diff,regression="ct")[1],'Nº de lags': adfuller(serie_diff,regression="ct")[2],'Critical_value':adfuller(serie_diff, regression="ct")[4]['5%']}

# 3.4.2.1.2 Test Augmented Dickey-Fuller - Com constante
adf_2 = {'adf':adfuller(serie_diff)[0],'pvalue':adfuller(serie_diff)[1],'Nº de lags': adfuller(serie_diff)[2],'Critical_value':adfuller(serie_diff)[4]['5%']}

# 3.4.2.1.3 Test Augmented Dickey-Fuller - Sem constante e Sem tendência 
adf_3 = {'adf':adfuller(serie_diff,regression="nc")[0],'pvalue':adfuller(serie_diff,regression="nc")[1],'Nº de lags': adfuller(serie_diff,regression="nc")[2],'Critical_value':adfuller(serie_diff, regression="nc")[4]['5%']}

# 3.4.3 Correlograma 
# 3.4.3.1 Autocorrelação
autocorrelação = acf(serie)

# 3.4.3.2 Autocorrelação Parcial 
autocorrelação_parcial = pacf(serie)

# 3.5 Modelando a Serie temporal 
# 3.5.1 Função de detecção de melhores modelos 
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

# 3.5.2 Descobrindo o melhor modelo 
# 3.5.2.1 Melhor modelo bic
modelo_bic = best_arima(serie,order=[4,1,4],criterion='Bic')
arima_bic = ARIMA(serie,order=(3,1,3))
model_bic = arima_bic.fit()
model_bic.summary()

# 3.5.2.2 Melhor modelo aic
modelo_aic = best_arima(serie,order=[4,1,4],criterion='Aic')
arima_aic = ARIMA(serie,order=(3,1,3))
model_aic = arima_aic.fit()
model_aic.summary()

# 3.5.2.3 Melhor modelo aicc
modelo_aicc = best_arima(serie,order=[4,1,4],criterion='Aicc')
arima_aicc = ARIMA(serie,order=[3,1,3])
model_aicc = arima_aicc.fit()
model_aicc.summary()

# 4. Criando Previsões
# 4.1 Modelos Autoregressivos 
# 4.1.1 Econtrando os melhores modelos preditivos 
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

# 4.1.2 Gerando previsões 
# 4.1.2.1 Gerando o melhor modelo arma para previsão 
modelo_rmse = best_forecast(serie,[4,1,4])
arima_rmse = ARIMA(serie,order=[3,0,4])
model_rmse = arima_rmse.fit()

# 4.1.2.2 Previsão: Forecast 
forecast_rmse = model_rmse.forecast(steps=7)

# 4.1.2.3 Previsão: Predict 
predict_rmse = model_rmse.predict(start=125,end=131) 

# 4.2 Naive bayes
# 4.2.1 Definindo as amostras de treino e teste
serie_train = serie_diff[0:len(serie_diff)*0.8]
serie_test = serie_diff[len(serie_diff)*0.8:]

# 4.2.1 Definindo o intervalo de armazenamento de dados 
n_bins = 10
bins = np.linspace(serie_train.min(), serie_train.max(), n_bins)
binned = np.digitize(serie_train, bins)

# 4.2.2 Criando uma serie de dados armazenados
serie_train1 = pd.Series(binned, index = serie_train.index)

# 4.2.3 Media de realizações da serie
bin_means = {}
for binn in range(1,n_bins+1):
    bin_means[binn] = serie_train[binned == binn].mean()

# 4.2.4 Dataframe com defasagens
lagged_list = []
for s in range(13):
    lagged_list.append(serie_train1.shift(s))
lagged_frame = pd.concat(lagged_list,1).dropna()

# 4.2.5 Criando as amostras de treino
train_X = lagged_frame.iloc[:,1:]
train_y = lagged_frame.iloc[:,0]

# 4.2.6 Modelando a Serie temporal 
# 4.2.6.1 Setando o modelo Naive Bayes Gaussiano
model = GaussianNB()

# 4.2.6.2 Ajustando o modelo com as amostras de treino 
model.fit(train_X, train_y)

# 4.2.7 Prevendo a serie 
prediction_frame = pd.DataFrame(np.nan, index = serie_test.index, columns = range(train_X.shape[1]))
predictions = pd.Series(index = serie_test.index)
prediction_frame.iloc[0,1:] = train_X.iloc[-1,:-1].values
prediction_frame.iloc[0,0] = train_y.iloc[-1]

# 4.2.8 Media das amostras de treino
def get_mean_from_class(prediction):
    return(bin_means[prediction[0]])

for i in range(len(test)):
    pred = model.predict(prediction_frame.iloc[i,:].values.reshape(1,-1))
    pred_num = get_mean_from_class(pred.reshape(-1))
    predictions.iloc[i] = pred_num
    try:
        prediction_frame.iloc[i+1,1:] = prediction_frame.iloc[i,:-1].values
        prediction_frame.iloc[i+1,0] = pred[0]
    except:
        pass

trend_test = np.arange(len(serie_train),len(serie_train)+len(serie_test)).reshape(-1,1)
final_prediction = predictions.cumsum()* ((serie_test+1)**(1/2)).reshape(-1)+train.iloc[-1]

trend_test = np.arange(len(train),len(train)+len(test)).reshape(-1,1)
final_prediction = predictions.cumsum()* ((trend_test+1)**(1/2)).reshape(-1)+train.iloc[-1]
rmse = np.sqrt(np.mean((test-final_prediction)**2))

# 4.3 Modelo de Autorregressivos com Sazonalidade 
# 3.5.1 Função de detecção de melhores modelos 
def best_arima(serie,order,criterion):
    Modelo,Bic,Aic,Aicc = [],[],[],[]
    for ar in range(order[0]+1):
        for i in range(order[1]+1):
            for ma in range(order[2]+1):
                for sar in range(order[3]+1):
                    for si in range(order[4]+1):
                        for sma in range(order[5]+1):
                            for saz in range(2,order[6]+1):
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

# 3.5.2 Descobrindo o melhor modelo 
# 3.5.2.1 Melhor modelo bic
modelo_bic = best_arima(serie,order=[1,1,1,4,1,4,12],criterion='Bic')
arima_bic = ARIMA(serie,order=(3,1,3))
model_bic = arima_bic.fit()
model_bic.summary()

# 3.5.2.2 Melhor modelo aic
modelo_bic = best_arima(serie,order=[1,1,1,4,1,4,12],criterion='Aic')
arima_aic = ARIMA(serie,order=(3,1,3))
model_aic = arima_aic.fit()
model_aic.summary()

# 3.5.2.3 Melhor modelo aicc
modelo_bic = best_arima(serie,order=[1,1,1,4,1,4,12],criterion='Aicc')
arima_aicc = ARIMA(serie,order=[3,1,3])
model_aicc = arima_aicc.fit()
model_aicc.summary()
# 4.3.1 Econtrando os melhores modelos preditivos 
def best_forecast(serie,order):
    Modelo, Rmse =[],[]
    for ar in range(order[0]+1):
        for i in range(order[1]+1):
            for ma in range(order[2]+1):
                for sar in range(order[3]+1):
                    for si in range(order[4]+1):
                        for sma in range(order[5]+1):
                            for saz in [3,4,6,12]:
                                arima = ARIMA(serie,order=(ar,i,ma),seasonal_order=(sar,si,sma,saz))
                                model = arima.fit()
                                Modelo.append("("+str(ar)+","+str(i)+","+str(ma)+")"+"("+str(sar)+","+str(si)+","+str(sma)+","+str(saz)+")")
                                Rmse.append(np.sqrt(model.mse))
    dic = {'Modelo':Modelo,'Rmse':Rmse}
    data = pd.DataFrame(dic)
    return data.sort_values(by=['Rmse'],ascending=True)            

# 4.1.2 Gerando previsões 
# 4.1.2.1 Gerando o melhor modelo arma para previsão 
modelo_rmse = best_forecast(serie,order=[1,1,1,4,1,4])
arima_rmse = ARIMA(serie,order=[3,0,4])
model_rmse = arima_rmse.fit()

# 4.1.2.2 Previsão: Forecast 
forecast_rmse = model_rmse.forecast(steps=7)

# 4.1.2.3 Previsão: Predict 
predict_rmse = model_rmse.predict(start=125,end=131) 
