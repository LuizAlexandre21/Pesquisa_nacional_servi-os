# Pacotes 
library(jsonlite)
library(dplyr)
library(urca)
library(stats)
library(forecast)

# 1. Coletando os dados 
# 1.1 Acesando o indice de serviço com a api do sidra/IBGE
dados = fromJSON('https://apisidra.ibge.gov.br/values/t/6443/n1/all/v/8676/p/all/c11046/40311/c12355/107071/d/v8676%201')

# 2. Limpeza e Ajuste dos dados 
# 2.1 Ajustes das colunas 
# 2.1.1 Renomeando as colunas
names(dados)<-dados[1,]
# 2.1.2 Removendo as linhas contendo nomes das colunas 
dados <- dados[-1,]

# 2.1.3 Limpando as colunas 
dados <- dados[,c('Nível Territorial','Valor','Mês','Atividades de serviços')]

# 2,2 Criando uma serie temporal 
serie = ts(data=as.numeric(dados$Valor),start=c(2011,1), end=c(2021,5),frequency=12)

# 3. Modelando a Serie Temporal 
# 3.1 Analisando o comportamento da serie temporal 
# 3.1.1 Teste de estacionariedade 
# 3.1.1.1 Test Augmented -Fuller - Com constante e com tendência 
adf_1 = ur.df(as.numeric(serie),type="trend")

# 3.1.1.2 Test Augmented Dickey-Fuller - Com constante
adf_2 = ur.df(as.numeric(serie),type="drift")

# 3.1.1.2 Test Augmented Dickey-Fuller - Com constante
adf_3 = ur.df(as.numeric(serie),type="none")

# 3.2.1 Teste de sazonalidade 

# 3.3 Correlograma 
# 3.3.1 Autocorrelação 
autocorrelação <- acf(as.numeric(serie),lag.max=36,plot=FALSE)

# 3.3.2 Autocorrelação Parcial 
autocorrelação_parcial <- pacf(as.numeric(serie),lag.max=36,plot=FALSE)

# 3.4 Diferenciando a serie 
# 3.4.1 Criando a serie diferenciada 
serie_diff = diff(as.numeric(serie))

# 3.4.2 Aplicando os testes 
# 3.4.2.1 Teste de estacionariedade 
# 3.4.2.1.1 Test Augmented Dickey-Fuller - Sem constante e sem tendência 
adf_1 = ur.df(serie_diff,type="trend")

# 3.4.2.1.2 Test Augmented -Fuller - Com constante e com tendência 
adf_2 = ur.df(serie_diff,type="drift")

# 3.4.2.1.3 Test Augmented Dickey-Fuller - Sem constante e Sem tendência 
adf_3 = ur.df(serie_diff,type="none")

# 3.4.3 Correlograma 
# 3.4.3.1 Autocorrelação
autocorrelação = acf(serie_diff)

# 3.4.3.2 Autocorrelação Parcial 
autocorrelação_parcial = pacf(serie_diff)

# 3.5 Modelando a Serie Temporal
# 3.5.1 Melhor modelo bic
modelo_bic=auto.arima(as.numeric(serie),max.p=4,max.q=4,max.d=1,ic=c("bic"))
summary(modelo_bic)

# 3.5.2 Melhor modelo aic 
modelo_aic=auto.arima(as.numeric(serie),max.p=4,max.q=4,max.d=1,ic=c("aic"))
summary(modelo_aic)

# 3.5.3 Melhor modelo aicc 
modelo_aicc=auto.arima(as.numeric(serie),max.p=4,max.q=4,max.d=1,ic=c("aicc"))
summary(modelo_aicc)

# 4. Criando Previsões
# 4.1.1 Econtrando os melhores modelos preditivos 
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

# 4.1.2 Gerando o melhor modelo modelo arma para previsão 
modelo_rmse<-best_forecast(serie,c(3,1,3))
modelo_rmse<-arima(serie,order=c(4,1,4))

# 4.1.3 Previsão: Forecast 
forecast_rmse = forecast(modelo_rmse,h=7)

# 4.1.4 Previsão: Predict 
predict_rmse = predict(modelo_rmse,n.ahead=7)