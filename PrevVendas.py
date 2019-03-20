#Leonardo Kamke.
#base de leitura http://www.elciofernando.com.br/material/3390f253ba6c.pdf
# https://www.vooo.pro/insights/guia-de-acesso-rapido-python-para-data-science-matplotlib/
# https://www.vooo.pro/insights/guia-de-acesso-rapido-python-numpy/
# https://www.vooo.pro/insights/guia-de-acesso-rapido-ao-pandas/

#Regressão Linear exemplos: https://bit.ly/2TNZ4Pt
#Perda média de regressão por erro absoluto


from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from sklearn import datasets, linear_model
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
import _csv
from pandas import csv
import numpy as np
import pandas as pd



#Calculo MAPE usado para medir a acuracidade da previsão
#o ABS retorna a dif entre 0 e o resultado da equação.
#O mean retorna a media da matriz
def mape(previsao, real):

    return np.mean(np.abs((real - previsao) / real)) * 100  #Calculo do MAPE - Erro Percentual Médio Absoluto


#dados do excel
#dados = open('C:\Temp\meuarquivo1.csv')
#skip_blank_lines=True       # ignora linhas vazias
dados = pd.read_csv("C:\Temp\meuarquivo1.csv", header=0, index_col=0)      # pode ser usado o "heade" ou "index_col" localiza,usa a primeira coloca como índice e organiza por ordem crescente.
#dados = pd.read_csv('C:\Temp\meuarquivo1.csv')

print(dados)


###                                       Testes abaixo - 
###

'''

x_dados = []
y_dados = []

for d in range(1, dados.shape[0]):

    # a propriedade shape mostra o tamanho de cada dimensão da matriz
    # iloc obtem linhas (ou colunas) em posições específicas no índice (portanto, somente números inteiros).

    x = dados.iloc[0:-1, d:]
    y = dados.iloc[d].values[0]

    x_dados.append(x)
    y_dados.append(y)

    #O append()método acrescenta um elemento ao final da lista.
    #EXEMPLO https://www.w3schools.com/python/ref_list_append.asp
    x_dados = np.array(x_dados)
    y_dados = np.array(y_dados)


    previsao = []
    previsao_ultima = []
    previsao_mediamovel = []
    real = []


                                            ## ESTUDAR MÉTODO DE TREINAMENTO ##
##Artigo explicando sobre train e test em aprendizagem https://towardsdatascience.com/train-test-split-and-cross-validation-in-python-80b61beca4b6

end = y_dados.shape[0]
for i in range(30, end):

    x_train = x_dados[:i, :]
    y_train = y_dados[:i]


    x_teste = x_dados[i, :]
    y_teste = y_dados[i]


    # .FIT(train) Treina o modelo para um determinado número de épocas (iterações em um conjunto de dados).

    model = LinearRegression(normalize=True) #documentação https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html
    model.fit(x_train,y_train)

    previsao.append(model.predict(x_teste))
    previsao_ultima.append(x_teste[-1])
    previsao_mediamovel.append(x_teste.mean())
    real.append(y_teste)

    previsao = np.array(previsao)
    previsao_ultima = np.array(previsao_ultima)
    previsao_mediamovel = np.array(previsao_mediamovel)
    real = np.array(real)

    # Imprime os erros e resultados de cálculos  na tela
    print
    "\nMean Absolute Percentage Error"
    print
    "MAPE Regressão Linear", mape(previsao, real)
    print
    "MAPE Último Valor", mape(previsao_ultima, real)
    print
    "MAPE Média Móvel", mape(previsao_mediamovel, real)

    print
    "\nMean Absolute Error"
    print
    "MAE Regressão Linear", mean_absolute_error(previsao, real)
    print
    "MAE Último Valor", mean_absolute_error(previsao_ultima, real)

    "MAE Média Móvel", mean_absolute_error(previsao_mediamovel, real)



    
 '''



