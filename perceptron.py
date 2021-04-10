import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn import svm
from sklearn import metrics

def Train_Perceptron(alfa, X_train, Y_train):
    
    '''Treina a rede perceptron dado um coeficiente de aprendizagem alfa 
    e o conjunto de treinamento X_train.
    Retorna: Parâmetros do hiperplano (w0, w1, w2) -> w0*x + w1*y + w2 = 0'''
    
    i = 0 #iteraciones
    w = np.zeros((2, 1)) #w1, w2
    b = 0
    error = 1
    error_min = 2*len(X_train)
    COTA = 500
    
    #A cota é para o caso em que o conjunto não seja linearmente separável
    while error > 0 and i < COTA:
        
        i_x = X_train.sample() #sacar 1 elemento aleatório del conjunto de entrenamiento.
        x = np.array([ i_x['xi'], i_x['yi'] ]) #(xi, yi)
        index = int(i_x.index.values)
        ClasseVerdadeira = Y_train[index]
        
        exitacion = np.dot(x.T, w) + b
        activacao = 1 if exitacion >= 0 else -1
        
        err = ClasseVerdadeira - activacao
        delta_w = alfa*(err)*x
        delta_b = alfa*(err)
        
        w = w + delta_w
        b = b + delta_b
        
        #O error é calculado sobre todo o conjunto de treinamento.
        error = CalcularError(X_train, Y_train, w, b)
        if error < error_min:
            error_min = error
            w_min = w
            b_min = b
        i += 1
        
    return w_min[0], w_min[1], b_min


def CalcularError(X_train, Y_train, w, b):
    
    num_erro = 0
    
    for index, row in X_train.iterrows():
        
        x = np.array([row[0], row[1]]) #(xi, yi)
        ClasseVerdadeira = Y_train[index]
    
        #ClasseVerdadeira = -1, 1
        exitacion = np.dot(x.T, w) + b
        activacao = 1 if exitacion >= 0 else -1
        error = ClasseVerdadeira - activacao
        
        if (error != 0):
            num_erro += 1
    
    return num_erro


w0_TP31, w1_TP31, w2_TP31 = Train_Perceptron(0.5, X1_train, y1_train) #Entreinamento Perceptron TP3-1
w0_TP32, w1_TP32, w2_TP32 = Train_Perceptron(0.5, X2_train, y2_train) #Entreinamento Perceptron TP3-2


def Classificacao_Perceptron(X_test, y_test, w, bias):
    
    N = X_test.shape[0]
    Acertos = 0
    Precision = 0
    
    for index, row in X_test.iterrows():
        
        x = np.array([row[0], row[1]]) #(xi, yi)
        RealClase = y_test[index]
        
        #ClasseVerdadeira = -1, 1
        exitacion = np.dot(x.T, w) - bias
        activacao = 1 if exitacion >= 0 else -1
        error = RealClase - activacao
        
        if(error == 0):
            Acertos += 1
            
    return Acertos/N



#Classificação Perceptron (TP31)
w0 = [w0_TP31, w1_TP31]
b = w2_TP31
Precision_Percep_TP31 = Classificacao_Perceptron(X1_test, y1_test, w0, b)
#print(Precision_Percep_TP31)

#Classificação Perceptron (TP32)
w0 = [w0_TP32, w1_TP32]
b = w2_TP32
Precision_Percep_TP32 = Classificacao_Perceptron(X1_test, y1_test, w0, b)
#print(Precision_Percep_TP32)
