import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np

def Entrenar_Perceptron_Simple(factor_aprendizaje, X_train, Y_train, epochs):

    ''' Algoritmo perceptron simple (diapositiva 40/60)
    Retorna los parametros del hiperplano (w0, w1,..., w[n-1]) -> w0*x + w1*y + w2*z + ... + w[n-1] = 0'''

    i = 0 
    w_shape = X_train.shape[1]
    w = np.zeros((w_shape+1, 1))
    error = 1
    error_min = 2*len(X_train)
    
    while error > 0 and i < epochs:
       
        i_x = X_train.sample() #saco 1 elemento aleatório del conjunto de entrenamiento.
        x = np.zeros(w.shape) #(xi, yi, zi,..., 1) (x : ksi_{i}^{u})
        x[-1] = 1
        x[:-1] = np.array(i_x.T)

        index   = int(i_x.index.values)
        zeta_mu = float(Y_train[index]) # Salida deseada
        O_mu    = float(np.dot(x.T, w)) # Excitacion
        activacao = 1 if O_mu >= 0 else -1 #Funcion Escalon
        
        #Actualizacion de los pesos (diapositiva 36/60)
        delta     = (factor_aprendizaje*(zeta_mu - activacao)*x
        w         = w + delta
        
        #Calculo del error sobre todo el conjunto de entrenamiento
        error = ErrorSimple(X_train, Y_train, w)
        if error < error_min:
            error_min = error
            w_min = w
        i += 1
        
    return w_min, error_min


def ErrorSimple(X_train, Y_train, w):
    Ctd_erros = 0 
    for index, row in X_train.iterrows():
        x = np.zeros((X_train.shape[1]+1, 1)) #(xi, yi, zi,...., 1) (x : ksi_{i}^{u})
        x[-1] = 1
        rows = np.array([float(row[i]) for i in range(X_train.shape[1])])
        rows = rows.reshape((X_train.shape[1], 1))
        x[:-1] = rows
        zeta_mu = Y_train[index] #Salida deseada
        O_mu = np.dot(x.T, w)
        activacao = 1 if O_mu >= 0 else -1  #clasificacao perceptron
        error = zeta_mu - activacao
        if (error != 0):
            Ctd_erros += 1
    return Ctd_erros


def Entrenar_Perceptron_Lineal(factor_aprendizaje, X_train, Y_train, epochs):

    '''Algoritmo para entrenar el perceptron simple 
        Retorna los parametros del hiperplano (w0, w1,..., w[n-1]) -> w0*x + w1*y + w2*z + ... + w[n-1] = 0'''
    
    i = 0
    w_shape = X_train.shape[1]
    w = np.zeros((w_shape+1, 1))
    b = 0
    J = []
    
    while i < epochs:

        errors = 0

        for index, row in X_train.iterrows():
            #Para cada elemento do conjunto de treinamento
            i_x = X_train.sample() #saco 1 elemento aleatório del conjunto de entrenamiento.
            x = np.zeros(w.shape) #(xi, yi, zi,..., 1) (x : ksi_{i}^{u})
            x[-1] = 1
            x[:-1] = np.array(i_x.T)
            zeta_mu = float(Y_train[index]) #Salida deseada
            O_mu = float(np.dot(x.T, w))

            #Actualizacion de los pesos (diapositiva 36/60)
            erro      = zeta_mu - O_mu
            delta     = (factor_aprendizaje*erro)*x
            w         = w + delta
            errors += (erro**2)

        J.append(errors)

        i += 1

    return w, J # w = (w0, w1,..., w[n-1]) , donde w[n-1] es el umbral.



def ReadFiles():

    file1 = "TP3-ej2-Salida-deseada.txt"
    file2 = "TP3-ej2-Conjuntoentrenamiento.txt"
    salida        = open(file1, "r")
    entrenamiento = open(file2, "r")

    '''[xi|yi|zi|Salida]'''

    xi = list()
    yi = list()
    zi = list()
    Salida = list()

    for line in salida:
        Salida.append(float(line))

    for line in entrenamiento:
        numbers = line.split()
        xi.append(float(numbers[0]))
        yi.append(float(numbers[1]))
        zi.append(float(numbers[2]))

    data = pd.DataFrame({'xi':xi, 'yi':yi, 'zi':zi, 'Salida':Salida})
    return data
 
        


if __name__ == "__main__":

    #--Leer Archivo
    data = ReadFiles()
    data_normalizado = ((data - data.min()))/(data.max()-data.min())
    X_data = data_normalizado.iloc[:,0:3]
    Clase_data = data_normalizado.iloc[:,3]
    #X_data = data.iloc[:,0:3]   # [xi, yi, zi]
    #Clase_data = data.iloc[:,3] # [clase real o clase deseada]
    #--Separacion entre conjunto de entrenamiento y testeo
    X_train, X_test, y_train, y_test = train_test_split(X_data, Clase_data, test_size = 0.2)
    factor_aprendizaje = 0.001
    epochs = 10
    W, J = Entrenar_Perceptron_Lineal(factor_aprendizaje, X_train, y_train, epochs)
    w0 = W[0]
    w1 = W[1]
    w2 = W[2]
    w3 = W[3]
    print('Parametros del hiperplano')
    print('w0 : {0}'.format(w0))
    print('w1 : {0}'.format(w1))
    print('w2 : {0}'.format(w2))
    print('w3 : {0}'.format(w3))
    print('J : {0}'.format(J))


    #-----------------------------------Exemplo Diapositivas----------------------------------------
    #xi = [1.1946, 0.8788, 1.1907, 1.4180, 0.2032, 2.7571, 4.7125, 3.9392, 1.2072, 3.4799, 0.4763]
    #yi = [3.8427, 1.6595, 1.6117, 3.8272, 1.9208, 1.0931, 2.8166, 1.1032, 0.8132, 1.9982, 0.1020]
    #Salida  = [1, 1, 1, 1, 1, -1, -1, -1, -1, -1, -1]
    #colores = ['red','red','red','red','red','blue','blue','blue','blue','blue','blue']
    #Data    = pd.DataFrame({'xi':xi, 'yi':yi, 'Salida':Salida})
    #X_data  = Data.iloc[:,0:2]
    #Clase_data = Data.iloc[:,2]
    #X_train, X_test, y_train, y_test = train_test_split(X_data, Clase_data, test_size=0)
    #factor_aprendizaje = 0.01
    #epochs = 150
    #W, error_min = Entrenar_Perceptron_Simple(factor_aprendizaje, X_train, y_train, epochs)
    #w0 = W[0]
    #w1 = W[1]
    #w2 = W[2]
    #print('Parametros del hiperplano')
    #print('w0 : {0}'.format(w0))
    #print('w1 : {0}'.format(w1))
    #print('w2 : {0}'.format(w2))
    #print('error : {0}'.format(error_min))
    #----------------------------------------------------------------------------------------------


    #-------------------------------------Problema 1-------------------------------------
    #xi = [-1,+1,-1,+1]
    #yi = [+1,-1,-1,+1]
    #Salida_AND = [-1,-1,-1,+1] 
    #Salida_XOR = [+1,+1,-1,-1]
    #set_AND = pd.DataFrame({'xi':xi, 'yi':yi, 'Salida':Salida_AND})
    #set_XOR = pd.DataFrame({'xi':xi, 'yi':yi, 'Salida':Salida_XOR})

    #X_AND = set_AND.iloc[:,0:2] # [xi, yi]
    #Clase_AND = set_AND.iloc[:,2] # [clase real]
    #X_XOR = set_XOR.iloc[:,0:2]
    #Clase_XOR = set_XOR.iloc[:,2]

    #--Separacion entre conjunto de entrenamiento y testeo
    #X_train, X_test, y_train, y_test = train_test_split(X_AND, Clase_AND, test_size = 0)

    #--Condiciones Iniciales
    #factor_aprendizaje = 0.2
    #epochs = 150
    #W, error_min = Entrenar_Perceptron_Simple(factor_aprendizaje, X_train, y_train, epochs)
    #w0 = W[0]
    #w1 = W[1]
    #w2 = W[2]
    #print('Parametros del hiperplano')
    #print('w0 : {0}'.format(w0))
    #print('w1 : {0}'.format(w1))
    #print('w2 : {0}'.format(w2))
    #print('error : {0}'.format(error_min))
    #-----------------------------------------------------------------------------------


    #-----------------------------------PLOT 2D-----------------------------------------
    #Solo funciona para el ejemplo del perceptron simple

    #--x.w0 + y.w1 + w2 = 0
    #--y.w1 = -w0.x - w2
    #--y    = -(w0/w1)x - (w2/w1) --> Recta en R².
    #--m = -w0/w1; b = -w2/w1
    #m = -(w0/w1)
    #b = -(w2/w1)
    #x = np.linspace(-5, 5, 100) 
    #y_perceptron = m*x + b

    #fig, ax = plt.subplots()
    #colorsAND = ['red','red','red','blue']
    #colorsXOR = ['blue','blue','red','red']
    #plt.scatter(xi, yi, c=colores)
    #plt.plot(x, y_perceptron, color='green')
    #plt.show()
