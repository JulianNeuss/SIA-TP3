import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np

def Entrenar_Perceptron(factor_aprendizaje, X_train, Y_train, epochs):

    ''' Algoritmo perceptron simple (diapositiva 40/60)
    Retorna los parametros del hiperplano (w0, w1, w2) -> w0*x + w1*y + w2 = 0'''
    
    i = 0 
    w = np.random.rand(2, 1) #w1, w2 (Problemas en 2D)
    b = 0
    error = 1
    error_min = 2*len(X_train)
    
    while error > 0 and i < epochs:
        
        i_x = X_train.sample() #saco 1 elemento aleatório del conjunto de entrenamiento.
        x = np.array([ i_x['xi'], i_x['yi'] ]) #(xi, yi)
        index = int(i_x.index.values)
        ClaseReal = Y_train[index]
        
        exitacion = np.dot(x.T, w) + b
        activacao = 1 if exitacion >= 0 else -1 #Funcion Escalon
        
        #Actualizacion de los pesos
        err = ClaseReal - activacao
        delta_w = factor_aprendizaje*(err)*x
        delta_b = factor_aprendizaje*(err)
        w = w + delta_w
        b = b + delta_b
        
        #Calculo del error sobre todo el conjunto de entrenamiento
        error = CalcularErro(X_train, Y_train, w, b)
        if error < error_min:
            error_min = error
            w_min = w
            b_min = b
        i += 1
        
    return w_min[0], w_min[1], b_min


def CalcularErro(X_train, Y_train, w, b):
    Ctd_erros = 0 #Cantidad de errores de clasificación con los parametros w y b.
    for index, row in X_train.iterrows():
        x = np.array([row[0], row[1]]) #(xi, yi)
        ClaseReal = Y_train[index] #Clase real
        exitacion = np.dot(x.T, w) + b
        activacao = 1 if exitacion >= 0 else -1  #clasificacao perceptron
        error = ClaseReal - activacao
        if (error != 0):
            Ctd_erros += 1
    return Ctd_erros


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
    # Leer Archivo

    data = ReadFiles()
    X_data = data.iloc[:,0:3]   # [xi, yi, zi]
    Clase_data = data.iloc[:,3] # [clase real]
    #Separacion entre conjunto de entrenamiento y testeo
    test_size = 20 #Porcentaje de elementos en el conjunto de testeo
    X_train, X_test, y_train, y_test = train_test_split(X_data, Clase_data, test_size = 0)
    factor_aprendizaje = 0.2
    epochs = 150
    #w0, w1, w2 = Entrenar_Perceptron(factor_aprendizaje, X_train, y_train, epochs)

    #-------------------------------------Problema 1-------------------------------------
    xi = [-1,+1,-1,+1]
    yi = [+1,-1,-1,+1]
    Salida_AND = [-1,-1,-1,+1] 
    Salida_XOR = [+1,+1,-1,-1]

    set_AND = pd.DataFrame({'xi':xi, 'yi':yi, 'Salida':Salida_AND})
    set_XOR = pd.DataFrame({'xi':xi, 'yi':yi, 'Salida':Salida_XOR})

    X_AND = set_AND.iloc[:,0:2] # [xi, yi]
    Clase_AND = set_AND.iloc[:,2]   # [clase real]

    X_XOR = set_XOR.iloc[:,0:2]
    Clase_XOR = set_XOR.iloc[:,2]

    #Separacion entre conjunto de entrenamiento y testeo
    test_size = 0 #Porcentaje de elementos en el conjunto de testeo
    X_train, X_test, y_train, y_test = train_test_split(X_AND, Clase_AND, test_size = 0)

    # Condiciones Iniciales
    factor_aprendizaje = 0.5
    epochs = 150
    w0, w1, w2 = Entrenar_Perceptron(factor_aprendizaje, X_train, y_train, epochs)

    #x.w0 + y.w1 + w2 = 0
    #y.w1 = -w0.x - w2
    #y    = -(w0/w1)x - (w2/w1) --> Recta en el R².
    #m = -w0/w1; b = -w2/w1
    m = -(w0/w1)
    b = -(w2/w1)
    x = np.linspace(-1.5, 1.5, 100) 
    y_perceptron = m*x + b

    fig, ax = plt.subplots()
    plt.style.use('seaborn-whitegrid')
    colorsAND = ['red','red','red','blue']
    colorsXOR = ['blue','blue','red','red']
    plt.scatter(xi, yi, c=colorsAND)
    plt.plot(x, y_perceptron, color='green')
    plt.show()
