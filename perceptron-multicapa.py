import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from perceptron import *




# 1.Inicializar el conjunto de pesos en valores ’peque ̃nos’ al azar.
    # TODO Necesito cantidad_de_w = K*(cantidad_nodos_capa_oculta_siguiente)
        # TODO necesitamos variables para cantidad_de_capas y nodos_x_capas 3-2-1 | 3-1


# 2.Tomar un ejemploξμal azar del conjunto de entrenamiento y aplicarlo a la capa 0:V0k=ξμkpara todok

# 3.Propagar la entrada hasta a capa de salidaVmi=g(hmi) =g(∑jwmijVm−1j) para todo m desde 1 hasta M.

# 4.Calcularδpara la capa de salidaδMi=g′(hMi)(ζμi−VMi)

# 5.RetropropagarδMiδm−1i=g′(hm−1i)∑jwmjiδmjpara todo m entre M y 2

# 6.Actualizar los pesos de las conexiones de acuerdow.nuevomij=w.viejomij+ ∆wmijdonde ∆wmij=ηδmiVm−1j

# 7.Calcular elerror. Sierror>COTA, ir a 2.













if __name__ == "__main__":
    # --Leer Archivo
    data = ReadFiles()
    data_normalizado = ((data - data.min())) / (data.max() - data.min())
    X_data = data_normalizado.iloc[:, 0:3]
    Clase_data = data_normalizado.iloc[:, 3]
    # X_data = data.iloc[:,0:3]   # [xi, yi, zi]
    # Clase_data = data.iloc[:,3] # [clase real o clase deseada]
    # --Separacion entre conjunto de entrenamiento y testeo
    X_train, X_test, y_train, y_test = train_test_split(X_data, Clase_data, test_size=0.6)
    factor_aprendizaje = 0.001
    epochs = 50
    W, J = Entrenar_Perceptron_No_Lineal(factor_aprendizaje, X_train, y_train, epochs)
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
