
def entrenar(theta, factor_aprendizaje, w1, w2, epochs, x1, x2, d, n_muestras):
    errores = True
    while errores and epochs < 100:
        errores = False
        for i in range(n_muestras):
            z = ((x1[i] * w1)+(x2[i] * w2)) - theta # calculamos z

            if z >= 0:
                z = 1
            else:
                z = -1

            if z != d[i]:
                errores = True
                # calcular errores
                error = (d[i] - z)
                # ajustar theta
                theta = theta + (-(factor_aprendizaje * error))
                # ajustar pesos
                w1 = w1 + (x1[i] * error * factor_aprendizaje)
                w2 = w2 + (x2[i] * error * factor_aprendizaje)
                epochs += 1

    return w1, w2, epochs, theta


if __name__ == "__main__":
    # Leer Archivo
    # archivo_excel = pd.read_excel("C:/Trabajo/PythonWorkbook/perceptron_data.xlsx")

    # Ejercicio 1
    entradasX = list()
    entradasX.append(-1)
    entradasX.append(1)
    entradasX.append(-1)
    entradasX.append(1)

    entradasY = list()
    entradasY.append(1)
    entradasY.append(-1)
    entradasY.append(-1)
    entradasY.append(1)

    # Salida esperada AND
    salidaAND = list()
    salidaAND.append(-1)
    salidaAND.append(-1)
    salidaAND.append(-1)
    salidaAND.append(1)

    # Salida esperada XOR
    salidaXOR = list()
    salidaXOR.append(1)
    salidaXOR.append(1)
    salidaXOR.append(-1)
    salidaXOR.append(-1)

    # Condiciones Iniciales
    theta = 0.2
    factor_aprendizaje = 0.2
    w1 = 0.5
    w2 = 0.5
    epochs = 0
    n_muestras = len(salidaXOR)
    w1, w2, epochs, theta = entrenar(theta, factor_aprendizaje, w1, w2, epochs, entradasX, entradasY, salidaXOR, n_muestras)
    print(w1)
    print(w2)
    print(epochs)
    print(theta)
