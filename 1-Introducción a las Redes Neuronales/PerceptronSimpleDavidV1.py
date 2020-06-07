import matplotlib.pyplot as plt
import numpy as np
#import cv2
import csv
import pandas as pd

datos=[]


def leer_Datos(file='trainAND.csv'):
    """
    Cargar los datos
    """
    data= pd.read_csv(file, header = None) #Load DataBase
    X = data.iloc[:,1:].values # Features
    S= data.iloc[:,0].values   # Target    
    return X, S
            

def inicializar_Parametros(dim):
    """
    Esta función da valores aleatorios de tamaño (dim,1) para los pesos w e inicializa b (teta) a 0
    
    Argumentos:
    dim -- Tamaño del vector w que se desea (esto es igual al número de parámetros)
    
    Returna:
    w -- vector inicializado de tamaño (dim, 1). Usar np.random.randn(dim, 1) para crear un vector de valores 
         aleatorios de tamaño (dim,1) 
    b -- Escalar inicializado en 0 
    """    
    np.random.seed(1)    
    w = np.random.randn(dim,1)  # se generan valores aleatorios a los pesos
    b = 0                       # se asigna a 0 el valor del bias

    
    #Con esto nos aseguramos que no hay error en la operación anterior:
    assert(w.shape == (dim, 1)) 
    assert(isinstance(b, float) or isinstance(b, int))
    
    return w, b

def forward(X, w, b):
    """
    
    Implementa la parte lineal de la propagación hacia adelante

    Arguments:
    X -- Conjunto de datos de un ejemplo de entrada: (1, número de características)
    w -- vector de pesos: numpy array de tamaño (número de características, 1)
    b -- peso teta como escalar

    Returns:
    Z -- nivel de activación de la neurona que se convierte en la entrada a la función de activación. 
    """
    
    ### Haga su código acá ### (≈ 1 line of code) 
    Z = np.dot(X,w) + b #XW + b
    
    ### Fin ###
    
    return Z

def activation_forward(X, w, b):
    
    """ 
    Calcula la función de activación Umbral llamando a la parte lineal de la propagación hacia adelante

    Argumentos:
    X -- Conjunto de datos de un ejemplo de entrada: (número de características, 1)
    w -- vector de pesos: numpy array de tamaño (número de características, 1)
    b -- peso teta como escalar

    Returna:
    y -- umbral(Z)
    """
    
    ### Haga su código acá ### (≈ 1 line of code) 
    
    Z= forward(X, w, b)
    
    ### Fin ###

    if(Z>0):
        Y=1
    else:
        Y=-1
    
    return Y


def compute_error(Y, S):
    """
    Implementa la función de error
    
    Arguments:
    Y -- Salida de la Red, Escalar
    S -- etiqueta del ejemplo de entrenamiento (1 o -1)

    Returns:
    error -- 1 si se equivocó 0 de lo contrario
    """
    
    ### Haga su código acá ### (≈ 4 line of code) 
    
    if (Y != S):
        error = 1
    else:
        error = 0
    
    ### Fin ###
           
    return error

def backward(X, S, error):
    
    """
    Implementa el proceso de aprendizaje actualizando los pesos w y bias (teta)

    Argumentos:
    X -- datos de tamaño (1, número de características)
    Y -- valor de etiqueta "true label" (conteniendo -1 o 1) 

    Return:
    dw -- delta_w, cambio del peso w. Del mismo tamaño que w
    db -- delta_b(teta), cambio del peso b(teta). Escalar
    
    """
    ### Haga su código acá (≈ 6 line of code) 
    if(error==1):
        dw = S*X[np.newaxis].T #w = w + dw; w = s*x.T; dw = s*x.T ('newaxis', Agregamos un nuevo eje para que haga bien la transpuesta )
        db = S                 #b = b + db; db = s
    else:
        dw=np.zeros((len(X),1))
        db=0


    return dw, db


def update_parameters(w, b, dw, db):
    
    """
    Realiza la actualización de los pesos

    Argumentos:
    w -- vector de pesos sin actualizar: numpy array de tamaño (número de características, 1)
    b -- peso teta como escalar sin actualizar   
    dw -- delta_w, cambio del peso w. Del mismo tamaño que w
    db -- delta_b(teta), cambio del peso b(teta). Escalar

    Return:
    w -- vector de pesos actualizado: numpy array de tamaño (número de características, 1)
    b -- peso teta como escalar actualizado
    """
    
    ### Haga su código acá ### (≈ 2 line of code) 
    w = w + dw
    b = b + db
    
    ### Fin ###
       
    return w, b

def prediccion(X, w, b):
    '''
    Predice la etiqueta 1 o -1 para el conjunto de parametros (w, b, X)
    
    Argumentos:
    X -- Conjunto de datos de un ejemplo de entrada: (1, número de características)
    w -- vector de pesos: numpy array de tamaño (número de características, 1)
    b -- peso teta como escalar
    
    Returns:
    Y -- La predicción
    '''

    ### Haga su código acá ### (≈ 1 line of code) 
    
    Y = activation_forward(X, w, b)
    
    ### Fin ###

    return Y

def simplePerceptron_model(X, S, num_iterations = 1000, min_error=0, print_cost=False):
    '''
        X = vector de ejemplos
        S = vector de salida
        m = número de ejemplos de entrenamiento
        costs = Acumulativo de los errores totales de cada iteración
    '''
    
    w, b = inicializar_Parametros(len(X[0,:]))
    m = len(X[:,0])
    errorTotal = 0.0
    costs = []
    contIter = 0

    for iterationNum in range(0, num_iterations):
        
        contIter = iterationNum + 1
        errorTotal = 0.0

        for exampleNum in range(0, m):
            
            #Calculo del nivel de activación y de la función de activación
            Y = activation_forward(X[exampleNum,:], w, b)

            #Calculo del error
            error = compute_error(Y, S[exampleNum])
            errorTotal += error

            #Calculo del aprendizaje
            dw, db = backward(X[exampleNum, :], S[exampleNum], error)

            #Actualizar Parámetros
            w, b = update_parameters(w, b, dw, db)

            error = 0

        #dividimos por m para normalizar
        #Por ejemplo, si se equivoco en todos los ejemplos, errorTotal es igual a m
        #En este caso el error total en esta iteración es 1, 100%
        costs.append(errorTotal/m)

        if(errorTotal/m<=min_error):
            break
    
    if(print_cost == True):
        plt.plot(costs)
        plt.ylabel('cost')
        plt.xlabel('iterations')
        plt.show()
    
    return w, b, costs, contIter

def pruebaModeloAnd(w, b):
    x1 = float(input("digite feature 1 "))
    x2 = float(input("digite feature 2 "))
    Y = prediccion(np.array([x1, x2]), w, b) #AND
    SalidaEsperada = input("digite la salida esperada ")
    print(f'Salida esperada {SalidaEsperada} el modelo predice: {Y} ')
    pruebaModeloAnd(w, b)

def pruebaModeloCancer(w, b):
    x1 = float(input("digite feature 1 "))
    x2 = float(input("digite feature 2 "))
    x3 = float(input("digite feature 3 "))
    x4 = float(input("digite feature 4 "))
    x5 = float(input("digite feature 5 "))
    x6 = float(input("digite feature 6 "))
    x7 = float(input("digite feature 7 "))
    x8 = float(input("digite feature 8 "))
    x9 = float(input("digite feature 9 "))
    
    Y = prediccion(np.array([x1, x2, x3, x4, x5, x6, x7, x8, x9]), w, b) #CANCER
    SalidaEsperada = input("digite la salida esperada ")
    print(f'Salida esperada {SalidaEsperada} el modelo predice: {Y} ')
    pruebaModeloCancer(w, b)

def getFilePathData(opcion):
    filePathBase = "D:/David A/Repositories/DiplomadoDeepLearning/1-Introducción a las Redes Neuronales/"
    fileName = 'trainAND.csv' if (opcion == "1") else 'breastCancer.csv'
    return f'{filePathBase}{fileName}'

def main_function():
    dataProcess = input("digite 1 para AND, 2 para Cancer ")    
    file_path = getFilePathData(dataProcess)
    data = leer_Datos(file_path)
    w, b, costs, contIter = simplePerceptron_model(data[0], data[1], num_iterations = 1000, min_error = 0.02, print_cost = True)
    print(w, b, costs, contIter)
    if(dataProcess == 1):
        pruebaModeloAnd(w, b)
    else:
        pruebaModeloCancer(w, b)    

if(__name__ == '__main__'):
    main_function()




