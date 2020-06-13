import matplotlib.pyplot as plt
import numpy as np
import cv2
import pandas as pd


def leer_Datos(file='trainAdaline.csv'):

    data= pd.read_csv(file, header = None)
    
    X= data.iloc[:,1:].values
    S= data.iloc[:,0].values
    S= S[:, np.newaxis]

    return X,S
            

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
    
    w = np.random.randn(dim, 1)
    b = 0

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
    
#    ### Haga su código acá ### (≈ 1 line of code) 
    
    Z =  np.dot(X,w)+b  # usar np.dot()
    
    ### Fin ###
    
    return Z

def activation_forward(X, w, b):
    
    """ 
    Calcula la función de activación lineal del Adaline Y= k*Z donde K=1, llamando a la parte lineal de la propagación hacia adelante

    Argumentos:
    X -- Conjunto de datos de un ejemplo de entrada: (1,número de características)
    w -- vector de pesos: numpy array de tamaño (número de características, 1)
    b -- peso teta como escalar

    Returna:
    Y -- k*(Z) donde K=1
    """
    
    K=1
    
    ### Haga su código acá ### (≈ 2 line of code) 
    
    Z= forward(X, w, b)
   
    Y= K*Z
    
    ### Fin ###
    
    return Y


def compute_error(Y, S):
    """
    Implementa la función del error cuadrático medio
    
    Arguments:
    Y -- Salida de la Red, Escalar
    S -- Salida del ejemplo de entrenamiento, Escalar

    Returns:
    error -- error cuadrático medio
    """
    
    ### Haga su código acá ### (≈ 1 line of code) 
    
    error=((S-Y)**2)/2
    
    ### Fin ###
       
    return error


def backward(X, Y, S, f):
    
    """
    Implementa el proceso de aprendizaje de la regla Delta actualizando los pesos w y bias (teta)

    Argumentos:
    X -- datos de tamaño (1, número de características)
    Y -- valor de salida Escalar 

    Return:
    dw -- delta_w, cambio del peso w. Del mismo tamaño que w
    db -- delta_b(teta), cambio del peso b(teta). Escalar
    
    """
    ### Haga su código acá ### (≈ 2 line of code) 
    
    dw = f*(S-Y)*X[np.newaxis].T
    db = f*(S-Y)
    
    ### Fin ###

    return dw, db

#######################################################


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
    
    w= w + dw
    b= b + db
    
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


# def adaline_model(X, S, num_iterations = 10000, min_error=0, print_cost=False):
        
#     w,b=inicializar_Parametros(len(X[0]))
        
#     errorTotal=0.0
    
#     m=len(X[0:,])
    
#     costs=[]
    
#     contIter=0
    
#     #
#     for i in range(0, num_iterations):
        
#         contIter+=1
#         errorTotal=0.0
        
#         for i in range(0,m):
            
#             # Calculo del nivel de Activación y de la función de Activación
            
#             Y=activation_forward(X[i,:], w, b)
            
#             # Calculo de Error
            
#             error=compute_error(Y, S[i])
# #            
#             errorTotal+=error
# #            
# #            # Calculo del aprendizaje con factor de aprendizaje 0.01
# #            
#             dw, db= backward(X[i,:], Y, S[i], 0.01)
# #                
# #            # Actualizar Parámetros
# #            
#             w, b= update_parameters(w, b, dw, db)
# #            
#             error=0
            
#         costs.append(errorTotal)
         
#         if (errorTotal<=min_error):
#             break
        
#     if(print_cost== True):
#         plt.plot(costs)
#         plt.ylabel('cost')
#         plt.xlabel('iterations')
#         plt.show()
        
#     return w, b, contIter
    


# #######################   Entrenar el modelo ##################################

# X, S=leer_Datos()
        
# w, b, contIter = adaline_model(X, S, num_iterations = 1000, min_error=0.00001, print_cost=True)
      
# print("número de Iteraciones",contIter)
# print("w:",w)
# print("b:",b)

#######################   Predicción ##########################################
#
#my_predicted=prediccion(np.array([1,0,1]),w,b) # 5 
#
#my_value_y= 5
#
#print ("y = " + str(np.squeeze(my_value_y)) + ", your model predicts a \"" + str(int(np.squeeze(np.around(my_predicted, decimals=0)))))


