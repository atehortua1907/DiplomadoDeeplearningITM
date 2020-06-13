import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def getDataFromCovid(filePath, geoId = 'CO'):
    data= pd.read_csv(filePath) #Load DataBase
    dataFrame = data.loc[data['geoId'] == geoId]
    dataFrame = dataFrame.sort_values(['month','day'], ascending=[True, True])
    return dataFrame



def getFeaturesAndTarget(dataFrame):

    X = dataFrame.iloc[:-1,4:6].values #Botamos el último dato
    S = dataFrame.iloc[1:,5].values #Obtenemos las muertes desde la segunda hasta la última
    return X, S

def getMaxValues(X, S):
    maxCases = np.amax(X[:,0])
    maxDeath = np.amax(X[:,1])
    maxOutPut = np.amax(S)

    return maxCases, maxDeath, maxOutPut

def normalizeData(X, S, maxCases, maxDeath, maxOutPut):
    X[:,0] = X[:,0]/maxCases
    X[:,1] = X[:,1]/maxDeath
    S = S/maxOutPut

    return X, S

def mainFunction():
    file_path = 'D:/David A/Repositories/DiplomadoDeepLearningITM/1-Introducción a las Redes Neuronales/ActividadCovid/Covid.csv'
    ColombianData = getDataFromCovid(file_path, 'CO')
    X, S = getFeaturesAndTarget(ColombianData)
    maxCases, maxDeath, maxOutPut = getMaxValues(X, S)
    X, S = normalizeData(X, S, maxCases, maxDeath, maxOutPut)
    data = 'dada'

if (__name__ == '__main__'):
    mainFunction()