#in1

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import random as rnd
%matplotlib inline

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
#print(os.listdir("../demo"))

#in2
#Descomprimir el archivo mnist_train.rar dentro de la carpeta Red_Neuronal_MNIST junto a este archivo .py
mnist = pd.read_csv('../Red_Neuronal_MNIST/mnist_train.csv')#Si se usa otra ubicacion cambiar el nombre de la carpeta a donde este el archivo csv

#in3

dataTrain = 60000

DatasetIn = np.array(mnist.iloc[0:dataTrain,1:])
DatasetOut = np.array(mnist.iloc[0:dataTrain,0])

def tratamientoEntradas():
    return DatasetIn/128-1
def tratamientoSalidas():
    y=np.zeros((len(DatasetOut),10))
    for i in range(len(DatasetOut)):
        for j in range(10):
            if DatasetOut[i]==j:
                y[i][j]=1
            else:
                y[i][j]=0.01
    return y

X = tratamientoEntradas()
Yd = tratamientoSalidas()

epsilon = 1e-9
epocas = 1000
eta = 0.35
capas = 3
Neuronas = np.array([len(X[0]),256,128,len(Yd[0])],np.int)
class NNA(object):
    def __init__(self):

        if capas>0:
            self.b1 = self.buildB(Neuronas[1])
            self.W1 = self.buildW(Neuronas[1], Neuronas[0])
            self.Z1 = np.zeros((Neuronas[1],1))
            self.A1 = np.zeros((Neuronas[1],1))
            if capas>1:
                self.b2 = self.buildB(Neuronas[2])
                self.W2 = self.buildW(Neuronas[2], Neuronas[1])
                self.Z2 = np.zeros((Neuronas[2],1))  
                self.A2 = np.zeros((Neuronas[2],1))
                if capas>2:
                    self.b3 = self.buildB(Neuronas[3])
                    self.W3 = self.buildW(Neuronas[3], Neuronas[2])
                    self.Z3 = np.zeros((Neuronas[3],1))  
                    self.A3 = np.zeros((Neuronas[3],1))
        
        self.SSE = 0
    def buildB(self, n):
        b=np.zeros((n,1))
        for i in range(n):
            b[i][0]=2*(rnd.random()-0.5)
        return b
    def buildW(self, n, n_ant):
        w=np.zeros((n_ant,n))
        for i in range(n_ant):
            for j in range(n):
                w[i][j]=2*(rnd.random()-0.5)
        #print(w)
        return w
    def forward(self, Yo):                      # Función de propagación hacia adelante
        if capas>0:
            self.Z1 = np.dot(self.W1.T,Yo) + self.b1      # Z1 = W1.Yo + W10
            #print("Z1 =", self.Z1)
            self.A1 = self.sig(self.Z1)                 # A1 = F(Z1)               
            #print("A1 =", self.A1)
            if capas>1:
                self.Z2 = np.dot(self.W2.T,self.A1) + self.b2 # Z2 = W2.Yo + W20
                #print("Z2 =", self.Z2)
                self.A2 = self.sig(self.Z2)                 # A2 = F(Z2)             F-> función de activación
                #print("A2 =", self.A2)
                if capas>2:
                    self.Z3 = np.dot(self.W3.T,self.A2) + self.b3 # Z3 = W3.Yo + W30
                    #print("Z3 =", self.Z3)
                    self.A3 = self.sig(self.Z3)                 # A3 = F(Z3)             F-> función de activación
                    #print("A3 =", self.A3)
                    return self.A3
                return self.A2
            return self.A1
        return 0
    def error(self, Ydi, aN):
        #print("error", (Yd - aN))
        # Calculo del error de la salida respecto a los valores deseados
        return Ydi - aN
    def errCuad(self, Ydi, aN): # Calculo del valor cuadrático
        err = self.error(Ydi, aN)
        #print("error cuadrado: ", np.sum(err**2)/2)
        return np.sum(err**2)/2, err
    def backpropagation(self, Xi, Ydi): # Retropropagación
        if capas>0:
            if capas>1:
                if capas>2:
                    # if siguiente capa else
                    delta3=np.dot(self.dSig(self.Z3),self.error(Ydi, self.A3))
                    dEdW3 = np.dot(delta3,self.A2.T)                             # componente de corrección para W2
                    #print("dEdW3",dEdW3)
                    dEdb3 = delta3
                    #print("dEdb3",dEdb3)
                    delta2 = np.dot(np.dot(self.dSig(self.Z2),self.W3),delta3)
                else:
                    delta2=np.dot(self.dSig(self.Z2),self.error(Ydi, self.A2))
                dEdW2 = np.dot(delta2,self.A1.T)                             # componente de corrección para W2
                #print("dEdW2",dEdW2)
                dEdb2 = delta2
                #print("dEdb2",dEdb2)
                #print(self.dSig(self.Z1),self.W2.T, delta2)
                delta1 = np.dot(np.dot(self.dSig(self.Z1),self.W2),delta2)
            else:
                delta1 = np.dot(self.dSig(self.Z1),self.error(Ydi, self.A1))
            dEdW1 = np.dot(delta1,Xi.T)
            #print("dEdW1",dEdW1)# componente de corrección para W3
            dEdb1 = delta1
            #print("delta 1",delta1)
        if capas>0:
            if capas>1:
                if capas>2:
                    return dEdW3, dEdb3, dEdW2, dEdb2, dEdW1, dEdb1
                return dEdW2, dEdb2, dEdW1, dEdb1
            return dEdW1, dEdb1
        return 0      
    def training(self):        # Entrenamiento de la red
        
        i=0
        errCua=1
        self.SSE = 0
        if capas>0:
            self.dW1 = np.zeros((Neuronas[0], Neuronas[1]))
            self.dB1 = np.zeros((Neuronas[1], 1))
            if capas>1:
                self.dW2 = np.zeros((Neuronas[1], Neuronas[2]))
                self.dB2 = np.zeros((Neuronas[2], 1))
                if capas>2:
                    self.dW3 = np.zeros((Neuronas[2], Neuronas[3]))
                    self.dB3 = np.zeros((Neuronas[3], 1))
        #         for _ in range(5000):
        ind = np.zeros(len(X),np.int)
        for i in range(len(X)):
            ind[i]=i
        
        while i < len(X): # Criterio de parada
            rnd.shuffle(ind)
            indx=ind[i]
            Xi = np.array([X[indx]]).T
            Ydi = np.array([Yd[indx]]).T
            #print(Xi,Ydi)
            aN = self.forward(Xi)
            errCua, err = self.errCuad(Ydi, aN)
            #print(i," -- ","error cuadrado: ", errCua, "Salida: ", aN)
            if capas>0:
                if capas>1:
                    if capas>2:
                        #if siguiente capa else
                        dEW3, dEB3, dEW2, dEB2, dEW1, dEB1 = self.backpropagation(Xi, Ydi)
                        self.dW3 = self.dW3 + dEW3.T
                        self.dB3 = self.dB3 + dEB3
                    else:
                        dEW2, dEB2, dEW1, dEB1 = self.backpropagation(Xi, Ydi)
                    self.dW2 = self.dW2 + dEW2.T
                    self.dB2 = self.dB2 + dEB2
                else:
                    dEW1, dEB1 = self.backpropagation(Xi, Ydi)
                self.dW1 = self.dW1 + dEW1.T
                self.dB1 = self.dB1 + dEB1
            self.SSE += errCua
            i+=1
        #print(ind)
        if capas>0:
            if capas>1:
                if capas>2:
                    return self.dW3, self.dB3, self.dW2, self.dB2, self.dW1, self.dB1, self.SSE/i
                return self.dW2, self.dB2, self.dW1, self.dB1, self.SSE/i
            return self.dW1, self.dB1, self.SSE/i
        return 0
    def dSig(self, s):
        df = self.sig(s.T[0])*(1-self.sig(s.T[0]))
        ds = np.diag(df)
        return ds
    def sig(self, s):
        return 1/(1+np.exp(-s))
    
    def train(self):
        mse = 1
        mse_ant = 10
        
        ep = 0
        while abs(mse-mse_ant) > epsilon**2 and mse > epsilon and ep<=epocas:
            mse_ant = mse
            deta = 1
            if capas>0:
                if capas>1:
                    if capas>2:
                        #if siguiente capa else
                        dw3, db3, dw2, db2, dw1, db1, mse = self.training()
                        self.W3 = self.W3 + eta*dw3
                        self.b3 = self.b3 + eta*db3
                    else:
                        dw2, db2, dw1, db1, mse = self.training()
                    self.W2 = self.W2 + eta*dw2
                    self.b2 = self.b2 + eta*db2
                else:
                    dw1, db1, mse = self.training()
                self.W1 = self.W1 + eta*dw1
                self.b1 = self.b1 + eta*db1
            if abs(mse-mse_ant) < epsilon**2 or mse < epsilon or ep==epocas:
                print("Epoca: ", ep)
                print("Error cuadratico medio: ", mse)
            ep += 1
        if capas>0:
            if capas>1:
                if capas>2:
                    return self.W3, self.b3, self.W2, self.b2, self.W1, self.b1
                return self.W2, self.b2, self.W1, self.b1
            return self.W1, self.b1
        return 0
nna = NNA()
if capas>0:
    if capas>1:
        if capas>2:
            #if siguiente capa else
            W3, B3, W2, B2, W1, B1 = nna.train()
            #print("Matriz W Capa 3 -- ", "B: ", B3,"W: ", W3)
        else:
            W2, B2, W1, B1 = nna.train()
        #print("Matriz W Capa 2 -- ", "B: ", B2,"W: ", W2)
    else:
        W1, B1 = nna.train()
    #print("Matriz W Capa 1 -- ", "B: ", B1,"W: ", W1)
else:
    print("No hay capas")
    
def leer_numero(grafo):
    numero=np.array([grafo]).T
    original_image=np.array(grafo).reshape(28,28)
    fig, axes = plt.subplots(nrows=1, ncols=1)
    axes.imshow(original_image, cmap ='gist_gray')
    fig.tight_layout()
    output = nna.forward(numero/128-1)
    maximo = output[0][0]
    Rta = 0
    print(0,": ", output[0][0])
    for y in range(9):
        print(y+1,": ", output[y+1][0])
        if maximo < output[y+1][0]:
            maximo = output[y+1][0]
            Rta = y+1
    return Rta
print("Red Entrenada!")

#in4
#Si se quiere revisar un Digito en especifico del dataset reemplazar "rnd.randint(0, dataTrain-1)"
#por el numero de la fila que tenga en el dataset dicho Digito
Resultado=leer_numero(mnist.iloc[rnd.randint(0, dataTrain-1),1:])
print("Respuesta: ", Resultado)
