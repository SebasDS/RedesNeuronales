import numpy as np
import random as rnd

#                               1                             2                             3                                  4                         5                                6                                  7                               8                           9                             10                              11                                12                       13                                14                                  15                      16                            17                                 18                          19                            20                              21                        22                                   23                        24                             25                            26                              27                                28                              29                            30                             31                                 32                           33                                 34                                35                            36                             37                           38                                    39                                     40
DatasetIn = np.array([[3702.62,3702.62,40.29,0.0032],[3702.62,3757.21,39.82,0.0032],[3757.21,3717.25,39.83,0.0032],[3717.25,3700.28,39.37,0.0032],[3700.28,3709.00,39.37,0.0032],[3709.00,3709.00,39.37,0.0032],[3709.00,3709.00,38.96,0.0032],[3709.00,3697.00,39.35,0.0032],[3697.00,3683.49,41.29,0.0032],[3683.49,3703.86,42.07,0.0032],[3703.86,3714.65,42.98,0.0032],[3714.65,3725.37,42.98,0.0032],[3725.37,3725.37,42.98,0.0032],[3725.37,3725.37,41.49,0.0032],[3725.37,3790.54,41.32,0.0032],[3790.54,3813.30,41.40,0.0032],[3813.30,3863.60,41.22,0.0032],[3863.60,3873.80,41.93,0.0032],[3873.80,3867.81,41.93,0.0032],[3867.81,3867.81,41.93,0.0032],[3867.81,3867.81,41.61,0.0032],[3867.81,3859.90,41.46,0.0032],[3859.90,3878.94,40.65,0.0032],[3878.94,3865.47,39.94,-0.0006],[3865.47,3842.34,38.09,-0.0006],[3842.34,3881.80,38.09,-0.0006],[3881.80,3881.80,38.09,-0.0006],[3881.80,3881.80,39.07,-0.0006],[3881.80,3843.75,40.33,-0.0006],[3843.75,3826.77,40.45,-0.0006],[3826.77,3837.79,41.06,-0.0006],[3837.79,3839.73,41.61,-0.0006],[3839.73,3824.25,41.61,-0.0006],[3824.25,3824.25,41.61,-0.0006],[3824.25,3824.25,40.57,-0.0006],[3824.25,3824.25,40.68,-0.0006],[3824.25,3856.32,41.20,-0.0006],[3856.32,3843.59,41.29,-0.0006],[3843.59,3854.47,41.37,-0.0006],[3854.47,3846.48,41.37,-0.0006]])
DatasetOut = np.array([[3757.21],[3717.25],[3700.28],[3709.00],[3709.00],[3709.00],[3697.00],[3683.49],[3703.86],[3714.65],[3725.37],[3725.37],[3725.37],[3790.54],[3813.30],[3863.60],[3873.80],[3867.81],[3867.81],[3867.81],[3859.90],[3878.94],[3865.47],[3842.34],[3881.80],[3881.80],[3881.80],[3843.75],[3826.77],[3837.79],[3839.73],[3824.25],[3824.25],[3824.25],[3824.25],[3856.32],[3843.59],[3854.47],[3846.48],[3846.48]])
Petroleo_Base=40

def tratamientoEntradas():
    x=np.zeros((len(DatasetIn),3))
    for i in range(len(DatasetIn)):
        x[i][0]=(DatasetIn[i][1]-DatasetIn[i][0])/DatasetIn[i][0]
        x[i][1]=DatasetIn[i][2]-Petroleo_Base
        x[i][2]=DatasetIn[i][3]*100
    return x

def tratamientoSalidas():
    y=np.zeros((len(DatasetOut),1))
    for i in range(len(DatasetOut)):
        y[i][0]=(DatasetOut[i][0]-DatasetIn[i][1])/DatasetIn[i][1]
    return y

X = tratamientoEntradas()
Yd = tratamientoSalidas()
epsilon = 1e-15
epocas = 50000
eta = 0.4
capas = 3
Neuronas = np.array([len(X[0]),4,4,len(Yd[0])],np.int)
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
                    self.A3 = self.Z3                 # A3 = F(Z3)             F-> función de activación
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
                    delta3=self.error(Ydi, self.A3)
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
                print("Error medio cuadratico: ", mse)
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
            print("Matriz W Capa 3 -- ", "B: ", B3,"W: ", W3)
        else:
            W2, B2, W1, B1 = nna.train()
        print("Matriz W Capa 2 -- ", "B: ", B2,"W: ", W2)
    else:
        W1, B1 = nna.train()
    print("Matriz W Capa 1 -- ", "B: ", B1,"W: ", W1)
else:
    print("No hay capas")
    
def predecir_dolar(dolarA, dolarH, petroleo, inflacion):
    xp=np.array([[(dolarH-dolarA)/dolarA],[petroleo-Petroleo_Base],[inflacion*100]])
    return (nna.forward(xp)+1)*dolarH

#In[]:

#Para evaluar el dolar de mañana usar la siguiente sintaxis:
#dolar=predecir_dolar(<dolar_ayer>,<dolar_hoy>,<Valor_del_petroleo_OPEP>,<Inflacion>)
#print("Dolar de Mañana: ", dolar[0][0])

#Ejemplo:
dolar1=predecir_dolar(3807.13,3763.82,39.79,-0.0006)
print("Dolar de Mañana: ",dolar1[0][0]) #Reultado esperado $3738.19

#Ejemplo de resultado:
# Epoca:  40184
# Error medio cuadratico:  3.1965641853214597e-18
# Matriz W Capa 3 --  B:  [[-0.16727844]] W:  [[-0.13129669]
#  [ 0.06892905]
#  [ 0.15090598]
#  [ 0.2443278 ]]
# Matriz W Capa 2 --  B:  [[0.29354281]
#  [0.46461715]
#  [0.11554135]
#  [0.55740059]] W:  [[ 0.72539613 -0.0057271  -0.6285104  -0.34168151]
#  [ 0.84283027  0.81343397  0.11073702  0.31861157]
#  [ 0.41199395  0.1909925   0.62418098 -0.66925566]
#  [ 0.68942662  0.27681046  0.6159709  -0.39525855]]
# Matriz W Capa 1 --  B:  [[-0.48982472]
#  [ 0.91743001]
#  [-0.58148838]
#  [ 0.70835093]] W:  [[-0.56783039  0.68986402  0.14038063  0.18757074]
#  [ 0.42337564  0.18228173 -0.72689974 -0.90114643]
#  [-0.51017277 -0.66055332  0.38583574 -0.73790893]]
# Dolar de Mañana:  3772.8361966985576