import numpy as np
import time
import pandas as pd
from sklearn.metrics.pairwise import polynomial_kernel as polykernel
from scipy.spatial.distance import cdist, pdist, squareform
test = pd.read_csv("data/dtest123.dat", delimiter = "  ", header = None, engine = "python")
train = pd.read_csv("data/zipcombo.dat", delimiter = " ", header = None, engine = "python")
train.shape
minitrain = pd.read_csv("data/dtrain123.dat", delimiter = "  ", header = None, engine = "python")
train = np.array(train)
xtrain = train[:,1:257]
ytrain = train[:,0]
one = np.array(minitrain)
xone = one[:, 1:257]
yone = one[:,0]
test = np.array(test)
xtest = test[:, 1:]
ytest = test[:,0]
class kernel_perceptron:
    def __init__(self, N):
        self.N = N
        self.alpha = np.zeros((self.N, 1))
        self.accs = []
        self.w = np.zeros((N, 1))
    def train(self, K, y, d = 2, epochs = 1, alt = True):
        if len(self.alpha) != len(K):
            self.N = len(K)
            self.alpha = np.zeros((len(K), 1))
        for epoch in range(epochs):
            js = np.arange(self.N)
            np.random.shuffle(js)
            for i in js:
                yhat = np.sign(np.sign((K[:,i].dot(self.alpha))))
                if yhat != y[i]:
                    self.alpha[i] += y[i]

    def predict(self, Ktest):
        return Ktest.T.dot(self.alpha).flatten()

class onevsall:
    def __init__(self, xtrain, ytrain, kernel = "poly", karg = "Default"):
        self.N = len(xtrain)
        self.models = []
        self.x = xtrain
        if kernel == "poly":
            degree = karg if karg != "Default" else 2
            self.K = polykernel(xtrain, degree = degree)
            self.K /= self.K.max()
        elif kernel == "rbf":
            c = karg if karg != "Default" else 1
            edist = squareform(pdist(xtrain, metric = "euclidean"))**2
            self.K = np.exp(-c * edist/edist.max())
        else:
            raise ValueError("Invalid Kernel Specified. Must be poly or rbf")
        self.y = ytrain
    
    def train(self):
        self.vals = range(int(max(self.y)+1))
        if len(self.models) == 0:
            self.ytemps = []
            for val in self.vals:
                ytemp = self.y.copy()
                ytemp[ytemp != val] = -1
                ytemp[ytemp == val] = 1
                self.add_model(ytemp)
                self.ytemps.append(ytemp.copy())
        for val in self.vals:
            self.models[val].train(self.K, self.ytemps[val], epochs = 1)

    def add_model(self, ytrain):
        model = kernel_perceptron(self.N)
        self.models.append(model)
    
    def predict(self, x, full = False, test = True):
        ktest = polykernel(self.x, x, degree = 2)
        preds = np.zeros((len(x), len(self.models)))
        for i, model in enumerate(self.models):
            preds[:,i] = model.predict(ktest)
        if full == True:
            return preds
        return preds.argmax(axis = 1)


class onevsone:
    
    def __init__(self, xtrain, ytrain, kernel = "poly", karg = "Default"):
        self.N = len(xtrain)
        self.models = []
        self.x = xtrain
        if kernel == "poly":
            degree = karg if karg != "Default" else 2
            self.K = polykernel(xtrain, degree = degree)
            self.K /= self.K.max()
        elif kernel == "rbf":
            c = karg if karg != "Default" else 1
            edist = squareform(pdist(xtrain, metric = "euclidean"))**2
            self.K = np.exp(-c * edist/edist.max())
        else:
            raise ValueError("Invalid Kernel Specified. Must be poly or rbf")
        self.y = ytrain
            
    
    def train(self, print_accs = True, alt = False):
        self.vals = range(int(max(self.y)+1))
        self.m = int(max(self.y)+1)
        m = self.m
        if len(self.models) == 0:
            self.Ks = [int(max(self.y+1))*[None] for i in range(int(max(self.y+1)))]
            self.ijs = [int(max(self.y+1))*[None] for i in range(int(max(self.y+1)))]
            self.ys = [int(max(self.y+1))*[None] for i in range(int(max(self.y+1)))]
            self.models = [int(max(self.y+1))*[None] for i in range(int(max(self.y+1)))]
            for i in range(0, m):
                for j in range(1+i, m):
                    ij = np.where((self.y==i)|(self.y==j))[0]
                    self.models[i][j] = kernel_perceptron(len(ij))
                    ij = np.where((self.y==i)|(self.y==j))[0]
                    ktemp = self.K[ij, :]
                    ktemp = ktemp[:,ij]
                    ktemp = ktemp.squeeze()
                    ytemp = self.y[ij]
                    ytemp[ytemp == i] = 1
                    ytemp[ytemp == j] = -1
                    self.Ks[i][j] = ktemp.copy()
                    self.ys[i][j] = ytemp.copy()
                    self.ijs[i][j] = ij
        for i in range(0, m):
            for j in range(1+i, m):
                self.models[i][j].train(self.Ks[i][j], self.ys[i][j])

    
    def predict(self, K):
        n = K.shape[1]
        preds = np.zeros((n, self.m, self.m)) 
        for i in range(0, self.m):
            for j in range(i+1, self.m):
                preds[:,i,j] = model.models[i][j].predict(K[self.ijs[i][j],:])
        scores = np.zeros((n, 10))
        for n, pred in enumerate(preds):
            for i in range(10):
                scores[n,i] += -np.sum(pred[:i, i]) + np.sum(pred[i,i:])
        return np.argmax(scores, axis = 1).flatten()


model = onevsall(xtrain, ytrain, "poly", 4)
for i in range(10):
    model.train()
    print(np.mean(model.predict(xtest) == ytest.flatten()))