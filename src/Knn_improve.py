import math
from typing import Counter
import numpy
from sklearn import metrics


class Knn_improve:
    ngh = None
    X_train = None
    y_train = None
    X_test = None
    y_test = None
    knn = None

    def __init__(self, ngh, X_train, y_train, X_test, y_test):
        self.ngh = ngh
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test

    def calcular_raios(self):
        e = 1e-20
        raios = []

        for i in range(len(self.X_train)):
            new_data = self.X_train.copy()
            new_data.pop(i)
            new_data_y = self.y_train.copy()
            new_data_y.pop(i)
            results = []

            for j in range(len(new_data)):
                r = 0

                for k in range(len(self.X_train[i])):
                    r += (self.X_train[i][k] - new_data[j][k]) ** 2

                results.append(math.sqrt(r))
        
            indexes = numpy.argsort(results)
            aux = 0

            while self.y_train[i] == new_data_y[indexes[aux]]:
                aux += 1

            raios.append(results[indexes[aux]] - e)

        return raios

    def treina_knn_improve(self,test ,ngh, raios):
        results = []

        for i in range(len(self.X_train)):
            r = 0

            for j in range(len(test)):
                r += (test[j] - self.X_train[i][j]) ** 2
            
            results.append(math.sqrt(r)/raios[i])
        
        indexes = numpy.argsort(results)
        indexes = indexes [0:ngh]
        res = [self.y_train[i] for i in indexes]
        final = Counter(res)

        return final.most_common(1)[0][0]

    def previsao(self, raios):
        result = []
        for i in range(len(self.X_test)):
            classe = self.treina_knn_improve(self.X_test[i], self.ngh, raios)
            result.append(classe)

        return result 

    def precisao(self, result):
        acc = metrics.accuracy_score(result, self.y_test)
        show = round(acc * 100)
        return show
