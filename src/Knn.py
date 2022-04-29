from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier

class Knn:
    mtr = ""
    ngh = None
    X_train = None
    y_train = None
    X_test = None
    y_test = None
    knn = None

    def __init__(self,mtr, ngh, X_train, y_train, X_test, y_test):
        self.mtr = mtr
        self.ngh = ngh
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test

    def treina_knn(self):
        self.knn = KNeighborsClassifier(n_neighbors=self.ngh, metric=self.mtr, algorithm='brute')
        self.knn = self.knn.fit(self.X_train, self.y_train)

    def previsao(self):
        result = self.knn.predict(self.X_test)
        return result

    def precisao(self):
        acc = metrics.accuracy_score(self.previsao(), self.y_test)
        show = round(acc * 100)   
        return f"{show}%"