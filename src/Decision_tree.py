from sklearn import tree
from sklearn import metrics

class Decision_tree:
    crit = ""
    X_train = None
    y_train = None
    X_test = None
    y_test = None
    tree = None

    def __init__(self, crit, X_train, y_train, X_test, y_test):
        self.crit = crit
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test

    def treina_arvore(self):
        self.tree = tree.DecisionTreeClassifier(criterion=self.crit)
        self.tree = self.tree.fit(self.X_train, self.y_train)

    def previsao(self):
        result = self.tree.predict(self.X_test)
        return result

    def precisao(self):
        acc = metrics.accuracy_score(self.previsao(), self.y_test)
        show = round(acc * 100)   
        return show

     
    
    

