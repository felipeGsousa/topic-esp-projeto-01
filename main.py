from sklearn import tree
from sklearn.model_selection import train_test_split
import pandas as pd

from src.Decision_tree import Decision_tree
from src.Knn import Knn
from src.Knn_improve import Knn_improve
import src.Result as Result

url = "database/abalone.data"
results_tree = open("results/results_tree.txt","w+")
results_knn= open("results/results_knn.txt","w+")
results_knn_improve = open("results/results_knn_improve.txt","w+")


dataset = pd.read_csv(url, header=None)
dataset = dataset.sort_values(by=0, axis=0, ascending=True, inplace=False, kind='quicksort', na_position='last', ignore_index=True, key=None)

columns = len(dataset.columns)

y = dataset[0]
X = dataset.loc[:,1:columns-1]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state=None, stratify=y)

results_tree.write(Result.create_title(y_test))
results_knn.write(Result.create_title(y_test))
results_knn_improve.write(Result.create_title(y_test))

#Decision Tree

crit_1 = "gini"
tree_1 = Decision_tree(crit_1,X_train,y_train,X_test,y_test)
tree_1.treina_arvore()
results_tree.write(Result.decision_tree_result(tree_1.precisao(), tree_1.previsao(), crit_1))

crit_2 = "entropy"
tree_2 = Decision_tree(crit_2,X_train,y_train,X_test,y_test)
tree_2.treina_arvore()
results_tree.write(Result.decision_tree_result(tree_2.precisao(), tree_2.previsao(), crit_2))

#KNN

n_1 = 9
n_2 = 19
n_3 = 130

metric_1 = 'manhattan'
metric_2 = 'minkowski'

knn_1 = Knn(metric_1,n_1,X_train,y_train,X_test,y_test)
knn_1.treina_knn()
results_knn.write(Result.knn_result(knn_1.precisao(), knn_1.previsao(), metric_1, n_1))

knn_2 = Knn(metric_1,n_2,X_train,y_train,X_test,y_test)
knn_2.treina_knn()
results_knn.write(Result.knn_result(knn_2.precisao(), knn_2.previsao(), metric_1, n_2))

knn_3 = Knn(metric_1,n_3,X_train,y_train,X_test,y_test)
knn_3.treina_knn()
results_knn.write(Result.knn_result(knn_3.precisao(), knn_3.previsao(), metric_1, n_3))

knn_4 = Knn(metric_2,n_1,X_train,y_train,X_test,y_test)
knn_4.treina_knn()
results_knn.write(Result.knn_result(knn_4.precisao(), knn_4.previsao(), metric_2, n_1))

knn_5 = Knn(metric_2,n_2,X_train,y_train,X_test,y_test)
knn_5.treina_knn()
results_knn.write(Result.knn_result(knn_5.precisao(), knn_5.previsao(), metric_2, n_2))

knn_6 = Knn(metric_2,n_3,X_train,y_train,X_test,y_test)
knn_6.treina_knn()
results_knn.write(Result.knn_result(knn_6.precisao(), knn_6.previsao(), metric_2, n_3))

#Knn Improve

n_4 = 86
n_5 = 302
n_6 = 250

train_X = X_train.values.tolist()
train_y = y_train.values.tolist()
test_X = X_test.values.tolist()
test_y = y_test.values.tolist()

knn_improve_1 = Knn_improve(n_4, train_X, train_y, test_X, test_y)
raios = knn_improve_1.calcular_raios()
result_knn_1 = knn_improve_1.previsao(raios)
results_knn_improve.write(Result.knn_improve_result(knn_improve_1.precisao(result_knn_1), result_knn_1, n_4))

knn_improve_2 = Knn_improve(n_5, train_X, train_y, test_X, test_y)
result_knn_2 = knn_improve_2.previsao(raios)
results_knn_improve.write(Result.knn_improve_result(knn_improve_2.precisao(result_knn_2), result_knn_2, n_5))

knn_improve_3 = Knn_improve(n_6, train_X, train_y, test_X, test_y)
result_knn_3 = knn_improve_3.previsao(raios)
results_knn_improve.write(Result.knn_improve_result(knn_improve_3.precisao(result_knn_3), result_knn_3, n_6))


results_tree.close()
results_knn.close()
results_knn_improve.close()