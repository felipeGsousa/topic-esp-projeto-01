def create_title(y_test):
    str_title = f"""##########################= Conjunto y teste =###########################

["""
    c = 0
    for i in range(len(list(y_test))):
        str_title += str(list(y_test)[i]) + " "
        c += 1
        if c == 37:
            str_title += "\n"
            str_title += " "
            c = 0
    str_title = str_title.rstrip() + "]"
    return str_title
    
def decision_tree_result(show, result, criterion):
    str_tree = f"""\n
##########################= Arvore de decisao =###########################

-Criterion = {criterion}

-Porcentagem: {show}%

-Resultado: 
{result}"""
    return str_tree


def knn_result(show, result, metric, neigh_number):
    str_knn = f"""\n
##############################=   KNN   =##################################

Metric = {metric}

Vizinhanca = {neigh_number}

Porcentagem: {show}%

Resultado: 
{result}"""
    return str_knn

def knn_improve_result(show, result, neigh_number):

    str_result = "["

    c = 0
    for i in range(len(result)):
        str_result += str(result[i]) + " "
        c += 1
        if c == 37:
            str_result += "\n"
            str_result += " "
            c = 0
    str_result = str_result.rstrip() + "]"

    str_knn_improve = f"""\n
##########################=   KNN Improve   =##############################

Vizinhanca = {neigh_number}

Porcentagem: {show}%

Resultado: 
{str_result}"""
    return str_knn_improve
