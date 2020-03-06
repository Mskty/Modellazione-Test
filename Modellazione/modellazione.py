import random

import joblib
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.preprocessing import StandardScaler, MinMaxScaler

from Persone_Fisiche.funzioni import *
from sklearn.utils import shuffle
from sklearn.metrics import make_scorer, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.tree import DecisionTreeClassifier

from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import NearMiss

"""
In questo file verranno elaborati i dati forniti dal dataset in forma definitiva allo scopo di trovare il modello 
predittivo migliore. La forma degli esempi non verrà modificata, ma sarà divisio il set inziale in training e test set,
allo scopo di valutare le performance dei vari modelli utilizzati. La feature colonna contenente la label di classse 
per ogni esempio è chiamata 'label'.
I modelli proposti includono i seguenti algoritmi forniti dalla libreria scikit-lean:
1. LogisticRegressorClassfier
2. SupportVectorMachineClassifier
3. LinearSupportVectorMachineClassifier
4. DecisionTreeClassifier
5. RandomForestClassifier
6. ExtremeGradientBoostingClassifier (xgboost)
7. NeuralNetwork MultiLayerPerceptronClassifier
"""

"""
------------------------------------------FUNZIONI----------------------------------------------------------------------
"""


def display_all_scores(scores):
    """
    Utilizza la dunzione display_scores per stampare l'array di risultati presenti in ogni key di scores
    :param scores: dictionary contenente array numerici di scores ritornati da un processo di cross_validation
    :return:
    """
    keys = scores.keys()
    for key in keys:
        print(key, ":")
        display_scores(scores[key])
        print("")  # Linea bianca


def cross_validation_scores(classifiers, X_train, y_train, fold=5, metric="f1", sample="None"):
    """
    Esegue cross validation e stampa i risultati per tutti i classificatori passati all'interno di classifiers
    :param classifiers:
    :param X_train:
    :param y_train:
    :param fold:
    :param metric: se sample == "None" utilizza la o le metriche specificate nella cross_validate
    :param sample: indica che tecnica di sampling è utilizzata, default nessuna, altrimenti i valori sono "SMOTE" o "NearMiss"
    :return:
    """
    if sample == "None":
        for classifier in classifiers:
            print(classifier.__class__.__name__, ": ")
            print(" ")
            display_all_scores(cross_validate(classifier, X_train, y_train, cv=fold, scoring=metric))
            print("\n")
    elif sample == "SMOTE":
        for classifier in classifiers:
            print(classifier.__class__.__name__, ": ")
            print(" ")
            display_all_scores(smote_cross_validation(classifier, X_train, y_train))
            print("\n")
    elif sample == "NearMiss":
        for classifier in classifiers:
            print(classifier.__class__.__name__, ": ")
            print(" ")
            display_all_scores(nearmiss_cross_validation(classifier, X_train, y_train))
            print("\n")


def smote_cross_validation(classifier, X, y, fold=5):
    """
    Esegue cross validation applicando SMOTE oversampling sui training folds
    Ritorna 5 diverse scoring metrics sui test set
    :param classifier:
    :param X:
    :param y:
    :param fold:
    :return:
    """

    kf = StratifiedKFold(n_splits=fold)
    accuracy = np.array([])
    precision = np.array([])
    recall = np.array([])
    f1 = np.array([])
    roc_auc = np.array([])
    scores = {'No Results': np.zeros(1)}
    for train_index, test_index in kf.split(X, y):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        X_train_oversampled, y_train_oversampled = SMOTE(random_state=42).fit_sample(X_train, y_train)
        classifier.fit(X_train_oversampled, y_train_oversampled)
        y_pred = classifier.predict(X_test)
        accuracy = np.append(accuracy, accuracy_score(y_test, y_pred))
        precision = np.append(precision, precision_score(y_test, y_pred))
        recall = np.append(recall, recall_score(y_test, y_pred))
        f1 = np.append(f1, f1_score(y_test, y_pred))
        roc_auc = np.append(roc_auc, roc_auc_score(y_test, y_pred))
        scores = {'test_accuracy': accuracy,
                  'test_precision': precision,
                  'test_recall': recall,
                  'test_f1_score': f1,
                  'test_roc_auc': roc_auc}
    return scores


def nearmiss_cross_validation(classifier, X, y, fold=5):
    """
    Esegue cross validation applicando NearMiss oversampling sui training folds
    Ritorna 5 diverse scoring metrics sui test set
    :param classifier:
    :param X:
    :param y:
    :param fold:
    :return:
    """
    kf = StratifiedKFold(n_splits=fold)
    accuracy = np.array([])
    precision = np.array([])
    recall = np.array([])
    f1 = np.array([])
    roc_auc = np.array([])
    scores = {'No Results': np.zeros(1)}
    for train_index, test_index in kf.split(X, y):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        X_train_oversampled, y_train_oversampled = NearMiss(random_state=42).fit_sample(X_train, y_train)
        classifier.fit(X_train_oversampled, y_train_oversampled)
        y_pred = classifier.predict(X_test)
        accuracy = np.append(accuracy, accuracy_score(y_test, y_pred))
        precision = np.append(precision, precision_score(y_test, y_pred))
        recall = np.append(recall, recall_score(y_test, y_pred))
        f1 = np.append(f1, f1_score(y_test, y_pred))
        roc_auc = np.append(roc_auc, roc_auc_score(y_test, y_pred))
        scores = {'test_accuracy': accuracy,
                  'test_precision': precision,
                  'test_recall': recall,
                  'test_f1_score': f1,
                  'test_roc_auc': roc_auc}
    return scores


def multiple_confusion_matrix(classifiers, X_train, y_train, X_test, y_test):
    """
    Fitta ogni classificatore in classifiers e stampa la confusion matrix, calcolata su
    X_test e y_test, per ogni classificatore.
    :param classifiers: lista di classificatori non fittati
    :param X_train:
    :param y_train:
    :param X_test:
    :param y_test:
    :return:
    """
    for classifier in classifiers:
        print(classifier.__class__.__name__, ": ")
        classifier.fit(X_train, y_train)
        print(confusion_matrix(y_test, classifier.predict(X_test)))
        print(classification_report(y_test, classifier.predict(X_test)))
        print("\n")


def trainset_feature_scaling(trainset: pd.DataFrame, minmax=False):
    """
    Utilizza uno StandardScaler o MinMaxScaler per normalizzare o standardizzare il trainset
    :param trainset: trainset pulito su cui fittare lo scaler e che verrò trasformato
    :param minmax: se True utilizza MinMaxScaler al posto di StandardScaler
    :return: trainset con le colonne non categoriche scalate e scaler
    """
    trainset = trainset.copy()
    # Rimozione features categoriche da non scalare
    categorical_trainset = trainset[["Telefono", "Deceduto", "CittadinanzaItaliana", "Estero", "NuovoContribuente", "label"]]
    trainset.drop(columns=["Telefono", "Deceduto", "CittadinanzaItaliana", "Estero", "NuovoContribuente", "label"], inplace=True)
    if minmax == False:
        # StandardScaler
        scaler = StandardScaler()
    else:
        # MinMaxScaler
        scaler = MinMaxScaler()
    scaled_features = scaler.fit_transform(trainset.values)
    scaled_trainset = pd.DataFrame(scaled_features, index=trainset.index, columns=trainset.columns)
    # Concatenazione scalati e categorici nel dataset originale
    scaled_trainset = pd.concat([scaled_trainset, categorical_trainset], axis=1, sort=False)
    # Salvo lo scaler su file con joblib
    joblib.dump(scaler, 'scaler.pkl')

    return scaled_trainset, scaler


def feature_scaling(data: pd.DataFrame, scaler) -> pd.DataFrame:
    """
    Utilizza lo scaler passato già fittato (sul training set precedentemente)  per normalizzare o standardizzare il DataFrame passato
    :param data:
    :param scaler: StandardScaler o MinMaxScaler
    :return:
    """
    data = data.copy()
    categorical_data = data[["Telefono", "Deceduto", "CittadinanzaItaliana", "Estero", "NuovoContribuente", "label"]]
    data.drop(columns=["Telefono", "Deceduto", "CittadinanzaItaliana", "Estero", "NuovoContribuente", "label"], inplace=True)
    scaled_features = scaler.transform(data.values)
    scaled_data = pd.DataFrame(scaled_features, index=data.index, columns=data.columns)
    # Concatenazione scalati e categorici nel dataset originale
    scaled_data = pd.concat([scaled_data, categorical_data], axis=1, sort=False)
    return scaled_data


def random_feature_selection(trainset: pd.DataFrame, testset: pd.DataFrame = None, test=False):
    """
    Ritorna X Y sia train che test senza un numero randomico di features
    :param trainset:
    :param testset:
    :param test: se True ritorna anche il test set, False utilizzato solo sul training (per cross validation)
    :return:
    """

    remove_n = random.randint(1, 10)
    columns = np.random.choice(trainset.drop(columns="label").columns.values, remove_n, replace=False)
    print("Feature rimosse:", columns)
    print("Feature rimaste:", trainset.drop(columns="label").drop(columns=columns).columns.values)
    if test == True:
        X_train, Y_train, X_test, Y_test = separazione_label(trainset.drop(columns=columns), testset.drop(columns=columns), test=True)
        return X_train, Y_train, X_test, Y_test
    else:
        X_train, Y_train = separazione_label(trainset.drop(columns=columns))
        return X_train, Y_train


def column_feature_selection(trainset: pd.DataFrame, columns, testset: pd.DataFrame = None, test=False):
    """
    Ritorna X Y sia train che test con le sole feature indicate + label
    :param trainset:
    :param testset:
    :param testset: array contenente i titoli delle colonne da mantenere
    :param test: se True ritorna anche il test set, False utilizzato solo sul training (per cross validation)
    :return:
    """
    columns = np.array(columns)
    columns = np.append(columns, 'label')
    print("Feature rimaste:", columns)
    if test == True:
        X_train, Y_train, X_test, Y_test = separazione_label(trainset[columns], testset[columns], test=True)
        return X_train, Y_train, X_test, Y_test
    else:
        X_train, Y_train = separazione_label(trainset[columns])
        return X_train, Y_train


def model_feature_selection(trainset: pd.DataFrame, model, testset: pd.DataFrame = None, test=False):
    """
    Valuta l'importanza delle feature secondo il modello passato, ritorna una lista delle colonne corrispondente alle features valutate più importanti
    :param trainset:
    :param testset:
    :param model: Modello che valuterà le features, deve supportare la funzionalità di feature_importance
    :param test:
    :return:
    """
    model_select = SelectFromModel(model, threshold='median')
    X_train, Y_train = separazione_label(trainset)
    X_trans = model_select.fit_transform(X_train, Y_train)
    print("We started with {0} features but retained only {1} of them!".format(X_train.shape[1], X_trans.shape[1]))
    columns_retained_FromMode = trainset.drop(columns="label").columns[model_select.get_support()].values
    return columns_retained_FromMode


def train_test_ruolo(clean_dataset_path):
    """
    Suddivisione standard in trainset e testset a partire dal dataframe preparato nelle fasi precedenti
    :param clean_dataset_path: percorso del file csv contenente il dataset pulito e preparato
    :return:
    """
    # Load Dataset
    df: pd.DataFrame = load_raw_data(clean_dataset_path)

    # Droppo le colonne che non serviranno nella elaborazione
    df.drop(columns=["idAnagrafica", "DataPrimaNotifica", "IndirizzoResidenza", "Provincia", "Pagato120Giorni"],
            inplace=True)

    # Divisione test out of sample e training in sample
    trainset = df.loc[(df.DataCaricoTitolo != '2017-03-08')]
    testset = df.loc[(df.DataCaricoTitolo == '2017-03-08')]

    # Drop della colonna una volta separati i due set
    trainset.drop(columns="DataCaricoTitolo", inplace=True)
    testset.drop(columns="DataCaricoTitolo", inplace=True)

    # Mescolo le righe del training set
    trainset = shuffle(trainset, random_state=42)

    # Salvo i dataset con UI:
    save_dataset(trainset)
    save_dataset(testset)


def separazione_label(trainset: pd.DataFrame, testset: pd.DataFrame = None, test=False):
    """
    Separa i dataframe trainset e testset nelle rispettive componenti input (features) e output (label)
    :param trainset:
    :param testset:
    :param test: se passato True ritorna anche la separazione per il testset
    :return:
    """
    X_test, Y_test = None, None
    Y_train = trainset['label'].to_numpy()
    X_train = trainset.drop(columns="label").to_numpy()
    if test == True:
        Y_test = testset['label'].to_numpy()
        X_test = testset.drop(columns="label").to_numpy()
        return X_train, Y_train, X_test, Y_test
    return X_train, Y_train





"""
--------------------------------------------MODELLAZIONE E STUDIO-------------------------------------------------------
"""
# Load original Dataset
#train_test_ruolo("datasets/CRCPFprepared.csv")
# trainset: 18557 totali, 3403 label 1, 15154 label 0
# testset: 4903 totali, 973 label 1, 3930 label 0

# Load Dataset
trainset: pd.DataFrame = load_raw_data("datasets/trainset.csv")  # 18557 entry
testset: pd.DataFrame = load_raw_data("datasets/testset.csv")  # 4903 entry

# Separazione risultato (label) dall'input (features)
X_train, Y_train, X_test, Y_test = separazione_label(trainset, testset, test=True)

# Dichiarazione classificatori "dirty"
logistic = skl.linear_model.LogisticRegression(solver="lbfgs", max_iter=1000)
linearsvm = svm.LinearSVC(max_iter=10000)
supportvectormachine = svm.SVC(kernel="rbf", gamma="scale")
forest = RandomForestClassifier(n_estimators=100)
xgboost = xgb.XGBClassifier()
tree = DecisionTreeClassifier()
mlp = MLPClassifier()
eclf1 = VotingClassifier(estimators=[('1', logistic), ('2', supportvectormachine), ('3', forest), ('4', xgboost), ('5', tree), ('6', mlp)])  # bagging

# Random Undersampling della classe 0
one_indices = trainset[trainset.label == 1].index
sample_size = sum(trainset.label == 1)  # Equivalent to len(data[data.Healthy == 0]), numero di titoli con label 1 3394
zero_indices = trainset[trainset.label == 0].index
random_indices = np.random.choice(zero_indices, sample_size, replace=False)
# Unisco gli 1 e 0
under_indices = one_indices.union(random_indices)
undersampletrain = trainset.loc[under_indices]  # nuovo training set con 50/50 di classe 1 e 0
X_trainund, Y_trainund = separazione_label(undersampletrain)

# Smote Oversampling della classe 1
smt = SMOTE()
X_trainsmt, Y_trainsmt = smt.fit_sample(X_train, Y_train)  # X e Y contengono ora le due classi in modo bilanciato [15124, 15124]

# Nearmiss oversampling della classe 1
nr = NearMiss()
X_trainnm, Y_trainnm = nr.fit_sample(X_train, Y_train)

# Standard Scaled dataset
scaler = joblib.load("scaler.pkl")
scaledtrain = feature_scaling(trainset, scaler)
scaledtest = feature_scaling(testset, scaler)
scaled_X_train, scaled_Y_train, scaled_X_test, scaled_Y_test = random_feature_selection(scaledtrain, scaledtest, test=True)
# Smote on scaled dataset
smt = SMOTE()
scaled_X_trainsmt, scaled_Y_trainsmt = smt.fit_sample(scaled_X_train, scaled_Y_train)

# Scorers
scoring = {'accuracy': make_scorer(accuracy_score),
           'precision': make_scorer(precision_score),
           'recall': make_scorer(recall_score),
           'f1_score': make_scorer(f1_score),
           'roc_auc': make_scorer(roc_auc_score)}


"""
#PARAMS FOR GRID SEARCH

forestparams = {
    'bootstrap': [True, False],
    'n_jobs': [-1],
    'max_depth': [10, 30, None],
    'max_features': ['auto', 'log2'],
    'min_samples_leaf': [1, 6],
    'min_samples_split': [2, 10],
    'n_estimators': [200, 500]
}

svmparams = {
    'C': [0.001, 0.01, 0.1, 1, 10],
    'gamma': [0.001, 0.01, 0.1, 1],
}
"""


"""
#TSNE
X_embedded50 = TSNE(n_components=2, perplexity=50.0, random_state=42).fit_transform(X_train)
scatter_plot_2d(X_embedded50[:,0],X_embedded50[:,1],Y_train,"Training set 2d TSNE", "Primo componente", "Secondo componente")
"""

"""
#SCALING. SMOTE e random feature selection

scaler=joblib.load("scaler.pkl")
scaledtrain=feature_scaling(trainset,scaler)
scaledtest=feature_scaling(testset,scaler)
X_train, Y_train, X_test, Y_test = random_feature_selection(scaledtrain, scaledtest, test=True)
smt = SMOTE()
X_trainsmt, Y_trainsmt = smt.fit_sample(X_train, Y_train)
multiple_confusion_matrix([logistic], X_trainsmt, Y_trainsmt, X_test, Y_test)

#column feature selection, SMOTE senza scaling
columns=['Telefono', 'Cap', 'CittadinanzaItaliana', 'NumeroTitoliAperti',
 'ImportoTitoliAperti', 'ImportoTitoliSaldati', 'NumeroTitoliRecenti',
 'TitoliCredito', 'RapportoImporto', 'RapportoDovutoAperti']
X_train, Y_train, X_test, Y_test = column_feature_selection(trainset, columns, testset, test=True)
smt = SMOTE()
X_trainsmt, Y_trainsmt = smt.fit_sample(X_train, Y_train)
multiple_confusion_matrix([logistic], X_trainsmt, Y_trainsmt, X_test, Y_test)


target_ids = range(len(trainset.label))
from matplotlib import pyplot as plt
colors = 'r', 'g'
for i, c, label in zip(target_ids, colors, target_ids):
    plt.scatter(X_2d[y == i, 0], X_2d[y == i, 1], c=c, label=label)
plt.legend()
"""
