import random

import imblearn
import joblib
from sklearn.dummy import DummyClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.svm import LinearSVC, SVC
from xgboost import XGBClassifier

from funzioni import *
from sklearn.utils import shuffle
from sklearn.metrics import make_scorer, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, \
    average_precision_score, roc_curve, auc
from sklearn.tree import DecisionTreeClassifier

from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import NearMiss, RandomUnderSampler

from sklearn.linear_model import Lasso, LogisticRegression

"""
In questo file verranno elaborati i dati forniti dal dataset riguardanti le Persone Giuridiche
in forma definitiva allo scopo di trovare il modello 
predittivo migliore. La forma degli esempi non verrà modificata, ma sarà divisio il set inziale in training e test set,
allo scopo di valutare le performance dei vari modelli utilizzati. La feature colonna contenente la label di classe 
per ogni esempio è chiamata 'label'.
I modelli proposti includono i seguenti algoritmi forniti dalla libreria scikit-lean:
1. LogisticRegressorClassfier
2. SupportVectorMachineClassifier
3. LinearSupportVectorMachineClassifier
4. DecisionTreeClassifier
5. RandomForestClassifier
6. ExtremeGradientBoostingClassifier (xgboost)
Vengono in oltre fornite funzioni per produrre grafici utili a visualizzare le performance di tali algoritmi in 
condizioni di dataset molto sbilanciati come in questo caso: la curva precision recall e quella roc-auc
"""

"""----------------------------------------------------CLASSI-----------------------------------------------------------
"""


class RandomDummy(BaseEstimator, ClassifierMixin):
    """
    Classificatore che fa previsioni completamente a caso
    """

    def __init__(self, randomfactor=0.5):
        self.randomfactor = randomfactor

    def fit(self, X, Y):
        return self

    def predict(self, X):
        array = [np.random.randint(2) for x in X]
        array = np.asarray(array)
        return array


"""-------------------------VISUALIZZAZIONE PRC ROC AUC--------------------------------------------------------------"""


def generate_figure_prc(xsize=9, ysize=9):
    """
    Genera una figura base per curve prc aggiungendo già il classificatore ottimale
    """
    fig_size = (xsize, ysize)
    figure, ax = plt.subplots(figsize=fig_size)
    ax.axis = ([[0, 1, 0, 1]])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision Recall Curves")
    # Aggiunta ottimale (rapporto t 2 f 8 circa)
    ax.plot([0, 1, 1], [1, 1, 0.23], 'g-', linewidth=3, label="Classificatore Perfetto (AP = 1.0)")
    # Aggiunta peggiore, valore calcolato addestrando un uniform_dummy su training set e
    # poi utilizzando la funzione skl.metrics.plot_precision_recall_curve(uniform_dummy, scaled_X_test, Y_test)
    ax.plot([0, 0, 1], [1, 0.23, 0.23], 'r-', linewidth=3, label="Selezione Casuale (AP = 0.23)")
    # Legenda
    ax.legend(loc="upper right", fontsize='small')
    figure.canvas.draw()
    return figure, ax


def generate_figure_rocauc(xsize=9, ysize=9):
    """
    Genera una figura base per curve rocauc aggiungendo già il classificatore ottimale
    """
    fig_size = (xsize, ysize)
    figure, ax = plt.subplots(figsize=fig_size)
    ax.axis = ([[0, 1, 0, 1]])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("Roc-Auc Curves")
    # Aggiunta ottimale
    ax.plot([0, 0, 1], [0, 1, 1], 'g-', linewidth=3, label="Classificatore Perfetto (AUC = 1.0)")
    # Aggiunta peggiore
    ax.plot([0, 1], [0, 1], 'r-', linewidth=3, label="Selezione Casuale (AUC = 0.5)")
    # Legenda
    ax.legend(loc="lower right", fontsize='small')
    figure.canvas.draw()
    return figure, ax


def add_prc(figure, ax, recalls: list, precisions: list, ap_score, label: str, color: str, line: str = "-"):
    ax.plot(recalls, precisions, color + line, label=label + " (AP = {:0.3f})".format(ap_score))
    ax.legend(loc="upper right", fontsize='small')
    figure.canvas.draw()
    return ax


def add_rocauc(figure, ax, true_positive: list, false_positive: list, auc_score, label: str, color: str,
               line: str = "-"):
    ax.plot(false_positive, true_positive, color + line, label=label + " (AUC = {:0.3f})".format(auc_score))
    ax.legend(loc="lower right", fontsize='small')
    figure.canvas.draw()
    return ax


def prc_scores(classifier, X_train, Y_train, X_test, Y_test):
    classifier.fit(X_train, Y_train)
    Y_pred = classifier.predict_proba(X_test)[:, 1]
    pos_label = classifier.classes_[1]
    precision, recall, _ = precision_recall_curve(Y_test, Y_pred, pos_label=pos_label)
    ap_score = average_precision_score(Y_test, Y_pred, pos_label=pos_label)
    return precision, recall, ap_score


def auc_scores(classifier, X_train, Y_train, X_test, Y_test):
    classifier.fit(X_train, Y_train)
    Y_pred = classifier.predict_proba(X_test)[:, 1]
    pos_label = classifier.classes_[1]
    fpr, tpr, _ = roc_curve(Y_test, Y_pred, pos_label=pos_label)
    roc_auc = auc(fpr, tpr)
    return fpr, tpr, roc_auc


"""--------------------------------------------FEATURE SELECTION-----------------------------------------------------"""


def lasso_feature_selection(columns, scaled_X_train, Y_train):
    sel = SelectFromModel(LogisticRegression(C=1, penalty='l1', solver="liblinear"))
    sel.fit(scaled_X_train, Y_train)
    # Features con false non sono state ritenute importanti
    selected_feat = columns[(sel.get_support())]
    print('total features: {}'.format((X_train.shape[1])))
    print('selected features: {}'.format(len(selected_feat)))
    print('features with coefficients shrank to zero: {}'.format(
        np.sum(sel.estimator_.coef_ == 0)))
    print(selected_feat)
    return selected_feat


def random_forest_feature_selection(columns, scaled_X_train, Y_train):
    sel = SelectFromModel(RandomForestClassifier(n_estimators=500))
    sel.fit(scaled_X_train, Y_train)
    # Features con false non sono state ritenute importanti
    selected_feat = columns[(sel.get_support())]
    print('total features: {}'.format((X_train.shape[1])))
    print('selected features: {}'.format(len(selected_feat)))
    print(selected_feat)
    print(columns)
    print(sel.estimator_.feature_importances_)
    return selected_feat


def cross_feature_selection(classifiers: list, scaled_trainset: pd.DataFrame, metric="f1"):
    """
    Esegue una cross validation sul training set per ogni classificatore con l'esclusione di 1 feature o 2 features
    provando ogni combinazione e salvando il risultato di mean e std variation
    :param classifiers:
    :param scaled_trainset:
    :param metric:
    :return:
    """
    trainset = scaled_trainset
    METRIC = metric
    FOLDS = 3

    # Esclusione di una sola feature
    features = list(scaled_trainset.drop(columns="label").columns.values)
    for classifier in classifiers:
        print("valutazione per ", classifier.__class__.__name__)
        scores = cross_val_score(classifier, scaled_trainset.drop(columns=["label"]), scaled_trainset["label"],
                                 cv=FOLDS, scoring=METRIC)
        print("feature rimossa: nessuna")
        display_scores(scores)

        for feature in features:
            scores = cross_val_score(classifier, scaled_trainset.drop(columns=["label", feature]),
                                     scaled_trainset["label"], cv=FOLDS, scoring=METRIC)
            print("feature rimossa: ", feature)
            display_scores(scores)


"""-------------------------------------UTILITY PER SELEZIONARE IL MODELLO------------------------------------------"""


def report(results, n_top=3):
    # Utility function to report best scores

    for i in range(1, n_top + 1):
        candidates = np.flatnonzero(results['rank_test_score'] == i)
        for candidate in candidates:
            print("Model with rank: {0}".format(i))
            print("Mean validation score: {0:.3f} (std: {1:.3f})".format(
                results['mean_test_score'][candidate],
                results['std_test_score'][candidate]))
            print("Parameters: {0}".format(results['params'][candidate]))
            print("")


def grid_search(param_grid, classifier, X_train, y_train, fold=5, metric="f1"):
    # effettua una cross validation del modello con grid search sulla lista dei parametri passata
    # ritorna i migliori parametri e il miglior risultato, media su tutti i fold
    # usa lo scoring di default del classificatore (accuracy)

    grid_search_cv = GridSearchCV(classifier, param_grid, cv=fold, iid=False, scoring=metric)

    start = time.perf_counter()
    grid_search_cv.fit(X_train, y_train)
    print("Esecuzione terminata in: ", time.perf_counter() - start)

    print("Migliori parametri:", grid_search_cv.best_params_)
    print("Miglior modello:", grid_search_cv.best_estimator_)
    cvres = grid_search_cv.cv_results_
    print("Risultati: \n")
    report(grid_search_cv.cv_results_)

    return grid_search_cv.best_estimator_, grid_search_cv.best_params_, grid_search_cv.best_score_


def random_search(param_distribution, num_iter, classifier, X_train, y_train, fold=5, metric="f1"):
    # effettua una cross validation del modello con random search sulla distribuzione dei parametri passata
    # ritorna i migliori parametri e il miglior risultato, media su tutti i fold
    # usa lo scoring di default del classificatore (accuracy)

    random_search_cv = RandomizedSearchCV(classifier, param_distribution, n_iter=num_iter, cv=fold, iid=False,
                                          scoring=metric)

    start = time.perf_counter()
    random_search_cv.fit(X_train, y_train)
    print("Esecuzione terminata in: ", time.perf_counter() - start)

    print("Migliori parametri:" + random_search_cv.best_params_)
    print("Miglior modello:" + random_search_cv.best_estimator_)
    cvres = random_search_cv.cv_results_
    print("Risultati: \n")
    report(random_search_cv.cv_results_)

    return random_search_cv.best_params_, random_search_cv.best_score_


def display_scores(scores):
    """
    Utility function to display cross validation scores
    :param scores: array di scores
    :return:
    """

    print("Scores:", scores)
    print("Mean:", scores.mean())
    print("Standard deviation:", scores.std())


def cross_validation(classifier, X_train, y_train, fold=5, metric="f1"):
    # usa lo scoring di default del classificatore (accuracy)
    # ritorna il valore medio della cross validation

    scores = cross_val_score(classifier, X_train, y_train, cv=fold, scoring=metric)  # default è stratified
    display_scores(scores)
    return scores


def rf_feat_importance(clf, df) -> pd.DataFrame:
    """
    Ritorna un dataframe con feature importance di un classificatore che supporta questa funzionalità.
    Bisogna passare il dataframe contenente le features già preprocessato
    :param clf: classificatore che supporta feature_importance
    :param df: dataset utilizzato per l'addestramento in formato Pandas Dataframe
    :return:
    """
    return pd.DataFrame({'cols': df.columns, 'imp':
        clf.feature_importances_}).sort_values('imp', ascending=False)


"""
------------------------------------------FUNZIONI CROSS VALIDATION-----------------------------------------------------
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
    categorical_trainset = trainset[
        ["Telefono", "Cessata", "PEC", "Estero", "NuovoContribuente", "label"]]
    trainset.drop(columns=["Telefono", "Cessata", "PEC", "Estero", "NuovoContribuente", "label"],
                  inplace=True)
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
    categorical_data = data[["Telefono", "Cessata", "PEC", "Estero", "NuovoContribuente", "label"]]
    data.drop(columns=["Telefono", "Cessata", "PEC", "Estero", "NuovoContribuente", "label"],
              inplace=True)
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
        X_train, Y_train, X_test, Y_test = separazione_label(trainset.drop(columns=columns),
                                                             testset.drop(columns=columns), test=True)
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


def learning_curve(X_train_scaled, Y_train, classifier, score, train_sizes: list):
    """
    Esegue learning curve su una pipeline che include l'undersampling del training set
    train_sizes fissato
    cross_validation folds fissato
    ratio dell'undersampling fissato
    scroer= f1 score fissato
    :param X_train_scaled: trainset con gia applicata feature scaling
    :param Y_train: labels per trainset
    :param classifier: classificatore
    :return:
    """
    # initial parameters
    RANDOM_STATE = 42
    TRAIN_SIZES = train_sizes
    K_FOLDS = 5
    SCORER = make_scorer(score)
    SAMPLING_RATIO = 1

    # parameters
    X = X_train_scaled
    Y = Y_train
    classifier = classifier

    # Make pipeline (non funziona per cambiamkento alla libreria per RandomUnderSampler
    """undersampler = imblearn.under_sampling.RandomUnderSampler(random_state=RANDOM_STATE, sampling_strategy=SAMPLING_RATIO)
    pipeline = Pipeline([('und',undersampler), ('cls',classifier)])
    pipeline.fit(X,Y).predict_proba(Y)"""

    train_sizes, train_scores, test_scores, fit_times, _ = skl.model_selection.learning_curve(classifier, X, Y,
                                                                                              cv=K_FOLDS,
                                                                                              train_sizes=TRAIN_SIZES,
                                                                                              return_times=True,
                                                                                              n_jobs=-1,
                                                                                              scoring=SCORER)
    print(train_sizes)
    print(train_scores)
    print(test_scores)
    print("mean train scores: \n", train_scores.mean(axis=1))
    print("mean test scores: \n", test_scores.mean(axis=1))
    plt.plot(train_sizes, train_scores.mean(axis=1), label="train")
    plt.plot(train_sizes, test_scores.mean(axis=1), label="test")


"""
--------------------------------------------MODELLAZIONE E STUDIO-------------------------------------------------------
"""
# Load original Dataset
# trainset: 5275 totali, 1216  label 1, 4059 label 0
# testset: 1276 totali, 291 label 1, 985 label 0

# Load Dataset
trainset: pd.DataFrame = load_raw_data("datasets/export_trainset_PG.csv")  # 18557 entry
testset: pd.DataFrame = load_raw_data("datasets/export_testset_PG.csv")  # 4903 entry

# Separazione risultato (label) dall'input (features)
X_train, Y_train, X_test, Y_test = separazione_label(trainset, testset, test=True)

# Dichiarazione classificatori "dirty"
logistic = skl.linear_model.LogisticRegression(solver="liblinear", max_iter=1000, class_weight="balanced")
linearsvm = svm.LinearSVC(max_iter=1000, class_weight="balanced", dual=False)
supportvectormachine = svm.SVC(kernel="rbf", gamma="scale", class_weight="balanced", probability=True)
tree = DecisionTreeClassifier(class_weight="balanced")
forest = RandomForestClassifier(n_estimators=100, n_jobs=-1, class_weight="balanced")
xgboost = xgb.XGBClassifier(scale_pos_weight=2)

# Dummy classifiers per baseline
most_frequent_dummy = DummyClassifier(strategy="most_frequent")
stratified_dummy = DummyClassifier(strategy="stratified")
uniform_dummy = DummyClassifier(strategy="uniform")
total_random_dummy = RandomDummy()

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
X_trainsmt, Y_trainsmt = smt.fit_sample(X_train,
                                        Y_train)  # X e Y contengono ora le due classi in modo bilanciato [15124, 15124]

# Standard Scaled dataset
data = trainset.copy()
categorical_data = data[["Telefono", "Cessata", "PEC", "Estero", "NuovoContribuente", "label"]]
data.drop(columns=["Telefono", "Cessata", "PEC", "Estero", "NuovoContribuente", "label"],
          inplace=True)
scaler = StandardScaler()
scaler.fit(data)
scaledtrain = feature_scaling(trainset, scaler)
scaledtest = feature_scaling(testset, scaler)
scaled_X_train, scaled_Y_train, scaled_X_test, scaled_Y_test = separazione_label(scaledtrain, scaledtest, test=True)

# Smote on scaled dataset
smt = SMOTE()
scaled_X_trainsmt, scaled_Y_trainsmt = smt.fit_sample(scaled_X_train, scaled_Y_train)

# Undersampling su scaled dataset
one_indices = scaledtrain[scaledtrain.label == 1].index
sample_size = sum(
    scaledtrain.label == 1)  # Equivalent to len(data[data.Healthy == 0]), numero di titoli con label 1 3394
zero_indices = trainset[scaledtrain.label == 0].index
random_indices = np.random.choice(zero_indices, sample_size, replace=False)
# Unisco gli 1 e 0
under_indices = one_indices.union(random_indices)
scaledundersampletrain = scaledtrain.loc[under_indices]  # nuovo training set con 50/50 di classe 1 e 0
scaled_X_trainund, scaled_Y_trainund = separazione_label(scaledundersampletrain)

# Scorers
scoring = {'accuracy': make_scorer(accuracy_score),
           'precision': make_scorer(precision_score),
           'recall': make_scorer(recall_score),
           'f1_score': make_scorer(f1_score),
           'roc_auc': make_scorer(roc_auc_score)}

# PARAMS FOR GRID SEARCH
params_logistic = {'penalty': ['l1', 'l2'], 'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000]}
params_tree = {'criterion': ['gini', 'entropy'],
               'splitter': ['best', 'random'],
               'min_samples_split': [2, 4, 8],
               'min_samples_leaf': [1, 2, 4],
               "max_depth": [2, 10, 14, None],
               'max_leaf_nodes': [2, 10, 100, None]}
params_forest = {
    'n_estimators': [100, 200, 500],
    'max_features': ['auto', 'sqrt', 'log2'],
    'max_depth': [2, 4, 10, 14, None],
    'criterion': ['gini', 'entropy'],
    'bootstrap': [True, False]
}
params_svm_rbf = {
    'C': [0.001, 0.01, 0.1, 1, 10, 100],
    'gamma': [0.001, 0.01, 0.1, 1, 'scale'],
}

params_svm_linear = {
    'penalty': ['l1', 'l2'],
    'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000],
    'loss': ['hinge', 'squared_hinge']
}

params_xgboost = {"learning_rate": [0.10, 0.20, 0.30],
                  "max_depth": [2, 4, 6, 10, 12, 15],
                  "min_child_weight": [1, 4, 10],
                  "gamma": [0.0, 0.1, 0.3, 0.5, 0.7, 1.0],
                  "colsample_bytree": [0.3, 0.4, 0.7, 1.0]}

# BEST VERSION OF ALGORITHMS AFTER GRID SEARCH
"""
ESEMPIO esecuzione e risultati grid search con rispettivi parametri:

bestlogistic, _, _=grid_search(params_logistic,logistic,scaled_X_train,Y_train)

stampa a schermo il miglior algoritmo di LogisticRegression ottenibile combinando i parametri in params_logistic
"""
bestlogistic = LogisticRegression(C=100, class_weight='balanced', dual=False,
                                  fit_intercept=True, intercept_scaling=1, l1_ratio=None,
                                  max_iter=1000, multi_class='auto', n_jobs=None, penalty='l2',
                                  random_state=None, solver='liblinear', tol=0.0001, verbose=0,
                                  warm_start=False)
bestlinearsvm = LinearSVC(C=0.1, class_weight='balanced', dual=False, fit_intercept=True,
                          intercept_scaling=1, loss='squared_hinge', max_iter=1000,
                          multi_class='ovr', penalty='l1', random_state=None, tol=0.0001,
                          verbose=0)
bestrbfsvm = SVC(C=1, break_ties=False, cache_size=200, class_weight='balanced', coef0=0.0,
                 decision_function_shape='ovr', degree=3, gamma='scale', kernel='rbf',
                 max_iter=-1, probability=True, random_state=None, shrinking=True,
                 tol=0.001, verbose=False)
besttree = DecisionTreeClassifier(ccp_alpha=0.0, class_weight='balanced', criterion='gini',
                                  max_depth=14, max_features=None, max_leaf_nodes=10,
                                  min_impurity_decrease=0.0, min_impurity_split=None,
                                  min_samples_leaf=4, min_samples_split=4,
                                  min_weight_fraction_leaf=0.0, presort='deprecated',
                                  random_state=None, splitter='random')
bestforest = RandomForestClassifier(bootstrap=False, ccp_alpha=0.0, class_weight='balanced',
                                    criterion='entropy', max_depth=10, max_features='sqrt',
                                    max_leaf_nodes=None, max_samples=None,
                                    min_impurity_decrease=0.0, min_impurity_split=None,
                                    min_samples_leaf=1, min_samples_split=2,
                                    min_weight_fraction_leaf=0.0, n_estimators=200,
                                    n_jobs=-1, oob_score=False, random_state=None, verbose=0,
                                    warm_start=False)
bestxgboost = XGBClassifier(base_score=0.5, booster='gbtree', colsample_bylevel=1,
                            colsample_bynode=1, colsample_bytree=0.4, gamma=1.0,
                            learning_rate=0.1, max_delta_step=0, max_depth=4,
                            min_child_weight=10, missing=None, n_estimators=100, n_jobs=1,
                            nthread=None, objective='binary:logistic', random_state=0,
                            reg_alpha=0, reg_lambda=1, scale_pos_weight=2, seed=None,
                            silent=None, subsample=1, verbosity=1)

"""# MAKE GRAPHS

-------------------------------------VANNO ESEGUITI SU CONSOLE IN QUANTO FANNO COMPARIRE LE FIGURE----------------------

# PRC
fig, ax = generate_figure_prc()

prec, rec, ap = prc_scores(logistic, scaled_X_train, Y_train, scaled_X_test, Y_test)
add_prc(fig, ax, rec, prec, ap, "Logistic", 'b', '-')
prec, rec, ap = prc_scores(bestlogistic, scaled_X_train, Y_train, scaled_X_test, Y_test)
add_prc(fig, ax, rec, prec, ap, "Best Logistic", 'b', '-.')

prec, rec, ap = prc_scores(supportvectormachine, scaled_X_train, Y_train, scaled_X_test, Y_test)
add_prc(fig, ax, rec, prec, ap, "SVM", 'm', '-')
prec, rec, ap = prc_scores(bestrbfsvm, scaled_X_train, Y_train, scaled_X_test, Y_test)
add_prc(fig, ax, rec, prec, ap, "Best SVM", 'm', '-.')

prec, rec, ap = prc_scores(tree, scaled_X_train, Y_train, scaled_X_test, Y_test)
add_prc(fig, ax, rec, prec, ap, "Decision Tree", 'y', '-')
prec, rec, ap = prc_scores(besttree, scaled_X_train, Y_train, scaled_X_test, Y_test)
add_prc(fig, ax, rec, prec, ap, "Best Decision Tree", 'y', '-.')

prec, rec, ap = prc_scores(forest, scaled_X_train, Y_train, scaled_X_test, Y_test)
add_prc(fig, ax, rec, prec, ap, "Random Forest", 'c', '-')
prec, rec, ap = prc_scores(bestforest, scaled_X_train, Y_train, scaled_X_test, Y_test)
add_prc(fig, ax, rec, prec, ap, "Best Random Forest", 'c', '-.')

prec, rec, ap = prc_scores(xgboost, scaled_X_train, Y_train, scaled_X_test, Y_test)
add_prc(fig, ax, rec, prec, ap, "Xgboost", 'k', '-')
prec, rec, ap = prc_scores(bestxgboost, scaled_X_train, Y_train, scaled_X_test, Y_test)
add_prc(fig, ax, rec, prec, ap, "Best Xgboost", 'k', '-.')

fig.show()

# AUC
fig, ax = generate_figure_rocauc()

fpr, tpr, aucs = auc_scores(logistic, scaled_X_train, Y_train, scaled_X_test, Y_test)
add_rocauc(fig, ax, tpr, fpr, aucs, "Logistic", 'b', '-')
fpr, tpr, aucs = auc_scores(bestlogistic, scaled_X_train, Y_train, scaled_X_test, Y_test)
add_rocauc(fig, ax, tpr, fpr, aucs, "Best Logistic", 'b', '-.')

fpr, tpr, aucs = auc_scores(supportvectormachine, scaled_X_train, Y_train, scaled_X_test, Y_test)
add_rocauc(fig, ax, tpr, fpr, aucs, "SVM", 'm', '-')
fpr, tpr, aucs = auc_scores(bestrbfsvm, scaled_X_train, Y_train, scaled_X_test, Y_test)
add_rocauc(fig, ax, tpr, fpr, aucs, "Best SVM", 'm', '-.')

fpr, tpr, aucs = auc_scores(tree, scaled_X_train, Y_train, scaled_X_test, Y_test)
add_rocauc(fig, ax, tpr, fpr, aucs, "Decision Tree", 'y', '-')
fpr, tpr, aucs = auc_scores(besttree, scaled_X_train, Y_train, scaled_X_test, Y_test)
add_rocauc(fig, ax, tpr, fpr, aucs, "Best Decision Tree", 'y', '-.')

fpr, tpr, aucs = auc_scores(forest, scaled_X_train, Y_train, scaled_X_test, Y_test)
add_rocauc(fig, ax, tpr, fpr, aucs, "Random Forest", 'c', '-')
fpr, tpr, aucs = auc_scores(bestforest, scaled_X_train, Y_train, scaled_X_test, Y_test)
add_rocauc(fig, ax, tpr, fpr, aucs, "Best Random Forest", 'c', '-.')

fpr, tpr, aucs = auc_scores(xgboost, scaled_X_train, Y_train, scaled_X_test, Y_test)
add_rocauc(fig, ax, tpr, fpr, aucs, "Xgboost", 'k', '-')
fpr, tpr, aucs = auc_scores(bestxgboost, scaled_X_train, Y_train, scaled_X_test, Y_test)
add_rocauc(fig, ax, tpr, fpr, aucs, "Best Xgboost", 'k', '-.')

fig.show()

# MAKE GRAPHS LASSO FEATURE SELECTION logistic e xgboost
features_lasso=lasso_feature_selection(scaledtrain.drop(columns="label").columns,scaled_X_train,Y_train)
lasso_scaled_train=scaledtrain[features_lasso]
lasso_scaled_test=scaledtest[features_lasso]
# PRC
fig, ax = generate_figure_prc()
prec, rec, ap = prc_scores(logistic, scaled_X_train, Y_train, scaled_X_test, Y_test)
add_prc(fig, ax, rec, prec, ap, "Logistic no feature selection", 'b', '-')
prec, rec, ap = prc_scores(logistic, lasso_scaled_train, Y_train, lasso_scaled_test, Y_test)
add_prc(fig, ax, rec, prec, ap, "Logistic Lasso feature selection", 'b', '-.')
prec, rec, ap = prc_scores(xgboost, scaled_X_train, Y_train, scaled_X_test, Y_test)
add_prc(fig, ax, rec, prec, ap, "Xgboost no feature selection", 'k', '-')
prec, rec, ap = prc_scores(xgboost, lasso_scaled_train, Y_train, lasso_scaled_test, Y_test)
add_prc(fig, ax, rec, prec, ap, "Xgboost Lasso feature selection", 'k', '-.')
ax.set_title("Precision Recall Curves \n Colonne rimosse da Lasso: ValoreTitolo ")
fig.show()
# AUC
fig, ax = generate_figure_rocauc()
fpr, tpr, aucs = auc_scores(logistic, scaled_X_train, Y_train, scaled_X_test, Y_test)
add_rocauc(fig, ax, tpr, fpr, aucs, "Logistic no feature selection", 'b', '-')
fpr, tpr, aucs = auc_scores(logistic, lasso_scaled_train, Y_train, lasso_scaled_test, Y_test)
add_rocauc(fig, ax, tpr, fpr, aucs, "Logistic Lasso feature selection", 'b', '-.')
fpr, tpr, aucs = auc_scores(xgboost, scaled_X_train, Y_train, scaled_X_test, Y_test)
add_rocauc(fig, ax, tpr, fpr, aucs, "Xgboost no feature selection", 'k', '-')
fpr, tpr, aucs = auc_scores(xgboost, lasso_scaled_train, Y_train, lasso_scaled_test, Y_test)
add_rocauc(fig, ax, tpr, fpr, aucs, "Xgboost Lasso feature selection", 'k', '-.')
ax.set_title("Roc-Auc curves \n Colonne rimosse da Lasso: ValoreTitolo, Deceduto ")
fig.show()

# MAKE GRAPHS RANDOM FOREST FEATURE SELECTION logistic e xgboost
features_forest=random_forest_feature_selection(scaledtrain.drop(columns="label").columns,scaled_X_train,Y_train)
forest_scaled_train=scaledtrain[features_forest]
forest_scaled_test=scaledtest[features_forest]
# PRC
fig, ax = generate_figure_prc()
prec, rec, ap = prc_scores(logistic, scaled_X_train, Y_train, scaled_X_test, Y_test)
add_prc(fig, ax, rec, prec, ap, "Logistic no feature selection", 'b', '-')
prec, rec, ap = prc_scores(logistic, forest_scaled_train, Y_train, forest_scaled_test, Y_test)
add_prc(fig, ax, rec, prec, ap, "Logistic Forest feature selection", 'b', '-.')
prec, rec, ap = prc_scores(xgboost, scaled_X_train, Y_train, scaled_X_test, Y_test)
add_prc(fig, ax, rec, prec, ap, "Xgboost no feature selection", 'k', '-')
prec, rec, ap = prc_scores(xgboost, forest_scaled_train, Y_train, forest_scaled_test, Y_test)
add_prc(fig, ax, rec, prec, ap, "Xgboost Forest feature selection", 'k', '-.')
ax.set_title("Precision Recall Curves \nColonne mantenute da Random Forest: \nValoreTitolo, ImportoTitoliSaldati, "
             "TotaleTitoliRecenti, Vetusta, Cap")
fig.show()
# AUC
fig, ax = generate_figure_rocauc()
fpr, tpr, aucs = auc_scores(logistic, scaled_X_train, Y_train, scaled_X_test, Y_test)
add_rocauc(fig, ax, tpr, fpr, aucs, "Logistic no feature selection", 'b', '-')
fpr, tpr, aucs = auc_scores(logistic, forest_scaled_train, Y_train, forest_scaled_test, Y_test)
add_rocauc(fig, ax, tpr, fpr, aucs, "Logistic Forest feature selection", 'b', '-.')
fpr, tpr, aucs = auc_scores(xgboost, scaled_X_train, Y_train, scaled_X_test, Y_test)
add_rocauc(fig, ax, tpr, fpr, aucs, "Xgboost no feature selection", 'k', '-')
fpr, tpr, aucs = auc_scores(xgboost, forest_scaled_train, Y_train, forest_scaled_test, Y_test)
add_rocauc(fig, ax, tpr, fpr, aucs, "Xgboost Forest feature selection", 'k', '-.')
ax.set_title("Roc-Auc Curves \n Colonne mantenute da Random Forest: \nValoreTitolo, ImportoTitoliSaldati, "
             "TotaleTitoliRecenti, Eta, Vetusta, Cap")
fig.show()

# MAKE GRAPHS CON E SENZA FEATURES CREATE IN PREPROCESSING
nopre_scaled_train=scaledtrain.drop(columns=['NuovoContribuente','Estero','RapportoImporto','RapportoDovutoAperti','label'])
nopre_scaled_test=scaledtest.drop(columns=['NuovoContribuente','Estero','RapportoImporto','RapportoDovutoAperti','label'])
# PRC
fig, ax = generate_figure_prc()
prec, rec, ap = prc_scores(logistic, scaled_X_train, Y_train, scaled_X_test, Y_test)
add_prc(fig, ax, rec, prec, ap, "Logistic tutte le features", 'b', '-')
prec, rec, ap = prc_scores(logistic, nopre_scaled_train, Y_train, nopre_scaled_test, Y_test)
add_prc(fig, ax, rec, prec, ap, "Logistic features rimosse", 'b', '-.')
prec, rec, ap = prc_scores(xgboost, scaled_X_train, Y_train, scaled_X_test, Y_test)
add_prc(fig, ax, rec, prec, ap, "Xgboost tutte le features", 'k', '-')
prec, rec, ap = prc_scores(xgboost, nopre_scaled_train, Y_train, nopre_scaled_test, Y_test)
add_prc(fig, ax, rec, prec, ap, "Xgboost features rimosse", 'k', '-.')
ax.set_title("Precision Recall Curves \n "
             "Colonne rimosse: \n"
             "NuovoContribuente,Cap,Estero,RapportoImporto,RapportoDovutoAperti")
fig.show()
# AUC
fig, ax = generate_figure_rocauc()
fpr, tpr, aucs = auc_scores(logistic, scaled_X_train, Y_train, scaled_X_test, Y_test)
add_rocauc(fig, ax, tpr, fpr, aucs, "Logistic tutte le features", 'b', '-')
fpr, tpr, aucs = auc_scores(logistic, nopre_scaled_train, Y_train, nopre_scaled_test, Y_test)
add_rocauc(fig, ax, tpr, fpr, aucs, "Logistic features rimosse", 'b', '-.')
fpr, tpr, aucs = auc_scores(xgboost, scaled_X_train, Y_train, scaled_X_test, Y_test)
add_rocauc(fig, ax, tpr, fpr, aucs, "Xgboost tutte le features", 'k', '-')
fpr, tpr, aucs = auc_scores(xgboost, nopre_scaled_train, Y_train, nopre_scaled_test, Y_test)
add_rocauc(fig, ax, tpr, fpr, aucs, "Xgboost features rimosse", 'k', '-.')
ax.set_title("Roc-Auc Curves \n "
             "Colonne rimosse: \n"
             "NuovoContribuente,Cap,Estero,RapportoImporto,RapportoDovutoAperti")
fig.show()"""
