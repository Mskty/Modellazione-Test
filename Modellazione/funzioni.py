import itertools
import tkinter as tk
import matplotlib.pyplot as plt
import pandas as pd
import sklearn as skl
import numpy as np
import seaborn as sns
import time
import datetime
import xgboost as xgb

from tkinter import filedialog
from joblib import dump, load
from mpl_toolkits.mplot3d import Axes3D
from sklearn import datasets, svm, model_selection
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.decomposition import PCA
from pandas import DataFrame
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score, cross_val_predict, GridSearchCV, \
    RandomizedSearchCV
from sklearn.neighbors import NearestNeighbors, KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, pairwise, precision_recall_curve
from sklearn import preprocessing
from sklearn import tree
from sklearn.neighbors import NearestNeighbors
from sklearn.naive_bayes import GaussianNB, BernoulliNB, MultinomialNB
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.preprocessing import LabelBinarizer, LabelEncoder, OneHotEncoder

# TUTTI I PARAMETRI NELLE FUNZIONI SONO PASSATI PER RIFERIMENTO, AVVIENE QUINDI SIDE EFFECT A MENO DI COPIA ESPLICITA

"""--------------------------------------------------UTILITIES PER DATASET-----------------------------------------------------"""


def set_dataset_display_properties(width=1000, columns=200, rows=200):
    """
    Imposta i valori di visualizzazione dei dataset pandas sul terminale
    :param width:
    :param columns:
    :param rows:
    :return:
    """

    pd.set_option('display.width', width)
    pd.set_option('display.max_columns', columns)
    pd.set_option('display.max_rows', rows)


def load_raw_data(raw_data_path) -> pd.DataFrame:
    """
    ritorna un pandas Dataframe dal file csv specificato dal percorso
    :param raw_data_path:
    :return:
    """
    set_dataset_display_properties()
    data = pd.read_csv(raw_data_path)

    # shuffle the dataset
    # data = data.sample(frac=1)
    return data


def load_raw_data_excel(raw_data_path) -> pd.DataFrame:
    """
        ritorna un pandas Dataframe dal file excel specificato dal percorso
        :param raw_data_path:
        :return:
        """
    set_dataset_display_properties()
    data = pd.read_excel(raw_data_path,
                         sheetname=0,
                         header=0,
                         index_col=False,
                         keep_default_na=True)

    # shuffle the dataset
    # data = data.sample(frac=1)
    return data


def save_dataset(data: pd.DataFrame):
    """
    Mostra un'interfaccia grafica per salvare il dataset in formato csv nella posizione desiderata nel filesystem
    :param data: Pandas Dataframe da salvare
    :return:
    """
    root = tk.Tk()
    canvas1 = tk.Canvas(root, width=300, height=300, bg='lightsteelblue2', relief='raised')
    canvas1.pack()

    def exportCSV():
        export_file_path = filedialog.asksaveasfilename(defaultextension='.csv')
        data.to_csv(export_file_path, index=None, header=True)

    saveAsButton_CSV = tk.Button(text='Export CSV', command=exportCSV, bg='green', fg='white',
                                 font=('helvetica', 12, 'bold'))
    canvas1.create_window(150, 150, window=saveAsButton_CSV)

    root.mainloop()


def show_data_info(data: pd.DataFrame):
    """
    Stampa diverse informazioni per il DataFrame passato
    :param data:
    :return:
    """

    print("\n----------INIZIO DATASET---------- \n")
    print("Formato: \n")
    print(data.shape)
    print("\nInfo e tipi: \n")
    print(data.info())
    print("\nDescrizione: \n")
    print(data.describe())
    print("\nPrime 5 righe: \n")
    print(data.head())
    print("\n----------FINE DATASET---------- \n")


def divide_features_target(data: pd.DataFrame):
    """
    Divide il dataset in 2 numpy array, X=features e Y=label, Y deve essere l'ultima colonna del DataFrame
    :param data:
    :return:
    """

    X = data.drop(data.columns[len(data.columns) - 1], axis=1).to_numpy()
    Y = data[data.columns[len(data.columns) - 1]].to_numpy()
    return X, Y


"""
def one_hot_encoder(features, data: pd.DataFrame) -> pd.DataFrame:
    # ritorna il dataset passato con le colonne selezionate in formato one hot

    cat_dum = pd.get_dummies(data[features].astype(str))
    data.drop(columns=["Sex", "Embarked", "Title"], inplace=True)
    data = data.join(cat_dum)
    return data
"""


def discrete_label_encoder(features, data: pd.DataFrame) -> pd.DataFrame:
    """
    Ritorna il DataFrame passato con le colonne selezionate in formato label discrete
    :param features:
    :param data:
    :return:
    """

    data = data.copy()
    encoder = LabelEncoder()
    for feature in features:
        data[feature] = encoder.fit_transform(data[feature])
    return data


"""
class CustomLabelBinarizer(BaseEstimator, TransformerMixin):
    def __init__(self, sparse_output=False):
        self.sparse_output = sparse_output

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        enc = LabelBinarizer(sparse_output=self.sparse_output)
        return enc.fit_transform(X)


def transformation_pipeline(numeric_features, categorical_features, data: pd.DataFrame) -> pd.DataFrame:
    # ritorna una copia del dataframe come matrice numpy trasformato a seconda della pipeline
    # di default usa standardscaler e one_hot_encoding, minmax scaler sarebbe preferibile se non ci fossero outliers
    # TODO salvare inputer in modo da utilizzarlo identico sul test set

    num_pipeline = Pipeline([
        ('selector', DataFrameSelector(numeric_features)),
        ('imputer', Imputer(strategy="median")),
        ('std_scaler', skl.preprocessing.StandardScaler()),
    ])
    cat_pipeline = Pipeline([
        ('selector', DataFrameSelector(categorical_features)),
        ('label_binarizer', OneHotEncoder()),
    ])
    full_pipeline = FeatureUnion(transformer_list=[
        ("num_pipeline", num_pipeline),
        ("cat_pipeline", cat_pipeline),
    ])

    # data_transformed = full_pipeline.fit_transform(data)
    data_transformed = num_pipeline.fit_transform(data)
    return data_transformed
"""

"""-------------------------------------GRAFICI E VISUALIZZAZIONE------------------------------------------"""


def bar_plot(x_data, y_data, title="", x_label="", y_label=""):
    # genera una nuova figura, da visualizzare poi a schermo con plt.show()

    plt.figure(figsize=(10, 5))
    ax = sns.barplot(x_data, y_data)
    plt.title(title)
    plt.xlabel(x_label, fontsize=12)
    plt.ylabel(y_label, fontsize=12)


def dist_plot(data, title="", x_label="", y_label=""):
    plt.figure(figsize=(10, 5))
    sns.set_style("whitegrid")
    ax = sns.distplot(data)


def correlation_heatmap(df):
    # correlation heatmap of dataset
    _, ax = plt.subplots(figsize=(10, 8))
    colormap = sns.diverging_palette(220, 10, as_cmap=True)

    _ = sns.heatmap(
        df.corr(),
        cmap=colormap,
        square=True,
        cbar_kws={'shrink': .9},
        ax=ax,
        annot=True,
        linewidths=0.1, vmax=1.0, linecolor='white',
        annot_kws={'fontsize': 12}
    )

    plt.title('Pearson Correlation of Features', y=1.05, size=15)


def scatter_plot_3d(x_data, y_data, z_data, class_label, title="", x_label="", y_label="", z_label=""):
    # genera una nuova figura, da visualizzare poi a schermo con plt.show()

    fig = plt.figure(figsize=(10, 5))
    ax = fig.add_subplot(111, projection='3d')
    plt.colorbar(ax.scatter(x_data, y_data, z_data, c=class_label))
    plt.title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_zlabel(z_label)


def scatter_plot_2d(x_data, y_data, class_label, title="", x_label="", y_label="", size: int = None):
    # genera una nuova figura, da visualizzare poi a schermo con plt.show()

    fig = plt.figure(figsize=(10, 5))
    ax = fig.add_subplot(111)
    if size == None:
        plt.colorbar(ax.scatter(x_data, y_data, c=class_label))
    else:
        plt.colorbar(ax.scatter(x_data, y_data, c=class_label, s=size))
    plt.title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    Stampa in un grafico 2d la confusion matrix
    si può scegliere di normalizzare i risultati tra 0 e 1 passando True al parametro normalize.
    :param cm: confusion matrix
    :param classes:
    :param normalize:
    :param title:
    :param cmap:
    :return:
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=0)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        # print("Normalized confusion matrix")
    else:
        True  # print('Confusion matrix, without normalization')

    # print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


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

    return grid_search_cv.best_params_, grid_search_cv.best_score_


def random_search(param_distribution, num_iter, classifier, X_train, y_train, fold=5, metric="f1"):
    # effettua una cross validation del modello con random search sulla distribuzione dei parametri passata
    # ritorna i migliori parametri e il miglior risultato, media su tutti i fold
    # usa lo scoring di default del classificatore (accuracy)

    random_search_cv = RandomizedSearchCV(classifier, param_distribution, n_iter=num_iter, cv=fold, iid=False, scoring=metric)

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


"""-------------------------------------VISUALIZZAZIONE METRICHE DI CLASSIFICAZIONE------------------------------------------"""


def plot_roc_curve(true_positive, false_positive, label=None):
    # fa comparire il grafico della roc_curve
    # richiede fpr, tpr, thresholds = roc_curve(y_train_true, y_proba_scores)

    plt.plot(true_positive, false_positive, linewidth=2, label=label)
    plt.plot([0, 1], [0, 1], 'k--', label="coin toss")
    plt.axis([0, 1, 0, 1])
    plt.legend(loc="lower right")
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')


def plot_precision_recall_vs_threshold(precisions, recalls, thresholds):
    # fa comparire il grafico della precision vs recall con threshold della decision function (default 0)
    # precisions, recalls, thresholds = precision_recall_curve(y_train_true, y_proba_scores)

    plt.plot(thresholds, precisions[:-1], "b-", label="Precision")
    plt.plot(thresholds, recalls[:-1], "g-", label="Recall")
    plt.axis([0, 1, 0, 1])
    plt.xlabel("Threshold")
    plt.legend(loc="upper left")


def plot_precision_recall(precision, recall):
    # fa comparire il grafico della precision vs recall con threshold della decision function (default 0)
    # precisions, recalls, thresholds = precision_recall_curve(y_train_true, y_proba_scores)

    plt.plot(recall, precision, "b-", label="Precision")
    plt.axis([0, 1, 0, 1])
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.legend(loc="upper left")


"""-------------------------------------SALVARE E CARICARE MODELLI------------------------------------------"""


def save_model(classifier, filename):
    """
    Salva il modello fittato in un file dal nome indicato
    :param classifier:
    :param filename:
    :return:
    """
    dump(classifier, filename + ".joblib")


def load_model(filename):
    """
    Carica il modello fittato dal file con il percorso indicato
    :param filename:
    :return: classificatore di scikit-learn fittato
    """
    classifier = load(filename + ".joblib")
    return classifier
