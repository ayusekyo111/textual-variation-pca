import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import plotly.express as px
import seaborn as sns
from matplotlib import pyplot as plt
np.set_printoptions(suppress=True)

total_features = ['feature_LOC', 'feature_ORG', 'feature_PER', 'feature_MISC', 'feature_TOTAL_EN', 'feature_LONGUEUR_MOTS', 'feature_ADP', 'feature_SCONJ', 'feature_CCONJ', 'feature_ADV', 'feature_PROPN', 'feature_NUM', 'feature_AUX', 'feature_VERB', 'feature_DET', 'feature_ADJ', 'feature_NOUN', 'feature_PRON', 'feature_PPER1S', 'feature_PPER2S', 'feature_PPER3', 'feature_INTJ', 'feature_SYM', 'feature_PUNCT', 'feature_X', 'feature_PAST_TENSE']

features = ['feature_LOC',
            'feature_ORG',
            'feature_PER',
            #'feature_MISC',
            'feature_TOTAL_EN',
            'feature_LONGUEUR_MOTS',
            'feature_ADP',
            'feature_SCONJ',
            'feature_CCONJ',
            'feature_ADV',
            'feature_PROPN',
            'feature_NUM',
            'feature_AUX',
            'feature_VERB',
            'feature_DET',
            'feature_ADJ',
            'feature_NOUN',
            'feature_PRON',
            'feature_PPER1S',
            'feature_PPER2S',
            'feature_PPER3',
            'feature_INTJ',
            #'feature_SYM',
            'feature_PUNCT',
            #'feature_X',
            'feature_PAST_TENSE']

absent_features = []

# petit moyen d'afficher automatiquement les features non sélectionnés dans les titres des figures (biplots)
for feature in total_features:
    if feature not in features:
        absent_feature = feature.replace("feature_", "")
        absent_features.append(absent_feature)

classif = ['feature_LOC',
            'feature_ORG',
            'feature_PER',
            #'feature_MISC',
            'feature_TOTAL_EN',
            'feature_LONGUEUR_MOTS',
            'feature_ADP',
            'feature_SCONJ',
            'feature_CCONJ',
            'feature_ADV',
            'feature_PROPN',
            'feature_NUM',
            'feature_AUX',
            'feature_VERB',
            'feature_DET',
            'feature_ADJ',
            'feature_NOUN',
            'feature_PRON',
            'feature_PPER1S',
            'feature_PPER2S',
            'feature_PPER3',
            'feature_INTJ',
            'feature_SYM',
            'feature_PUNCT',
            'feature_X',
            'label']


data = pd.read_csv("../tableurs/chunks_features.csv") # on charge les données

scaler = StandardScaler() # on instancie notre standardiseur
data_standard = scaler.fit_transform(data[features]) # on standardise les données

n_components = 6 #len(features) # on définit le nombre de composantes
pca = PCA(n_components=n_components) # on instancie notre pca avec le nombre de composantes choisi

data_pca = pca.fit_transform(data_standard) # formatage pour les biplots et scatter plot
pca_fit = pca.fit(data_standard) # formatage pour les scree plots

eigenvalues = pca.explained_variance_
variance = pca_fit.explained_variance_ratio_ # variance expliquée par chaque dimension


def visu_scree_plot(DONNES_PCA):
    """Fonction qui génère un scree plot (courbe de décroissance des valeurs propres)"""
    plt.ylabel("Eigenvalues")
    plt.xlabel("Nombre de variables")
    plt.ylim(0, max(DONNES_PCA.explained_variance_))
    plt.axhline(y=1, color="r", linestyle="--")
    plt.plot(DONNES_PCA.explained_variance_, "ro-")
    return plt.show()

def visu_scree_plot_inverse():
    """Fonction qui génère un scree plot inversé (courbe des valeurs propres cumulées, en pourcentage)"""
    var = np.cumsum(np.round(variance, decimals=3) * 100)
    plt.ylabel("Pourcentage de variance expliquée")
    plt.xlabel("Nombre de variables")
    plt.ylim(min(var), 100.5)
    plt.axhline(y=80, color="r", linestyle="--")
    plt.plot(var, "ro-")
    return plt.show()

# on prépare les labels de nos axes, qui sont les composantes principales et leur variance expliquée en pourcentage
# correspond à : {'0': 'PC1 (35.0%)', '1': 'PC2 (13.4%)', '2': 'PC3 (10.8%)', '3': 'PC4 (7.1%)', '4': 'PC5 (6.0%)', '5': 'PC6 (4.5%)'}
labels = {
    str(i): f"PC{i+1} ({var:.1f}%)"
    for i, var in enumerate(pca.explained_variance_ratio_ * 100)
}


def visu_scatter_matrix(LABELS, DONNEES_PCA, COMPOSANTES, CSV):
    """
    Fonction qui génère une matrice scatter avec toutes les composantes principales (s'ouvre dans le navigateur).
    :param LABELS: les composantes et leurs variances expliquées en % (pour nos axes)
    :param DONNEES_PCA: les données transformées
    :param COMPOSANTES: le nombre de composantes
    :param CSV: le fichier csv d'où les données ont été chargées (pour la colonne "label" = catégories textuelles)
    :return:
    """
    fig = px.scatter_matrix(
        DONNEES_PCA,
        labels=LABELS,
        dimensions=range(COMPOSANTES),
        color=CSV["label"],
        #title=f"Figure : {n_components} composantes principales, {len(features)} features (sans {', '.join(absent_features)})"
    )
    fig.update_traces(diagonal_visible=False)
    return fig.show()


loadings = pca.components_.T # poids de l'association entre les variables d'origine et les composantes principales
colonnes = ["PC1", "PC2", "PC3", "PC4", "PC5", "PC6"] # liste de nos PC

# on charge les loadings dans un dataframe, avec les caractéristiques en index
loadings_dataframe = pd.DataFrame(loadings, columns=colonnes, index=features) #loadings_dataframe.to_csv('loadings_features.csv', encoding='utf-8')

def visu_biplot(NUM1, NUM2, PCA, DONNEES_PCA, FEATURES, LOADINGS, CSV, LABELS):
    """
    Fonction qui génère un biplot pour 2 composantes de notre choix.
    :param NUM1: choix de la 1ère composante par l'utilisateur
    :param NUM2: choix de la 2ème composante par l'utilisateur
    :param PCA: la pca instanciée
    :param DONNEES_PCA: les données transformées
    :param FEATURES: la liste des caractéristiques
    :param LOADINGS: le dataframe de la matrice transposée des loadings (poids des variables d'origine dans les composantes principales)
    :param CSV: le fichier csv d'où les données ont été chargées (pour la colonne "label" = catégories textuelles)
    :param LABELS: les composantes et leurs variances expliquées en % (pour nos axes)
    :return:
    """
    PCname1 = "PC" + str(NUM1)
    PCname2 = "PC" + str(NUM2)

    NUM1 = int(NUM1) - 1
    NUM2 = int(NUM2) - 1

    PCnum1 = PCA.fit_transform(DONNEES_PCA)[:,NUM1]
    PCnum2 = PCA.fit_transform(DONNEES_PCA)[:,NUM2]

    scalePCnum1 = 1.0/(PCnum1.max() - PCnum1.min())
    scalePCnum2 = 1.0/(PCnum2.max() - PCnum2.min())

    plt.figure(figsize=(13, 8))

    for i, feature in enumerate(FEATURES): # pour chacune de nos caractéristiques
        # on n'affiche que les caractéristiques qui ont des loadings supérieurs à 0.2 (valeur absolue)
        if abs(LOADINGS[PCname1].iloc[i]) >= 0.2 or abs(LOADINGS[PCname2].iloc[i]) >= 0.2:
            plt.arrow(0, 0,
                      LOADINGS[PCname1].iloc[i],
                      LOADINGS[PCname2].iloc[i],
                      head_width=0.01,
                      head_length=0.01)
            plt.text(LOADINGS[PCname1].iloc[i] * 1.15,
                     LOADINGS[PCname2].iloc[i] * 1.15,
                     feature, fontsize=12)
        else: continue

    sns.scatterplot(x=PCnum1 * scalePCnum1,
                    y=PCnum2 * scalePCnum2,
                    hue=CSV["label"],
                    palette="bright")

    plt.xlabel(f'{LABELS.get(str(NUM1))}', fontsize=15)
    plt.ylabel(f'{LABELS.get(str(NUM2))}', fontsize=15)
    #plt.title(f"Biplot : {n_components} composantes principales, {len(features)} features (sans {', '.join(absent_features)}), sans sequoia (multisources) et fqb (questions)", fontsize=15)
    return plt.show()


if __name__ == '__main__':
    print("\nEigenvalues :\n", eigenvalues)
    print("\nVariance expliquée :\n", variance)

    #visu_scree_plot(pca_fit)
    #visu_scree_plot_inverse()
    #visu_scatter_matrix(labels, data_pca, n_components, data)
    visu_biplot(5, 6, pca, data_pca, features, loadings_dataframe, data, labels)
