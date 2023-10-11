import glob, re, csv
import pandas as pd
from pathlib import Path
from nltk.tokenize import word_tokenize
from nltk import RegexpTokenizer
from collections import defaultdict
from sem.tokenisers import FrenchTokeniser

"""
import spacy
def return_token(sentence):
    nlp = spacy.load('fr_core_news_sm')
    doc = nlp(sentence) # tokenise la phrase
    return [X.text for X in doc] # retourne le texte de chaque token
"""

tokenizer = FrenchTokeniser()
tokenizer_pour_code = RegexpTokenizer(r'''\w'|\w+|[^\w\s]''')


def formatage_annotations():
    """
    Fonction qui renvoie :
    - une liste des noms de documents
    - un dictionnaire des annotations
    :return liste_docs, dict_annotations:
    """
    liste_docs = []
    dict_annotations = {}
    # On récupère les annotations des fichiers ann
    for f in glob.glob('../documents/gold-wikiner-tagset/*.*'):
        liste_ann = []
        if f.endswith(".ann"):
            liste_inter = []
            fichier = open(f, "r").readlines()
            f = f.replace("../documents/gold-wikiner-tagset/", "")
            document = f.replace(".sample.ann", "")
            liste_docs.append(document)

            for ligne in fichier:
                ligne = ligne.replace("\n", "")  # supprime le retour chariot à la fin de chaque phrase
                ligne = re.sub('(.*)\t(.*) (.*) (.*)\t(.*)', '\\1\t\\2\t\\3\t\\4\t\\5', ligne)  # sépare chaque élément d'une annotation avec une tabulation (en tenant compte des 2 qui y sont déjà)
                ligne = ligne.split("\t")  # on split sur les tabulations
                #ligne[0] = re.sub("(T)(\d+)", "\\1.\\2", ligne[0])
                #print(ligne[0])
                liste_inter.append(ligne)
            liste_ann.append(liste_inter)

        # Parcours des annotations
        for liste_temp in liste_ann:
            for liste in liste_temp:
                # Essais des différents tokenisers :
                # entite = return_token(liste[4]) --> SpaCy
                # entite = word_tokenize(liste[4]) --> NLTK
                # entite = tokenizer_pour_code.tokenize(liste[4]) --> regex NLTK
                entite = [liste[4][span.start: span.end] for span in tokenizer.word_spans(liste[4])]
                # print(entite)
                nb_tokens = len(entite)  # on compte le nombre de tokens de chaque EN
                liste.append(nb_tokens)  # on l'ajoute après chaque EN
            dict_annotations[document] = liste_temp

    return liste_docs, dict_annotations


def generer_fichiers_ann_tab(dictionnaire_annotations):
    """
    Fonction qui génère des fichiers ann des samples (version avec tabulations + nb de tokens / EN).
    """
    for cle, valeur in dictionnaire_annotations.items():
        new_doc = ""
        for element in valeur:
            new_ligne = "\t".join(map(str, element))
            new_doc += new_ligne + "\n"
        nouveau_fichier = open('{}.sample_tab.ann'.format(cle), 'w')
        nouveau_fichier.write(new_doc)
        nouveau_fichier.close()
    return None


def generer_format_condense(dict_annotations):
    """
    Fonction qui permet d'écrire des fichiers txt au format condensé (= codes des EN à la place des EN).
    Génère des fichiers txt, ne retourne rien.
    :param dict_annotations:
    :return None:
    """
    # On récupère le texte des fichiers txt
    for f in glob.glob('../documents/gold-wikiner-tagset/*.*'):
        if f.endswith(".txt"):
            new_data = ""
            data = open(f, "r").readlines()  # Path(f).read_text()

            # On récupère le nom de chaque fichier, qui sera le nom de chaque colonne de notre dataframe
            f = f.replace("../documents/gold-wikiner-tagset/", "")
            document = f.replace(".sample.txt", "")
            valeurs = dict_annotations.get(document)

            # On lit la colonne des annotations du document correspondant
            for phrase in data:
                for une_annotation in valeurs:
                    # On remplace l'EN entière par sa classe d'EN
                    phrase = re.sub(une_annotation[4], une_annotation[0], phrase)
                new_data += phrase

        # ---- Pour générer les fichier txt des documents (version condensée) ---- #
        nouveau_fichier = open('{}_condense.txt'.format(document), 'w')
        nouveau_fichier.write(new_data)
        nouveau_fichier.close()
    return None


def chunking(dict_annotations):
    """
    Fonction qui permet de créer des chunks de 200 tokens maximum à partir d'un dictionnaire.
    Renvoie deux dictionnaires.
    :param dict_annotations:
    :return dict_chunks_EN, dict_chunks_normaux:
    """
    dict_chunks_EN = {}
    dict_chunks_normaux = {}
    telephone = re.compile('\s+(\d\d)\s+(\d\d)\s+(\d\d)\s+(\d\d)\s+(\d\d)')

    # On récupère le texte des fichiers txt
    for f in glob.glob('../documents/corpus_condense_code/*.txt'):
        data = Path(f).read_text()

        data = telephone.sub(' \\1\\2\\3\\4\\5', data)
        #data = [data[span.start: span.end] for span in tokenizer.word_spans(data)]
        #data = word_tokenize(data) --> NLTK
        data = tokenizer_pour_code.tokenize(data)

        f = f.replace("../documents/corpus_condense_code/", "")
        document = f.replace("_condense.txt", "")
        print(document)
        print(data)
        valeurs = dict_annotations.get(document)  # on transforme les valeurs du document correspondant en liste
        dico_document = {sous_liste[0]: sous_liste[1:] for sous_liste in
                         valeurs}  # pour transformer chaque premier élément d'une sous-liste (= une annotation entière) en clé, et le reste de la sous-liste en valeurs de cette clé
        #print(dico_document)
        chunks_EN = []
        chunks_normaux = []

        chunk_EN = []
        chunk_normal = []

        for token in data:  # on lit chaque token

            if dico_document.get(token) is not None:  # si le token est une clé du dico, alors c'est que c'est une EN
                token_encode = dico_document.get(token)  # on transforme les valeurs de l'EN correspondante en liste
                test = token_encode[4] + len(
                    chunk_normal)  # on crée une variable test en additionnant la longueur de l'EN (en tokens) + la longueur du chunk courant

                if test < 200:  # si la valeur de test est inférieure à 200
                    token = re.sub(token, dico_document[token][0],
                                   token)  # on remplace le code de l'EN par son type d'EN (MISC, LOC, ORG ou PER)
                    chunk_EN.append(token)  # on ajoute ce token (= type) au chunk courant
                    # print(tokenizer.tokenize(token_encode[3]), token_encode[4]) # pour voir l'EN tokenisée et sa longueur
                    # entite_str = ' '.join(tokenizer.tokenize(token_encode[3]))
                    entite_str = [token_encode[3][span.start: span.end] for span in
                                  tokenizer.word_spans(token_encode[3])]
                    #print(entite_str)
                    for element in entite_str:
                        chunk_normal.append(element)  # on ajoute l'EN littérale tokenisée


                else:  # sinon, résultat du test dépasse 200 (= le chunk dépasserait 200 avec l'ajout de cette EN)
                    chunks_EN.append(chunk_EN)  # on ajoute le chunk courant version EN à la liste de chunks EN
                    chunks_normaux.append(chunk_normal)  # on ajoute le chunk courant version normale à la liste de chunks normaux

                    # On réinitialise les 2 listes de chunks
                    chunk_EN = []
                    chunk_normal = []

                    token = re.sub(token, dico_document[token][0], token)  # on remplace le code de l'EN par son type d'EN (MISC, LOC, ORG ou PER)
                    chunk_EN.append(token)  # on ajoute ce token (= type de l'EN) au chunk courant
                    entite_str = [token_encode[3][span.start: span.end] for span in tokenizer.word_spans(token_encode[3])]

                    # print(entite_str)#entite_str = ' '.join(tokenizer.tokenize(token_encode[3]))
                    for element in entite_str:
                        # print(element)
                        chunk_normal.append(element)  # on ajoute l'EN littérale tokenisée

            else:  # token normal (pas une EN)

                if len(chunk_normal) < 200:  # si la longueur du chunk courant est inférieur à 200
                    # On ajoute le token à nos deux chunks (version EN et version normale)
                    chunk_EN.append(token)
                    chunk_normal.append(token)

                else:  # on a atteint un chunk de 200 tokens
                    chunks_EN.append(chunk_EN)  # on ajoute le chunk courant version EN à la liste de chunks EN
                    chunks_normaux.append(chunk_normal)  # on ajoute le chunk courant version normale à la liste de chunks normaux

                    # On réinitialise les 2 listes de chunks
                    chunk_EN = []
                    chunk_normal = []
                    # On ajoute le token à nos deux chunks (version EN et version normale)
                    chunk_EN.append(token)
                    chunk_normal.append(token)

        # Si on veut ajouter le dernier chunk (nettement inférieur à 200)
        #chunks_EN.append(chunk_EN)  # on ajoute le chunk courant final version EN à la liste de chunks EN
        #chunks_normaux.append(chunk_normal)  # on ajoute le chunk courant final version normale à la liste de chunks normaux

        # On ajoute les chunks à nos dictionnaires de chunks (version EN et version normale), avec les documents en clés
        dict_chunks_EN[document] = chunks_EN
        dict_chunks_normaux[document] = chunks_normaux

    return dict_chunks_EN, dict_chunks_normaux


def generer_chunks_fichier_csv(dictionnaire_chunks_normaux, dictionnaire_chunks_EN):
    """
    Fonction qui génère le dictionnaire final permettant de générer un fichier csv, à partir d'un dictionnaire de chunks.
    :param dictionnaire_chunks_normaux, dictionnaire_chunks_EN:
    :return None:
    """
    dataframe = pd.DataFrame()

    liste_documents = []
    liste_nb_tokens = []
    liste_chunks_EN = []
    liste_chunks_normaux = []
    liste_chunks_tokens = []

    site_internet = re.compile('(www)\s\.\s(\w+)\s\.\s([a-z]+)')
    #site_internet2 = re.compile('(http)\s(:)\s(\/)\s(\/)\s(www)?\s\.\s(\w+|\W+)\s\.\s([a-z]+)')

    for document, chunks in dictionnaire_chunks_normaux.items():
        print(document)
        print("nombre de chunks :", len(chunks))
        for chunk in chunks:
            liste_documents.append(document)
            liste_chunks_tokens.append(chunk)
            print(len(chunk))#, chunk)
            chunk_str = ' '.join(chunk)
            #chunk_str = site_internet2.sub('\\1\\2.\\3.\\4.\\5.\\6.\\7', chunk_str)
            chunk_str = site_internet.sub('\\1.\\2.\\3', chunk_str)
            liste_nb_tokens.append(len(chunk_str.split()))
            liste_chunks_normaux.append(chunk_str)

    for document, chunks in dictionnaire_chunks_EN.items():
        #print(document)
        #print("nombre de chunks :", len(chunks))
        for chunk in chunks:
            #print(len(chunk), chunk)
            chunk_str = ' '.join(chunk)
            chunk_str = site_internet.sub('\\1.\\2.\\3', chunk_str)
            liste_chunks_EN.append(chunk_str)

    dataframe["document"] = liste_documents
    dataframe["chunks_normaux"] = liste_chunks_normaux
    dataframe["chunks_EN"] = liste_chunks_EN
    dataframe["chunks_tokens"] = liste_chunks_tokens
    dataframe["nb_tokens"] = liste_nb_tokens

    #dataframe.to_csv('chunks_features_25.csv', index=False, encoding='utf-8')
    return None


if __name__ == '__main__':
    liste_documents, dico_annotations = formatage_annotations()

    #print("LISTE_DOCUMENTS :")
    #print(liste_documents)

    print("DICO_ANNOTATIONS :")
    print(dico_annotations)
    """for cle, valeur in dico_annotations.items():
        print(cle)
        for i in range(4):
            print(valeur[i])"""
    #generer_fichiers_ann_tab(dico_annotations)
    #generer_format_condense(dico_annotations)

    chunks_EN, chunks_normaux = chunking(dico_annotations)

    generer_chunks_fichier_csv(chunks_normaux, chunks_EN)

