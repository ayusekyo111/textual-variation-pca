import glob, re, csv
from transformers import AutoTokenizer, AutoModelForTokenClassification
from transformers import pipeline
import spacy
import pandas as pd
from flair.data import Sentence
from flair.models import SequenceTagger


def annotation_pos_BERT():
    """
    Fonction permettant d'annoter en partie du discours du texte (sur du texte brut) avec camemBERT. Génère des fichiers txt.
    (modèle : https://huggingface.co/gilf/french-camembert-postag-model)
    :param None:
    :return None:
    """
    tokenizer = AutoTokenizer.from_pretrained("gilf/french-camembert-postag-model", return_offsets_mapping=True)
    model = AutoModelForTokenClassification.from_pretrained("gilf/french-camembert-postag-model")
    nlp_token_class = pipeline('ner', model=model, tokenizer=tokenizer, grouped_entities=True)
    for f in glob.glob('../FENEC/*.txt'):
        print(f)
        fichier = open(f, "r").readlines()
        f = f.replace("../FENEC/", "")
        document = f.replace(".sample.txt", "")
        nouveau_fichier = open('{}_brut_pos_tagging.txt'.format(document), 'a+')
        for ligne in fichier:
            ligne_pos = nlp_token_class(ligne)
            for element in ligne_pos:
                nouveau_fichier.write(str(element))
                nouveau_fichier.write("\n")
    return None


def annotation_pos_spaCy():
    """
    Fonction permettant d'annoter en partie du discours du texte (sur du texte brut) avec spaCy. Génère des fichiers txt.
    :param None:
    :return None:
    """
    nlp = spacy.load("fr_core_news_sm")
    data = pd.read_csv("./chunks_features_25.csv")
    colonne_chunks = data["chunks_normaux"]
    liste_traits = []
    for ligne in colonne_chunks:
        annotations = []
        doc = nlp(ligne)
        #print(doc.text)
        for token in doc:
            # print(token.text, token.pos_, token.morph)
            # print()
            annotation = token.text + "\t" + token.pos_ + "\t" + str(token.morph)
            #print(annotation)
            annotations.append(annotation)
        liste_traits.append(annotations)
    #print(annotations)
    print(len(liste_traits))

    data['annotation_spacy_past'] = liste_traits

    #data.to_csv('chunks_features_25.csv', index=False, encoding='utf-8')
    return None


def annotation_pos_Flair():
    """
    Fonction permettant d'annoter en partie du discours du texte avec Flair.
    (modèle : https://huggingface.co/qanastek/pos-french)
    :param None:
    :return None:
    """
    data = pd.read_csv("./chunks_features_25.csv")
    colonne_chunks = data["chunks_normaux"]
    annotations = []
    liste_pos = []
    model = SequenceTagger.load("qanastek/pos-french")
    for ligne in colonne_chunks:
        # Annotation FLAIR
        ligne_pos = Sentence(ligne)
        model.predict(ligne_pos)
        annotations.append(ligne_pos.to_tagged_string())

    for annotation in annotations:
        annotation = annotation.split("→")
        annotation = annotation[1]
        liste_pos.append(annotation)

    data['annotation_FLAIR'] = liste_pos

    #data.to_csv('chunks_features_25.csv', index=False, encoding='utf-8')

    return None


if __name__ == '__main__':
    #annotation_pos_BERT()
    annotation_pos_spaCy()
    #annotation_pos_Flair()