import pandas as pd
import ast

def extract_pos_from_csv():
    """
    Jeu d'étiquette de Flair : https://huggingface.co/qanastek/pos-french-camembert-flair
    :return:
    """
    data = pd.read_csv("./chunks_features_25.csv")
    colonne_annotations = data["annotation_FLAIR"]
    liste_finale = []
    for annotation in colonne_annotations:
        annotation = annotation.split(",")
        liste_pos = []
        for element in annotation:
            element = element.replace("[", "")
            element = element.replace("]", "")
            element = element.replace("\n", "")
            element = element.replace(" \"", "")
            element = element.replace("\"", "")
            element = element.split("/")
            element = [x for x in element if x != '']
            if len(element) == 1: element.clear() # pour enlever les annotations "vides" (ex : '  ', PUNCT)
            element = list(filter(None, element)) # la fonction clear() laisse des éléments vides (ex : '') dans les listes ; on les retire
            if len(element) != 2: continue # pour ignorer les annotations qui ne sont pas égales à 2 (= pas une paire)
            liste_pos.append(element[1]) # on ne garde que l'annotation
        liste_finale.append(liste_pos)
    data['pos_FLAIR'] = liste_finale
    #data.to_csv('chunks_features_25.csv', index=False, encoding='utf-8')
    return None

def extract_trait_past_from_csv():
    data = pd.read_csv("../tableurs/chunks_features.csv")
    colonne_annotations = data["annotation_spacy_past"]
    liste_finale = []
    count_past = 0
    count_imp = 0
    i = 0
    for annotation in colonne_annotations:
        #print("\nChunk n°", i)
        count_par_chunk = 0
        label = data.loc[data.index, 'label'].values[i]
        document = data.loc[data.index, 'document'].values[i]
        i += 1
        liste_traits = []
        annotation = ast.literal_eval(annotation) # pour lire une chaîne de caractères qui a la forme d'une liste sous forme de liste
        for avant_element, element in zip(annotation, annotation[1:]): # pour pouvoir afficher l'élément qui est juste avant celui qui nous intéresse dans la liste
            element = element.split("\t")
            if "Tense=Past" in element[2]:
                #print("1)", label, "|", document, "|", avant_element)
                #print("2)", label, "|", document, "|", element[0], "|", element[1], "|", element[2])
                print(element[0], "\t", element[1], "\t", element[2])
                liste_traits.append("PAST")
                count_past += 1
                count_par_chunk += 1
            elif "Tense=Imp" in element[2]:
                liste_traits.append("IMP")
                count_imp += 1
                count_par_chunk += 1
            else:
                liste_traits.append("non")
        print("past_tense dans ce chunk :", count_par_chunk)
        liste_finale.append(liste_traits)
        #print("Chunk n°", len(liste_finale), "\n")
    #print("\nTotal de past_tense :", count_past, "\nTotal de imp_tense :", count_imp, )
    data['trait_spacy_past'] = liste_finale
    #data.to_csv('chunks_features_2.csv', index=False, encoding='utf-8')
    return None

if __name__ == "__main__":
    #extract_pos_from_csv()
    extract_trait_past_from_csv()
