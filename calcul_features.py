import pandas as pd
import numpy as np

labels = ["prose", "parole", "informations", "encyclopedie", "poesie"]


def lire_fichier_csv(fichier_csv):
    dataframe = pd.read_csv(fichier_csv)
    dataframe.index = np.arange(1, len(dataframe) + 1) # pour commencer l'index à 1 et non à 0 (important pour récupérer les bonnes valeurs de la colonne nb_tokens)
    return dataframe

# 1
def count_LOC(dataframe, dataframe_pour_count) :
    colonne_chunks = dataframe["chunks_EN"]
    liste_LOC = []
    liste_nb = []
    i = 0
    dico_labels = {}
    for chunk in colonne_chunks:
        nb_tokens = dataframe.loc[dataframe.index,'nb_tokens'].values[i]
        nb_LOC = chunk.count("LOC")
        moyenne_LOC = nb_LOC / nb_tokens
        liste_LOC.append(moyenne_LOC)
        liste_nb.append(nb_LOC)
        # Pour calculer le nombre d'occurrences de ce type par genre
        label = dataframe.loc[dataframe.index, 'label'].values[i]
        if label in dico_labels.keys():
            dico_labels[label] += nb_LOC
        else : dico_labels[label] = nb_LOC
        i += 1
    print("count_LOC :\n", dico_labels)
    dataframe['feature_LOC'] = liste_LOC
    dataframe_pour_count['count_LOC'] = liste_nb
    return dataframe, dataframe_pour_count

# 2
def count_ORG(dataframe, dataframe_pour_count) :
    colonne_chunks = dataframe["chunks_EN"]
    liste_ORG = []
    liste_nb = []
    i = 0
    dico_labels = {}
    for chunk in colonne_chunks:
        nb_tokens = dataframe.loc[dataframe.index,'nb_tokens'].values[i]
        nb_ORG = chunk.count("ORG")
        moyenne_ORG = nb_ORG / nb_tokens
        liste_ORG.append(moyenne_ORG)
        liste_nb.append(nb_ORG)
        # Pour calculer le nombre d'occurrences de ce type par genre
        label = dataframe.loc[dataframe.index, 'label'].values[i]
        if label in dico_labels.keys():
            dico_labels[label] += nb_ORG
        else:
            dico_labels[label] = nb_ORG
        i += 1
    print("count_ORG :\n", dico_labels)
    dataframe['feature_ORG'] = liste_ORG
    dataframe_pour_count['count_ORG'] = liste_nb
    return dataframe, dataframe_pour_count

# 3
def count_PER(dataframe, dataframe_pour_count) :
    colonne_chunks = dataframe["chunks_EN"]
    liste_PER = []
    liste_nb = []
    i = 0
    dico_labels = {}
    for chunk in colonne_chunks:
        nb_tokens = dataframe.loc[dataframe.index,'nb_tokens'].values[i]
        nb_PER = chunk.count("PER")
        moyenne_PER = nb_PER / nb_tokens
        liste_PER.append(moyenne_PER)
        liste_nb.append(nb_PER)
        # Pour calculer le nombre d'occurrences de ce type par genre
        label = dataframe.loc[dataframe.index, 'label'].values[i]
        if label in dico_labels.keys():
            dico_labels[label] += nb_PER
        else:
            dico_labels[label] = nb_PER
        i += 1
    print("count_PER :\n", dico_labels)
    dataframe['feature_PER'] = liste_PER
    dataframe_pour_count['count_PER'] = liste_nb
    return dataframe, dataframe_pour_count

# 4
def count_MISC(dataframe, dataframe_pour_count) :
    colonne_chunks = dataframe["chunks_EN"]
    liste_MISC = []
    liste_nb = []
    i = 0
    dico_labels = {}
    for chunk in colonne_chunks:
        nb_tokens = dataframe.loc[dataframe.index,'nb_tokens'].values[i]
        nb_MISC = chunk.count("MISC")
        moyenne_MISC = nb_MISC / nb_tokens
        liste_MISC.append(moyenne_MISC)
        liste_nb.append(nb_MISC)
        # Pour calculer le nombre d'occurrences de ce type par genre
        label = dataframe.loc[dataframe.index, 'label'].values[i]
        if label in dico_labels.keys():
            dico_labels[label] += nb_MISC
        else:
            dico_labels[label] = nb_MISC
        i += 1
    print("count_MISC :\n", dico_labels)
    dataframe['feature_MISC'] = liste_MISC
    dataframe_pour_count['count_MISC'] = liste_nb
    return dataframe, dataframe_pour_count

# 5
def count_TOTAL_EN(dataframe, dataframe_pour_count) :
    colonne_chunks = dataframe["chunks_EN"]
    liste_TOTAL_EN = []
    liste_nb = []
    i = 0
    dico_labels = {}
    for chunk in colonne_chunks:
        nb_tokens = dataframe.loc[dataframe.index,'nb_tokens'].values[i]
        nb_LOC = chunk.count("LOC")
        nb_MISC = chunk.count("MISC")
        nb_ORG = chunk.count("ORG")
        nb_PER = chunk.count("PER")
        total_EN = nb_LOC + nb_MISC + nb_ORG + nb_PER
        moyenne_TOTAL_EN = total_EN / nb_tokens
        liste_TOTAL_EN.append(moyenne_TOTAL_EN)
        liste_nb.append(total_EN)
        # Pour calculer le nombre d'occurrences de ce type par genre
        label = dataframe.loc[dataframe.index, 'label'].values[i]
        if label in dico_labels.keys():
            dico_labels[label] += total_EN
        else:
            dico_labels[label] = total_EN
        i += 1
    print("count_TOTAL_EN :\n", dico_labels)
    dataframe['feature_TOTAL_EN'] = liste_TOTAL_EN
    dataframe_pour_count['count_TOTAL_EN'] = liste_nb
    return dataframe, dataframe_pour_count

# 6
def count_LONGUEUR_MOTS(dataframe) :
    colonne_chunks = dataframe["chunks_EN"]
    liste_LONGUEUR_MOTS = []
    i = 0
    for chunk in colonne_chunks:
        nb_tokens = dataframe.loc[dataframe.index,'nb_tokens'].values[i]
        moyenne_LONGUEUR_MOTS = sum(len(mot) for mot in chunk) / nb_tokens
        liste_LONGUEUR_MOTS.append(moyenne_LONGUEUR_MOTS)
        i += 1
    dataframe['feature_LONGUEUR_MOTS'] = liste_LONGUEUR_MOTS
    return dataframe

# 7
def count_ADP(dataframe, dataframe_pour_count) :
    colonne_annotations = dataframe["pos_FLAIR"]
    liste_ADP = []
    liste_nb = []
    i = 0
    dico_labels = {}
    for annotation in colonne_annotations:
        nb_tokens = dataframe.loc[dataframe.index, 'nb_tokens'].values[i]
        nb_PREP = annotation.count("PREP")
        nb_PART = annotation.count("PART")
        total_ADP = nb_PREP + nb_PART
        moyenne_ADP = total_ADP / nb_tokens
        liste_ADP.append(moyenne_ADP)
        liste_nb.append(total_ADP)
        # Pour calculer le nombre d'occurrences de ce type par genre
        label = dataframe.loc[dataframe.index, 'label'].values[i]
        if label in dico_labels.keys():
            dico_labels[label] += total_ADP
        else:
            dico_labels[label] = total_ADP
        i += 1
    print("count_ADP :\n", dico_labels)
    dataframe['feature_ADP'] = liste_ADP
    dataframe_pour_count['count_ADP'] = liste_nb
    return dataframe, dataframe_pour_count

# 8
def count_SCONJ(dataframe, dataframe_pour_count) :
    colonne_annotations = dataframe["pos_FLAIR"]
    liste_SCONJ = []
    liste_nb = []
    i = 0
    dico_labels = {}
    for annotation in colonne_annotations:
        nb_tokens = dataframe.loc[dataframe.index, 'nb_tokens'].values[i]
        nb_SCONJ = annotation.count("COSUB")
        moyenne_SCONJ = nb_SCONJ / nb_tokens
        liste_SCONJ.append(moyenne_SCONJ)
        liste_nb.append(nb_SCONJ)
        # Pour calculer le nombre d'occurrences de ce type par genre
        label = dataframe.loc[dataframe.index, 'label'].values[i]
        if label in dico_labels.keys():
            dico_labels[label] += nb_SCONJ
        else:
            dico_labels[label] = nb_SCONJ
        i += 1
    print("count_SCONJ :\n", dico_labels)
    dataframe['feature_SCONJ'] = liste_SCONJ
    dataframe_pour_count['count_SCONJ'] = liste_nb
    return dataframe, dataframe_pour_count

# 9
def count_CCONJ(dataframe, dataframe_pour_count) :
    colonne_annotations = dataframe["pos_FLAIR"]
    liste_CCONJ = []
    liste_nb = []
    i = 0
    dico_labels = {}
    for annotation in colonne_annotations:
        nb_tokens = dataframe.loc[dataframe.index, 'nb_tokens'].values[i]
        nb_CCONJ = annotation.count("COCO")
        moyenne_CCONJ = nb_CCONJ / nb_tokens
        liste_CCONJ.append(moyenne_CCONJ)
        liste_nb.append(nb_CCONJ)
        # Pour calculer le nombre d'occurrences de ce type par genre
        label = dataframe.loc[dataframe.index, 'label'].values[i]
        if label in dico_labels.keys():
            dico_labels[label] += nb_CCONJ
        else:
            dico_labels[label] = nb_CCONJ
        i += 1
    print("count_CCONJ :\n", dico_labels)
    dataframe['feature_CCONJ'] = liste_CCONJ
    dataframe_pour_count['count_CCONJ'] = liste_nb
    return dataframe, dataframe_pour_count

# 10
def count_ADV(dataframe, dataframe_pour_count) :
    colonne_annotations = dataframe["pos_FLAIR"]
    liste_ADV = []
    liste_nb = []
    i = 0
    dico_labels = {}
    for annotation in colonne_annotations:
        nb_tokens = dataframe.loc[dataframe.index, 'nb_tokens'].values[i]
        nb_ADV = annotation.count("ADV")
        moyenne_ADV = nb_ADV / nb_tokens
        liste_ADV.append(moyenne_ADV)
        liste_nb.append(nb_ADV)
        # Pour calculer le nombre d'occurrences de ce type par genre
        label = dataframe.loc[dataframe.index, 'label'].values[i]
        if label in dico_labels.keys():
            dico_labels[label] += nb_ADV
        else:
            dico_labels[label] = nb_ADV
        i += 1
    print("count_ADV :\n", dico_labels)
    dataframe['feature_ADV'] = liste_ADV
    dataframe_pour_count['count_ADV'] = liste_nb
    return dataframe, dataframe_pour_count

# 11
def count_PROPN(dataframe, dataframe_pour_count) :
    colonne_annotations = dataframe["pos_FLAIR"]
    liste_PROPN = []
    liste_nb = []
    i = 0
    dico_labels = {}
    for annotation in colonne_annotations:
        nb_tokens = dataframe.loc[dataframe.index, 'nb_tokens'].values[i]
        nb_PROPN = annotation.count("PROPN")
        nb_XFAMIL = annotation.count("XFAMIL")
        total_PROPN = nb_PROPN + nb_XFAMIL
        moyenne_PROPN = total_PROPN / nb_tokens
        liste_PROPN.append(moyenne_PROPN)
        liste_nb.append(total_PROPN)
        # Pour calculer le nombre d'occurrences de ce type par genre
        label = dataframe.loc[dataframe.index, 'label'].values[i]
        if label in dico_labels.keys():
            dico_labels[label] += total_PROPN
        else:
            dico_labels[label] = total_PROPN
        i += 1
    print("count_PROPN :\n", dico_labels)
    dataframe['feature_PROPN'] = liste_PROPN
    dataframe_pour_count['count_PROPN'] = liste_nb
    return dataframe, dataframe_pour_count

# 12
def count_NUM(dataframe, dataframe_pour_count) :
    colonne_annotations = dataframe["pos_FLAIR"]
    liste_NUM = []
    liste_nb = []
    i = 0
    dico_labels = {}
    for annotation in colonne_annotations:
        nb_tokens = dataframe.loc[dataframe.index, 'nb_tokens'].values[i]
        nb_NUM = annotation.count("NUM")
        nb_CHIF = annotation.count("CHIF")
        total_NUM = nb_NUM + nb_CHIF
        moyenne_NUM = total_NUM / nb_tokens
        liste_NUM.append(moyenne_NUM)
        liste_nb.append(total_NUM)
        # Pour calculer le nombre d'occurrences de ce type par genre
        label = dataframe.loc[dataframe.index, 'label'].values[i]
        if label in dico_labels.keys():
            dico_labels[label] += total_NUM
        else:
            dico_labels[label] = total_NUM
        i += 1
    print("count_NUM :\n", dico_labels)
    dataframe['feature_NUM'] = liste_NUM
    dataframe_pour_count['count_NUM'] = liste_nb
    return dataframe, dataframe_pour_count

# 13
def count_AUX(dataframe, dataframe_pour_count) :
    colonne_annotations = dataframe["pos_FLAIR"]
    liste_AUX = []
    liste_nb = []
    i = 0
    dico_labels = {}
    for annotation in colonne_annotations:
        nb_tokens = dataframe.loc[dataframe.index, 'nb_tokens'].values[i]
        nb_AUX = annotation.count("AUX")
        moyenne_AUX = nb_AUX / nb_tokens
        liste_AUX.append(moyenne_AUX)
        liste_nb.append(nb_AUX)
        # Pour calculer le nombre d'occurrences de ce type par genre
        label = dataframe.loc[dataframe.index, 'label'].values[i]
        if label in dico_labels.keys():
            dico_labels[label] += nb_AUX
        else:
            dico_labels[label] = nb_AUX
        i += 1
    print("count_AUX :\n", dico_labels)
    dataframe['feature_AUX'] = liste_AUX
    dataframe_pour_count['count_AUX'] = liste_nb
    return dataframe, dataframe_pour_count

# 14
def count_VERB(dataframe, dataframe_pour_count) :
    colonne_annotations = dataframe["pos_FLAIR"]
    liste_VERB = []
    liste_nb = []
    i = 0
    dico_labels = {}
    for annotation in colonne_annotations:
        nb_tokens = dataframe.loc[dataframe.index, 'nb_tokens'].values[i]
        nb_VERB = annotation.count("VERB")

        nb_VPPFS = annotation.count("VPPFS")
        nb_VPPFP = annotation.count("VPPFP")
        nb_VPPMS = annotation.count("VPPMS")
        nb_VPPMP = annotation.count("VPPMP")

        nb_VPPRE = annotation.count("VPPRE")

        total_VERB = nb_VERB + nb_VPPFS + nb_VPPFP + nb_VPPMS + nb_VPPMP + nb_VPPRE
        moyenne_VERB = total_VERB / nb_tokens
        liste_VERB.append(moyenne_VERB)
        liste_nb.append(total_VERB)
        # Pour calculer le nombre d'occurrences de ce type par genre
        label = dataframe.loc[dataframe.index, 'label'].values[i]
        if label in dico_labels.keys():
            dico_labels[label] += total_VERB
        else:
            dico_labels[label] = total_VERB
        i += 1
    print("count_VERB :\n", dico_labels)
    dataframe['feature_VERB'] = liste_VERB
    dataframe_pour_count['count_VERB'] = liste_nb
    return dataframe, dataframe_pour_count

# 15
def count_DET(dataframe, dataframe_pour_count) :
    colonne_annotations = dataframe["pos_FLAIR"]
    liste_DET = []
    liste_nb = []
    i = 0
    dico_labels = {}
    for annotation in colonne_annotations:
        nb_tokens = dataframe.loc[dataframe.index, 'nb_tokens'].values[i]
        nb_DET = annotation.count("DET")
        nb_DETMS = annotation.count("DETMS")
        nb_DETFS = annotation.count("DETFS")
        total_DET = nb_DET + nb_DETMS + nb_DETFS
        moyenne_DET = total_DET / nb_tokens
        liste_DET.append(moyenne_DET)
        liste_nb.append(total_DET)
        # Pour calculer le nombre d'occurrences de ce type par genre
        label = dataframe.loc[dataframe.index, 'label'].values[i]
        if label in dico_labels.keys():
            dico_labels[label] += total_DET
        else:
            dico_labels[label] = total_DET
        i += 1
    print("count_DET :\n", dico_labels)
    dataframe['feature_DET'] = liste_DET
    dataframe_pour_count['count_DET'] = liste_nb
    return dataframe, dataframe_pour_count

# 16
def count_ADJ(dataframe, dataframe_pour_count) :
    colonne_annotations = dataframe["pos_FLAIR"]
    liste_ADJ = []
    liste_nb = []
    i = 0
    dico_labels = {}
    for annotation in colonne_annotations:
        nb_tokens = dataframe.loc[dataframe.index, 'nb_tokens'].values[i]
        nb_ADJ = annotation.count("ADJ")
        nb_ADJMS = annotation.count("ADJMS")
        nb_ADJMP = annotation.count("ADJMP")
        nb_ADJFS = annotation.count("ADJFS")
        nb_ADJFP = annotation.count("ADJFP")
        total_ADJ = nb_ADJ + nb_ADJMS + nb_ADJMP + nb_ADJFS + nb_ADJFP
        moyenne_ADJ = total_ADJ / nb_tokens
        liste_ADJ.append(moyenne_ADJ)
        liste_nb.append(total_ADJ)
        # Pour calculer le nombre d'occurrences de ce type par genre
        label = dataframe.loc[dataframe.index, 'label'].values[i]
        if label in dico_labels.keys():
            dico_labels[label] += total_ADJ
        else:
            dico_labels[label] = total_ADJ
        i += 1
    print("count_ADJ :\n", dico_labels)
    dataframe['feature_ADJ'] = liste_ADJ
    dataframe_pour_count['count_ADJ'] = liste_nb
    return dataframe, dataframe_pour_count

# 17
def count_NOUN(dataframe, dataframe_pour_count) :
    colonne_annotations = dataframe["pos_FLAIR"]
    liste_NOUN = []
    liste_nb = []
    i = 0
    dico_labels = {}
    for annotation in colonne_annotations:
        nb_tokens = dataframe.loc[dataframe.index, 'nb_tokens'].values[i]
        nb_NOUN = annotation.count("NOUN")
        nb_NMS = annotation.count("NMS")
        nb_NMP = annotation.count("NMP")
        nb_NFS = annotation.count("NFS")
        nb_NFP = annotation.count("NFP")
        total_NOUN = nb_NOUN + nb_NMS + nb_NMP + nb_NFS + nb_NFP
        moyenne_NOUN = total_NOUN / nb_tokens
        liste_NOUN.append(moyenne_NOUN)
        liste_nb.append(total_NOUN)
        # Pour calculer le nombre d'occurrences de ce type par genre
        label = dataframe.loc[dataframe.index, 'label'].values[i]
        if label in dico_labels.keys():
            dico_labels[label] += total_NOUN
        else:
            dico_labels[label] = total_NOUN
        i += 1
    print("count_NOUN :\n", dico_labels)
    dataframe['feature_NOUN'] = liste_NOUN
    dataframe_pour_count['count_NOUN'] = liste_nb
    return dataframe, dataframe_pour_count

# 18
def count_PRON(dataframe, dataframe_pour_count) :
    colonne_annotations = dataframe["pos_FLAIR"]
    liste_PRON = []
    liste_nb = []
    i = 0
    dico_labels = {}
    for annotation in colonne_annotations:
        nb_tokens = dataframe.loc[dataframe.index, 'nb_tokens'].values[i]
        nb_PRON = annotation.count("PRON")
        nb_PINT = annotation.count("PINT")

        nb_PDEMMS = annotation.count("PDEMMS")
        nb_PDEMMP = annotation.count("PDEMMP")
        nb_PDEMFS = annotation.count("PDEMFS")
        nb_PDEMFP = annotation.count("PDEMFP")

        nb_PINDMS = annotation.count("PINDMS")
        nb_PINDMP = annotation.count("PINDMP")
        nb_PINDFS = annotation.count("PINDFS")
        nb_PINDFP = annotation.count("PINDFP")

        nb_PPOBJMS = annotation.count("PPOBJMS")
        nb_PPOBJMP = annotation.count("PPOBJMP")
        nb_PPOBJFS = annotation.count("PPOBJFS")
        nb_PPOBJFP = annotation.count("PPOBJFP")

        nb_PREFS = annotation.count("PREFS")
        nb_PREF = annotation.count("PREF")
        nb_PREFP = annotation.count("PREFP")
        nb_PREL = annotation.count("PREL")

        nb_PRELMS = annotation.count("PRELMS")
        nb_PRELMP = annotation.count("PRELMP")
        nb_PRELFS = annotation.count("PRELFS")
        nb_PRELFP = annotation.count("PRELFP")

        total_PRON = nb_PRON +nb_PINT +nb_PDEMMS +nb_PDEMMP +nb_PDEMFS +nb_PDEMFP+nb_PINDMS +nb_PINDMP +nb_PINDFS +nb_PINDFP +nb_PPOBJMS +nb_PPOBJMP +nb_PPOBJFS +nb_PPOBJFP +nb_PREFS +nb_PREF +nb_PREFP +nb_PREL +nb_PRELMS +nb_PRELMP +nb_PRELFS +nb_PRELFP

        moyenne_PRON = total_PRON / nb_tokens
        liste_PRON.append(moyenne_PRON)
        liste_nb.append(total_PRON)
        # Pour calculer le nombre d'occurrences de ce type par genre
        label = dataframe.loc[dataframe.index, 'label'].values[i]
        if label in dico_labels.keys():
            dico_labels[label] += total_PRON
        else:
            dico_labels[label] = total_PRON
        i += 1
    print("count_PRON :\n", dico_labels)
    dataframe['feature_PRON'] = liste_PRON
    dataframe_pour_count['count_PRON'] = liste_nb
    return dataframe, dataframe_pour_count

# 19
def count_PPER1S(dataframe, dataframe_pour_count) :
    colonne_annotations = dataframe["pos_FLAIR"]
    liste_PPER1S = []
    liste_nb = []
    i = 0
    dico_labels = {}
    for annotation in colonne_annotations:
        nb_tokens = dataframe.loc[dataframe.index, 'nb_tokens'].values[i]
        nb_PPER1S = annotation.count("PPER1S")
        moyenne_PPER1S = nb_PPER1S / nb_tokens
        liste_PPER1S.append(moyenne_PPER1S)
        liste_nb.append(nb_PPER1S)
        # Pour calculer le nombre d'occurrences de ce type par genre
        label = dataframe.loc[dataframe.index, 'label'].values[i]
        if label in dico_labels.keys():
            dico_labels[label] += nb_PPER1S
        else:
            dico_labels[label] = nb_PPER1S
        i += 1
    print("count_PPER1S :\n", dico_labels)
    dataframe['feature_PPER1S'] = liste_PPER1S
    dataframe_pour_count['count_PPER1S'] = liste_nb
    return dataframe, dataframe_pour_count

# 20
def count_PPER2S(dataframe, dataframe_pour_count) :
    colonne_annotations = dataframe["pos_FLAIR"]
    liste_PPER2S = []
    liste_nb = []
    i = 0
    dico_labels = {}
    for annotation in colonne_annotations:
        nb_tokens = dataframe.loc[dataframe.index, 'nb_tokens'].values[i]
        nb_PPER2S = annotation.count("PPER2S")
        moyenne_PPER2S = nb_PPER2S / nb_tokens
        liste_PPER2S.append(moyenne_PPER2S)
        liste_nb.append(nb_PPER2S)
        # Pour calculer le nombre d'occurrences de ce type par genre
        label = dataframe.loc[dataframe.index, 'label'].values[i]
        if label in dico_labels.keys():
            dico_labels[label] += nb_PPER2S
        else:
            dico_labels[label] = nb_PPER2S
        i += 1
    print("count_PPER2S :\n", dico_labels)
    dataframe['feature_PPER2S'] = liste_PPER2S
    dataframe_pour_count['count_PPERS2S'] = liste_nb
    return dataframe, dataframe_pour_count

# 21
def count_PPER3(dataframe, dataframe_pour_count) :
    colonne_annotations = dataframe["pos_FLAIR"]
    liste_PPER3 = []
    liste_nb = []
    i = 0
    dico_labels = {}
    for annotation in colonne_annotations:
        nb_tokens = dataframe.loc[dataframe.index, 'nb_tokens'].values[i]
        nb_PPER3MS = annotation.count("PPER3MS")
        nb_PPER3MP = annotation.count("PPER3MP")
        nb_PPER3FS = annotation.count("PPER3FS")
        nb_PPER3FP = annotation.count("PPER3FP")
        total_PPER3 = nb_PPER3MS + nb_PPER3MP + nb_PPER3FS + nb_PPER3FP
        moyenne_PPER3 = total_PPER3 / nb_tokens
        liste_PPER3.append(moyenne_PPER3)
        liste_nb.append(total_PPER3)
        # Pour calculer le nombre d'occurrences de ce type par genre
        label = dataframe.loc[dataframe.index, 'label'].values[i]
        if label in dico_labels.keys():
            dico_labels[label] += total_PPER3
        else:
            dico_labels[label] = total_PPER3
        i += 1
    print("count_PPER3 :\n", dico_labels)
    dataframe['feature_PPER3'] = liste_PPER3
    dataframe_pour_count['count_PPER3'] = liste_nb
    return dataframe, dataframe_pour_count

# 22
def count_INTJ(dataframe, dataframe_pour_count) :
    colonne_annotations = dataframe["pos_FLAIR"]
    liste_INTJ = []
    liste_nb = []
    i = 0
    dico_labels = {}
    for annotation in colonne_annotations:
        nb_tokens = dataframe.loc[dataframe.index, 'nb_tokens'].values[i]
        nb_INTJ = annotation.count("INTJ")
        moyenne_INTJ = nb_INTJ / nb_tokens
        liste_INTJ.append(moyenne_INTJ)
        liste_nb.append(nb_INTJ)
        # Pour calculer le nombre d'occurrences de ce type par genre
        label = dataframe.loc[dataframe.index, 'label'].values[i]
        if label in dico_labels.keys():
            dico_labels[label] += nb_INTJ
        else:
            dico_labels[label] = nb_INTJ
        i += 1
    print("count_INTJ :\n", dico_labels)
    dataframe['feature_INTJ'] = liste_INTJ
    dataframe_pour_count['count_INTJ'] = liste_nb
    return dataframe, dataframe_pour_count

# 23
def count_SYM(dataframe, dataframe_pour_count) :
    colonne_annotations = dataframe["pos_FLAIR"]
    liste_SYM = []
    liste_nb = []
    i = 0
    dico_labels = {}
    for annotation in colonne_annotations:
        nb_tokens = dataframe.loc[dataframe.index, 'nb_tokens'].values[i]
        nb_SYM = annotation.count("SYM")
        moyenne_SYM = nb_SYM / nb_tokens
        liste_SYM.append(moyenne_SYM)
        liste_nb.append(nb_SYM)
        # Pour calculer le nombre d'occurrences de ce type par genre
        label = dataframe.loc[dataframe.index, 'label'].values[i]
        if label in dico_labels.keys():
            dico_labels[label] += nb_SYM
        else:
            dico_labels[label] = nb_SYM
        i += 1
    print("count_SYM :\n", dico_labels)
    dataframe['feature_SYM'] = liste_SYM
    dataframe_pour_count['count_SYM'] = liste_nb
    return dataframe, dataframe_pour_count

# 24
def count_PUNCT(dataframe, dataframe_pour_count) :
    colonne_annotations = dataframe["pos_FLAIR"]
    liste_PUNCT = []
    liste_nb = []
    i = 0
    dico_labels = {}
    for annotation in colonne_annotations:
        nb_tokens = dataframe.loc[dataframe.index, 'nb_tokens'].values[i]
        nb_PUNCT = annotation.count("PUNCT")
        nb_YPFOR = annotation.count("YPFOR")
        total_PUNCT = nb_PUNCT + nb_YPFOR
        moyenne_PUNCT = total_PUNCT / nb_tokens
        liste_PUNCT.append(moyenne_PUNCT)
        liste_nb.append(total_PUNCT)
        # Pour calculer le nombre d'occurrences de ce type par genre
        label = dataframe.loc[dataframe.index, 'label'].values[i]
        if label in dico_labels.keys():
            dico_labels[label] += total_PUNCT
        else:
            dico_labels[label] = total_PUNCT
        i += 1
    print("count_PUNCT :\n", dico_labels)
    dataframe['feature_PUNCT'] = liste_PUNCT
    dataframe_pour_count['count_PUNCT'] = liste_nb
    return dataframe, dataframe_pour_count

# 25
def count_X(dataframe, dataframe_pour_count) :
    colonne_annotations = dataframe["pos_FLAIR"]
    liste_X = []
    liste_nb = []
    i = 0
    dico_labels = {}
    for annotation in colonne_annotations:
        nb_tokens = dataframe.loc[dataframe.index, 'nb_tokens'].values[i]
        nb_MOTINC = annotation.count("MOTINC")
        nb_X = annotation.count("X")
        total_X = nb_MOTINC + nb_X
        moyenne_X = total_X / nb_tokens
        liste_X.append(moyenne_X)
        liste_nb.append(total_X)
        # Pour calculer le nombre d'occurrences de ce type par genre
        label = dataframe.loc[dataframe.index, 'label'].values[i]
        if label in dico_labels.keys():
            dico_labels[label] += total_X
        else:
            dico_labels[label] = total_X
        i += 1
    print("count_X :\n", dico_labels)
    dataframe['feature_X'] = liste_X
    dataframe_pour_count['count_X'] = liste_nb
    return dataframe, dataframe_pour_count

# 26
def count_PAST_TENSE(dataframe, dataframe_pour_count) :
    colonne_annotations = dataframe["trait_spacy_past"]
    liste_PAST = []
    liste_nb = []
    i = 0
    dico_labels = {}
    for annotation in colonne_annotations:
        nb_tokens = dataframe.loc[dataframe.index, 'nb_tokens'].values[i]
        nb_PAST = annotation.count("PAST")
        nb_IMP = annotation.count("IMP")
        total_PAST = nb_PAST + nb_IMP
        moyenne_PAST = total_PAST / nb_tokens
        liste_PAST.append(moyenne_PAST)
        liste_nb.append(total_PAST)
        # Pour calculer le nombre d'occurrences de ce type par genre
        label = dataframe.loc[dataframe.index, 'label'].values[i]
        if label in dico_labels.keys():
            dico_labels[label] += total_PAST
        else:
            dico_labels[label] = total_PAST
        i += 1
    print("count_PAST_TENSE :\n", dico_labels)
    dataframe['feature_PAST_TENSE'] = liste_PAST
    dataframe_pour_count['count_PAST_TENSE'] = liste_nb
    return dataframe, dataframe_pour_count


if __name__ == '__main__':
    dataframe = lire_fichier_csv("../tableurs/chunks_features.csv")
    dataframe_pour_count = pd.DataFrame(index=dataframe['label'])

    dataframe, dataframe_pour_count = count_LOC(dataframe, dataframe_pour_count) # 1
    dataframe, dataframe_pour_count = count_ORG(dataframe, dataframe_pour_count) # 2
    dataframe, dataframe_pour_count = count_PER(dataframe, dataframe_pour_count) # 3
    dataframe, dataframe_pour_count = count_MISC(dataframe, dataframe_pour_count) # 4
    dataframe, dataframe_pour_count = count_TOTAL_EN(dataframe, dataframe_pour_count) # 5
    dataframe = count_LONGUEUR_MOTS(dataframe) # 6
    dataframe, dataframe_pour_count = count_ADP(dataframe, dataframe_pour_count) # 7
    dataframe, dataframe_pour_count = count_SCONJ(dataframe, dataframe_pour_count) # 8
    dataframe, dataframe_pour_count = count_CCONJ(dataframe, dataframe_pour_count) # 9
    dataframe, dataframe_pour_count = count_ADV(dataframe, dataframe_pour_count) # 10
    dataframe, dataframe_pour_count = count_PROPN(dataframe, dataframe_pour_count) # 11
    dataframe, dataframe_pour_count = count_NUM(dataframe, dataframe_pour_count) # 12
    dataframe, dataframe_pour_count = count_AUX(dataframe, dataframe_pour_count) # 13
    dataframe, dataframe_pour_count = count_VERB(dataframe, dataframe_pour_count) # 14
    dataframe, dataframe_pour_count = count_DET(dataframe, dataframe_pour_count) # 15
    dataframe, dataframe_pour_count = count_ADJ(dataframe, dataframe_pour_count) # 16
    dataframe, dataframe_pour_count = count_NOUN(dataframe, dataframe_pour_count)  # 17
    dataframe, dataframe_pour_count = count_PRON(dataframe, dataframe_pour_count) # 18
    dataframe, dataframe_pour_count = count_PPER1S(dataframe, dataframe_pour_count) # 19
    dataframe, dataframe_pour_count = count_PPER2S(dataframe, dataframe_pour_count) # 20
    dataframe, dataframe_pour_count = count_PPER3(dataframe, dataframe_pour_count) # 21
    dataframe, dataframe_pour_count = count_INTJ(dataframe, dataframe_pour_count) # 22
    dataframe, dataframe_pour_count = count_SYM(dataframe, dataframe_pour_count) # 23
    dataframe, dataframe_pour_count = count_PUNCT(dataframe, dataframe_pour_count) # 24
    dataframe, dataframe_pour_count = count_X(dataframe, dataframe_pour_count) # 25
    dataframe, dataframe_pour_count = count_PAST_TENSE(dataframe, dataframe_pour_count) # 26
    #print(list(dataframe.columns.values[8:]))

    #dataframe.to_csv('chunks_features_3.csv', index=False, encoding='utf-8')
    #dataframe_pour_count.to_csv('count_EN_POS_chunks.csv', encoding='utf-8')