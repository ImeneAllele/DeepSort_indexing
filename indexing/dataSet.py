
import pandas as pd  # pour upload dataset
import numpy as np



def upload_data():  # importer le data set
    df = pd.read_csv('object.txt', sep=" ", header=None)  # {importer le dataset dans une dataframe }
    #df.pop(20)  # supprimier la derniere colonne vide
    #df.pop(1)  # supprimier  2 eme colonne _____ id de l'objet estimer
    #df = df.sort_values(by=[0, 2])  # order les data set selon id de l'obejt et le camera qui detecter l'objet
    #df = df.astype(int)  # convertir les valeur de  data set to int
    # df=df[(df.index < 50) ]   # choisir quelque ligne pour le premier test
    #df.to_csv('nvdataset.txt', sep='\t', index=False)  # sauvgrder dataset
    matrix = df.to_numpy()  # convertir dataframe to numpy
    #print( matrix)
    #m=np.mean(matrix, 0)
    #print("m", m)

    return matrix


#t= np.load('/home/etudiant/PycharmProjects/application/output/video1_test_outputdata.npy',allow_pickle=True)
#print(t)




