import numpy as np

from indexing.distance import *
import time
from scipy.spatial import distance


# initilisation vr global



global nb_comparaison
nb_comparaison=0
global nb_distance
nb_distance =0
global nombre_regrpm




def nouedintrne(tree):
    if tree.right is None:
        return 0
    else:
        # print(node)
        return 1 + nouedintrne(tree.left) + nouedintrne(tree.right)


def taille(tree): # total nodes number
    if tree.right is None:
        return 1
    else:
        # print(node)
        return 1 + taille(tree.left) + taille(tree.right)

def get_height(tree):
        if tree is None :
            return 0
        else :
            return 1 + max(
               get_height( tree.left) if tree.left else -1,
               get_height( tree.right) if tree.right else -1
            )

def noued_niveau(tree):   # nombre des noued par niveau
    if tree is None :
        L=[0]
        return L
    else :
        if tree.right is  None :
            return [1]
        else :
            L= fusion(noued_niveau(tree.right), noued_niveau(tree.left))
    return L





def fusion(l1,l2):
    l=[]
    i=0
    l.append(1)
    if (len(l1)  > len(l2)):
      while (i < len(l2)):
          l.append(l1[i]+ l2[i])
          i+=1

      while (i < len(l1)):
          l.append(l1[i])
          i+=1
    else:
        while (i < len(l1)):
            l.append(l1[i] + l2[i])
            i += 1

        while (i < len(l2)):
            l.append(l2[i])
            i += 1
    return l


def nombre_feuille(tree):
    if tree is None :
        return 0
    else:
        if tree.right is None :
              return 1 + nombre_feuille(tree.right) + nombre_feuille(tree.left)
        else:
            return nombre_feuille(tree.right)+nombre_feuille(tree.left)




def EvaluationSAC_guache_droite(Tree, ListeSAC_Gauche, ListeSAC_Droite , LG ,LD):
    if Tree.right.pivotD is None :
        L = []

        for v in Tree.right.sac:
            #print("v evaluation ", len(v))
            L.append(v)
        #ListeSAC_Droite.append(Tree.right.sac)
        ListeSAC_Droite.append(L)
        LD.append(Tree.right.size)
    else :
        (ListeSAC_Gauche, ListeSAC_Droite , LG ,LD) = EvaluationSAC_guache_droite(Tree.right, ListeSAC_Gauche, ListeSAC_Droite , LG ,LD)

    if Tree.left.pivotG is None :
        L = []
        for v in Tree.left.sac:
          L.append(v)
        #ListeSAC_Gauche.append(Tree.left.sac)
        ListeSAC_Gauche.append(L)
        LG.append(Tree.left.size)
    else :
        (ListeSAC_Gauche, ListeSAC_Droite , LG ,LD) = EvaluationSAC_guache_droite(Tree.left, ListeSAC_Gauche, ListeSAC_Droite , LG ,LD)

    return (ListeSAC_Gauche, ListeSAC_Droite , LG ,LD)



def EvaluationSac (ListeSacs):
    assert type(ListeSacs) is list
    ListeSize = []
    somme = 0
    for v in ListeSacs :
          S = len(v)
       ##print(S, end='  ')
          somme += S
          ListeSize.append(S)
    MinSize = min(ListeSize)
    MaxSize = max(ListeSize)
    AvgSize = somme/len(ListeSacs)
    return (MinSize, MaxSize, AvgSize, ListeSize)


def rechche_liste(liste , label):
    t=False
    #p=None
    #ep=None  #L'EMPLACEMENT DE PROFIL DANS LA LISTE
    i=0
    while(i < len(liste)):
        #a=liste[i]
        if (liste[i]== label) :
            t=True
            #p=a[1]
            #ep=i
            break
        else :
            i+=1
    return  t


def recherche_offline(tree , objet, seuil_similarité , k):
    global nb_comparaison
    global nb_distance
    global existe_profils
    sac = []
    label = None
    existe_profils = False
    listK= None
    nb_noued= None
    # global pnt_tree  # ponteur de la racine
    # X = None
    start_time = time.time()  # start time of the loop

    if tree == None:
        return existe_profils , label , 0


    else:

        while (tree.left != None):

            # print(" pivotG", tree.pivotG)
            # d1 = bhattacharyya(tree.pivotG, rep)  # la distance de rep avec le PG   ( bahatacharya )
            # d1 = ecludienne_distance(tree.pivotG, rep)  # la distance de rep avec le PG   ( ecludienne )
            d1 = cosinus_distance(tree.pivotG, objet)
            nb_distance += 1
            # ("dist pG", d1, tree.pivotG)
            # print(" pivotD", tree.pivotD)
            # d2 = bhattacharyya(tree.pivotD, rep)  # la distance de rep avec le PD   *( bahatacharya )
            # d2 = ecludienne_distance(tree.pivotD, rep)  # la distance de rep avec le PD   ( ecludienne )
            d2 = cosinus_distance(tree.pivotD, objet)
            nb_distance += 1
            # print("dist pD", d2,tree.pivotD)
            # print("distanceD", d2, d1)
            if (d1 < d2):  # <
                nb_comparaison += 1

                return recherche_offline(tree.left,objet, seuil_similarité,k)

            else:

                return recherche_offline(tree.right,objet,seuil_similarité,k)
        # jsuque atteindre la derniere feuille

        # print("tree.rep", tree.rep)
        # print("rep", rep)
        # dis = bhattacharyya(tree.rep,rep)  # cvalcule de disatance entre rep et rep de derniere feuille  ( bahatacharya )
        # dis = ecludienne_distance(tree.rep, rep)  # cvalcule de disatance entre rep et rep de derniere feuille  ( ecludienne )

        dis = cosinus_distance(tree.rep, objet)
        # print("distance rech 1" ,dis , tree.etiquette)
        nb_distance += 1


        if (dis <= seuil_similarité):  # comparaison avec le seuil de similarité fixé  ( if the same profils )
            nb_comparaison += 1
            existe_profils = True
            label = tree.etiquette  # recupurer etiquette

            pointeur = tree  # l'emplecment de node
            print("lennnn", len(pointeur.sac))
            for v in pointeur.sac:
                # print("v evaluation ", len(v))
                sac.append(v)

            listK, nb_noued , nb_dist=kNN(pointeur, objet,sac , k)
            nb_distance+= nb_dist

        else:
            existe_profils = False


        end_time = time.time() - start_time


        return   existe_profils , label , listK, nb_distance , nb_comparaison,nb_noued, end_time


req=[ 0.12510008, -0.10789373, -0.13098833 ,-0.02617935,  0.06903525 , 0.19022463,
 -0.0330794,  -0.04694448  ,0.03098982, -0.05622067 , 0.01206111, -0.03175758,
 -0.08346848, -0.09734762 ,-0.09605233 ,-0.02023652 , 0.15136221 ,-0.08174489,
 -0.01518932 ,-0.09388317, -0.03687492, -0.08064314 ,-0.0182355 , -0.08802549,
 -0.00920027 , 0.02482709 ,-0.06521109  ,0.18695073 , 0.14932604 ,-0.0482179,
 -0.03149826 , 0.10174313, -0.1018279  , 0.03829882  ,0.03052727  ,0.09691444,
  0.22394176 ,-0.09688115  ,0.23414965, -0.07520223 , 0.02003306 ,-0.02033318,
  0.1405109 , -0.07165491 , 0.08136281 ,-0.0197759  , 0.14740229, -0.10467197,
  0.03761575,  0.01867484 ,-0.09632779 , 0.10131069, -0.05757577  ,0.04089801,
 -0.02094378,  0.169834  , -0.14606954,  0.00942972 , 0.04436302, -0.09838986,
 -0.00479182,  0.1189533 , -0.06543209 ,-0.05466866 ,-0.06563212 ,-0.01366851,
 -0.0317462 , -0.02375698 , 0.15703483 , 0.01805638 , 0.07361256 ,-0.0535559,
  0.09351198,  0.0102226 ,  0.10951872 , 0.06545097 , 0.08256231, -0.06586042,
  0.0976291 , -0.07808077 , 0.16904643 ,-0.0685697 , -0.08675254 , 0.04431532,
  0.10765827, -0.09434238, -0.02979772 , 0.08029356 ,-0.03261329,  0.02581364,
  0.09704752 ,-0.05552957,  0.08181867, -0.0234168 ,  0.16191061, -0.11067985,
  0.03931686,  0.09048839 ,-0.01290988,  0.01462779 , 0.0777008 , -0.04754048,
 -0.06048294, -0.072851  ,  0.19181775,  0.1039472 , -0.0835558 , -0.01874777,
 -0.01847519 , 0.02999556,  0.00246788, -0.08144104 , 0.01619684 , 0.22443266,
 -0.0245896 , -0.02623818,  0.01873908 , 0.07493879 ,-0.08954689, -0.09813711,
 -0.04690684, -0.01167587 ,-0.03613986, -0.03891705 ,-0.10427255 ,-0.00224551,
  0.13957697, -0.09011631]


objet= [ [ 150.12510008, -0.10789373, -0.13098833 ,-0.02617935,  0.06903525 , 0.19022463,
 -0.0330794,  -0.04694448  ,0.03098982, -0.05622067 , 0.01206111, -0.03175758,
 -0.08346848, -0.09734762 ,-0.09605233 ,-0.02023652 , 0.15136221 ,-0.08174489,
 -0.01518932 ,-0.09388317, -0.03687492, -0.08064314 ,-0.0182355 , -0.08802549,
 -0.00920027 , 0.02482709 ,-0.06521109  ,0.18695073 , 0.14932604 ,-0.0482179,
 -0.03149826 , 0.10174313, -0.1018279  , 0.03829882  ,0.03052727  ,0.09691444,
  0.22394176 ,-0.09688115  ,0.23414965, -0.07520223 , 0.02003306 ,-0.02033318,
  0.1405109 , -0.07165491 , 0.08136281 ,-0.0197759  , 0.14740229, -0.10467197,
  0.03761575,  0.01867484 ,-0.09632779 , 0.10131069, -0.05757577  ,0.04089801,
 -0.02094378,  0.169834  , -0.14606954,  0.00942972 , 0.04436302, -0.09838986,
 -0.00479182,  0.1189533 , -0.06543209 ,-0.05466866 ,-0.06563212 ,-0.01366851,
 -0.0317462 , -0.02375698 , 0.15703483 , 0.01805638 , 0.07361256 ,-0.0535559,
  0.09351198,  0.0102226 ,  0.10951872 , 0.06545097 , 0.08256231, -0.06586042,
  0.0976291 , -0.07808077 , 0.16904643 ,-0.0685697 , -0.08675254 , 0.04431532,
  0.10765827, -0.09434238, -0.02979772 , 0.08029356 ,-0.03261329,  0.02581364,
  0.09704752 ,-0.05552957,  0.08181867, -0.0234168 ,  0.16191061, -0.11067985,
  0.03931686,  0.09048839 ,-0.01290988,  0.01462779 , 0.0777008 , -0.04754048,
 -0.06048294, -0.072851  ,  0.19181775,  0.1039472 , -0.0835558 , -0.01874777,
 -0.01847519 , 0.02999556,  0.00246788, -0.08144104 , 0.01619684 , 0.22443266,
 -0.0245896 , -0.02623818,  0.01873908 , 0.07493879 ,-0.08954689, -0.09813711,
 -0.04690684, -0.01167587 ,-0.03613986, -0.03891705 ,-0.10427255 ,-0.00224551,
  0.13957697, -0.09011631] ,[ 0.12510008, -0.10789373, -0.13098833 ,-0.02617935,  0.06903525 , 0.19022463,
 -0.0330794,  -0.04694448  ,0.03098982, -0.05622067 , 0.01206111, -0.03175758,
 -0.08346848, -0.09734762 ,-0.09605233 ,-0.02023652 , 0.15136221 ,-0.08174489,
 -0.01518932 ,-0.09388317, -0.03687492, -0.08064314 ,-0.0182355 , -0.08802549,
 -0.00920027 , 0.02482709 ,-0.06521109  ,0.18695073 , 0.14932604 ,-0.0482179,
 -0.03149826 , 0.10174313, -0.1018279  , 0.03829882  ,0.03052727  ,0.09691444,
  0.22394176 ,-0.09688115  ,0.23414965, -0.07520223 , 0.02003306 ,-0.02033318,
  0.1405109 , -0.07165491 , 0.08136281 ,-0.0197759  , 0.14740229, -0.10467197,
  0.03761575,  0.01867484 ,-0.09632779 , 0.10131069, -0.05757577  ,0.04089801,
 -0.02094378,  0.169834  , -0.14606954,  0.00942972 , 0.04436302, -0.09838986,
 -0.00479182,  0.1189533 , -0.06543209 ,-0.05466866 ,-0.06563212 ,-0.01366851,
 -0.0317462 , -0.02375698 , 0.15703483 , 0.01805638 , 0.07361256 ,-0.0535559,
  0.09351198,  0.0102226 ,  0.10951872 , 10.06545097 , 0.08256231, -0.06586042,
  0.0976291 , -0.07808077 , 0.16904643 ,-0.0685697 , -0.08675254 , 0.04431532,
  0.10765827, -0.09434238, -0.02979772 , 0.08029356 ,-0.03261329,  0.02581364,
  0.09704752 ,-0.05552957,  0.08181867, -0.0234168 ,  0.16191061, -0.11067985,
  0.03931686,  0.09048839 ,-0.01290988,  0.01462779 , 0.0777008 , -0.04754048,
 -0.06048294, -0.072851  ,  0.19181775,  0.1039472 , -0.0835558 , -0.01874777,
 -0.01847519 , 0.02999556,  0.00246788, -0.08144104 , 0.01619684 , 0.22443266,
 -0.0245896 , -0.02623818,  0.01873908 , 0.07493879 ,-0.08954689, -0.09813711,
 -0.04690684, -0.01167587 ,-0.03613986, -0.03891705 ,-0.10427255 ,-0.00224551,
  0.13957697, -0.09011631]]

def elem_sort(elem):
    return  elem[1]

def kNN ( pointeur, req , sac ,  k ):
    nb_noued= 0
    nb_distance=0
    liste_k_objet=[]
    print("len",len(sac), pointeur.size, type(sac) , type(req))
    print(sac[1])
    if ( len(sac) >= k) :
         for i in range(0 , len(sac)):
             nb_noued+=1

             if (len(sac) == 1):
                 dis = np.array(distance.cosine(req, sac))
                 t = (sac, dis)
             else:

                 dis = np.array(distance.cosine(req, sac[i]))
                 t = (sac[i], dis)

             nb_distance+=1

             liste_k_objet.append(t)


         liste_k_objet.sort(key=elem_sort)
         list_final = []
         for i in range(0, k):
             list_final.append(liste_k_objet[i])


    else :
        print("111111111111")
        for i in range(0, len(sac)):

            nb_noued += 1
            if (len(sac) == 1):
                dis = np.array(distance.cosine(req, sac))
                t = (sac, dis)
            else:

                dis = np.array(distance.cosine(req, sac[i]))
                t = (sac[i], dis)

            nb_distance += 1

            liste_k_objet.append(t)

        #liste_k_objet.sort(key=elem_sort)
        rest_k = k - len(sac)
        print(rest_k , "is pro", pointeur.is_profils )
        while ( pointeur.is_profils != 1  and   rest_k >= 0):
            print(("//////////////"))
            if (pointeur.fils == 0) : # filsgauche
                print("22222222222222")
                nb_noued += 1
                if pointeur.parent.right.sac is not None:

                    sac = pointeur.parent.right.sac

                else:
                    while pointeur.parent.right.sac is  None :
                        pointeur = pointeur.parent
                        sac = pointeur.parent.right.sac
                #sac = pointeur.parent.right.sac
                for i in range(0, len(sac)):
                    if (len(sac) == 1):
                        dis = np.array(distance.cosine(req, sac))
                        t = (sac, dis)
                    else:

                        dis = np.array(distance.cosine(req, sac[i]))
                        t = (sac[i], dis)

                    nb_distance += 1

                    liste_k_objet.append(t)

                #liste_k_objet.sort(key=elem_sort)
                rest_k = k - len(sac)
                pointeur = pointeur.parent

            else :  # fils droite
                print("3333333333")
                nb_noued += 1
                if  pointeur.parent.left.sac is not None:

                     sac = pointeur.parent.left.sac

                else :
                    pointeur= pointeur.parent
                    sac = pointeur.parent.left.sac
                for i in range(0, len(sac)):

                    if (len(sac) == 1):
                        dis = np.array(distance.cosine(req, sac))
                        t = (sac, dis)
                    else:

                        dis = np.array(distance.cosine(req, sac[i]))
                        t = (sac[i], dis)

                    liste_k_objet.append(t)

                #liste_k_objet.sort(key=elem_sort)
                rest_k = k - len(sac)
                pointeur = pointeur.parent
    liste_k_objet.sort(key=elem_sort)
    print(len(liste_k_objet))
    list_final=[]
    for i in range (0,k):
        list_final.append(liste_k_objet[i])


    return list_final , nb_noued , nb_distance
"""
data= np.load('/home/etudiant/PycharmProjects/Golden_version/4_outputdata.npy' ,allow_pickle=True)
a =data[0]
print(len(a[2]))

"""