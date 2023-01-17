import numpy as np

from indexing.dataSet import *
from indexing.distance import *
from indexing.evaluation import  *
import time
import sys

sys.setrecursionlimit(50000)




global label_tree      ####### liste des labels dans arbre
label_tree=[]

global etiquette
etiquette=[]

global etiquette1
etiquette1=[]

global list_rech
list_rech=[]
global nb_comparaison
nb_comparaison=0

global nb_distance
nb_distance=0

global lab  # etiquette unique pour chaque profils ( compteur )
lab = 1

global seuil
seuil = 0.7# 400 # la valeur fixer pour  appliquer la fusion avec le profils sinon avec la racine

global seuil_similarité  # le suil pour indiquer si le meme profils ou non  utilise dans la rechrche
seuil_similarité = 0.3# 300

max_size = 10000# la taile maximal de sac de profils # fixé selon la machine utilisé (sqrt(n))

global existe_profils
existe_profils = False

global pointeur  # pour le profils deja existé
pointeur = None


global Tree
Tree = None
global nb_fusion
nb_fusion = 0


class Node:  # la creation de noued de l'arbre d'indexation
    # Initialize the attributes of Node
    def __init__(self, sac, left, right, pivotG, pivotD, rep, etiquette,  size,parent, is_profils, fils):
        self.etiquette = etiquette  # identifiant de profils
        self.left = left  # Left Child
        self.right = right  # Right Child
        self.sac = [] # Node Data ( sac)
        self.pivotG = pivotG  # pivot gauche utilisé dans la rechercche
        self.pivotD = pivotD  # le mme que gauche
        self.rep = rep  # le centre de noued = vecteur pour la fusion
        self.size = size  # nombre d'objet  par noued
        self.parent = parent  # le parent de noueds
        self.is_profils = is_profils  # 1 si le  noued est le debut de profils none sinon
        self.fils = fils  # 0 fils gauche 1 fils droite

    ###############################################################################"

    def PrintTree(self):
        if self.left:
            self.left.PrintTree()
        print(
            self.etiquette)

        if self.right:
            self.right.PrintTree()

    def display(self):
        lines, *_ = self._display_aux()
        for line in lines:
            print(line)

    def _display_aux(self):
        """Returns list of strings, width, height, and horizontal coordinate of the root."""
        # No child.
        if self.right is None and self.left is None:
            line = '%s' % self.etiquette
            width = len(line)
            height = 1
            middle = width // 2
            return [line], width, height, middle

        # Only left child.
        if self.right is None:
            lines, n, p, x = self.left._display_aux()
            s = '%s' % self.etiquette
            u = len(s)
            first_line = (x + 1) * ' ' + (n - x - 1) * '_' + s
            second_line = x * ' ' + '/' + (n - x - 1 + u) * ' '
            shifted_lines = [line + u * ' ' for line in lines]
            return [first_line, second_line] + shifted_lines, n + u, p + 2, n + u // 2

        # Only right child.
        if self.left is None:
            lines, n, p, x = self.right._display_aux()
            s = '%s' % self.etiquette
            u = len(s)
            first_line = s + x * '_' + (n - x) * ' '
            second_line = (u + x) * ' ' + '\\' + (n - x - 1) * ' '
            shifted_lines = [u * ' ' + line for line in lines]
            return [first_line, second_line] + shifted_lines, n + u, p + 2, u // 2

        # Two children.
        left, n, p, x = self.left._display_aux()
        right, m, q, y = self.right._display_aux()
        s = '%s' % self.etiquette
        u = len(s)
        first_line = (x + 1) * ' ' + (n - x - 1) * '_' + s + y * '_' + (m - y) * ' '
        second_line = x * ' ' + '/' + (n - x - 1 + u + y) * ' ' + '\\' + (m - y - 1) * ' '
        if p < q:
            left += [n * ' '] * (q - p)
        elif q < p:
            right += [m * ' '] * (p - q)
        zipped_lines = zip(left, right)
        lines = [first_line, second_line] + [a + u * ' ' + b for a, b in zipped_lines]
        return lines, n + m + u, max(p, q) + 2, n + u // 2

################################################"#######################"

    """
    def recherche2(tree, rep):
        print("debut rech2")
        global list_rech
        global nb_comparaison
        global nb_distance
        global nombre_regrpm
        global existe_profils
        global pointeur
        global lab
        existe_profils = False
        # global pnt_tree  # ponteur de la racine
        # X = None
        start_time = time.time()  # start time of the loop
        if tree == None:
            #labelle = lab
            #lab += 1
            print( "tree is none ")
        else:

            while (tree.left != None):

                # print(" pivotG", tree.pivotG)
                # d1 = bhattacharyya(tree.pivotG, rep)  # la distance de rep avec le PG   ( bahatacharya )
                # d1 = ecludienne_distance(tree.pivotG, rep)  # la distance de rep avec le PG   ( ecludienne )
                d1 = cosinus_distance(tree.pivotG, rep)
                nb_distance += 1
                # ("dist pG", d1, tree.pivotG)
                # print(" pivotD", tree.pivotD)
                # d2 = bhattacharyya(tree.pivotD, rep)  # la distance de rep avec le PD   *( bahatacharya )
                # d2 = ecludienne_distance(tree.pivotD, rep)  # la distance de rep avec le PD   ( ecludienne )
                d2 = cosinus_distance(tree.pivotD, rep)
                nb_distance += 1
                # print("dist pD", d2,tree.pivotD)
                # print("distanceD", d2, d1)
                if (d1 < d2):  # <
                    nb_comparaison += 1

                    return tree.left.recherche2(rep)

                else:

                    return tree.right.recherche2(rep)
            # jsuque atteindre la derniere feuille

            # print("tree.rep", tree.rep)
            # print("rep", rep)
            # dis = bhattacharyya(tree.rep,rep)  # cvalcule de disatance entre rep et rep de derniere feuille  ( bahatacharya )
            # dis = ecludienne_distance(tree.rep, rep)  # cvalcule de disatance entre rep et rep de derniere feuille  ( ecludienne )

            dis = cosinus_distance(tree.rep, rep)
            print("distance2", dis)
            nb_distance += 1

            if (dis <= seuil_similarité):  # comparaison avec le seuil de similarité fixé  ( if the same profils )
                nb_comparaison += 1
                existe_profils = True
                labelle = tree.etiquette  # recupurer etiquette
                pointeur = tree  # l'emplecment de node

            else:
                nb_comparaison += 1
                existe_profils = False
                labelle = lab  # etiquette le nv profils
                lab = lab + 1  # compteur des etiquette +1
                pointeur = tree  # l'emplecment de node


                #labelle = lab  # etiquette le nv profils
                #lab = lab + 1  # compteur des etiquette +1
            # print("dis", dis,"tree", tree.etiquette, "new",labelle)
        # labelle=str(labelle)
        # print("existe", existe_profils)
        end_time = time.time() - start_time
        list_rech.append(end_time)
        # print("le temp de rechrche ", end_time)

        print("existe rechrche2 =",existe_profils , labelle , pointeur )
        return existe_profils, labelle, pointeur
    """
    ###################################################################" la fonction de rechrche de profils ( arbre )

    def recherche(tree, rep):
        #print("debut rech 1")
        global Tree
        global list_rech
        global nb_comparaison
        global  nb_distance
        global nombre_regrpm
        global existe_profils
        global pointeur
        global lab
        existe_profils = False
        #global pnt_tree  # ponteur de la racine
        # X = None
        start_time = time.time()  # start time of the loop
        if tree == None:

            label = lab
            lab+=1
            tree, pointeur = Node.create_tree(Tree, rep, label, existe_profils, pointeur)
            Tree=tree

            #Tree.display()

        else:


            while (tree.left != None):

                if (tree.is_profils == 1):
                    # le premier nouerd qui indique le debut de profils aprées le partitionnement
                    pointeur = tree

                # print(" pivotG", tree.pivotG)
                # d1 = bhattacharyya(tree.pivotG, rep)  # la distance de rep avec le PG   ( bahatacharya )
                # d1 = ecludienne_distance(tree.pivotG, rep)  # la distance de rep avec le PG   ( ecludienne )
                d1 = cosinus_distance(tree.pivotG, rep)
                nb_distance += 1
                #("dist pG", d1, tree.pivotG)
                # print(" pivotD", tree.pivotD)
                # d2 = bhattacharyya(tree.pivotD, rep)  # la distance de rep avec le PD   *( bahatacharya )
                # d2 = ecludienne_distance(tree.pivotD, rep)  # la distance de rep avec le PD   ( ecludienne )
                d2 = cosinus_distance(tree.pivotD, rep)
                nb_distance += 1
                #print("dist pD", d2,tree.pivotD)
                # print("distanceD", d2, d1)
                if (d1 < d2):  # <
                    nb_comparaison+=1

                    return tree.left.recherche(rep)

                else:

                    return tree.right.recherche(rep)
            # jsuque atteindre la derniere feuille

            # print("tree.rep", tree.rep)
            # print("rep", rep)
            # dis = bhattacharyya(tree.rep,rep)  # cvalcule de disatance entre rep et rep de derniere feuille  ( bahatacharya )
            # dis = ecludienne_distance(tree.rep, rep)  # cvalcule de disatance entre rep et rep de derniere feuille  ( ecludienne )

            dis = cosinus_distance(tree.rep, rep)
            #print("distance rech 1" ,dis , tree.etiquette)
            nb_distance += 1

            if (tree.is_profils == 1):
                # le premier nouerd qui indique le debut de profils aprées le partitionnement
                pointeur = tree

            if (dis <= seuil_similarité):  # comparaison avec le seuil de similarité fixé  ( if the same profils )
                nb_comparaison += 1
                existe_profils = True
                label = tree.etiquette  # recupurer etiquette
                pointeur = tree  # l'emplecment de node

            else:
                existe_profils = False

                label = lab  # etiquette le nv profils
                lab = lab + 1  # compteur des etiquette +1

            end_time = time.time() - start_time
            list_rech.append(end_time)

        ########################" creation de noued de profil  if not existe


            tree, pointeur=Node.create_tree(Tree, rep, label, existe_profils, pointeur)
            Tree = tree


        return  label, pointeur

        ###############################################  inserer un profils deja existé
    def partionnement(tree,pointeur, sac,label):
            #print("partitionnment")

            global nb_comparaison
            """"
            if (tree.sac is None):
                existe , lab ,p= tree.recherche2(rep)
                #print( " len sac rechrche 2", len(tree.sac) , existe , lab , p)
                tree=p
     
            if (pointeur.sac is None ): # LA PREMIERE INSERTION DES VETEUR DANS CE NOUED

                if (len(sac)  > max_size):
                    # appliquer vision de sac
                    pointeur.sac=sac
                    pointeur.size=len(sac)
                    pointeur.division_sac(sac , label)

                else :
                    # insertion direct
                    pointeur.insertion_direct(pointeur,sac)
             """


            if ((pointeur.size + len(sac)) > max_size):  # apliquer partitionnement de noeud

                    nb_comparaison += 1

                    #size_reste= max_size - pointeur.size
                    for i in range(0,len(sac)):
                        pointeur.sac.append(sac[i])

                    sac=pointeur.sac
                    size_t= len(sac)

                    if (size_t % 2 == 0):

                        # sac1= pointeur.sac[0:int(pointeur.size / 2)]
                        # sac2= pointeur.sac[(int(pointeur.size / 2)):pointeur.size]

                        sacG = []
                        sacD = []
                        v = 0
                        while (v < int(size_t / 2)):
                            sacG.append(sac[v])
                            v += 1
                        while (v < size_t ):
                            sacD.append(sac[v])
                            v += 1
                    else:

                        # sac1 = pointeur.sac[0:int(pointeur.size // 2)]
                        # sac2 = pointeur.sac[int(pointeur.size // 2):int(pointeur.size )]
                        sacG = []
                        sacD = []
                        v = 0
                        while (v < int(size_t // 2)):
                            sacG.append(sac[v])
                            v += 1
                        while (v < size_t ):
                            sacD.append(sac[v])
                            v += 1



                    #sacG = sac [ 0:size_reste]
                    #print('sac G TREE', len(sacG))

                    #sacD= sac[size_reste : len(sac)]

                    #print("sac droite reste ", len(sacD))
                    #pointeur.insertion_direct(pointeur,sacG)
                    #gauche
                    rep1= sacG[0]
                    pointeur.left = Node([], None, None, None, None, rep1, pointeur.etiquette, len(sacG), pointeur, None,
                                     0)  # fils gauche

                    for i in range(0, len(sacG)) :

                            pointeur.left.sac.append(sac[i])
                    #droite
                    rep2=  sacD[0]
                    pointeur.right = Node([], None, None, None, None, rep2, label,
                                      len(sacD), pointeur, None, 1)  # fils droite
                    for  i in range(0 , len(sacD)):

                         pointeur.right.sac.append(sacD[i])


                    pointeur.pivotG = rep1  # laison pour la rechrche
                    pointeur.pivotD = rep2  # meme laison
                    pointeur.rep = np.mean((rep1,rep2), 0)  #   cas noued intrne de profil pour le test de fusion ou regroupement
                    #tree.rep= np.maximum(tree.rep , rep)

                    pointeur.sac = []
                    pointeur.size=0
                    # tree.display()
                    # return  tree
                    pointeur.right.division_sac(sacD , label)
                    pointeur.left.division_sac(sacG, label)
                    #print("profil divise")
                    #pointeur.right.display()
            else:

                pointeur.insertion_direct(pointeur,sac)
                # tree.display()
                # return  tree

        ##########################################################################################"

    def insertion_direct(tree,pointeur, sac ):

            # ajouter sac nauveau a le meme node
            # print("inserer le profils a le meme noeud ")
            #tree.rep = np.mean((tree.rep , rep), 0)
            #print( 'insert directly  len sac ', len(sac))

            #print('insert directly  len p.sac ', len(pointeur.sac) , pointeur.size)
            #tree.rep = np.maximum(tree.rep, rep)

            """
            print(pointeur.sac)
            list = []
            list.append(sac)
            list.append(pointeur.sac)
            pointeur.sac = list
            print(pointeur.sac)
            print(" len  pointeur . sac= ", len(pointeur.sac) )
            pointeur.size = pointeur.size + len(sac)  # + 1 pour le sac nauveau
            
            #####################
            list=[]
            if (len(pointeur.sac)== 128):
                list.append(pointeur.sac)
                for v in sac:
                # print("v evaluation ", len(v))
                   list.append(v)
                pointeur.sac=list
            else :
                for v in sac:
                # print("v evaluation ", len(v))
                   pointeur.sac.append(v)
            """


            for v in range(0 , len(sac)):
                # print("v evaluation ", len(v))
                pointeur.sac.append(sac[v])

            pointeur.size= len(pointeur.sac)




    ############################################################# devision sac sur 2
    def division_sac(pointeur, sac , label):

        global nb_comparaison

        if (pointeur.size  > max_size):  # cas N 2

            #print('appliquer devison sac', "sac=", len(sac), pointeur.size)
            nb_comparaison += 1
            if (pointeur.size % 2 == 0):

                #sac1= pointeur.sac[0:int(pointeur.size / 2)]
                #sac2= pointeur.sac[(int(pointeur.size / 2)):pointeur.size]

                sac1=[]
                sac2=[]
                v=0
                while (v < int(pointeur.size / 2) ):
                    sac1.append(sac[v])
                    v+=1
                while (v < pointeur.size):
                    sac2.append(sac[v])
                    v+=1
            else :

                #sac1 = pointeur.sac[0:int(pointeur.size // 2)]
                #sac2 = pointeur.sac[int(pointeur.size // 2):int(pointeur.size )]
                sac1 = []
                sac2 = []
                v = 0
                while (v < int(pointeur.size // 2)):
                    sac1.append(sac[v])
                    v += 1
                while (v < pointeur.size):
                    sac2.append(sac[v])
                    v += 1

            #print('sacD = ', len(sac2), " sacG =", len(sac1))

            pointeur.left = Node([], None, None, None, None, sac1[0] , label, len(sac1), pointeur, None,
                             0)  # fils gauche   np.max(np.array(sac1), 0)

            #pointeur.left.sac.append(sac1)
            for i in range(0, len(sac1)):
                pointeur.left.sac.append(sac1[i])
            # print(" noeud droite a ete cree ")


            pointeur.right = Node([], None, None, None, None, sac2[0] , label,
                              len(sac2), pointeur, None, 1)  # fils droite  np.max(np.array(sac2), 0)

            for i in range(0, len(sac2)):
                pointeur.right.sac.append(sac2[i])

            #pointeur.right.sac.append(sac2)


            pointeur.sac=None
            pointeur.size=0
            #tree.pivotG =  np.mean(sac1,0)  # laison pour la rechrche
            # tree.pivotG =  np.max(np.array(sac1))
            pointeur.pivotG = sac1[0]


            #tree.pivotD = np.mean(sac2,0)  # meme laison
            #tree.pivotD=  np.max(np.array(sac2))


            pointeur.pivotD= sac2[0]

            pointeur.rep = np.mean((pointeur.pivotD , pointeur.pivotG),0)   # la creation de nouvelle centre pour le profils partionner
            #tree.rep = np.maximum(tree.pivotG,tree.pivotD)
            ### appel
            pointeur.right.division_sac( sac2 , label)
            pointeur.left.division_sac( sac1 , label)
            return pointeur

    ##################################################################################### profils n'existe pas
    def fusion(tree, rep, lab):    # avec le noued de profil
        global nb_fusion
        #global pnt_tree  # ponteur de la racine
        # dis = bhattacharyya(tree.rep, rep)  # bahatacharya distnace
        # dis = ecludienne_distance(tree.rep, rep)   # ecludienne distance
        #dis = cosinus_distance(tree.rep, rep)


        if (nb_fusion == 0):
            noued_intrm = Node(None, None, None, rep, tree.rep, None, None, 0, None, None, None)
            sac=rep.tolist()

            nv_profils = Node([], None, None, None, None, rep, lab, 1, noued_intrm, 1, 0)
            nv_profils.sac.append(sac)

            #nv_profils.division_sac(sac, rep, lab)  # si la taille ta sac est sup ra cmax

            #if (tree.sac is not None):
                #tree.division_sac(tree.sac , tree.rep , tree.etiquette)

            noued_intrm.left = nv_profils
            noued_intrm.right = tree
            tree.parent = noued_intrm
            tree.fils = 1
            tree = noued_intrm
            #pnt_tree = tree
            #tree.display()


            return tree ,nv_profils
        else:
            noued_intrmd = Node(None, None, None, rep, tree.rep, None, None, 0, None, None, tree.fils)
            sac = rep.tolist()
            nv_profils = Node([], None, None, None, None, rep, lab, 1, noued_intrmd, 1, 0)
            nv_profils.sac.append(sac)
            #nauveau_profils.division_sac(sac, rep, lab)
            noued_intrmd.left = nv_profils
            noued_intrmd.right = tree
            #if (tree.sac is not None):
                #tree.division_sac(tree.sac , tree.rep , tree.etiquette)

            parent = tree.parent

            tree.fils = 1
            tree.parent = noued_intrmd
            tree = parent


            if (noued_intrmd.fils == 0):

                tree.left = noued_intrmd

            else:

                tree.right = noued_intrmd

            noued_intrmd.parent = tree

            return tree, nv_profils

        return  tree ,nv_profils

    def regroupment(tree, rep, lab):   # avec la racine tj
        global nb_comparaison
        # utilise la moyenne entre les deux vecteur
        # centre = (tree.rep + rep) / 2
        if (nb_fusion == 0):
            racine = Node(None, None, None, rep, tree.rep, None, None, 0, None, None, None)
            sac = rep.tolist()
            nv_profils = Node([], None, None, None, None, rep, lab, 1, racine, 1, 0)
            nv_profils.sac.append(sac)
            #nv_profils.division_sac(sac , rep , lab)
            #if (tree.sac is not None):
                #tree.division_sac(tree.sac , tree.rep , tree.etiquette)

            racine.left = nv_profils
            racine.right = tree
            tree.parent = racine
            tree.fils = 1
            tree = racine

            return tree ,nv_profils


        else:
            racine = Node(None, None, None, None, None, None, None,
                          0, None, 0, None)  # la creation de noeud pour nauveau profils
            sac = rep.tolist()
            nv_profils  = Node([], None, None, None, None, rep, lab,
                           1, racine, 1, None)  # la creation de noued pour nauveau profils
            nv_profils.sac.append(sac)
            #profils.division_sac(sac, rep, lab)  # si la taille ta sac est sup ra cmax
            #if (tree.sac is not None):
                #tree.division_sac(tree.sac , tree.rep , tree.etiquette)


            d1 = cosinus_distance(tree.pivotG, rep)
            d2 = cosinus_distance(tree.pivotD, rep)

            if (d1 < d2):
                nb_comparaison += 1

                #centre = np.mean((tree.pivotG, rep),0)  # appliquer le max ou moyenne entre les deux vecteur pour creer  un nv centre
                racine.left = nv_profils
                racine.right = tree  # pnt_tree
                racine.pivotG = rep   #left
                racine.pivotD = tree.pivotG                   #centre   #right
                tree.fils = 1  # pnt_tree.fils=1
                nv_profils.fils = 0
                pnt_tree = racine  # pointeur gllobal
                tree = racine
                # tree=pnt_tree

                return tree ,nv_profils
            else:
                # print("droite")
                #centre = np.mean((tree.pivotD,  rep),0)  # appliquer le max  ou moyenne entre les deux vecteur pour creer  un nv centre
                racine.right = nv_profils
                racine.left = tree
                racine.pivotG = tree.pivotD              #centre
                racine.pivotD = rep
                tree.fils = 0
                nv_profils .fils = 1
                #pnt_tree = racine
                tree = racine

            return tree ,nv_profils
        return tree ,nv_profils

    ################################################# creation de l'arbre
    def create_tree(tree, rep , label  , existe_profils, pointeur):
      global nb_fusion
      #global pnt_tree
      if (tree == None):
          #print("la creation de l'arbre pour la premiere fois ", label)  # cas n 1
          sac = rep.tolist()
          tree = Node([], None, None, None, None, rep, label, 1, None, 1, None)
          tree.sac.append(sac)
          #tree.division_sac(sac, rep, lab)  # si la taille ta sac est sup ra cmax
          p=tree
          return tree  ,p



      else:

           if (existe_profils== True):

               #tree.display()
               return tree, pointeur

               """"
               print( "appliquer partionnment")
               pointeur.partionnement(sac , rep , label )
               print("is pofils ", tree.is_profils, tree.etiquette)
               """



           else :

              dis = cosinus_distance(pointeur.rep, rep)
              #print("dist nv profil et tree profil", dis)

              if ((nb_fusion==0)):

                  tree , p= pointeur.fusion( rep , label )    # avec le noued de profil
                  nb_fusion += 1
                  return  tree , p
              else :
                  if (dis <= seuil):
                      #print('appliquer fusion')                 # avec le noued de profil
                      tree,p=pointeur.fusion( rep, label)
                      nb_fusion += 1

                      return Tree,p



                  else :
                      #print( 'appliquer regroupement')

                      tree,p = Tree.regroupment( rep, label)    # avec la racine
                      nb_fusion += 1

                      return tree, p



    ####################################################

def main_new_up_down():
    matrix = upload_data()
    print(matrix)
    print("fin data new upDown")
    tree = None
    for i_line, line in enumerate(matrix):
        existe_profils, labelle, pointeur = Node.recherche(tree, np.array(matrix[i_line]).astype(float))
        #print("lab", labelle , "exute",existe_profils , "rep ",np.array(matrix[i_line]).astype(float))
        tree = Node.create_tree(tree, np.array(matrix[i_line]).astype(float), np.array(matrix[i_line]).astype(float),
                                labelle, existe_profils, pointeur)


    tree.display()
    hauteur = get_height(tree)
    global lab

    print("la hauteur de la'rbre =", hauteur, 'nombre des objets ', lab-1)
    nouedintern=nouedintrne(tree)
    print('noued intrne', nouedintern)
    nombre_node=taille(tree)
    print('nombre des noued ', nombre_node)
    niveau= noued_niveau(tree)
    print("level",niveau)

    print("nombre comparaison", nb_comparaison)
    feuille = nombre_feuille(tree)
    print('nombre feuille ', feuille)
    print("nb disatnce ", nb_distance)
    l1=[]
    l2=[]
    Lg,Ld= EvaluationSAC_guache_droite(tree,l1,l2)
    print("liste",Lg,Ld)


