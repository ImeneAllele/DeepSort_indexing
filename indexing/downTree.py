
from indexing.distance import  *
from indexing.dataSet import  *
import time
from indexing.evaluation import  *

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

global racine
racine = None


global tree
tree=None
global Tree
Tree = None

#global seuil
#seuil = 1 # 400 # la valeur fixer pour  appliquer la fusion ou non

global seuil_similarité  # le suil pour indiquer si le meme profils ou non  utilise dans la rechrche
seuil_similarité = 0.2

max_size = 10000# la taile maximal de sac de profils # fixé selon la machine utilisé (sqrt(n))

global existe_profils
existe_profils = False

global pointeur  # pour le profils deja existé
pointeur = None

global nombre_regrpm
nombre_regrpm=0



class Node:  # la creation de noued de l'arbre d'indexation
    # Initialize the attributes of Node
    def __init__(self, sac, left, right, pivotG, pivotD, rep, etiquette, size,parent , is_profils , fils):
        self.etiquette = etiquette  # identifiant de profils
        self.left = left  # Left Child
        self.right = right  # Right Child
        self.sac = []  # Node Data ( sac)
        self.pivotG = pivotG  # pivot gauche utilisé dans la rechercche
        self.pivotD = pivotD  # le mme que gauche
        self.rep = rep  # le centre de noued = vecteur pour la fusion
        self.size = size  # nombre d'objet  par noued
        self.parent = parent # le parent de noueds
        self.is_profils= is_profils   # 1 si le  noued est le debut de profils none sinon
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

    ########################################################################################################## rechrche
    def create_tree(tree, rep , label  , existe_profils, pointeur):
      global nombre_regrpm
      if (tree== None):
          sac = rep.tolist()
          tree = Node([], None, None, None, None, rep, label, 1, None, 1, None)
          tree.sac.append(sac)
          # tree.division_sac(sac, rep, lab)  # si la taille ta sac est sup ra cmax
          p = tree
          return tree, p


      else:
           if (existe_profils== True):
               #print( "point", pointeur)
               #pointeur.partionnement(sac , rep , label )

                   return tree, pointeur



           else :
              #print("pointeur ", pointeur)
              if (nombre_regrpm==0):
                  tree, p= pointeur.regroupement( rep , label )
                  nombre_regrpm += 1
                  return tree, p
              else :

                  tree, p = pointeur.regroupement(rep, label)
                  nombre_regrpm += 1

                  return Tree, p

    def partionnement(tree,pointeur, sac,label):
        global nb_comparaison
        if ((pointeur.size + len(sac)) > max_size):  # apliquer partitionnement de noeud

            nb_comparaison += 1

            # size_reste= max_size - pointeur.size
            for i in range(0, len(sac)):
                pointeur.sac.append(sac[i])

            sac = pointeur.sac
            size_t = len(sac)

            if (size_t % 2 == 0):



                sacG = []
                sacD = []
                v = 0
                while (v < int(size_t / 2)):
                    sacG.append(sac[v])
                    v += 1
                while (v < size_t):
                    sacD.append(sac[v])
                    v += 1
            else:


                sacG = []
                sacD = []
                v = 0
                while (v < int(size_t // 2)):
                    sacG.append(sac[v])
                    v += 1
                while (v < size_t):
                    sacD.append(sac[v])
                    v += 1


            pointeur.left = Node([], None, None, None, None, pointeur.rep, pointeur.etiquette, len(sacG), pointeur,
                                 None,
                                 0)  # fils gauche

            for i in range(0, len(sacG)):
                pointeur.left.sac.append(sac[i])
            # droite

            pointeur.right = Node([], None, None, None, None, sacD[0], label,
                                  len(sacD), pointeur, None, 1)  # fils droite
            for i in range(0, len(sacD)):
                pointeur.right.sac.append(sacD[i])

            pointeur.pivotG = sacG[0]  # laison pour la rechrche
            pointeur.pivotD = sacD[0]  # meme laison
            pointeur.rep = np.mean((sacG[0], sacD[0]),
                                   0)  # cas noued intrne de profil pour le test de fusion ou regroupement


            pointeur.sac = []
            pointeur.size = 0

            pointeur.right.division_sac(sacD, label)
            pointeur.left.division_sac(sacG, label)

        else:

            pointeur.insertion_direct(pointeur, sac)
#########################################################################################"


    def division_sac(pointeur, sac, label):

        global nb_comparaison

        if (pointeur.size > max_size):  # cas N 2

            # print('appliquer devison sac', "sac=", len(sac), pointeur.size)
            nb_comparaison += 1
            if (pointeur.size % 2 == 0):

                # sac1= pointeur.sac[0:int(pointeur.size / 2)]
                # sac2= pointeur.sac[(int(pointeur.size / 2)):pointeur.size]

                sac1 = []
                sac2 = []
                v = 0
                while (v < int(pointeur.size / 2)):
                    sac1.append(sac[v])
                    v += 1
                while (v < pointeur.size):
                    sac2.append(sac[v])
                    v += 1
            else:

                # sac1 = pointeur.sac[0:int(pointeur.size // 2)]
                # sac2 = pointeur.sac[int(pointeur.size // 2):int(pointeur.size )]
                sac1 = []
                sac2 = []
                v = 0
                while (v < int(pointeur.size // 2)):
                    sac1.append(sac[v])
                    v += 1
                while (v < pointeur.size):
                    sac2.append(sac[v])
                    v += 1

            # print('sacD = ', len(sac2), " sacG =", len(sac1))

            pointeur.left = Node([], None, None, None, None, sac1[0], label, len(sac1), pointeur, None,
                                 0)  # fils gauche   np.max(np.array(sac1), 0)

            # pointeur.left.sac.append(sac1)
            for i in range(0, len(sac1)):
                pointeur.left.sac.append(sac1[i])
            # print(" noeud droite a ete cree ")

            pointeur.right = Node([], None, None, None, None, sac2[0], label,
                                  len(sac2), pointeur, None, 1)  # fils droite  np.max(np.array(sac2), 0)

            for i in range(0, len(sac2)):
                pointeur.right.sac.append(sac2[i])

            # pointeur.right.sac.append(sac2)

            pointeur.sac = None
            pointeur.size = 0
            # tree.pivotG =  np.mean(sac1,0)  # laison pour la rechrche
            # tree.pivotG =  np.max(np.array(sac1))
            pointeur.pivotG = sac1[0]

            # tree.pivotD = np.mean(sac2,0)  # meme laison
            # tree.pivotD=  np.max(np.array(sac2))

            pointeur.pivotD = sac2[0]

            pointeur.rep = np.mean((pointeur.pivotD, pointeur.pivotG),
                                   0)  # la creation de nouvelle centre pour le profils partionner
            # tree.rep = np.maximum(tree.pivotG,tree.pivotD)
            ### appel
            pointeur.right.division_sac(sac2, label)
            pointeur.left.division_sac(sac1, label)
            return pointeur

    ##########################################################################################"

    def insertion_direct(tree,pointeur, sac ) : # insertion_direct(tree,pointeur, sac ):
        for v in range(0, len(sac)):
            # print("v evaluation ", len(v))
            pointeur.sac.append(sac[v])

        pointeur.size = len(pointeur.sac)

    ###############################################################################

    def regroupement(tree,  rep , lab ):
        #print(" debut profis ", tree.is_profils , tree.etiquette)
        global  nombre_regrpm


        if (nombre_regrpm==0):
            noued_intrm = Node(None, None, None, rep, tree.rep, None, None, 0, None, None, None)
            sac = rep.tolist()

            nv_profils = Node([], None, None, None, None, rep, lab, 1, noued_intrm, 1, 0)
            nv_profils.sac.append(sac)

            noued_intrm.left = nv_profils
            noued_intrm.right = tree
            tree.parent = noued_intrm
            tree.fils = 1
            tree = noued_intrm
            # pnt_tree = tree
            # tree.display()

            return tree, nv_profils
        else:
            noued_intrmd = Node(None, None, None, rep, tree.rep, None, None, 0, None, None, tree.fils)
            sac = rep.tolist()
            nv_profils = Node([], None, None, None, None, rep, lab, 1, noued_intrmd, 1, 0)
            nv_profils.sac.append(sac)
            # nauveau_profils.division_sac(sac, rep, lab)
            noued_intrmd.left = nv_profils
            noued_intrmd.right = tree
            # if (tree.sac is not None):
            # tree.division_sac(tree.sac , tree.rep , tree.etiquette)

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

        return tree, nv_profils

    def search_tree(tree, rep ):
        global Tree
        global list_rech
        global nb_distance
        global nombre_regrpm
        global existe_profils
        global pointeur
        global lab
        existe_profils = False
        global nb_comparaison
        #X = None
        start_time = time.time()  # start time of the loop
        if tree == None:
            label = lab
            lab += 1
            tree, pointeur = Node.create_tree(Tree, rep, label, existe_profils, pointeur)
            Tree = tree

        else:

            while (tree.left != None):
                if (tree.is_profils == 1):
                    # le premier nouerd qui indique le debut de profils aprées le partitionnement
                    pointeur = tree


                #print(" pivotG", tree.pivotG)
                #d1 = bhattacharyya(tree.pivotG, rep)  # la distance de rep avec le PG   ( bahatacharya )
                #d1 = ecludienne_distance(tree.pivotG, rep)  # la distance de rep avec le PG   ( ecludienne )
                d1 = cosinus_distance(tree.pivotG, rep)
                nb_distance+=1
                # print("dist pG", d1)
                # print(" pivotD", tree.pivotD)
                #d2 = bhattacharyya(tree.pivotD, rep)  # la distance de rep avec le PD   *( bahatacharya )
                #d2 = ecludienne_distance(tree.pivotD, rep)  # la distance de rep avec le PD   ( ecludienne )
                d2 = cosinus_distance(tree.pivotD, rep)
                nb_distance += 1
                # print("distanceD", d2, d1)
                if (d1 < d2):  # <
                    nb_comparaison += 1

                    return tree.left.search_tree(rep)

                else:

                    return tree.right.search_tree(rep)
            # jsuque atteindre la derniere feuille

            #print("tree.rep", tree.rep)
            #print("rep", rep)
            #dis = bhattacharyya(tree.rep,rep)  # cvalcule de disatance entre rep et rep de derniere feuille  ( bahatacharya )
            #dis = ecludienne_distance(tree.rep, rep)  # cvalcule de disatance entre rep et rep de derniere feuille  ( ecludienne )

            dis = cosinus_distance(tree.rep, rep)
            nb_distance += 1
            if (tree.is_profils==1):
                pointeur = tree



            #print("dist bahta  rechrche feulle ", dis)

            if (dis < seuil_similarité):  # comparaison avec le seuil de similarité fixé  ( if the same profils )
                nb_comparaison += 1

                existe_profils = True
                label = tree.etiquette  # recupurer etiquette
                pointeur = tree  # l'emplecment de node

            else:
                existe_profils= False
                label = lab  # etiquette le nv profils
                lab = lab + 1  # compteur des etiquette +1

            end_time = time.time() - start_time
            list_rech.append(end_time)

            tree, pointeur = Node.create_tree(Tree, rep, label, existe_profils, pointeur)
            Tree = tree
        return  label, pointeur









"""
def main_down():
    matrix = upload_data()
    print(matrix)
    print("fin data")
    #tree = None
    global tree
    for i_line, line in enumerate(matrix):
        existe_profils, labelle, pointeur = Node.search_tree(tree, np.array(matrix[i_line]).astype(float))

        tree = Node.create_tree(tree, np.array(matrix[i_line]).astype(float), np.array(matrix[i_line]).astype(float),
                                labelle, existe_profils, pointeur)
        #tree.display()
    tree.display()
    hauteur =get_height(tree)
    global lab
    print("la hauteur de la'rbre =", hauteur , 'nombre des objets ', lab-1)
"""

