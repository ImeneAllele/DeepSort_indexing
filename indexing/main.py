
from dataSet import *
import time
from  UpdownTree import  *
from downTree import  *
global nb_distance
nb_distance=0
from newUpDown import *


if __name__ == "__main__":

    start_time = time.time()
    #print( "down")
    #main_down()
    #print("updown")
    #main_up_down()
    #print( "new updonw")
    main_new_up_down()
    end_time = time.time() - start_time
    print("le temp de rechrche ", end_time)



"""  
   v1 = premier vecteur
   envoiyer le premier vecteur au fog
   if (tree== None) :
   return 1  
   else :
   existe_profils, labelle, pointeur, X=rechreche ( v1)
   # existe_profils :variable booleen  true si le profils avec le vecteur v1 existe false sinon
   #labelle   : ettiquette de profils
   #pointeur  : si le profils existe return l'mplecement de profils , none sinon
   #X    :  x=0 si le profils dans la liste none sinon
   envoyer ( labelle , camera )
   traking _ camera
   envoyer ( rep , sac , labelle ) ----- fog


   rep= calculer la moyenne ou bien la variance  pour le sac ----- le vecteur represantant



#######################
    tree = None
    cap = cv2.VideoCapture("3.mp4")
    tree=detect(tree,cap , 0.7, 10)
    tree.display()

   """