
import numpy as np

from scipy.spatial import distance
from indexing.evaluation import *




def bhattacharyya(h1, h2):

  """
    Calculates the Byattacharyya distance of two histograms.'''
  """
  def normalize(h):
    return h / np.sum(h)

  d = (1 - np.sum(np.sqrt(np.multiply(normalize(h1), normalize(h2)))) ** 2)

  return d


def ecludienne_distance(x, y):  # choix de distance

    d = (np.sqrt(np.sum((x - y) ** 2)))  # pour la distance eclidien
    return d



def cosinus_distance(x,y):
    d =np.array(distance.cosine(x,y))
    return d







