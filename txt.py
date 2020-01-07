import sys
sys.getdefaultencoding()
import pickle as pkl
import numpy as np
np.set_printoptions(threshold=1000000000000000)
path = './data/vocab.pkl'
file = open(path, 'rb')
vocab = pkl.load(file)
print(vocab)
inf = str(vocab)
obj_path = './data/vocab.txt'
ft = open(obj_path, 'w')
ft.write(inf)