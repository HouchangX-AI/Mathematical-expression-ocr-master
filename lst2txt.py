import os

new = 'im2latex_formulas.norm.txt'
f_new = open(new, 'w')
with open('/Users/wangxiyao/Desktop/im2latex-Tensorflow/data/im2latex_formulas.norm.lst') as f:
    lines = f.readlines()
    for line in lines:
        f_new.write(line)
f_new.close()

