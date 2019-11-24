import os, sys

n = 200000
readFile = open("./SNLI/s1.train")
lines = readFile.readlines()

readFile.close()
w = open("./SNLI/s1_200k.train", 'w')

w.writelines([items for items in lines[:n]])
w.close()

readFile = open("./SNLI/s2.train")
lines = readFile.readlines()

readFile.close()
w = open("./SNLI/s2_200k.train", 'w')

w.writelines([items for items in lines[:n]])
w.close()

readFile = open("./SNLI/labels.train")
lines = readFile.readlines()

readFile.close()
w = open("./SNLI/labels_200k.train", 'w')

w.writelines([items for items in lines[:n]])
w.close()


