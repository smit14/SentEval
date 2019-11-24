import os, sys
readFile = open("./SNLI/s1.train")
lines = readFile.readlines()

readFile.close()
w = open("./SNLI/ss1.train", 'w')

w.writelines([items for items in lines[:50000]])
w.close()
