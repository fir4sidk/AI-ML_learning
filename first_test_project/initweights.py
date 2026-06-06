import random
import numpy as np
import csv
import math
dmodel=128
h=4
dk=dmodel//h
dv=dmodel//h
dff=dmodel*4
def createweight(path,d):
  w=np.random.randn(*d) * 0.1
  np.save(path,w)
  
# Synthetic Arithmetic Dataset (Addition & Subtraction)

def newdataset():
    dataset =[]
    for __ in range (200):
        a=random.randint(0,40)
        b=random.randint(0,40)
        add=(f"{a:02d}+{b:02d}={(a+b):02d}EOS")
        if a>b:
            sub=(f"{a:02d}-{b:02d}={(a-b):02d}EOS")
        else:
            sub=(f"{b:02d}-{a:02d}={(b-a):02d}EOS")
        ta=(add[:-3],add[1:])
        ts=(sub[:-3],sub[1:])
        dataset.append(ta)
        dataset.append(ts)
    datafile=open("dataset.csv","w")
    datafilewriter=csv.writer(datafile,delimiter=";")
    datafilewriter.writerows(dataset)

def newembeding(d):
    vocab = [
    "0", "1", "2", "3", "4", "5", "6", "7", "8", "9", ".",

    "+", "-", "x", "=","EOS"
]

    createweight("embeding",(len(vocab),d))
    f=open("vocab.csv","w")
    wr=csv.writer(f,delimiter=";")
    wr.writerow(vocab)
    f.close()
newdataset()
newembeding(dmodel)
for i in range (1,2):
    for j in range (1,3):
      createweight("beta"+str(i)+"_"+str(j),(1,dmodel))
      createweight("gamma"+str(i)+"_"+str(j),(1,dmodel))
createweight("wu",(dmodel,dff))
createweight("bu",(1,dff))
createweight("wd",(dff,dmodel))
createweight("bd",(1,dmodel))
createweight("wq",(h,dmodel,dk))
createweight("wk",(h,dmodel,dk))
createweight("wv",(h,dmodel,dv))
createweight("wo",(dv*h,dmodel))