import random
import numpy as np
import csv
import math
dmodel=32
dd=8
def createweight(path,d):
  f=open(path,"w")
  w=np.random.randn(d[0],d[1]) * 0.1
  file_=csv.writer(f,delimiter=";")
  file_.writerows(w)
  f.close()
# Synthetic Arithmetic Dataset (Addition & Subtraction)

def newdataset():
    dataset =[]
    for __ in range (200):
        a=random.randint(0,9)
        b=random.randint(0,9)
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

    createweight("embeding.csv",(len(vocab),d))
    f=open("vocab.csv","w")
    wr=csv.writer(f,delimiter=";")
    wr.writerow(vocab)
    f.close()
newdataset()
newembeding(dmodel)
for i in range (1,2):
    for j in range (1,3):
      createweight("beta"+str(i)+"_"+str(j)+".csv",(1,dmodel))
      createweight("gamma"+str(i)+"_"+str(j)+".csv",(1,dmodel))
createweight("wu.csv",(dmodel,dmodel*2))
createweight("bu.csv",(1,dmodel*2))
createweight("wd.csv",(dmodel*2,dmodel))
createweight("bd.csv",(1,dmodel))
createweight("wq.csv",(dmodel,dd))
createweight("wk.csv",(dmodel,dd))
createweight("wv.csv",(dmodel,dmodel))