import random
import numpy as np
import csv
import math
dmodel=128
dd=32
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
        ta=(f"{a:02d}+{b:02d}=",f"+{b:02d}={(a+b):02d}")
        tc=(f"{a:02d}-{b:02d}=",f"-{b:02d}={(a-b):02d}")
        dataset.append(ta)
        dataset.append(tc)
    datafile=open("dataset.csv","w")
    datafilewriter=csv.writer(datafile,delimiter=";")
    datafilewriter.writerows(dataset)

def newembeding(d):
    vocab = [
    "0", "1", "2", "3", "4", "5", "6", "7", "8", "9", ".",

    "+", "-", "x", "/", "=", "(", ")", "[", "]", ",","|",

    "^", "**", "sqrt", "%", "!", "log", "ln",

    "sin", "cos", "tan", "arcsin", "arccos", "arctan",

    "pi", "e"
]

    createweight("embeding.csv",(len(vocab),d))
    f=open("vocab.csv","w")
    wr=csv.writer(f,delimiter=";")
    wr.writerow(vocab)
    f.close()

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