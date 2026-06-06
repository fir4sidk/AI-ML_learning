import random
import cupy as np
import csv
import math
dmodel=128
h=4
dk=dmodel//h
dv=dmodel//h
dff=dmodel*4
L=2
def createweight(path,d):
  w=np.random.randn(*d) * 0.1
  np.save(path,w)
  
# Synthetic Arithmetic Dataset (Addition & Subtraction)

def newdataset():
    batch_size=1000
    dataset = [(str(i%10)+str((i+1)%10)+str((i+2)%10)+str((i+3)%10),
            str((i+1)%10)+str((i+2)%10)+str((i+3)%10)+str((i+4)%10))
           for i in range(batch_size)]
    datafile=open("dataset.csv","w")
    datafilewriter=csv.writer(datafile,delimiter=";")
    datafilewriter.writerows(dataset)

def newembeding(d):
    vocab = [f"{i}" for i in range(10)]

    createweight("embeding",(len(vocab),d))
    f=open("vocab.csv","w")
    wr=csv.writer(f,delimiter=";")
    wr.writerow(vocab)
    f.close()

newdataset()
newembeding(dmodel)
createweight("beta1",(L,1,dmodel))
createweight("beta2",(L,1,dmodel))
createweight("gamma1",(L,1,dmodel))
createweight("gamma2",(L,1,dmodel))
createweight("wu",(L,dmodel,dff))
createweight("bu",(L,1,dff))
createweight("wd",(L,dff,dmodel))
createweight("bd",(L,1,dmodel))
createweight("wq",(L,h,dmodel,dk))
createweight("wk",(L,h,dmodel,dk))
createweight("wv",(L,h,dmodel,dv))
createweight("wo",(L,dv*h,dmodel))