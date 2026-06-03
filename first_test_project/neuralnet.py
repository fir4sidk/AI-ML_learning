import numpy as np
import csv
import math


dmodel=32
dd=8
def changeweight(wold,dw,lr):
    norm=np.linalg.norm(dw)
    if norm>1:
        dw=dw * (1/norm)
    return (wold - (lr * dw )).copy()

def changefiles(w,path):
    f=open(path,"w")
    file_=csv.writer(f,delimiter=";")
    file_.writerows(w)
    f.close()
def importweight(path):
  f=open(path,"r")
  r=csv.reader(f,delimiter=";")
  M=np.array([x for x in r],dtype=float)
  f.close()
  return M
def importvocab(path):
    f=open(path,"r")
    r=csv.reader(f,delimiter=";")
    v=list(r)[0]
    M = {x:int(i) for i,x in enumerate(v)}
    f.close()
    return M
def softmax(m):
  max=np.max(m,axis=2,keepdims=True)
  sum=np.sum(np.exp(m-max),axis=2,keepdims=True)
  res=np.exp(m-max) / sum
  return res
def masking(m):
  s=m.shape
  c=m.copy()
  ind=np.triu_indices(s[1], k=1)
  c[:,ind[0],ind[1]]=-9999
  return c
def LN(x,gamma,beta):
    epsilon=10**(-5)
    a=(x-np.mean(x, axis=-1, keepdims=True))
    b=np.sqrt(np.var(x, axis=-1, keepdims=True)+epsilon)
    hat_x=a / b
    return hat_x * gamma + beta
def MLP(E,wu,bu,wd,bd):
    global FU , FA , FFN
    FU=E @ wu + bu
    FA=np.maximum(0,FU)
    FFN=FA @ wd+ bd
    return FFN
def attention(E,dd,Q,K,V):
  global M,SMA
  KT=np.transpose(K,axes=(0,2,1))
  M=masking(Q @ KT)
  SMA=softmax(M / math.sqrt(dd))
  O=SMA @ V
  return O

def PE(seq_len, dmodel):
    i = np.arange(seq_len)[:, np.newaxis]  # Shape (seq_len, 1)
    j = np.arange(dmodel)[np.newaxis, :]   # Shape (1, dmodel)

    # Compute all angles at once
    angle = i / (10000 ** (2 * j / dmodel))

    # Allocate matrix and apply sin to even columns, cos to odd columns
    p = np.zeros((seq_len, dmodel))
    p[:, 0::2] = np.sin(angle[:, 0::2])
    p[:, 1::2] = np.cos(angle[:, 1::2])
    return p
def embeding(tokens,vocab,We):
  lind=list(map(vocab.get,tokens))
  E=We[lind]
  return E

def unembeding(FR,vocab):
    i=np.argmax(FR,axis=1)
    res=""
    for j in i:
        res+=vocab[j]
    print(res)
def hat(x):
    epsilon=10**(-5)
    a=(x-np.mean(x, axis=-1, keepdims=True))
    b=np.sqrt(np.var(x, axis=-1, keepdims=True)+epsilon)
    hat_x=a / b
    return hat_x
#gradient of the input of softmax (full matrix)
def dsoftmax(output,doutput):
    dss= output * doutput
    rowsum=np.sum(dss, axis=-1 , keepdims=True)
    return output * (doutput - rowsum)
#gradient of the input of the normalisation layer (full matrix)
def dBLN(dLN,X,H,gamma):
    var=np.var(X,axis=-1,keepdims=True)
    m=np.mean(X,axis=-1,keepdims=True)
    epsilon=10**(-5)
    a=(X-m)
    b=np.sqrt(var+epsilon)
    hat_X=a / b
    dX=(1/(H*b)) * (H*dLN - dLN - hat_X*np.sum(dLN * hat_X , axis=-1 , keepdims=True))
    dX=dX * gamma
    return dX






def dgamma(Xh,dPR):
    G=dPR * Xh
    dg=np.sum(G,axis=1)
    return dg


def dbeta(dPR):
    db=np.sum(dPR,axis=1)
    return db

def dembed (postokens,We,dE):
    w=np.zeros((400,We.shape[0],We.shape[1]))
    w_re=np.reshape(w,(400*We.shape[0],We.shape[1]))
    dE_re=np.reshape(dE,(400*dE.shape[1],dE.shape[2]))
    postokens_re=np.reshape(postokens,(postokens.size))
    np.add.at(w_re,postokens_re, dE_re)
    return w

#def masking0(X):
 #   for i in range(x.shape[0]):
#        for j in range(x.shape[1]):

def Trueres(tokensout,vocab,n):
    TR=np.zeros((n,len(vocab)))
    lind=list(map(vocab.get,tokensout))
    TR[list(range(len(lind))),lind]=1
    return TR


def saveweights(wq , wk , wv , beta2 , beta1,gamma2,gamma1,wu,bu,wd,bd,We):
    changefiles(We,"embeding.csv")
    changefiles(wu,"wu.csv")
    changefiles(wd,"wd.csv")
    changefiles(bu,"bu.csv")
    changefiles(bd,"bd.csv")
    changefiles(wq,"wq.csv")
    changefiles(wk,"wk.csv")
    changefiles(wv,"wv.csv")
    changefiles(gamma2,"gamma1_2.csv")
    changefiles(beta2,"beta1_2.csv")
    changefiles(gamma1,"gamma1_1.csv")
    changefiles(beta1,"beta1_1.csv")
def loadweights():
    wq=importweight("wq.csv")
    wk=importweight("wk.csv")
    wv=importweight("wv.csv")
    beta2=importweight("beta1_2.csv")
    gamma2=importweight("gamma1_2.csv")
    beta1=importweight("beta1_1.csv")
    gamma1=importweight("gamma1_1.csv")
    wu=importweight("wu.csv")
    bu=importweight("bu.csv")
    wd=importweight("wd.csv")
    bd=importweight("bd.csv")
    We=importweight("embeding.csv")
    vocab=importvocab("vocab.csv")
    datafile=open("dataset.csv","r")
    dfilew=csv.reader(datafile,delimiter=";")
    dataset=[tuple(row) for row in dfilew]
    return (wq , wk , wv , beta2 , beta1,gamma2,gamma1,wu,bu,wd,bd,We,vocab,dataset)

def changeweights(lr, We, wk, wq, wv, wu, wd, bu, bd, gamma2, beta2, gamma1, beta1,
    dWe, dWk, dWq, dWv, dWu, dWd, dBu, dBd, dg2, dbeta2, dg1, dbeta1):

    We = changeweight(We, dWe, lr)
    wk = changeweight(wk, dWk, lr)
    wq = changeweight(wq, dWq, lr)
    wv = changeweight(wv, dWv, lr)
    wu = changeweight(wu, dWu, lr)
    wd = changeweight(wd, dWd, lr)
    bu = changeweight(bu, dBu, lr)
    bd = changeweight(bd, dBd, lr)
    gamma2 = changeweight(gamma2, dg2, lr)
    beta2 = changeweight(beta2, dbeta2, lr)
    gamma1 = changeweight(gamma1, dg1, lr)
    beta1 = changeweight(beta1, dbeta1, lr)
    return (wq , wk , wv , beta2 , beta1,gamma2,gamma1,wu,bu,wd,bd,We)

def gradientcalc(FinalResult, TR, We, x2, gamma2, dmodel, FU, wu, wd, RA,
    x1, gamma1, V, SMA, dd, K, Q, wv, wq, wk, postokens,E,RM):
    dz = FinalResult - TR


    dRM = dz @ We


    dg2 = dgamma(hat(x2), dRM)


    dbeta2 = dbeta(dRM)


    dX2 = dBLN(dRM, x2, dmodel, gamma2)


    dWd = np.transpose(np.maximum(0, FU),axes=(0,2,1)) @ dX2


    dBd = np.sum(dX2, axis=1, keepdims=True)


    dRA = dX2 + (((dX2 @ np.transpose(wd)) * (FU > 0)) @ np.transpose(wu))


    dAF = dX2 @ np.transpose(wd)


    dFU = dAF * (FU > 0)


    dWu = np.transpose(RA, axes=(0, 2, 1)) @ dFU


    dBu = np.sum(dFU, axis=1, keepdims=True)


    dg1 = dgamma(hat(x1), dRA)


    dbeta1 = dbeta(dRA)


    dX1 = dBLN(dRA, x1, dmodel, gamma1)


    dA = dX1


    dSMA = dA @ np.transpose(V, axes=(0, 2, 1))


    dV = np.transpose(SMA, axes=(0, 2, 1)) @ dA


    dM = (1 / math.sqrt(dd)) * dsoftmax(SMA, dSMA)


    dUM = dM  # masking0(dM)


    dQ = dUM @ K


    dK = np.transpose(dUM, axes=(0, 2, 1)) @ Q


    dEv = dV @ np.transpose(wv)


    dEq = dQ @ np.transpose(wq)


    dEk = dK @ np.transpose(wk)


    dE = dEk + dEv + dEq


    dWv = np.transpose(E, axes=(0, 2, 1)) @ dV


    dWq = np.transpose(E, axes=(0, 2, 1)) @ dQ


    dWk = np.transpose(E, axes=(0, 2, 1)) @ dK


    dWep = np.transpose(dz, axes=(0, 2, 1)) @ RM


    dWee = dembed(postokens, We, dE)

    dWe = dWee + dWep


    return (
        dz, dRM, dg2, dbeta2, dX2, dWd, dBd, dRA, dAF, dFU, dWu, dBu,
        dg1, dbeta1, dX1, dA, dSMA, dV, dM, dUM, dQ, dK, dEv, dEq, dEk,
        dE, dWv, dWq, dWk, dWep, dWee, dWe
    )


def accumulate_weights(gWe, gwk, gwq, gwv, gwu, gwd, gbu, gbd, ggamma2, gbeta2, ggamma1, gbeta1,
                       dWe, dWk, dWq, dWv, dWu, dWd, dBu, dBd, dg2, dbeta2, dg1, dbeta1,n):
    gWe = np.sum(dWe,axis=0) / n
    gwk = np.sum(dWk,axis=0) / n
    gwq = np.sum(dWq,axis=0) / n
    gwv = np.sum(dWv,axis=0) / n
    gwu = np.sum(dWu,axis=0) / n
    gwd = np.sum(dWd,axis=0) / n
    gbu = np.sum(dBu,axis=0) / n
    gbd = np.sum(dBd,axis=0) / n
    ggamma2 = np.sum(dg2,axis=0) / n
    gbeta2 = np.sum(dbeta2,axis=0) / n
    ggamma1 = np.sum(dg1,axis=0) / n
    gbeta1 = np.sum(dbeta1,axis=0) / n
    return (gWe, gwk, gwq, gwv, gwu, gwd, gbu, gbd, ggamma2, gbeta2, ggamma1, gbeta1)
def posvoc(tokens,vocab):
  lind=np.array(list(map(vocab.get,tokens)))
  return lind


batch_size=len(dataset)
seq_len=8
(wq , wk , wv , beta2 , beta1,gamma2,gamma1,wu,bu,wd,bd,We,vocab,dataset)=loadweights()
step=0
P=PE(seq_len,dmodel)
import re
pattern = r'[a-zA-Z\.]+|\*\*|.'
posinput=np.zeros((batch_size,seq_len))
posoutput=np.zeros((batch_size,seq_len))
TR=np.zeros((batch_size,seq_len,len(vocab.keys())))
for i in range(len(dataset)):
  inp=dataset[i][0]
  output=dataset[i][1]
  tokens=re.findall(pattern,inp)
  tokensout=re.findall(pattern,output)
  posinput[i]=posvoc(tokens,vocab)
  posoutput[i]=posvoc(tokensout,vocab)
  TR[i]=Trueres(tokensout,vocab,seq_len)
seq_len=len(posinput[0])
posoutput=np.int32(posoutput + (np.arange(len(dataset))*We.shape[0])[:,np.newaxis])

while step<200000:
    avgloss=0
    (gWe, gwk, gwq, gwv, gwu, gwd, gbu, gbd,ggamma2, gbeta2, ggamma1, gbeta1)=(0,0,0,0,0,0,0,0,0,0,0,0)
    ET=np.zeros((400,seq_len,dmodel))
    ET=(We[np.int16(posinput)])
    ET+=P

    K=ET @ wk
    Q=ET @ wq
    V=ET @ wv
    #attentioon ======
    A=attention(ET,dd,Q,K,V)

    x1=ET+A

    RA=LN(x1,gamma1,beta1)

    M=MLP(RA,wu,bu,wd,bd)

    x2=RA+M

    RM=LN(x2,gamma2,beta2)

    z=RM @ We.T

    FinalResult=softmax(z)



    #calculating the loss for each token
    l = TR * np.log(FinalResult + 1e-15)
     #final loss calculation =======================
    loss= -np.sum(l,axis=(1,2))
    #average loss =================================
    avgloss = -np.sum(l)/len(dataset)
    (dz, dRM, dg2, dbeta2, dX2, dWd, dBd, dRA, dAF, dFU, dWu, dBu,
        dg1, dbeta1, dX1, dA, dSMA, dV, dM, dUM, dQ, dK, dEv, dEq, dEk,
        dE, dWv, dWq, dWk, dWep, dWee, dWe)=gradientcalc(
                                                    FinalResult, TR, We, x2, gamma2, dmodel, FU, wu, wd, RA,
                                                    x1, gamma1, V, SMA, dd, K, Q, wv, wq, wk, posoutput,ET,RM)
        #calculating the sum of the gradients
    (gWe, gwk, gwq, gwv, gwu, gwd, gbu, gbd,
         ggamma2, gbeta2, ggamma1, gbeta1)=accumulate_weights(gWe, gwk, gwq, gwv, gwu, gwd, gbu, gbd, ggamma2,
          gbeta2, ggamma1, gbeta1,dWe, dWk, dWq, dWv, dWu, dWd, dBu, dBd, dg2, dbeta2, dg1, dbeta1,batch_size)
    if step <=150000:
      lr=0.0005
    else:
      lr=0.0001


    (wq , wk , wv , beta2 , beta1,
            gamma2,gamma1,wu,bu,wd,bd,We)=changeweights(lr, We, wk, wq, wv, wu, wd, bu, bd, gamma2, beta2, gamma1, beta1,
                                                        gWe, gwk, gwq, gwv, gwu, gwd, gbu, gbd, ggamma2, gbeta2, ggamma1, gbeta1)
    step+=1
    if step >= 200000:
        print(f"Reached 200k steps ceiling. Saving final checkpoint and exiting.")
        saveweights(wq , wk , wv , beta2 , beta1, gamma2, gamma1, wu, bu, wd, bd, We)
        break
    if step % 10000 ==0:
        print("learning rate =",lr,"last avgloss=",avgloss)
        print("setep=",step)
        saveweights(wq , wk , wv , beta2 , beta1,gamma2,gamma1,wu,bu,wd,bd,We)
        print("saved")