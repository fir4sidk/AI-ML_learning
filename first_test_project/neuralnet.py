import numpy as np
import csv
import math


dmodel=128
dd=32
def changeweight(wold,dw,lr):
    return np.clip((wold - (lr * dw )).copy(),-1,1)
    
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
    M=[x for x in r]
    f.close()
    return M[0]
def softmax(m):
  max=np.max(m,axis=1,keepdims=True)
  sum=np.sum(np.exp(m-max),axis=1,keepdims=True)
  res=np.exp(m-max) / sum 
  return res
def masking(m):
  s=m.shape
  c=m.copy()
  c[np.triu_indices(s[0], k=1)]=-9999
  return c
def LN(x,gamma,beta):
    epsilon=10**(-5)
    a=(x-np.mean(x, axis=-1, keepdims=True))
    b=np.sqrt(np.var(x, axis=-1, keepdims=True)+epsilon)
    hat_x=a / b
    return hat_x * gamma + beta
def MLP(E,wu,bu,wd,bd):
    global FU , FA , FFN
    FU=np.dot(E,wu) + bu
    FA=np.maximum(0,FU)
    FFN=np.dot(FA,wd) + bd
    return FFN
def attention(E,dd,Q,K,V):
  global M,SMA
  KT=np.transpose(K)
  M=masking(np.dot(Q,KT))
  SMA=softmax(M / math.sqrt(dd))
  O=np.dot(SMA,V)
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
  lind=list()
  for i in tokens:
    lind.append(vocab.index(i))
  E=[We[x,:] for x in lind]
  return np.array(E)
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
    rowsum=np.sum(dss, axis=1 , keepdims=True)
    return output * (doutput - rowsum)
#gradient of the input of the normalisation layer (full matrix)
def dBLN(dLN,X,H,gamma):
    var=np.var(X,axis=1,keepdims=True)
    m=np.mean(X,axis=1,keepdims=True)
    epsilon=10**(-5)
    a=(X-m)
    b=np.sqrt(var+epsilon)
    hat_X=a / b
    dX=(1/(H*b)) * (H*dLN - dLN - hat_X*np.sum(dLN * hat_X , axis=1 , keepdims=True))
    dX=dX * gamma
    return dX






def dgamma(Xh,dPR):
    G=dPR * Xh
    dg=np.sum(G,axis=0)
    return dg


def dbeta(dPR):
    db=np.sum(dPR,axis=0)
    return db

def dembed (tokens,vocab,We,dE):
    w=np.zeros(We.shape)
    lind=list()
    for i in tokens:
        lind.append(vocab.index(i))
    for i in range(len(lind)):
        w[lind[i]]+=dE[i]
    return w

#def masking0(X):
 #   for i in range(x.shape[0]):
#        for j in range(x.shape[1]):

def Trueres(tokensout,vocab,n):
    TR=np.zeros((n,len(vocab)))
    lind=list()
    for i in tokensout:
        lind.append(vocab.index(i))
    for j in range (len(tokensout)):
        TR[j,lind[j]]=1
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
    x1, gamma1, V, SMA, dd, K, Q, wv, wq, wk, tokens, vocab):
    dz=FinalResult-TR
    dRM=dz @ We
    dg2=dgamma(hat(x2),dRM)
    dbeta2=dbeta(dRM)
    dX2=dBLN(dRM,x2,dmodel,gamma2)
    dWd=np.maximum(0,FU).T @ dX2
    dBd=np.sum(dX2,axis=0,keepdims=True)
    dRA=dX2 + (((dX2 @ wd.T)* (FU>0))@ wu.T)
    dAF=dX2 @ wd.T
    dFU=dAF * (FU>0)
    dWu=RA.T @ dFU
    dBu=np.sum(dFU,axis=0,keepdims=True)
    dg1=dgamma(hat(x1),dRA)
    dbeta1=dbeta(dRA)
    dX1=dBLN(dRA,x1,dmodel,gamma1)
    dA=dX1
    dSMA=dA @ V.T
    dV=SMA.T @ dA
    dM=(1/math.sqrt(dd)) * dsoftmax(SMA,dSMA)
    dUM=dM #masking0(dM)
    dQ=dUM @ K
    dK= dUM.T @ Q
    dEv=dV @ wv.T
    dEq=dQ @ wq.T
    dEk=dK @ wk.T
    dE = dEk +dEv + dEq
    dWv= E.T @ dV
    dWq= E.T @ dQ
    dWk=E.T @ dK
    dWep=dz.T @ RM
    dWee=dembed(tokens,vocab,We,dE)
    dWe=dWee+dWep

    return (
        dz, dRM, dg2, dbeta2, dX2, dWd, dBd, dRA, dAF, dFU, dWu, dBu,
        dg1, dbeta1, dX1, dA, dSMA, dV, dM, dUM, dQ, dK, dEv, dEq, dEk,
        dE, dWv, dWq, dWk, dWep, dWee, dWe
    )


def accumulate_weights(gWe, gwk, gwq, gwv, gwu, gwd, gbu, gbd, ggamma2, gbeta2, ggamma1, gbeta1,
                       dWe, dWk, dWq, dWv, dWu, dWd, dBu, dBd, dg2, dbeta2, dg1, dbeta1,n):
    gWe += (dWe/n)
    gwk += (dWk/n)
    gwq += (dWq/n)
    gwv += (dWv/n)
    gwu += (dWu/n)
    gwd += (dWd/n)
    gbu += (dBu/n)
    gbd += (dBd/n)
    ggamma2 += (dg2/n)
    gbeta2 += (dbeta2/n)
    ggamma1 += (dg1/n)
    gbeta1 += (dbeta1/n)
    return (gWe, gwk, gwq, gwv, gwu, gwd, gbu, gbd, ggamma2, gbeta2, ggamma1, gbeta1)
(wq , wk , wv , beta2 , beta1,gamma2,gamma1,wu,bu,wd,bd,We,vocab,dataset)=loadweights()
step=0
while step<1600:
    avgloss=0
    (gWe, gwk, gwq, gwv, gwu, gwd, gbu, gbd,ggamma2, gbeta2, ggamma1, gbeta1)=(0,0,0,0,0,0,0,0,0,0,0,0)
    for i in range(len(dataset)):
        #input / output processing : tokenizing / position embeding
        inp=dataset[i][0]
        output=dataset[i][1]
        import re
        pattern = r'[a-zA-Z\.]+|\*\*|.'
        tokens=re.findall(pattern,inp)
        tokensout=re.findall(pattern,output)
        FE=embeding(tokens,vocab,We)
        E=FE + PE(len(tokens),dmodel)
        
        # Forward Pass 
        K=np.dot(E,wk)
        Q=np.dot(E,wq)
        V=np.dot(E,wv)
        A=attention(E,dd,Q,K,V)
        x1=E+A
        RA=LN(x1,gamma1,beta1)
        M=MLP(RA,wu,bu,wd,bd)
        x2=RA+M
        RM=LN(x2,gamma2,beta2)
        z=RM @ We.T
        FinalResult=softmax(z)

        #initialising propability as 1 to the target result's tokens
        TR=Trueres(tokensout,vocab,len(tokensout))

        #calculating the loss for each token
        l = TR * np.log(FinalResult + 1e-15)
        #final loss calculation =======================
        loss = -np.sum(l)
        #adding it to sum of losses 
        avgloss+=loss
        
        #calculating the gradients
        (dz, dRM, dg2, dbeta2, dX2, dWd, dBd, dRA, dAF, dFU, dWu, dBu,
        dg1, dbeta1, dX1, dA, dSMA, dV, dM, dUM, dQ, dK, dEv, dEq, dEk,
        dE, dWv, dWq, dWk, dWep, dWee, dWe)=gradientcalc(
                                                    FinalResult, TR, We, x2, gamma2, dmodel, FU, wu, wd, RA, 
                                                    x1, gamma1, V, SMA, dd, K, Q, wv, wq, wk, tokens, vocab)


        #calculating the sum of the gradients
        (gWe, gwk, gwq, gwv, gwu, gwd, gbu, gbd,
         ggamma2, gbeta2, ggamma1, gbeta1)=accumulate_weights(gWe, gwk, gwq, gwv, gwu, gwd, gbu, gbd, ggamma2,
          gbeta2, ggamma1, gbeta1,dWe, dWk, dWq, dWv, dWu, dWd, dBu, dBd, dg2, dbeta2, dg1, dbeta1,len(dataset))
        #setting a higher learning rate for the first 2000 step (warmup steps) then dividing it by half 
        if step <2000:
            lr=0.01
        else:
            lr=0.005
        if i % 20 ==0:
            #updating the weights after every micro-batch (20 data item)
            #i made it update the weights based on the average of gradients
            (wq , wk , wv , beta2 , beta1,
            gamma2,gamma1,wu,bu,wd,bd,We)=changeweights(lr, We, wk, wq, wv, wu, wd, bu, bd, gamma2, beta2, gamma1, beta1,
                                                        gWe, gwk, gwq, gwv, gwu, gwd, gbu, gbd, ggamma2, gbeta2, ggamma1, gbeta1)
            (gWe, gwk, gwq, gwv, gwu, gwd, gbu, gbd,ggamma2, gbeta2, ggamma1, gbeta1)=(0,0,0,0,0,0,0,0,0,0,0,0)


    #calculating the average loss for the actual epoche
    avgloss=avgloss/len(dataset)
    #here step means epoche
    step+=1
    #logging the las average of loss after an amount of epoches and saving the weights to files
    if step % 200 ==0:
        print("last avgloss=",avgloss)
        print(step) 
        saveweights(wq , wk , wv , beta2 , beta1,gamma2,gamma1,wu,bu,wd,bd,We)
    if avgloss <0.3:
        saveweights(wq , wk , wv , beta2 , beta1,gamma2,gamma1,wu,bu,wd,bd,We)
        break
#inp=str(input("give a arithmetic formula: "))
#import re
#pattern = r'[a-zA-Z\d\.]+|\*\*|.'
#tokens=re.findall(pattern,inp)
#FE=embeding(tokens,vocab,We)
#E=FE + PE(len(tokens),dmodel)
#K=np.dot(E,wk)
#Q=np.dot(E,wq)
#V=np.dot(E,wv)
#A=attention(E,dd,Q,K,V)
#x1=E+A
#RA=LN(x1,gamma1,beta1)
#M=MLP(RA,wu,bu,wd,bd)
#x2=RA+M
#RM=LN(x2,gamma2,beta2)
#z=RM @ We.T
#FinalResult=softmax(z)
#unembeding(FinalResult,vocab)
