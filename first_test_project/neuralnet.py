import cupy as np
import csv


dmodel=128
h=4
dk=dmodel//h
dv=dmodel//h
B1=0.9
B2=0.999


def mt(m,t,B1,gt,pow):
  return (B1 * m) + (1-B1) * (gt ** pow)

def hat_mt(mt,B1,t):
  return mt / (1-(B1 ** t))


def changeweight(wold,dw,lr,B1,B2,t,mv,n):
    eps=1e-8
    mv[n][0]=mt(mv[n][0],t,B1,dw,1)
    mv[n][1]=mt(mv[n][1],t,B2,dw,2)
    hat_m=hat_mt(mv[n][0],B1,t)
    hat_v=hat_mt(mv[n][1],B2,t)
    wnew=wold - ((lr * hat_m)/(np.sqrt(hat_v)+eps))
    return wnew



def importvocab(path):
    f=open(path,"r")
    r=csv.reader(f,delimiter=";")
    v=list(r)[0]
    M = {x:int(i) for i,x in enumerate(v)}
    f.close()
    return M
def softmax(m):
  max=np.max(m,axis=-1,keepdims=True)
  sum=np.sum(np.exp(m-max),axis=-1,keepdims=True)
  res=np.exp(m-max) / sum
  return res
  
def masking(m):
  s=m.shape
  c=m.copy()
  ind=np.triu_indices(s[2], k=1)
  c[:,:,ind[0],ind[1]]=-9999
  return c

def LN(x,gamma,beta):
    epsilon=10**(-5)
    a=(x-np.mean(x, axis=-1, keepdims=True))
    b=np.sqrt(np.var(x, axis=-1, keepdims=True)+epsilon)
    hat_x=a / b
    return hat_x * gamma + beta
def MLP(E,wu,bu,wd,bd):
    FU=E @ wu + bu
    FA=np.maximum(0,FU)
    FFN=FA @ wd+ bd
    return FFN,FU , FA
def attention(E,dk,Wqt,Wkt,Wvt,Wo,h):
  K= E[:,None,:,:] @ Wkt
  Q= E[:,None,:,:] @ Wqt
  V= E[:,None,:,:] @ Wvt
  KT=np.transpose(K,(0,1,3,2))
  M=masking(Q @ KT)
  SMA=softmax(M / np.sqrt(dk))
  O=SMA @ V
  con=np.reshape(O.transpose(0,2,1,3),(O.shape[0],O.shape[2],dk*h))
  A=con @ Wo
  return (A,SMA,Q,K,V,con)

def PE(seq_len, dmodel):
    i = np.arange(seq_len)[:, np.newaxis]  # Shape (seq_len, 1)
    j = np.arange(dmodel)[np.newaxis, :]   # Shape (1, dmodel)


    angle = i / (10000 ** (2 * j / dmodel))

    p = np.zeros((seq_len, dmodel))
    p[:, 0::2] = np.sin(angle[:, 0::2])
    p[:, 1::2] = np.cos(angle[:, 1::2])
    return p


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

def dembed (postokens,We,dE,batch_size):
    w=np.zeros((batch_size,We.shape[0],We.shape[1]))
    w_re=np.reshape(w,(batch_size*We.shape[0],We.shape[1]))
    dE_re=np.reshape(dE,(batch_size*dE.shape[1],dE.shape[2]))
    postokens_re=np.reshape(postokens,(postokens.size))
    
    try:
        np.add.at(w_re,postokens_re, dE_re)
    except:
        import cupyx
        cupyx.scatter_add(w_re,postokens_re, dE_re)
    return w


def Trueres(tokensout,vocab,n):
    TR=np.zeros((n,len(vocab)))
    lind=list(map(vocab.get,tokensout))
    TR[list(range(len(lind))),lind]=1
    return TR


def saveweights(wq , wk , wv , beta2 , beta1,gamma2,gamma1,wu,bu,wd,bd,We,wo):
    np.save("embeding.npy", We)
    np.save("wu.npy", wu)
    np.save("wd.npy", wd)
    np.save("bu.npy", bu)
    np.save("bd.npy", bd)
    np.save("wq.npy", wq)
    np.save("wk.npy", wk)
    np.save("wv.npy", wv)
    np.save("gamma1_2.npy", gamma2)
    np.save("beta1_2.npy", beta2)
    np.save("gamma1_1.npy", gamma1)
    np.save("beta1_1.npy", beta1)
    np.save("wo",wo)
def loadweights():
    wo=np.load("wo.npy")
    wq=np.load("wq.npy")
    wk=np.load("wk.npy")
    wv=np.load("wv.npy")
    beta2=np.load("beta1_2.npy")
    gamma2=np.load("gamma1_2.npy")
    beta1=np.load("beta1_1.npy")
    gamma1=np.load("gamma1_1.npy")
    wu=np.load("wu.npy")
    bu=np.load("bu.npy")
    wd=np.load("wd.npy")
    bd=np.load("bd.npy")
    We=np.load("embeding.npy")
    vocab=importvocab("vocab.csv")
    datafile=open("dataset.csv","r")
    dfilew=csv.reader(datafile,delimiter=";")
    dataset=[tuple(row) for row in dfilew]
    return (wo,wq , wk , wv , beta2 , beta1,gamma2,gamma1,wu,bu,wd,bd,We,vocab,dataset)

def changeweights(lr, We, wk, wq, wv, wu, wd, bu, bd, gamma2, beta2, gamma1, beta1,wo,
    dWe, dWk, dWq, dWv, dWu, dWd, dBu, dBd, dg2, dbeta2, dg1, dbeta1,dwo,B1,B2,t,mv):
    
    We = changeweight(We, dWe, lr,B1,B2,t,mv,0)
    wk = changeweight(wk, dWk, lr,B1,B2,t,mv,1)
    wq = changeweight(wq, dWq, lr,B1,B2,t,mv,2)
    wo=changeweight(wo,dwo,lr,B1,B2,t,mv,3)
    wv = changeweight(wv, dWv, lr,B1,B2,t,mv,4)
    wu = changeweight(wu, dWu, lr,B1,B2,t,mv,5)
    wd = changeweight(wd, dWd, lr,B1,B2,t,mv,6)
    bu = changeweight(bu, dBu, lr,B1,B2,t,mv,7)
    bd = changeweight(bd, dBd, lr,B1,B2,t,mv,8)
    gamma2 = changeweight(gamma2, dg2, lr,B1,B2,t,mv,9)
    beta2 = changeweight(beta2, dbeta2, lr,B1,B2,t,mv,10)
    gamma1 = changeweight(gamma1, dg1, lr,B1,B2,t,mv,11)
    beta1 = changeweight(beta1, dbeta1, lr,B1,B2,t,mv,12)
    
    return (wq , wk , wv , beta2 , beta1,gamma2,gamma1,wu,bu,wd,bd,We,wo)
    
def gradientcalc(FinalResult, TR, We, x2, gamma2, dmodel, FU, wu, wd, RA,
    x1, gamma1, V, SMA, dk, K, Q, wv, wq, wk,Wo, postokens,E,RM,con,batch_size,h):
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

    dWo=con.transpose(0,2,1) @ dA
    dcon=dA @ Wo.T
    do=np.array(np.split(con,h,axis=-1)).transpose(1,0,2,3)
    dV=SMA.transpose(0,1,3,2) @ do
    dSMA=do @ V.transpose(0,1,3,2)
    dM=(1/np.sqrt(dk)) * (dsoftmax(SMA,dSMA))
    dQ=dM @ K
    dK= dM.transpose(0,1,3,2) @ Q
    
    dWv = np.transpose(E, axes=(0, 2, 1))[:,None,:,:] @ dV


    dWq = np.transpose(E, axes=(0, 2, 1))[:,None,:,:] @ dQ


    dWk = np.transpose(E, axes=(0, 2, 1))[:,None,:,:] @ dK

    dEk=dK @ wk.transpose(0,2,1)
    dEq=dQ @ wq.transpose(0,2,1)
    dEv=dV @ wv.transpose(0,2,1)
    dE=np.sum(dEk+dEq+dEv,axis=1)
    dWep = np.transpose(dz, axes=(0, 2, 1)) @ RM

    
    dWee = dembed(postokens, We, dE,batch_size)

    dWe = dWee + dWep


    return (dg2, dbeta2, dWd, dBd, dWu, dBu,
        dg1, dbeta1,
        dWv, dWq, dWk, dWe,dWo)
def initmoments(s0,s1,s2,s3,s4,s5,s6,s7,s8,s9,s10,s11,s12):
    m0=np.zeros(s0);v0=np.zeros(s0)
    m1=np.zeros(s1);v1=np.zeros(s1)
    m2=np.zeros(s2);v2=np.zeros(s2)
    m3=np.zeros(s3);v3=np.zeros(s3)
    m4=np.zeros(s4);v4=np.zeros(s4)
    m5=np.zeros(s5);v5=np.zeros(s5)
    m6=np.zeros(s6);v6=np.zeros(s6)
    m7=np.zeros(s7);v7=np.zeros(s7)
    m8=np.zeros(s8);v8=np.zeros(s8)
    m9=np.zeros(s9);v9=np.zeros(s9)
    m10=np.zeros(s10);v10=np.zeros(s10)
    m11=np.zeros(s11);v11=np.zeros(s11)
    m12=np.zeros(s12);v12=np.zeros(s12)
    return [[m0,v0],[m1,v1],[m2,v2],[m3,v3],[m4,v4],[m5,v5],[m6,v6],[m7,v7],[m8,v8],[m9,v9],[m10,v10],[m11,v11],[m12,v12]]

def accumulate_weights(dg2, dbeta2, dWd, dBd, dWu, dBu,dg1, dbeta1,dWv, dWq, dWk, dWe,dWo,n):
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
    gwo=np.sum(dWo,axis=0) /n
    return (gWe, gwk, gwq, gwv, gwu, gwd, gbu, gbd, ggamma2, gbeta2, ggamma1, gbeta1,gwo)
def posvoc(tokens,vocab):
  lind=np.array(list(map(vocab.get,tokens)))
  return lind

import re
pattern = r'[a-zA-Z\.]+|\*\*|.'

(wo,wq , wk , wv , beta2 , beta1,gamma2,gamma1,wu,bu,wd,bd,We,vocab,dataset)=loadweights()
mv = initmoments(We.shape,wk.shape,wq.shape,wo.shape,wv.shape,wu.shape,wd.shape,bu.shape,bd.shape,gamma2.shape,beta2.shape,gamma1.shape,beta1.shape)
batch_size=len(dataset)
seq_len=len(re.findall(pattern,dataset[0][0]))
step=0
P=PE(seq_len,dmodel)

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
posoutput=(posoutput + (np.arange(len(dataset))*We.shape[0])[:,np.newaxis]).astype(np.int32)
posinput=np.array(posinput)
posoutput=np.array(posoutput)
while step<200000:
    step+=1
    avgloss=0
    (gWe, gwk, gwq, gwv, gwu, gwd, gbu, gbd,ggamma2, gbeta2, ggamma1, gbeta1)=(0,0,0,0,0,0,0,0,0,0,0,0)
    ET=np.zeros((batch_size,seq_len,dmodel))
    ET=(We[(posinput).astype(np.int16)])
    ET+=P

    #attentioon ======
    (A,SMA,Q,K,V,con)=attention(ET,dk,wq,wk,wv,wo,h)

    x1=ET+A

    RA=LN(x1,gamma1,beta1)

    (M,FU,FA)=MLP(RA,wu,bu,wd,bd)

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
    (dg2, dbeta2, dWd, dBd, dWu, dBu,
        dg1, dbeta1,
        dWv, dWq, dWk, dWe,dWo)=gradientcalc(FinalResult, TR, We, x2, gamma2, dmodel, FU, wu, wd, RA,
                                      x1, gamma1, V, SMA, dk, K, Q, wv, wq, wk,wo, posoutput,ET,RM,con,batch_size,h)
        #calculating the sum of the gradients
    (gWe, gwk, gwq, gwv, gwu, gwd, gbu, gbd,
         ggamma2, gbeta2, ggamma1, gbeta1,gwo)=accumulate_weights(dg2, dbeta2, dWd, dBd, dWu, dBu,dg1, dbeta1,dWv, dWq, dWk, dWe,dWo,batch_size)
    
    


    if step <=150000:
      lr=0.0001
    else:
      lr=0.0001


    (wq , wk , wv , beta2 , beta1,
            gamma2,gamma1,wu,bu,wd,bd,We,wo)=changeweights(lr, We, wk, wq, wv, wu, wd, bu, bd, gamma2, beta2, gamma1, beta1,wo,
                                                        gWe, gwk, gwq, gwv, gwu, gwd, gbu, gbd, ggamma2, gbeta2, ggamma1, gbeta1,gwo,B1,B2,step,mv)
    
    if step >= 200000:
        print(f"Reached 200k steps ceiling. Saving final checkpoint and exiting.")
        saveweights(wq , wk , wv , beta2 , beta1, gamma2, gamma1, wu, bu, wd, bd, We,wo)
        break
    if step % 1000 ==0:
        print("learning rate =",lr,"last avgloss=",avgloss)
        print("setep=",step)
        saveweights(wq , wk , wv , beta2 , beta1,gamma2,gamma1,wu,bu,wd,bd,We,wo)
        print("saved")