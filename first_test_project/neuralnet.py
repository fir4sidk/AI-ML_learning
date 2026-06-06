import cupy as np
import csv


dmodel=128
h=4
dk=dmodel//h
dv=dmodel//h
dff=dmodel*4
L=2
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
    dg=np.sum(G,axis=1,keepdims=True)
    return dg


def dbeta(dPR):
    db=np.sum(dPR,axis=1,keepdims=True)
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
    np.save("gamma2.npy", gamma2)
    np.save("beta2.npy", beta2)
    np.save("gamma1.npy", gamma1)
    np.save("beta1.npy", beta1)
    np.save("wo",wo)
def loadweights():
    wo=np.load("wo.npy")
    wq=np.load("wq.npy")
    wk=np.load("wk.npy")
    wv=np.load("wv.npy")
    beta2=np.load("beta2.npy")
    gamma2=np.load("gamma2.npy")
    beta1=np.load("beta1.npy")
    gamma1=np.load("gamma1.npy")
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
    x1, gamma1, V, SMA, dk, K, Q, wv, wq, wk,Wo, postokens,E,RM,con,batch_size,h,seq_len,dff,L):
    dz = FinalResult - TR
    #matrices init
    dRM=np.zeros((L,batch_size,seq_len,dmodel))
    dg2=np.zeros((L,batch_size,1,dmodel))
    dbeta2=np.zeros((L,batch_size,1,dmodel))
    dX2=np.zeros((L,batch_size,seq_len,dmodel))
    dWd=np.zeros((L,batch_size,dff,dmodel))
    dBd=np.zeros((L,batch_size,1,dmodel))
    dRA=np.zeros((L,batch_size,seq_len,dmodel))
    dAF=np.zeros((L,batch_size,seq_len,dff))
    dFU=np.zeros((L,batch_size,seq_len,dff))
    dWu=np.zeros((L,batch_size,dmodel,dff))
    dBu=np.zeros((L,batch_size,1,dff))
    dg1=np.zeros((L,batch_size,1,dmodel))
    dbeta1=np.zeros((L,batch_size,1,dmodel))
    dX1=np.zeros((L,batch_size,seq_len,dmodel))
    dA=np.zeros((L,batch_size,seq_len,dmodel))
    dWo=np.zeros((L,batch_size,dmodel,dmodel))
    dcon=np.zeros((L,batch_size,seq_len,dmodel))
    do=np.zeros((L,batch_size,h,seq_len,dmodel//h))
    dSMA=np.zeros((L,batch_size,h,seq_len,seq_len))
    dM=np.zeros((L,batch_size,h,seq_len,seq_len))
    dK=np.zeros((L,batch_size,h,seq_len,dmodel//h))
    dQ=np.zeros((L,batch_size,h,seq_len,dmodel//h))
    dV=np.zeros((L,batch_size,h,seq_len,dmodel//h))
    dWv=np.zeros((L,batch_size,h,dmodel,dmodel//h))
    dWq=np.zeros((L,batch_size,h,dmodel,dmodel//h))
    dWk=np.zeros((L,batch_size,h,dmodel,dmodel//h))
    dEq=np.zeros((L,batch_size,h,seq_len,dmodel))
    dEk=np.zeros((L,batch_size,h,seq_len,dmodel))
    dEv=np.zeros((L,batch_size,h,seq_len,dmodel))
    dE=np.zeros((L,batch_size,seq_len,dmodel))
    #=======================================
    dRM[L-1] = dz @ We
    for i in range (L-1,-1,-1):
    #hidden layer start

      dg2[i] = dgamma(hat(x2[i]), dRM[i])


      dbeta2[i] = dbeta(dRM[i])


      dX2[i] = dBLN(dRM[i], x2[i], dmodel, gamma2[i])


      dWd[i] = np.transpose(np.maximum(0, FU[i]),axes=(0,2,1)) @ dX2[i]


      dBd[i] = np.sum(dX2[i], axis=1, keepdims=True)


      dRA[i] = dX2[i] + (((dX2[i] @ np.transpose(wd[i])) * (FU[i] > 0)) @ np.transpose(wu[i]))


      dAF[i] = dX2[i] @ np.transpose(wd[i])


      dFU[i] = dAF[i] * (FU[i] > 0)


      dWu[i] = np.transpose(RA[i], axes=(0, 2, 1)) @ dFU[i]


      dBu[i] = np.sum(dFU[i], axis=1, keepdims=True)


      dg1[i] = dgamma(hat(x1[i]), dRA[i])


      dbeta1[i] = dbeta(dRA[i])


      dX1[i] = dBLN(dRA[i], x1[i], dmodel, gamma1[i])


      dA[i] = dX1[i]

      dWo[i]=con[i].transpose(0,2,1) @ dA[i]
      dcon[i]=dA[i] @ Wo[i].T
      do[i]=np.array(np.split(dcon[i],h,axis=-1)).transpose(1,0,2,3)
      dV[i]=SMA[i].transpose(0,1,3,2) @ do[i]
      dSMA[i]=do[i] @ V[i].transpose(0,1,3,2)
      dM[i]=(1/np.sqrt(dk)) * (dsoftmax(SMA[i],dSMA[i]))
      dQ[i]=dM[i] @ K[i]
      dK[i]= dM[i].transpose(0,1,3,2) @ Q[i]
      
      dWv[i] = np.transpose(E[i], axes=(0, 2, 1))[:,None,:,:] @ dV[i]


      dWq[i] = np.transpose(E[i], axes=(0, 2, 1))[:,None,:,:] @ dQ[i]


      dWk[i] = np.transpose(E[i], axes=(0, 2, 1))[:,None,:,:] @ dK[i]

      dEk[i]=dK[i] @ wk[i].transpose(0,2,1)
      dEq[i]=dQ[i] @ wq[i].transpose(0,2,1)
      dEv[i]=dV[i] @ wv[i].transpose(0,2,1)
      dE[i]=np.sum(dEk[i]+dEq[i]+dEv[i],axis=1)+dX1[i]
      if i > 0:
          dRM[i-1]=dE[i]
    #hidden layer end
    dWep = np.transpose(dz, axes=(0, 2, 1)) @ RM[L-1]

    
    dWee = dembed(postokens, We, dE[0],batch_size)

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
    gwk = np.sum(dWk,axis=1) / n
    gwq = np.sum(dWq,axis=1) / n
    gwv = np.sum(dWv,axis=1) / n
    gwu = np.sum(dWu,axis=1) / n
    gwd = np.sum(dWd,axis=1) / n
    gbu = np.sum(dBu,axis=1) / n
    gbd = np.sum(dBd,axis=1) / n
    ggamma2 = np.sum(dg2,axis=1) / n
    gbeta2 = np.sum(dbeta2,axis=1) / n
    ggamma1 = np.sum(dg1,axis=1) / n
    gbeta1 = np.sum(dbeta1,axis=1) / n
    gwo=np.sum(dWo,axis=1) /n
    return (gWe, gwk, gwq, gwv, gwu, gwd, gbu, gbd, ggamma2, gbeta2, ggamma1, gbeta1,gwo)
def posvoc(tokens,vocab):
  lind=np.array(list(map(vocab.get,tokens)))
  return lind

import re
pattern = r'\d'

(wo,wq , wk , wv , beta2 , beta1,gamma2,gamma1,wu,bu,wd,bd,We,vocab,dataset)=loadweights()
mv = initmoments(We.shape,wk.shape,wq.shape,wo.shape,wv.shape,wu.shape,wd.shape,bu.shape,bd.shape,gamma2.shape,beta2.shape,gamma1.shape,beta1.shape)
batch_size=len(dataset)
seq_len=len(re.findall(pattern,dataset[0][0]))
print(re.findall(pattern, dataset[0][0]))
print(re.findall(pattern, dataset[0][1]))
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
while step<100000:
    step+=1
    avgloss=0
    #matrices initialization for multi layers
    A=np.zeros((L,batch_size,seq_len,dmodel))
    con=np.zeros((L,batch_size,seq_len,dmodel))
    SMA=np.zeros((L,batch_size,h,seq_len,seq_len))
    Q=np.zeros((L,batch_size,h,seq_len,dmodel//h))
    K=np.zeros((L,batch_size,h,seq_len,dmodel//h))
    V=np.zeros((L,batch_size,h,seq_len,dmodel//h))
    x1=np.zeros((L,batch_size,seq_len,dmodel))
    RA=np.zeros((L,batch_size,seq_len,dmodel))
    M=np.zeros((L,batch_size,seq_len,dmodel))
    FU=np.zeros((L,batch_size,seq_len,dff))
    FA=np.zeros((L,batch_size,seq_len,dff))
    x2=np.zeros((L,batch_size,seq_len,dmodel))
    RM=np.zeros((L,batch_size,seq_len,dmodel))
    #embeding
    ET=np.zeros((L+1,batch_size,seq_len,dmodel))
    ET[0]=(We[(posinput).astype(np.int16)])
    ET[0]+=P
    #============================================

    for i in range (L):
      #attentioon ======
        (A[i],SMA[i],Q[i],K[i],V[i],con[i])=attention(ET[i],dk,wq[i],wk[i],wv[i],wo[i],h)
        
        x1[i]=ET[i]+A[i]
        
        RA[i]=LN(x1[i],gamma1[i],beta1[i])
        
        (M[i],FU[i],FA[i])=MLP(RA[i],wu[i],bu[i],wd[i],bd[i])
        
        x2[i]=RA[i]+M[i]
        
        RM[i]=LN(x2[i],gamma2[i],beta2[i])
        ET[i+1]=RM[i]
    
    
    z=RM[L-1] @ We.T

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
                                      x1, gamma1, V, SMA, dk, K, Q, wv, wq, wk,wo, posoutput,ET,RM,con,batch_size,h,seq_len,dff,L)
    
        #calculating the sum of the gradients
    (gWe, gwk, gwq, gwv, gwu, gwd, gbu, gbd,
         ggamma2, gbeta2, ggamma1, gbeta1,gwo)=accumulate_weights(dg2, dbeta2, dWd, dBd, dWu, dBu,dg1, dbeta1,dWv, dWq, dWk, dWe,dWo,batch_size)
    


    if step <= 500:
        lr = 0.0001 * (step / 500)  # warmup
    elif step <= 30000:
        lr = 0.0001
    elif step <= 100000:
        lr = 0.00003
    else:
        lr = 0.00001


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