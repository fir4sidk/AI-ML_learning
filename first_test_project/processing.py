def Trueres(tokensout,vocab,n):
    TR=np.zeros((n,len(vocab)))
    lind=list(map(vocab.get,tokensout))
    TR[list(range(len(lind))),lind]=1
    return TR

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


def readres(FR,vocab):
    import numpy
    pos = np.argmax(FinalResult, axis=1).get()
    o=list(vocab.keys())
    vo=numpy.asarray(o)
    v=vo[pos]
    print(v)

def masking1(m):
  s=m.shape
  c=m.copy()
  ind=np.triu_indices(s[1], k=1)
  c[:,ind[0],ind[1]]=-9999
  return c

def attention1(E,dk,Wqt,Wkt,Wvt,Wo,h):
  K= E[None,:,:] @ Wkt
  Q= E[None,:,:] @ Wqt
  V= E[None,:,:] @ Wvt
  KT=np.transpose(K,(0,2,1))
  M=masking1(Q @ KT)
  SMA=softmax(M / np.sqrt(dk))
  O=SMA @ V
  con=np.reshape(O.transpose(1,0,2),(O.shape[1],dk*h))
  A=con @ Wo
  return (A,SMA,Q,K,V,con)

def posvoc(tokens,vocab):
  lind=np.array(list(map(vocab.get,tokens)))
  return lind


import cupy as np
import re
pattern = r'[a-zA-Z\.]+|\*\*|.'
inp="12+14=26"
output="2+14=26EOS"
seq_len=len(re.findall(pattern,inp))
step=0
P=PE(seq_len,dmodel)
tokens=re.findall(pattern,inp)
tokensout=re.findall(pattern,output)
posinput=posvoc(tokens,vocab)
posoutput=posvoc(tokensout,vocab)
TR=Trueres(tokensout,vocab,seq_len)

avgloss=0
(gWe, gwk, gwq, gwv, gwu, gwd, gbu, gbd,ggamma2, gbeta2, ggamma1, gbeta1)=(0,0,0,0,0,0,0,0,0,0,0,0)
ET=np.zeros((batch_size,seq_len,dmodel))
ET=(We[(posinput).astype(np.int16)])
ET+=P
(A,SMA,Q,K,V,con)=attention1(ET,dk,wq,wk,wv,wo,h)

x1=ET+A

RA=LN(x1,gamma1,beta1)

(M,FU,FA)=MLP(RA,wu,bu,wd,bd)

x2=RA+M

RM=LN(x2,gamma2,beta2)

z=RM @ We.T

FinalResult=softmax(z)

readres(FinalResult,vocab)