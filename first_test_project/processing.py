dmodel=128
h=4
dk=dmodel//h
dv=dmodel//h
L=2
def PE(seq_len, dmodel):
    i = np.arange(seq_len)[:, np.newaxis]  # Shape (seq_len, 1)
    j = np.arange(dmodel)[np.newaxis, :]   # Shape (1, dmodel)


    angle = i / (10000 ** (2 * j / dmodel))

    p = np.zeros((seq_len, dmodel))
    p[:, 0::2] = np.sin(angle[:, 0::2])
    p[:, 1::2] = np.cos(angle[:, 1::2])
    return p

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
def importvocab(path):
    f=open(path,"r")
    r=csv.reader(f,delimiter=";")
    v=list(r)[0]
    M = {x:int(i) for i,x in enumerate(v)}
    f.close()
    return M

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


def readres(FR,vocab):
    import numpy
    pos = np.argmax(FR, axis=1).get()
    o=list(vocab.keys())
    vo=numpy.asarray(o)
    v=vo[pos]
    print(v)
def masking(m):
  s=m.shape
  c=m.copy()
  ind=np.triu_indices(s[2], k=1)
  c[:,:,ind[0],ind[1]]=-9999
  return c

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

def posvoc(tokens,vocab):
  lind=np.array(list(map(vocab.get,tokens)))
  return lind


import cupy as np
import re
import csv

(wo,wq , wk , wv , beta2 , beta1,gamma2,gamma1,wu,bu,wd,bd,We,vocab,dataset)=loadweights()
batch_size=1
pattern = r'\d'
inp="1234"
print(f"input = {inp}")
seq_len=len(re.findall(pattern,inp))
step=0
P=PE(seq_len,dmodel)
tokens=re.findall(pattern,inp)
posinput=posvoc(tokens,vocab)

avgloss=0
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
ET[0]=(We[(posinput).astype(np.int16)])[None]
ET[0]+=P
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

FinalResult=softmax(z).reshape(seq_len,len(vocab.keys()))
readres(FinalResult,vocab)