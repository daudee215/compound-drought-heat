"""Testable temporal network, metrics and finite-sample conformal primitives."""
import math,random
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from sklearn.metrics import average_precision_score,f1_score,precision_score,recall_score,brier_score_loss,confusion_matrix,precision_recall_curve


def seed_all(seed):
 random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
 if torch.cuda.is_available():torch.cuda.manual_seed_all(seed)
 torch.use_deterministic_algorithms(True)
 torch.set_num_threads(2)


class CausalConv(nn.Module):
 def __init__(self,inside,outside,dilation,kernel=3):
  super().__init__();self.left=(kernel-1)*dilation;self.conv=nn.Conv1d(inside,outside,kernel,dilation=dilation)
 def forward(self,x):return self.conv(F.pad(x,(self.left,0)))


class Block(nn.Module):
 def __init__(self,inside,outside,dilation,dropout):
  super().__init__();self.main=nn.Sequential(CausalConv(inside,outside,dilation),nn.ReLU(),nn.Dropout(dropout),
   CausalConv(outside,outside,dilation),nn.ReLU(),nn.Dropout(dropout))
  self.short=nn.Conv1d(inside,outside,1) if inside!=outside else nn.Identity()
 def forward(self,x):return F.relu(self.main(x)+self.short(x))


class TCN(nn.Module):
 def __init__(self,features=5,channels=16,dilations=(1,2,4,8),dropout=.2):
  super().__init__();layers=[]
  for d in dilations:layers.append(Block(features,channels,d,dropout));features=channels
  self.network=nn.Sequential(*layers);self.head=nn.Linear(channels,1)
 def forward(self,x):return self.head(self.network(x)[:,:,-1]).squeeze(-1)


def dropout_only(model):
 model.eval()
 for module in model.modules():
  if isinstance(module,nn.Dropout):module.train()
 return model


def chosen_threshold(y,p):
 precision,recall,thresholds=precision_recall_curve(y,p)
 f=2*precision[:-1]*recall[:-1]/np.maximum(precision[:-1]+recall[:-1],1e-12)
 return float(thresholds[int(np.argmax(f))]) if len(thresholds) and np.any(y) else .5


def classification_metrics(y,p,threshold):
 y=np.asarray(y,dtype=int);p=np.clip(p,0,1);prediction=(p>=threshold).astype(int)
 tn,fp,fn,tp=confusion_matrix(y,prediction,labels=[0,1]).ravel()
 return dict(n=len(y),events=int(y.sum()),prevalence=float(y.mean()),average_precision=float(average_precision_score(y,p)) if y.sum() else None,
  f1=float(f1_score(y,prediction,zero_division=0)),precision=float(precision_score(y,prediction,zero_division=0)),
  recall=float(recall_score(y,prediction,zero_division=0)),brier=float(brier_score_loss(y,p)),tn=int(tn),fp=int(fp),fn=int(fn),tp=int(tp),threshold=threshold)


def conformal_quantile(scores,alpha=.1):
 scores=np.asarray(scores,dtype=float);n=len(scores)
 if n==0:raise ValueError('Empty conformal calibration set')
 k=math.ceil((n+1)*(1-alpha))
 return float(np.partition(scores,k-1)[k-1]) if k<=n else float('inf')


def conformal_sets(probability,q):
 p=np.asarray(probability);q0,q1=(q,q) if np.isscalar(q) else q
 return np.stack([p<=q0,1-p<=q1],axis=-1)


def conformal_metrics(y,sets):
 y=np.asarray(y,dtype=int);size=sets.sum(axis=1);covered=sets[np.arange(len(y)),y]
 return dict(n=len(y),events=int(y.sum()),coverage=float(covered.mean()),mean_set_size=float(size.mean()),
  ambiguity_rate=float((size==2).mean()),empty_rate=float((size==0).mean()),singleton_rate=float((size==1).mean()))
