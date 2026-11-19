"""Antecedent-only TCN forecasts; isolated validation, probability and set calibration."""
import json,time,platform
import joblib
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader,TensorDataset
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from common import ROOT,CFG,utc_now
from methods import TCN,seed_all,dropout_only,chosen_threshold,classification_metrics


def logits(model,X,device,batch=512):
 result=[]
 with torch.no_grad():
  for begin in range(0,len(X),batch):result.append(model(X[begin:begin+batch].to(device)).cpu().numpy())
 return np.concatenate(result)


def mc_probabilities(model,X,device,platt,seed):
 seed_all(seed);dropout_only(model);values=[]
 for _ in range(CFG['mc_passes']):
  z=logits(model,X,device);values.append(platt.predict_proba(z[:,None])[:,1])
 model.eval();values=np.asarray(values,dtype='float32')
 return values.mean(axis=0),values.std(axis=0),values


def summaries(X):
 return np.concatenate([X[:,:,-1],X[:,:,-7:].mean(axis=2),X[:,:,-28:].mean(axis=2),X.mean(axis=2),X.std(axis=2)],axis=1)


def main():
 begin=time.perf_counter();data=np.load(ROOT/'data/processed/learning_arrays.npz');samples=pd.read_csv(ROOT/'data/processed/samples.csv',parse_dates=['date'])
 window=np.lib.stride_tricks.sliding_window_view(data['features'],CFG['lookback_days'],axis=0)
 X=window[samples.anchor.to_numpy()-CFG['lookback_days']+1,samples.cell.to_numpy()].copy()
 assert X.shape[1:]==(5,CFG['lookback_days']) and np.isfinite(X).all()
 y=samples.label.to_numpy(dtype='float32');parts={s:np.where(samples.split==s)[0] for s in CFG['periods']}
 train,valid,pc,cc,test=[parts[s] for s in ['train','validation','probability_calibration','conformal_calibration','test']]
 for name,ids in parts.items():
  if y[ids].sum()<1 or y[ids].sum()==len(ids):raise RuntimeError(f'{name} has no two-class support; do not fabricate scores')
 mean=X[train].mean(axis=(0,2));std=X[train].std(axis=(0,2));std=np.maximum(std,1e-6)
 Xn=(X-mean[None,:,None])/std[None,:,None];Xt=torch.from_numpy(Xn.astype('float32'))
 np.savez(ROOT/'models/training_scaler.npz',mean=mean,std=std)
 device=torch.device('cuda' if torch.cuda.is_available() else 'cpu');logs=[];seed_metrics=[]
 positive_weight=float((len(train)-y[train].sum())/y[train].sum())
 for seed in CFG['seeds']:
  seed_all(seed);model=TCN(channels=CFG['tcn_channels'],dilations=CFG['tcn_dilations'],dropout=CFG['dropout']).to(device)
  optimizer=torch.optim.Adam(model.parameters(),lr=CFG['learning_rate'])
  loss_fn=nn.BCEWithLogitsLoss(pos_weight=torch.tensor(positive_weight,device=device))
  loader=DataLoader(TensorDataset(Xt[train],torch.from_numpy(y[train])),batch_size=CFG['batch_size'],shuffle=True,num_workers=0,
   generator=torch.Generator().manual_seed(seed))
  best=-np.inf;stale=0
  for epoch in range(CFG['max_epochs']):
   model.train();loss_total=0
   for batch,target in loader:
    optimizer.zero_grad();output=model(batch.to(device));loss=loss_fn(output,target.to(device));loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(),1.0);optimizer.step();loss_total+=float(loss.item())*len(batch)
   model.eval();z=logits(model,Xt[valid],device);p=1/(1+np.exp(-np.clip(z,-40,40)));ap=float(average_precision_score(y[valid],p))
   logs.append(dict(seed=seed,epoch=epoch,training_weighted_bce=loss_total/len(train),validation_average_precision=ap))
   print('TCN',seed,'epoch',epoch,'loss',round(logs[-1]['training_weighted_bce'],4),'validation AP',round(ap,4),flush=True)
   if ap>best+1e-5:
    best=ap;stale=0;torch.save(model.state_dict(),ROOT/f'models/tcn_{seed}.pt')
   else:stale+=1
   if stale>=CFG['patience']:break
  model.load_state_dict(torch.load(ROOT/f'models/tcn_{seed}.pt',map_location=device,weights_only=True));model.eval()
  zpc=logits(model,Xt[pc],device)
  platt=LogisticRegression(C=1.,max_iter=1000).fit(zpc[:,None],y[pc])
  joblib.dump(platt,ROOT/f'models/platt_{seed}.joblib')
  ppc=platt.predict_proba(zpc[:,None])[:,1];threshold=chosen_threshold(y[pc],ppc)
  ptest=platt.predict_proba(logits(model,Xt[test],device)[:,None])[:,1]
  seed_metrics.append(dict(model='TCN deterministic calibrated',seed=seed,validation_best_ap=best,**classification_metrics(y[test],ptest,threshold)))
 pd.DataFrame(logs).to_csv(ROOT/'metrics/training_log.csv',index=False)
 pd.DataFrame(seed_metrics).to_csv(ROOT/'metrics/seed_variation.csv',index=False)
 primary=CFG['seeds'][0];model=TCN(channels=CFG['tcn_channels'],dilations=CFG['tcn_dilations'],dropout=CFG['dropout']).to(device)
 model.load_state_dict(torch.load(ROOT/f'models/tcn_{primary}.pt',map_location=device,weights_only=True));model.eval()
 platt=joblib.load(ROOT/f'models/platt_{primary}.joblib')
 ppc,_,_=mc_probabilities(model,Xt[pc],device,platt,primary+100)
 threshold=chosen_threshold(y[pc],ppc)
 out=samples.loc[np.r_[cc,test]].copy();ids=out.index.to_numpy().copy();out['sample_id']=ids
 p,spread,passes=mc_probabilities(model,Xt[ids],device,platt,primary+200)
 out['tcn_probability']=p;out['dropout_std']=spread
 entropy=lambda p:-(p*np.log(np.clip(p,1e-9,1))+(1-p)*np.log(np.clip(1-p,1e-9,1)))
 out['predictive_entropy']=entropy(p);out['dropout_mutual_information']=np.maximum(entropy(p)-entropy(passes).mean(axis=0),0)
 # Exactly the same antecedent window is available to the non-neural baseline.
 summary=summaries(X)
 lr=make_pipeline(StandardScaler(),LogisticRegression(max_iter=2000,C=1,class_weight='balanced',random_state=primary))
 lr.fit(summary[train],y[train]);joblib.dump(lr,ROOT/'models/logistic.joblib')
 lpc=lr.decision_function(summary[pc]);cal=LogisticRegression(C=1,max_iter=1000).fit(lpc[:,None],y[pc])
 lp=cal.predict_proba(lr.decision_function(summary[ids])[:,None])[:,1]
 out['logistic_probability']=lp
 lr_threshold=chosen_threshold(y[pc],cal.predict_proba(lpc[:,None])[:,1])
 out['persistence_probability']=data['label'][out.anchor,out.cell]
 dates=pd.DatetimeIndex(samples.date);doy=dates.day_of_year.to_numpy();climatology=[]
 for d in range(1,367):
  delta=np.abs(doy[train]-d);keep=np.minimum(delta,366-delta)<=15
  climatology.append((y[train][keep].sum()+1)/(keep.sum()+2))
 out['climatology_probability']=np.asarray(climatology)[doy[ids]-1]
 climate_threshold=chosen_threshold(y[pc],np.asarray(climatology)[doy[pc]-1])
 out.to_csv(ROOT/'metrics/predictions.csv',index=False)
 thresholds=dict(tcn_probability=threshold,logistic_probability=lr_threshold,persistence_probability=.5,climatology_probability=climate_threshold)
 (ROOT/'models/thresholds.json').write_text(json.dumps(thresholds,indent=2))
 record=dict(recorded_utc=utc_now(),elapsed_seconds=time.perf_counter()-begin,device=str(device),hardware=platform.processor(),
  seeds=CFG['seeds'],primary_seed_policy='First predeclared seed, not best test seed',class_positive_weight=positive_weight,
  input_channels=['Tmax minus max(local calendar p90,30C)','log1p daily rain','trailing90d rain / (local p20+1)','sin season','cos season'],
  architecture=dict(channels=CFG['tcn_channels'],dilations=CFG['tcn_dilations'],kernel_size=3,convolutions_per_block=2,dropout=CFG['dropout'],
  receptive_field=1+4*sum(CFG['tcn_dilations']),lookback_days=CFG['lookback_days'],batch_norm=False),
  task='7-day-ahead cell-day compound-condition forecast; antecedent observations only; retrospective reanalysis availability, not operational lead-time validation',
  probability_calibration='Platt scaling and F1 threshold on 2017-2018 development cells; conformal 2019-2020 untouched',mc_passes=CFG['mc_passes'])
 (ROOT/'models/training_manifest.json').write_text(json.dumps(record,indent=2))
 print(pd.DataFrame(seed_metrics).to_string(index=False),flush=True)


if __name__=='__main__':main()
