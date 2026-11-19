"""Blocked uncertainty diagnostics, split-conformal sets and episode evaluation."""
import json
import geopandas as gpd
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import average_precision_score
from common import ROOT,CFG
from methods import classification_metrics,conformal_quantile,conformal_sets,conformal_metrics


def episode_counts(y,p):
 def intervals(a):
  diff=np.diff(np.r_[0,a,0].astype(int));return list(zip(np.flatnonzero(diff==1),np.flatnonzero(diff==-1)))
 actual,predicted=intervals(y),intervals(p)
 overlap=np.zeros((len(actual),len(predicted)))
 for i,(s,e) in enumerate(actual):
  for j,(ps,pe) in enumerate(predicted):overlap[i,j]=max(0,min(e,pe)-max(s,ps))
 matched=0
 if overlap.size:
  a,b=linear_sum_assignment(-overlap);matched=int((overlap[a,b]>0).sum())
 tp=matched;fn=len(actual)-matched;fp=len(predicted)-matched
 return dict(observed_episodes=len(actual),predicted_episodes=len(predicted),tp=tp,fn=fn,fp=fp,
  tn=None,precision=tp/max(tp+fp,1),recall=tp/max(tp+fn,1),f1=2*tp/max(2*tp+fp+fn,1))


def bootstrap(y,p,threshold,blocks):
 rng=np.random.default_rng(CFG['seeds'][0]);unique=np.unique(blocks);result=[]
 for _ in range(CFG['bootstrap_replicates']):
  picked=rng.choice(unique,len(unique),replace=True);ids=np.concatenate([np.where(blocks==b)[0] for b in picked])
  if not y[ids].sum():continue
  m=classification_metrics(y[ids],p[ids],threshold);result.append([m['average_precision'],m['f1'],m['brier']])
 if not result:return {}
 low,high=np.quantile(result,[.025,.975],axis=0)
 return dict(bootstrap_blocks=len(unique),valid_replicates=len(result),ap_low=low[0],ap_high=high[0],f1_low=low[1],f1_high=high[1],brier_low=low[2],brier_high=high[2])


def main():
 pred=pd.read_csv(ROOT/'metrics/predictions.csv',parse_dates=['date']);thresholds=json.loads((ROOT/'models/thresholds.json').read_text())
 if 'training_distance' not in pred:
  import joblib
  from sklearn.neighbors import NearestNeighbors
  from train import summaries
  arrays=np.load(ROOT/'data/processed/learning_arrays.npz');samples=pd.read_csv(ROOT/'data/processed/samples.csv')
  window=np.lib.stride_tricks.sliding_window_view(arrays['features'],CFG['lookback_days'],axis=0)
  selected=window[samples.anchor.to_numpy()-CFG['lookback_days']+1,samples.cell.to_numpy()]
  vectors=summaries(selected);scaler=joblib.load(ROOT/'models/logistic.joblib')[0]
  fit=np.flatnonzero(samples.split.eq('train'));fit=fit[::max(1,len(fit)//2000)]
  neighbour=NearestNeighbors(n_neighbors=1,n_jobs=2).fit(scaler.transform(vectors[fit]))
  pred['training_distance']=neighbour.kneighbors(scaler.transform(vectors[pred.sample_id]),return_distance=True)[0][:,0]
  pred.to_csv(ROOT/'metrics/predictions.csv',index=False)
 cells=gpd.read_file(ROOT/'data/processed/cells.gpkg');centres=gpd.points_from_xy(cells.lon,cells.lat)
 centres=gpd.GeoSeries(centres,crs=4326).to_crs(CFG['area_crs'])
 block_codes=(np.floor(centres.x/50000).astype(int).astype(str)+'_'+np.floor(centres.y/50000).astype(int).astype(str)).to_numpy()
 pred['spatial_block']=block_codes[pred.cell]
 arrays=np.load(ROOT/'data/processed/learning_arrays.npz')
 ratio=arrays['rain90'][pred.target,pred.cell];ref=arrays['drought_threshold'][pred.target,pred.cell]
 pred['drought_severity']=np.where(ratio<.5*ref,'severe_precipitation_deficit',np.where(ratio<ref,'precipitation_deficit','not_deficit'))
 pred['season']=np.asarray(['DJF','DJF','MAM','MAM','MAM','JJA','JJA','JJA','SON','SON','SON','DJF'])[pred.date.dt.month-1]
 calibration=pred[pred.split.eq('conformal_calibration')];test=pred[pred.split.eq('test')].copy()
 scores=np.where(calibration.label.to_numpy()==1,1-calibration.tcn_probability,calibration.tcn_probability)
 q=conformal_quantile(scores,CFG['conformal_alpha'])
 weekly=calibration.date.dt.day_of_week.eq(0).to_numpy();q_weekly=conformal_quantile(scores[weekly],CFG['conformal_alpha'])
 q_class=[conformal_quantile(scores[calibration.label.to_numpy()==c],CFG['conformal_alpha']) for c in [0,1]]
 scenarios={'pooled_90':q,'weekly_thinned_90':q_weekly,'class_conditional_90':q_class,'pooled_95':conformal_quantile(scores,.05)}
 rows=[]
 for scenario,cut in scenarios.items():
  sets=conformal_sets(test.tcn_probability,cut)
  test[scenario+'_no_event']=sets[:,0];test[scenario+'_event']=sets[:,1]
  for grouping,values in [('all',np.repeat('all',len(test))),('geography',test.geography.to_numpy()),('year',test.date.dt.year.to_numpy()),
   ('season',test.season.to_numpy()),('label',test.label.to_numpy()),('drought_severity',test.drought_severity.to_numpy())]:
   for group in np.unique(values):
    keep=values==group
    rows.append(dict(scenario=scenario,grouping=grouping,group=str(group),**conformal_metrics(test.label.to_numpy()[keep],sets[keep])))
 pd.DataFrame(rows).to_csv(ROOT/'metrics/conformal_coverage.csv',index=False)
 (ROOT/'models/conformal.json').write_text(json.dumps(dict(alpha=CFG['conformal_alpha'],n_calibration=len(calibration),calibration_events=int(calibration.label.sum()),
  weekly_n=int(weekly.sum()),q_pooled=q,q_weekly=q_weekly,q_class_conditional=q_class,
  score='1 - probability assigned to true reanalysis-derived label',guarantee='No unconditional coverage guarantee claimed: temporal/spatial dependence and domain shift violate exchangeability'),indent=2))
 metric_rows=[];episode_rows=[]
 for name,threshold in thresholds.items():
  for grouping,values in [('all',np.repeat('all',len(test))),('geography',test.geography.to_numpy()),('year',test.date.dt.year.to_numpy()),('season',test.season.to_numpy())]:
   for group in np.unique(values):
    keep=values==group;selected=test[keep]
    metric_rows.append(dict(model=name,grouping=grouping,group=str(group),**classification_metrics(selected.label.to_numpy(),selected[name].to_numpy(),threshold)))
  selected=metric_rows[-1]
  overall=classification_metrics(test.label.to_numpy(),test[name].to_numpy(),threshold)
  interval=bootstrap(test.label.to_numpy(),test[name].to_numpy(),threshold,test.spatial_block.to_numpy())
  metric_rows.append(dict(model=name,grouping='spatial_bootstrap',group='all',**overall,**interval))
  tp=fp=fn=observed=predicted=0
  for cell,part in test.groupby('cell'):
   part=part.sort_values('target');m=episode_counts(part.label.to_numpy(),(part[name].to_numpy()>=threshold).astype(int))
   episode_rows.append(dict(model=name,cell=cell,**m))
   tp+=m['tp'];fp+=m['fp'];fn+=m['fn'];observed+=m['observed_episodes'];predicted+=m['predicted_episodes']
  episode_rows.append(dict(model=name,cell='all',observed_episodes=observed,predicted_episodes=predicted,tp=tp,fp=fp,fn=fn,tn=None,
   precision=tp/max(tp+fp,1),recall=tp/max(tp+fn,1),f1=2*tp/max(2*tp+fp+fn,1)))
 pd.DataFrame(metric_rows).to_csv(ROOT/'metrics/discrimination.csv',index=False);pd.DataFrame(episode_rows).to_csv(ROOT/'metrics/episode_evaluation.csv',index=False)
 # Fully specified uncertainty diagnostic, with spatial blocks as resampling units.
 error=np.abs(test.tcn_probability.to_numpy()-test.label.to_numpy());spread=test.dropout_std.to_numpy()
 rho=float(spearmanr(spread,error).statistic);rng=np.random.default_rng(CFG['seeds'][0]);unique=test.spatial_block.unique();draws=[];conditional={0:[],1:[]}
 for _ in range(CFG['bootstrap_replicates']):
  ids=np.concatenate([np.flatnonzero(test.spatial_block.to_numpy()==b) for b in rng.choice(unique,len(unique),replace=True)])
  draws.append(spearmanr(spread[ids],error[ids]).statistic)
  for label in [0,1]:
   chosen=ids[test.label.to_numpy()[ids]==label]
   if len(chosen)>2 and np.unique(spread[chosen]).size>1:conditional[label].append(spearmanr(spread[chosen],error[chosen]).statistic)
 # Boundary proximity is an offline diagnostic and never enters the predictor sequence.
 proximity=np.empty(len(test));distance=[]
 for _,part in test.groupby('cell'):
  a=part.label.to_numpy();transitions=np.flatnonzero(np.diff(a)!=0)+1
  if not len(transitions):distance.extend([(int(i),np.nan) for i in part.index]);continue
  d=np.min(np.abs(np.arange(len(part))[:,None]-transitions[None,:]),axis=1)
  distance.extend((int(i),1/(1+float(v))) for i,v in zip(part.index,d))
 proxy=pd.Series(dict(distance)).reindex(test.index).to_numpy();keep=np.isfinite(proxy)
 info=dict(correlation='Spearman',x='MC-dropout probability standard deviation',y='absolute error |mean probability - binary rule label|',
  rho=rho,spatial_bootstrap_low=float(np.nanquantile(draws,.025)),spatial_bootstrap_high=float(np.nanquantile(draws,.975)),
  raw_pairs=len(test),spatial_blocks=len(unique),independent_sample_size='Not identified; cell-days are correlated',
  boundary_proximity_definition='1 / (1 + days to nearest label transition within cell); offline diagnostic',
  boundary_proximity_spearman=float(spearmanr(spread[keep],proxy[keep]).statistic) if keep.any() else None)
 info['std_vs_mean_probability_spearman']=float(spearmanr(spread,test.tcn_probability).statistic)
 info['mechanical_relationship_warning']='On non-event days absolute error equals predicted event probability; a high pooled spread/error correlation can be mechanical and does not establish useful error detection.'
 info['conditional_error_correlation']={}
 for label in [0,1]:
  chosen=test.label.to_numpy()==label
  info['conditional_error_correlation'][str(label)]=dict(n=int(chosen.sum()),rho=float(spearmanr(spread[chosen],error[chosen]).statistic),
   spatial_blocks=int(test.loc[chosen,'spatial_block'].nunique()),valid_resamples=len(conditional[label]),
   spatial_bootstrap_low=float(np.nanquantile(conditional[label],.025)) if conditional[label] else None,
   spatial_bootstrap_high=float(np.nanquantile(conditional[label],.975)) if conditional[label] else None)
 boundary_draws=[];mean_draws=[]
 for _ in range(CFG['bootstrap_replicates']):
  ids=np.concatenate([np.flatnonzero(test.spatial_block.to_numpy()==b) for b in rng.choice(unique,len(unique),replace=True)])
  mean_draws.append(spearmanr(spread[ids],test.tcn_probability.to_numpy()[ids]).statistic)
  selected=ids[np.isfinite(proxy[ids])]
  if len(selected)>2:boundary_draws.append(spearmanr(spread[selected],proxy[selected]).statistic)
 info['std_vs_mean_probability_interval']=[float(v) for v in np.nanquantile(mean_draws,[.025,.975])]
 info['boundary_proximity_interval']=[float(v) for v in np.nanquantile(boundary_draws,[.025,.975])] if boundary_draws else None
 info['boundary_pairs']=int(keep.sum());info['boundary_spatial_blocks']=int(test.loc[keep,'spatial_block'].nunique())
 (ROOT/'metrics/uncertainty_correlation.json').write_text(json.dumps(info,indent=2))
 test['absolute_error']=error;test['boundary_proximity']=proxy
 test.to_csv(ROOT/'metrics/test_predictions_with_sets.csv',index=False)
 # Same trained predictions, alternate drought threshold: definition sensitivity only.
 alt=classification_metrics(test.alternate_label.to_numpy(),test.tcn_probability.to_numpy(),thresholds['tcn_probability'])
 (ROOT/'metrics/alternate_definition.json').write_text(json.dumps(dict(**alt,interpretation='Fixed primary predictions and primary validation-selected threshold; no refit or alternate-label calibration',
  label_agreement=float((test.label==test.alternate_label).mean())),indent=2))
 print(pd.DataFrame(metric_rows).query("grouping == 'all'")[['model','n','events','prevalence','average_precision','f1','brier']].to_string(index=False),flush=True)
 print(pd.DataFrame(rows).query("grouping == 'all'").to_string(index=False),flush=True)


if __name__=='__main__':main()
