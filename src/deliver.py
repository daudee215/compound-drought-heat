"""Exact native-grid GIS products, 2022 case-study figures and measured reporting."""
import hashlib,importlib.metadata,json,platform,sys
import xml.etree.ElementTree as ET
import geopandas as gpd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from figure_style import COLORS, INK, SEQUENTIAL, UNCERTAINTY
import numpy as np
import pandas as pd
import rasterio
from rasterio.transform import from_origin
from rasterio.shutil import copy as rio_copy
from rio_cogeo.cogeo import cog_validate
from scipy import ndimage
from sklearn.metrics import precision_recall_curve
from sklearn.calibration import calibration_curve
from common import ROOT,CFG,utc_now

NOTICE='Independent reconstruction · © EuroGeographics for the administrative boundaries'


def table(a):
 def value(v):
  if pd.isna(v):return '—'
  if isinstance(v,(float,np.floating)):return f'{v:.4f}'
  return str(v).replace('|','/').replace('\n',' ')
 return '| '+' | '.join(a.columns)+' |\n| '+' | '.join(['---']*len(a.columns))+' |\n'+'\n'.join('| '+' | '.join(value(v) for v in row)+' |' for row in a.itertuples(index=False,name=None))


def raster_profile(cells):
 step=CFG['grid_degrees'];west=cells.lon.min()-step/2;north=cells.lat.max()+step/2
 width=round((cells.lon.max()-cells.lon.min())/step)+1;height=round((cells.lat.max()-cells.lat.min())/step)+1
 return dict(crs='EPSG:4326',transform=from_origin(west,north,step,step),width=width,height=height)


def image_values(values,cells,profile):
 array=np.full((profile['height'],profile['width']),np.nan,dtype='float32')
 for i,row in enumerate(cells.itertuples()):
  r,c=rasterio.transform.rowcol(profile['transform'],row.lon,row.lat);array[r,c]=values[i]
 return array


def cog(name,values,cells,profile,units,timestamp):
 path=ROOT/'maps'/name;tmp=path.with_suffix('.working.tif');array=image_values(values,cells,profile)
 with rasterio.open(tmp,'w',driver='GTiff',count=1,dtype='float32',nodata=-9999,tiled=True,blockxsize=256,blockysize=256,compress='deflate',**profile) as dst:
  dst.write(np.where(np.isfinite(array),array,-9999),1)
  dst.update_tags(units=units,valid_time=timestamp,provenance='new reconstruction; native ERA5 0.25 degree pilot',
   model='7-day antecedent-only TCN; first predeclared seed; 30 MC passes',warning='Not full Mediterranean coverage; use clipped land weights for area; cell support includes boundary/ocean fractions')
 rio_copy(tmp,path,driver='COG',BLOCKSIZE=256,COMPRESS='DEFLATE',OVERVIEWS='AUTO');tmp.unlink()
 return path


def maps(cells,data,pred):
 profile=raster_profile(cells);index=pd.DatetimeIndex(data['time']);weights=cells.land_area_km2.to_numpy();case=index.year==2022
 daily=pd.DataFrame(dict(date=index[case],compound_land_km2=data['label'][case]@weights,
  precipitation_deficit_land_km2=(data['rain90'][case]<data['drought_threshold'][case])@weights,
  hot_land_km2=(data['tmax'][case]>data['heat_threshold'][case])@weights,
  supported_land_denominator_km2=weights.sum()))
 daily.to_csv(ROOT/'metrics/case2022_daily_extent.csv',index=False)
 excess=np.maximum(data['tmax'][case]-data['heat_threshold'][case],0)
 intensity=(excess*data['label'][case])@weights
 daily['compound_area_weighted_heat_excess_c']=np.divide(intensity,daily.compound_land_km2.to_numpy(),out=np.full(len(daily),np.nan),where=daily.compound_land_km2.to_numpy()>0)
 daily.to_csv(ROOT/'metrics/case2022_daily_extent.csv',index=False)
 monthly=daily.set_index('date').resample('MS').agg(mean_compound_km2=('compound_land_km2','mean'),max_compound_km2=('compound_land_km2','max'),
  mean_precipitation_deficit_km2=('precipitation_deficit_land_km2','mean'),mean_hot_km2=('hot_land_km2','mean'))
 monthly['supported_land_denominator_km2']=weights.sum();monthly.to_csv(ROOT/'metrics/case2022_monthly_extent.csv')
 for timestamp in pd.date_range('2022-01-31','2022-12-31',freq='ME'):
  ti=index.get_loc(timestamp);date=timestamp.date().isoformat();rows=pred[pred.date.eq(timestamp)].set_index('cell')
  values={'compound_label':(data['label'][ti],'binary rule label, 0/1'),
   'probability':(rows.tcn_probability.reindex(range(len(cells))).to_numpy(),'mean calibrated MC forecast probability; not physical drought risk'),
   'dropout_std':(rows.dropout_std.reindex(range(len(cells))).to_numpy(),'MC probability standard deviation; approximate model spread'),
   'conformal_set_size':((rows.pooled_90_no_event.astype(int)+rows.pooled_90_event.astype(int)).reindex(range(len(cells))).to_numpy(),'number of included classes, 0/1/2'),
   'conformal_ambiguity':((rows.pooled_90_no_event&rows.pooled_90_event).astype(float).reindex(range(len(cells))).to_numpy(),'both-class inclusion indicator; empty sets also require caution'),
   'conformal_empty':((~rows.pooled_90_no_event&~rows.pooled_90_event).astype(float).reindex(range(len(cells))).to_numpy(),'empty prediction-set indicator; no class included'),
   'observation_coverage':(np.ones(len(cells)),'1=finite daily ERA5 source at selected native cell; excluded cells nodata'),
   'forecast_coverage':(np.isin(np.arange(len(cells)),rows.index).astype(float),'1=test-eligible forecast; 0=excluded geographic buffer'),
   'modis_available':(np.zeros(len(cells)),'0=MODIS granules not acquired; NOT a measured cloud fraction')}
  for variable,(v,units) in values.items():cog(f'{date}_{variable}.tif',v,cells,profile,units,date)
 peak=pd.Timestamp(daily.loc[daily.compound_land_km2.idxmax(),'date']);ti=index.get_loc(peak);rows=pred[pred.date.eq(peak)].set_index('cell')
 for name,v,units in [('compound_label',data['label'][ti],'binary compound-rule label'),
  ('probability',rows.tcn_probability.reindex(range(len(cells))).to_numpy(),'mean forecast probability'),
  ('dropout_std',rows.dropout_std.reindex(range(len(cells))).to_numpy(),'MC probability standard deviation'),
  ('heat_excess_c',np.maximum(data['tmax'][ti]-data['heat_threshold'][ti],0),'degree Celsius above declared local heat threshold'),
  ('precipitation_deficit_fraction',np.where(data['drought_threshold'][ti]>0,np.maximum(1-data['rain90'][ti]/np.maximum(data['drought_threshold'][ti],1e-9),0),np.nan),'relative deficit below local seasonal p20; undefined if reference threshold is zero')]:
  cog('case2022_peak_'+name+'.tif',v,cells,profile,units,str(peak.date()))
 longest=np.zeros(len(cells));current=np.zeros(len(cells))
 for row in data['label'][case]:current=np.where(row>0,current+1,0);longest=np.maximum(longest,current)
 cog('case2022_duration_days.tif',longest,cells,profile,'longest consecutive compound-rule run within 2022, days','2022-01-01/2022-12-31')
 # Space-time connected support is a dependence diagnostic, not a verified event inventory.
 shape=(int(case.sum()),profile['height'],profile['width']);cube=np.zeros(shape,dtype=bool)
 for t,values in enumerate(data['label'][case]):cube[t]=np.nan_to_num(image_values(values,cells,profile),nan=0)>0
 _,clusters=ndimage.label(cube,ndimage.generate_binary_structure(3,1))
 episode_info=dict(year=2022,supported_land_km2=float(weights.sum()),peak_date=str(peak.date()),peak_rule_extent_km2=float(daily.compound_land_km2.max()),
  rule_positive_cell_days=int(data['label'][case].sum()),space_time_connected_components=int(clusters),
  component_definition='6-neighbour native-grid space-time connectivity, including all acquired cells; not guaranteed independent episodes',
  month_end_map_policy='Exact last day of every month, not monthly averages. Additional peak map selected by labelled extent, not model scores.')
 (ROOT/'metrics/case2022.json').write_text(json.dumps(episode_info,indent=2))
 return profile,daily,episode_info


def plot_map(values,cells,profile,pilot,title,label,path,cmap=SEQUENTIAL,vmin=None,vmax=None):
 array=image_values(values,cells,profile);bounds=rasterio.transform.array_bounds(profile['height'],profile['width'],profile['transform'])
 fig,ax=plt.subplots(figsize=(8,6));im=ax.imshow(array,extent=[bounds[0],bounds[2],bounds[1],bounds[3]],cmap=cmap,interpolation='nearest',vmin=vmin,vmax=vmax)
 pilot.boundary.plot(ax=ax,color=INK,linewidth=.9)
 ax.set(title=title,xlabel='Longitude (EPSG:4326)',ylabel='Latitude');fig.colorbar(im,ax=ax,label=label,shrink=.8)
 fig.text(.1,.02,NOTICE+'\nNative cell support shown; area counts use clipped land fractions.',fontsize=7)
 fig.tight_layout(rect=(0,.07,1,1));fig.savefig(ROOT/'figures'/path,dpi=180);plt.close(fig)


def figures(cells,data,pred,profile,daily,case):
 pilot=gpd.read_file(ROOT/'data/processed/pilot.gpkg');target=gpd.read_file(ROOT/'data/processed/target_domain.gpkg')
 fig,ax=plt.subplots(figsize=(8,6));target.plot(ax=ax,color='#EFE6EF',edgecolor=COLORS[2],linewidth=.7);pilot.plot(ax=ax,color=COLORS[0])
 ax.set(title='Western Mediterranean design domain and executed pilot',xlabel='Longitude',ylabel='Latitude')
 fig.text(.1,.025,NOTICE+'\nLight: boundary-only design target. Plum: southern Valencian pilot; no full-domain forecasts.',fontsize=8)
 fig.tight_layout(rect=(0,.07,1,1));fig.savefig(ROOT/'figures/study_boundary.png',dpi=180);plt.close(fig)
 fig,ax=plt.subplots(figsize=(10,4));ax.plot(daily.date,daily.compound_land_km2,label='Compound-rule extent',color=COLORS[0]);ax.plot(daily.date,daily.precipitation_deficit_land_km2,label='90-day precipitation deficit',color=COLORS[1])
 ax.set(ylabel='Grid-derived extent, km²',title=f'2022 pilot conditions · Supported land {case["supported_land_km2"]:,.0f} km²');ax.legend();fig.tight_layout();fig.savefig(ROOT/'figures/case2022_extent.png',dpi=180);plt.close(fig)
 index=pd.DatetimeIndex(data['time']);fig,axes=plt.subplots(3,4,figsize=(12,8));maximum=max(1,max(data['label'][(index.year==2022)&(index.month==m)].sum(axis=0).max() for m in range(1,13)))
 bounds=rasterio.transform.array_bounds(profile['height'],profile['width'],profile['transform'])
 for m,ax in enumerate(axes.flat,1):
  values=data['label'][(index.year==2022)&(index.month==m)].sum(axis=0)
  im=ax.imshow(image_values(values,cells,profile),extent=[bounds[0],bounds[2],bounds[1],bounds[3]],vmin=0,vmax=maximum,cmap=SEQUENTIAL,interpolation='nearest')
  pilot.boundary.plot(ax=ax,color=INK,linewidth=.5);ax.set_title(pd.Timestamp(2022,m,1).strftime('%B'),fontsize=10);ax.set_xlabel('');ax.set_ylabel('');ax.tick_params(labelsize=8)
 fig.suptitle('2022 compound-rule days per native cell · monthly sequence');fig.subplots_adjust(bottom=.09,right=.88,hspace=.4)
 cax=fig.add_axes([.91,.2,.015,.6]);fig.colorbar(im,cax=cax,label='Rule-labelled days/month');fig.text(.1,.025,NOTICE,fontsize=8)
 fig.savefig(ROOT/'figures/case2022_monthly_sequence.png',dpi=180);plt.close(fig)
 peak=pd.Timestamp(case['peak_date']);part=pred[pred.date.eq(peak)].set_index('cell')
 for name,label in [('tcn_probability','Mean calibrated forecast probability'),('dropout_std','MC probability standard deviation')]:
  plot_map(part[name].reindex(range(len(cells))).to_numpy(),cells,profile,pilot,'2022 peak rule-labelled extent · '+str(peak.date()),label,'case2022_peak_'+name+'.png',cmap=UNCERTAINTY if name=='dropout_std' else SEQUENTIAL)
 fig,axes=plt.subplots(1,3,figsize=(14,4.5));colors=COLORS;reliability=[]
 for (name,color) in zip(['tcn_probability','logistic_probability','persistence_probability','climatology_probability'],colors):
  p,r,_=precision_recall_curve(pred.label,pred[name]);axes[0].step(r,p,where='post',label=name.replace('_probability',''),color=color)
  observed,expected=calibration_curve(pred.label,pred[name],n_bins=8,strategy='quantile')
  for ax in axes[1:]:ax.plot(expected,observed,'o-',label=name.replace('_probability',''),color=color,markersize=3)
  for grouping,values in [('all',np.repeat('all',len(pred))),('year',pred.date.dt.year.to_numpy()),('geography',pred.geography.to_numpy()),('country',np.repeat('ES',len(pred)))]:
   for group in np.unique(values):
    keep=values==group;p=pred[name].to_numpy()[keep];y=pred.label.to_numpy()[keep]
    edges=np.unique(np.quantile(p,np.linspace(0,1,9)));bins=np.searchsorted(edges[1:-1],p,side='right')
    for b in np.unique(bins):
     selected=bins==b;reliability.append(dict(model=name,grouping=grouping,group=str(group),bin=int(b),n=int(selected.sum()),events=int(y[selected].sum()),
      mean_probability=float(p[selected].mean()),observed_fraction=float(y[selected].mean())))
 pd.DataFrame(reliability).to_csv(ROOT/'metrics/reliability_bins.csv',index=False)
 axes[0].axhline(pred.label.mean(),color='#666',ls='--',label='Test prevalence');axes[0].set(xlabel='Recall',ylabel='Precision',ylim=(0,1),title='Held-out precision–recall');axes[0].legend(fontsize=8)
 for ax in axes[1:]:ax.plot([0,1],[0,1],color='#777',ls='--');ax.set(xlabel='Mean predicted probability',ylabel='Observed event fraction')
 axes[1].set(xlim=(0,1),ylim=(0,1),title='Reliability · full probability range')
 axes[2].set(xlim=(0,.15),ylim=(0,.15),title='Reliability · rare-event zoom 0–0.15')
 fig.tight_layout();fig.savefig(ROOT/'figures/pr_and_reliability.png',dpi=180);plt.close(fig)
 fig,axes=plt.subplots(1,3,figsize=(12,4))
 rng=np.random.default_rng(CFG['seeds'][0]);pick=rng.choice(len(pred),min(4000,len(pred)),replace=False);small=pred.iloc[pick]
 for ax,x,label in zip(axes,['absolute_error','boundary_proximity','training_distance'],['Absolute prediction error','1/(1 + days to label transition)','Distance to training summaries']):
  ax.scatter(small[x],small.dropout_std,s=3,alpha=.18,color=COLORS[0]);ax.set(xlabel=label,ylabel='MC probability std.')
 fig.suptitle('Dropout spread diagnostics · correlated samples, no causal interpretation');fig.tight_layout();fig.savefig(ROOT/'figures/uncertainty_diagnostics.png',dpi=180);plt.close(fig)
 coverage=pd.read_csv(ROOT/'metrics/conformal_coverage.csv');selected=coverage[coverage.grouping.eq('label')]
 fig,ax=plt.subplots(figsize=(9,4));labels=selected.scenario+' / '+selected['group'].map({'0':'non-event','1':'event'});ax.barh(labels,selected.coverage,color=[COLORS[0],COLORS[1]]*4)
 for pos,row in enumerate(selected.itertuples()):
  ax.text(min(row.coverage+.01,.99),pos,f'{100*row.coverage:.1f}%',va='center',ha='left' if row.coverage<.9 else 'right',fontsize=8)
  ax.plot(.95 if row.scenario=='pooled_95' else .9,pos,marker='|',markersize=18,color='#333')
 ax.set(xlim=(0,1),xlabel='Empirical class coverage · dark tick = nominal level',title='Conformal coverage · marginal coverage does not ensure event coverage');fig.tight_layout();fig.savefig(ROOT/'figures/conformal_class_coverage.png',dpi=180);plt.close(fig)
 years=pd.read_csv(ROOT/'metrics/events_by_cell_year.csv');fig,ax=plt.subplots(figsize=(9,4));annual=years.groupby('year').compound_days.sum();ax.bar(annual.index,annual.values,color=COLORS[0]);ax.set(xlabel='Year',ylabel='Rule-positive cell-days (not independent events)',title='Pilot compound-rule counts, fixed 1991–2010 reference');fig.tight_layout();fig.savefig(ROOT/'figures/yearly_event_counts.png',dpi=180);plt.close(fig)
 episodes=pd.read_csv(ROOT/'metrics/observed_cell_episodes.csv',parse_dates=['start'])
 episodes['year']=episodes.start.dt.year
 durations=episodes.groupby(['geography','year']).agg(cell_episodes=('duration_days','size'),mean_episode_days=('duration_days','mean'),maximum_episode_days=('duration_days','max'))
 counts=years.groupby(['geography','year']).agg(eligible_cell_days=('eligible_days','sum'),compound_cell_days=('compound_days','sum'))
 counts['event_rate']=counts.compound_cell_days/counts.eligible_cell_days
 counts=counts.join(durations);counts['cell_episodes']=counts.cell_episodes.fillna(0).astype(int)
 counts.to_csv(ROOT/'metrics/events_and_episodes_by_region_year.csv')


def project():
 root=ET.Element('qgis',version='3.40.0',projectname='Compound drought heat pilot');ET.SubElement(root,'title').text='Independent compound-condition forecast pilot'
 tree=ET.SubElement(root,'layer-tree-group',name='',checked='Qt::Checked');layers=ET.SubElement(root,'projectlayers')
 entries=[('Pilot boundary','../data/processed/pilot.gpkg|layername=region','ogr','vector')]
 entries += [(p.stem,'../maps/'+p.name,'gdal','raster') for p in sorted((ROOT/'maps').glob('case2022_peak*.tif'))]
 for i,(name,path,provider,kind) in enumerate(entries):
  assert (ROOT/'qgis'/path.split('|')[0]).resolve().exists()
  uid='layer_'+str(i);ET.SubElement(tree,'layer-tree-layer',id=uid,name=name,source=path,checked='Qt::Checked')
  node=ET.SubElement(layers,'maplayer',type=kind);ET.SubElement(node,'id').text=uid;ET.SubElement(node,'layername').text=name;ET.SubElement(node,'datasource').text=path;ET.SubElement(node,'provider').text=provider
 ET.indent(root);ET.ElementTree(root).write(ROOT/'qgis/compound_pilot.qgs',encoding='utf-8',xml_declaration=True)
 ET.parse(ROOT/'qgis/compound_pilot.qgs')
 (ROOT/'qgis/native_inspection.md').write_text('XML and datasource existence checked. Native QGIS loading/rendering is unverified; no working QGIS connection was established. GeoTIFFs retain CRS/units/time metadata.\n')
 (ROOT/'qgis/probability.qml').write_text('''<!DOCTYPE qgis PUBLIC 'http://mrcc.com/qgis.dtd' 'SYSTEM'><qgis version="3.40.0"><pipe><rasterrenderer type="singlebandpseudocolor" band="1" opacity="1"><rastershader><colorrampshader minimumValue="0" maximumValue="1" colorRampType="INTERPOLATED"><item value="0" color="#25364e" label="0" alpha="255"/><item value="0.5" color="#4b8d97" label="0.5" alpha="255"/><item value="1" color="#e0c05a" label="1" alpha="255"/></colorrampshader></rastershader></rasterrenderer></pipe></qgis>''')


def report(case):
 domain=json.loads((ROOT/'metrics/domain.json').read_text());m=pd.read_csv(ROOT/'metrics/discrimination.csv');c=pd.read_csv(ROOT/'metrics/conformal_coverage.csv');s=pd.read_csv(ROOT/'metrics/split_support.csv');seed=pd.read_csv(ROOT/'metrics/seed_variation.csv');ep=pd.read_csv(ROOT/'metrics/episode_evaluation.csv');cor=json.loads((ROOT/'metrics/uncertainty_correlation.json').read_text())
 overall=m[m.grouping.eq('all')][['model','n','events','prevalence','average_precision','f1','recall','precision','brier','tn','fp','fn','tp']]
 intervals=m[m.grouping.eq('spatial_bootstrap')][['model','bootstrap_blocks','valid_replicates','ap_low','ap_high','f1_low','f1_high','brier_low','brier_high']]
 tcn=overall[overall.model.eq('tcn_probability')].iloc[0];persistence=overall[overall.model.eq('persistence_probability')].iloc[0]
 pooled=c[c.scenario.eq('pooled_90')&c.grouping.eq('all')].iloc[0]
 rare=c[c.scenario.eq('pooled_90')&c.grouping.eq('label')&c['group'].eq('1')].iloc[0]
 conditional=c[c.scenario.eq('class_conditional_90')&c.grouping.eq('label')&c['group'].eq('1')].iloc[0]
 text=f'''# Results and limitations

**Independent open-data reconstruction. No original scripts or historical performance have been verified.** This is a local retrospective research pilot, not full-domain climate-change attribution or operational drought forecasting. © EuroGeographics for the administrative boundaries.

The main finding is negative: the TCN did not improve on simpler forecasts. Its test AP is {tcn.average_precision:.4f} and its validation-chosen F1 is {tcn.f1:.4f}, versus AP {persistence.average_precision:.4f} and F1 {persistence.f1:.4f} for antecedent-rule persistence. Pooled 90%-nominal conformal sets cover {100*pooled.coverage:.2f}% of all test cell-days but **{round(rare.coverage*rare.n)} of {int(rare.n)} event cell-days**. Class-conditional calibration improves event coverage to {100*conditional.coverage:.2f}%, still below its nominal level. These results do not support operational use or a claim that marginal coverage protects rare compound events.

## Executed footprint and source support

The boundary-only six-region Spanish design target measures **{domain['target_area_km2']:,.2f} km²** in EPSG:3035. The southern Valencian pilot measures **{domain['pilot_area_km2']:,.2f} km²**; **{domain['grid_supported_land_km2']:,.2f} km²** are represented by {domain['cells']} retained native 0.25° ERA5 cells. Cells with <50 km² of pilot land are excluded, leaving {domain['excluded_coastal_land_km2']:.2f} km² outside the executed support. Geographic linework is densified before equal-area measurement so clipped-cell weights reconcile with the boundary. No prediction is extrapolated over the much larger design target.

Public daily ERA5 is delivered through Open-Meteo, explicit model, UTC, no statistical elevation downscaling. It differs from preferred direct ERA5-Land retrieval. All 1991–2022 daily dates and units are verified and archived. The baseline is 1991–2010, not a standard 30-year normal. NASA collection metadata were checked, but MODIS LST/NDVI granules were not acquired, so no cloud-gap analysis or MODIS-informed forecast is claimed. Independent JRC drought reports provide regional narrative context only; no pixel-level independent validation is claimed.

![Domain](figures/study_boundary.png)

## Task and partitions

Predict the compound label seven days ahead from 56 daily antecedent vectors, with full raw dependencies reaching 144 days back. Heat requires >local seasonal p90 and >30°C for at least three days ending on the target day; drought uses a trailing-90-day precipitation total <local seasonal p20. This is not SPEI or measured soil-moisture drought. All scalers, checkpoints, probability mappings and F1 thresholds avoid conformal/test data. Northern cells are excluded from every supervised fitting/calibration stage, with a separating native row. Local pre-2011 reference climate is known in both geographies. Reanalysis release latency is not modelled; valid-time causality is not operational data availability.

The first calibration design had no positive examples in 2018; fitting stopped. The revised ordered periods were chosen from feasibility counts before any model test scores. See DESIGN_DECISIONS.md. Sparse positives are a major limitation, not something to hide with a headline accuracy.

{table(s)}

Table: eligible cell-time samples and rule-positive counts. Nearby days/cells are correlated, so these are not independent event denominators. Full raw-history embargoes separate all temporal stages.

## Model comparison

TCNs use four causal residual blocks, two kernel-3 convolutions at each dilation 1/2/4/8, 16 channels, dropout 0.2 and weighted binary cross-entropy. Receptive field is 61 positions over a 56-day input. There is no batch normalisation; dropout alone is activated at inference. Early stopping uses 2015–2016 AP. Platt scaling and the F1 operating point use 2017–2018 development data. The first predeclared seed supplies the main model; seeds are not chosen using test scores. The logistic model receives summaries of the same information window. Persistence sees the antecedent compound state, not the future label. Climatology is fitted only on training labels.

{table(overall)}

Table: held-out 2021–2022 cell-time metrics at probability-calibration-selected thresholds. AP must be compared with the shown rare-event prevalence. Cell-time confusion matrices use exactly the shown n; there is no implicit class balancing of evaluation.

{table(intervals)}

Table: descriptive 95% spatial-block bootstrap intervals, 50 km projected blocks, {CFG['bootstrap_replicates']} attempted resamples. Replicates without any events are excluded; the shown valid count matters. These intervals do not account for every episode or source uncertainty.

{table(seed[['seed','validation_best_ap','average_precision','f1','brier']])}

Table: deterministic calibrated TCN seed variation, same test denominator. Main MC-dropout evaluation is separate. The failed test operating point is retained; thresholds are not retuned on 2022.

![PR and reliability](figures/pr_and_reliability.png)

Figure: test PR curves and quantile-bin reliability in a displayed rare-event range. The PR reference is the actual test prevalence. Probability reliability is empirical and can vary under shift.

{table(ep[ep.cell.eq('all')][['model','observed_episodes','predicted_episodes','tp','fp','fn','precision','recall','f1']])}

Table: one-to-one temporal-overlap episode matching, summed over test cells. Episode TN is undefined and not fabricated; episode counts are not automatically independent meteorological events.

## Conformal reliability

LAC nonconformity is 1−p(true class). The corrected quantile uses rank ceil((n+1)(1−alpha)), with infinite threshold if that rank exceeds n. Prediction sets may be empty. We implemented and unit-tested this rule directly; no MAPIE execution is asserted. Calibration samples are entirely separate from training and Platt/threshold selection.

{table(c[c.grouping.eq('all')][['scenario','n','events','coverage','mean_set_size','ambiguity_rate','empty_rate','singleton_rate']])}

{table(c[c.grouping.eq('label')][['scenario','group','n','events','coverage','mean_set_size']])}

Tables: marginal and event/non-event set coverage. High marginal coverage can coexist with poor event coverage because non-events dominate. Weekly thinning and class-conditional calibration are sensitivity scenarios, not proofs of exchangeability or conditional coverage. Class-conditional event calibration is especially weak with only 13 positive cell-days. Geography/year/season/severity tables are in metrics/conformal_coverage.csv.

![Conformal coverage](figures/conformal_class_coverage.png)

## Uncertainty diagnostics

Thirty stochastic dropout passes use the same antecedent inputs. Mean calibrated probability, probability standard deviation, predictive entropy and dropout mutual-information approximation are retained. This is only an approximate model-uncertainty diagnostic. It does not quantify ERA5 observation error, label-definition uncertainty, unacquired MODIS gaps or all aleatoric variability.

Spearman correlation between dropout standard deviation and absolute probability error is **{cor['rho']:.4f}**, with spatial-block bootstrap interval [{cor['spatial_bootstrap_low']:.4f}, {cor['spatial_bootstrap_high']:.4f}]. The raw denominator is {cor['raw_pairs']:,} cell-days across {cor['spatial_blocks']} spatial blocks; effective independent sample size is not identified. The separate boundary diagnostic uses 1/(1+days to the nearest same-cell label transition), computed offline, never supplied as a predictor. Training distance is nearest distance in standardised antecedent-summary space to a fixed training-only subset. No missing-MODIS diagnostic is fabricated, since those predictors were absent throughout.

That high pooled correlation is not evidence of useful error detection: non-events dominate, and on a non-event day absolute error equals predicted event probability. Dropout spread also varies with mean probability, making a strong pooled relationship partly mechanical. Class-conditional correlations and their spatial resampling intervals are retained in uncertainty_correlation.json. They do not establish an independent meteorological sample size or causal uncertainty/error relation.

![Uncertainty](figures/uncertainty_diagnostics.png)

## 2022 case study

Over supported pilot land, the peak grid-derived compound-rule extent is **{case['peak_rule_extent_km2']:,.2f} km²** on **{case['peak_date']}**. There are {case['rule_positive_cell_days']} positive native cell-days and {case['space_time_connected_components']} six-neighbour connected space-time support components in 2022. Components are a dependence diagnostic, not an independently validated event catalogue. The denominator is {case['supported_land_km2']:,.2f} km², not the large design domain. Climate fields are represented as piecewise native cells; coastal area fractions are clipped to the boundary. Monthly/daily area tables expose the exact denominator. No validated area-confidence interval is available.

![Extents](figures/case2022_extent.png)

![Sequence](figures/case2022_monthly_sequence.png)

![Probability](figures/case2022_peak_tcn_probability.png)

![Dropout](figures/case2022_peak_dropout_std.png)

Figures: actual 2022 rule extent, monthly native-cell counts and model outputs on the peak labelled-extent date. The peak was selected by the observed reanalysis rule extent, not the most favourable prediction. File-per-time COGs show exact month-end days, while the figure sequence shows monthly counts; these are explicitly different summaries. Geographic-buffer cells have labels but no evaluated model predictions and remain nodata in forecast maps.

The [JRC August 2022 report](https://publications.jrc.ec.europa.eu/repository/handle/JRC130493) describes persistent precipitation deficits and heatwaves across Europe, including Spain. The [western Mediterranean February report](https://publications.jrc.ec.europa.eu/repository/handle/JRC128743) describes Iberian precipitation deficits and regional differences. They establish independent context, not numerical agreement with this small pilot's thresholds or extent. There is no causal attribution design here.

## Completion and limitations

Completed: real native-grid pilot acquisition/QA; Dask/xarray daily cube; physical labels and alternative; embargoed temporal/geographic split; three causal TCN seeds and non-neural/rule baselines; MC-dropout; direct split conformal and sensitivity scenarios; discrimination/reliability/episode diagnostics; validated COGs; source hashes, tests and reproduction commands.

Partial: only about 11,000 km² of the roughly 282,000 km² design domain is executed; the reference is 20 years; the target is a precipitation-deficit proxy; event support is sparse and clustered; direct ERA5-Land, soil moisture/SPEI, MODIS predictors, cloud-QA/missingness sensitivity and independent numerical drought/station validation are not executed. Native QGIS rendering is unverified. Exported probabilities and coverage are research results, not operational guarantees.

Computation dates remain in provenance files, not figure/table stamps. Historical figures are confined to historical_claims_unverified.md. No online publication is claimed.
'''
 (ROOT/'RESULTS_AND_LIMITATIONS.md').write_text(text,encoding='utf-8')


def manifest():
 checks=[]
 for p in sorted((ROOT/'maps').glob('*.tif')):
  valid,errors,warnings=cog_validate(p,strict=True,quiet=True)
  with rasterio.open(p) as src:
   assert src.crs.to_epsg()==4326 and src.res==(.25,.25) and src.nodata==-9999
   checks.append(dict(file=p.name,cog_valid=valid,crs=str(src.crs),width=src.width,height=src.height,overview_levels=str(src.overviews(1)),errors=str(errors)))
  assert valid,(p,errors)
 pd.DataFrame(checks).to_csv(ROOT/'metrics/cog_validation.csv',index=False)
 source=pd.read_csv(ROOT/'data_manifest.csv')
 for row in source.itertuples():assert hashlib.sha256((ROOT/row.file).read_bytes()).hexdigest()==row.sha256
 inputs=[]
 for folder in ['src','tests','data/raw','data/processed','data/download_manifest','models','metrics','maps','figures','qgis']:
  for p in sorted((ROOT/folder).rglob('*')):
   if p.is_file() and '__pycache__' not in p.parts and p.name!='run_manifest.json':inputs.append(dict(file=p.relative_to(ROOT).as_posix(),sha256=hashlib.sha256(p.read_bytes()).hexdigest(),bytes=p.stat().st_size))
 for name in ['config.yaml','requirements-lock.txt','environment.yml','METHODS.md','EVENT_DEFINITION.md','data_manifest.csv','figure_style.json','FIGURE_STYLE.md']:
  p=ROOT/name;inputs.append(dict(file=name,sha256=hashlib.sha256(p.read_bytes()).hexdigest(),bytes=p.stat().st_size))
 versions={p:importlib.metadata.version(p) for p in ['numpy','pandas','torch','scikit-learn','xarray','dask','geopandas','rasterio','rio-cogeo']}
 (ROOT/'metrics/run_manifest.json').write_text(json.dumps(dict(recorded_utc=utc_now(),config=CFG,seeds=CFG['seeds'],software=versions,python=sys.version,
  platform=platform.platform(),source_responses_verified=len(source),cogs_validated=len(checks),files=inputs,
  provenance='new reconstruction; no original work or historical metrics verified'),indent=2))
 status=[('Native-grid pilot pipeline','done'),('Causal TCN/baselines/MC-dropout','done'),('Conformal sets and coverage diagnostics','done'),('GIS exports and reproducibility tests','done'),
  ('Full target-domain model','partial'),('30-year reference and soil-moisture/SPEI definition','partial'),('MODIS predictor integration and cloud QA','partial'),
  ('Independent numerical climate validation','partial'),('Direct CDS ERA5-Land source','blocked'),('Native QGIS rendered inspection','blocked')]
 pd.DataFrame(status,columns=['deliverable','status']).to_csv(ROOT/'metrics/completion_status.csv',index=False)
 print(f'{len(checks)} COGs validated; {len(source)} source-response hashes verified',flush=True)


def main():
 cells=gpd.read_file(ROOT/'data/processed/cells.gpkg');data=np.load(ROOT/'data/processed/learning_arrays.npz')
 pred=pd.read_csv(ROOT/'metrics/test_predictions_with_sets.csv',parse_dates=['date'])
 profile,daily,case=maps(cells,data,pred);figures(cells,data,pred,profile,daily,case);project();report(case);manifest()


if __name__=='__main__':main()
