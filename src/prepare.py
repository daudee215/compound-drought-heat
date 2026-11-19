"""Daily climate cube, causal physical labels and embargoed sample partitions."""
import json
import geopandas as gpd
import numpy as np
import pandas as pd
import xarray as xr
from scipy import ndimage
from common import ROOT,CFG


def calendar_day(index):
 day=index.day_of_year.to_numpy()-((index.is_leap_year)&(index.month>2)).astype(int)
 day[(index.month==2)&(index.day==29)]=59
 return day


def compound_rule(tmax,heat_threshold,rain90,rain_threshold,minimum=3):
 hot=(tmax>heat_threshold)&np.isfinite(tmax)
 runs=pd.DataFrame(hot).rolling(minimum,min_periods=minimum).sum().to_numpy()==minimum
 return (runs&(rain90<rain_threshold)&np.isfinite(rain90)).astype('uint8')


def eligible_samples(index,cells):
 raw_history=CFG['lookback_days']+CFG['drought_window_days']-2
 result=[]
 for name,(start,end) in CFG['periods'].items():
  for anchor in range(raw_history,len(index)-CFG['lead_days']):
   target=anchor+CFG['lead_days']
   if index[anchor-raw_history]<pd.Timestamp(start) or index[target]>pd.Timestamp(end):continue
   for cell in range(len(cells)):
    if name!='test' and cells.geography.iloc[cell]!='development':continue
    if name=='test' and cells.geography.iloc[cell]=='buffer':continue
    result.append((name,cell,anchor,target,anchor-raw_history))
 return pd.DataFrame(result,columns=['split','cell','anchor','target','raw_history_start'])


def main():
 cells=gpd.read_file(ROOT/'data/processed/cells.gpkg')
 index=pd.date_range(CFG['source_start'],CFG['source_end']);temperature=[];precipitation=[]
 for row in cells.itertuples():
  payload=json.loads((ROOT/f'data/raw/{row.cell_id}_1991_2022.json').read_text(encoding='utf-8'))
  assert pd.DatetimeIndex(pd.to_datetime(payload['daily']['time'])).equals(index)
  temperature.append(payload['daily']['temperature_2m_max']);precipitation.append(payload['daily']['precipitation_sum'])
 temp=np.asarray(temperature,dtype='float32').T;rain=np.asarray(precipitation,dtype='float32').T
 assert np.isfinite(temp).all() and np.isfinite(rain).all() and (rain>=0).all()
 rain90=pd.DataFrame(rain).rolling(CFG['drought_window_days'],min_periods=CFG['drought_window_days']).sum().to_numpy()
 day=calendar_day(index);baseline=(index>=CFG['baseline_start'])&(index<=CFG['baseline_end'])
 hq=np.empty((365,len(cells)));rq=np.empty_like(hq);alternate=np.empty_like(hq);counts=[]
 for d in range(1,366):
  delta=np.abs(day-d);season=(np.minimum(delta,365-delta)<=CFG['seasonal_window_days']//2)&baseline
  hq[d-1]=np.quantile(temp[season],CFG['heat_percentile'],axis=0)
  rq[d-1]=np.nanquantile(rain90[season],CFG['drought_percentile'],axis=0)
  alternate[d-1]=np.nanquantile(rain90[season],CFG['alternate_drought_percentile'],axis=0)
  counts.append(dict(calendar_day=d,n_temperature_reference_days=int(season.sum()),n_complete_rain_reference_days=int(np.isfinite(rain90[season,0]).sum())))
 pd.DataFrame(counts).to_csv(ROOT/'metrics/reference_sample_counts.csv',index=False)
 heat=np.maximum(hq[day-1],CFG['heat_minimum_c']);drought=rq[day-1]
 label=compound_rule(temp,heat,rain90,drought,CFG['heat_minimum_days'])
 alt=compound_rule(temp,heat,rain90,alternate[day-1],CFG['heat_minimum_days'])
 label_valid=np.isfinite(rain90)&np.isfinite(temp)&np.isfinite(drought)
 channels=np.stack([temp-heat,np.log1p(rain),rain90/(drought+1),
  np.broadcast_to(np.sin(2*np.pi*day[:,None]/365),temp.shape),
  np.broadcast_to(np.cos(2*np.pi*day[:,None]/365),temp.shape)],axis=-1).astype('float32')
 # Regular spatial cube with missing ocean/out-of-pilot cells; no fine-grid interpolation.
 lat=np.sort(cells.lat.unique());lon=np.sort(cells.lon.unique());shape=(len(index),len(lat),len(lon))
 variables={}
 for name,values in [('tmax_c',temp),('precipitation_mm',rain),('rain90_mm',rain90),('compound',np.where(label_valid,label,np.nan)),('compound_alternate',np.where(label_valid,alt,np.nan)),('label_coverage',label_valid.astype(float))]:
  cube=np.full(shape,np.nan,dtype='float32')
  for c,row in enumerate(cells.itertuples()):cube[:,np.where(lat==row.lat)[0][0],np.where(lon==row.lon)[0][0]]=values[:,c]
  variables[name]=(('time','y','x'),cube)
 dataset=xr.Dataset(variables,coords=dict(time=index,y=lat,x=lon),attrs=dict(crs='EPSG:4326',timezone='UTC',
  source='ERA5 native 0.25° via Open-Meteo daily aggregation; not direct CDS files',provenance='new independent reconstruction'))
 dataset=dataset.chunk(dict(time=365,y=1,x=1))
 dataset.attrs['Conventions']='CF-1.8'
 dataset['x'].attrs.update(standard_name='longitude',units='degrees_east')
 dataset['y'].attrs.update(standard_name='latitude',units='degrees_north')
 dataset['tmax_c'].attrs['units']='degree_Celsius';dataset['precipitation_mm'].attrs['units']='mm';dataset['rain90_mm'].attrs['units']='mm'
 dataset.to_netcdf(ROOT/'data/processed/climate_cube.nc',engine='h5netcdf')
 samples=eligible_samples(index,cells)
 samples['label']=label[samples.target,samples.cell];samples['alternate_label']=alt[samples.target,samples.cell]
 assert label_valid[samples.target,samples.cell].all()
 samples['date']=index[samples.target];samples['geography']=cells.geography.to_numpy()[samples.cell]
 samples.to_csv(ROOT/'data/processed/samples.csv',index=False)
 np.savez_compressed(ROOT/'data/processed/learning_arrays.npz',features=channels,label=label,alternate=alt,
  rain90=rain90,drought_threshold=drought,heat_threshold=heat,tmax=temp,precipitation=rain,time=index.values,
  reference_heat=hq,reference_rain=rq,label_valid=label_valid)
 yearly=[];episodes=[]
 for c,row in enumerate(cells.itertuples()):
  starts=np.flatnonzero(np.diff(np.r_[0,label[:,c],0].astype(int))==1)
  ends=np.flatnonzero(np.diff(np.r_[0,label[:,c],0].astype(int))==-1)
  for start,end in zip(starts,ends):
   episodes.append(dict(cell=c,cell_id=row.cell_id,start=str(index[start].date()),end=str(index[end-1].date()),duration_days=end-start,
    geography=row.geography,land_area_km2=row.land_area_km2))
  for year in range(2011,2023):
   keep=index.year==year
   yearly.append(dict(cell_id=row.cell_id,year=year,geography=row.geography,eligible_days=int(keep.sum()),
    compound_days=int(label[keep,c].sum()),alternate_days=int(alt[keep,c].sum()),event_rate=float(label[keep,c].mean())))
 pd.DataFrame(yearly).to_csv(ROOT/'metrics/events_by_cell_year.csv',index=False)
 pd.DataFrame(episodes).to_csv(ROOT/'metrics/observed_cell_episodes.csv',index=False)
 samples.groupby(['split','geography']).agg(n=('label','size'),events=('label','sum'),prevalence=('label','mean')).to_csv(ROOT/'metrics/split_support.csv')
 print(samples.groupby(['split','geography']).label.agg(['size','sum','mean']).to_string(),flush=True)


if __name__=='__main__':main()
