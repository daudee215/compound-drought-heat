"""Native-grid pilot; no representative-point extrapolation to the target domain."""
import argparse,json,math,time
import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import box
from common import ROOT,CFG,dirs,fetch,lineage,write_manifest


def domains():
 url='https://gisco-services.ec.europa.eu/distribution/v2/nuts/geojson/NUTS_RG_10M_2024_4326_LEVL_2.geojson'
 p,_=fetch('NUTS_RG_10M_2024_4326_LEVL_2.geojson',url)
 nuts=gpd.read_file(p)
 target=nuts[nuts.NUTS_ID.isin(CFG['target_nuts'])].copy()
 pilot=nuts[nuts.NUTS_ID.eq(CFG['pilot_nuts'])].copy()
 if CFG.get('pilot_north_boundary'):
  pilot['geometry']=pilot.geometry.intersection(box(-180,-90,180,CFG['pilot_north_boundary']))
 pilot['study_label']='Southern Valencian native-grid pilot, clipped by declared northern latitude'
 target_dense=target.copy();target_dense['geometry']=target_dense.geometry.segmentize(.01)
 pilot_dense=pilot.copy();pilot_dense['geometry']=pilot_dense.geometry.segmentize(.01)
 target['area_km2']=target_dense.to_crs(CFG['area_crs']).area/1e6
 pilot['area_km2']=pilot_dense.to_crs(CFG['area_crs']).area/1e6
 target.to_file(ROOT/'data/processed/target_domain.gpkg',layer='regions',driver='GPKG')
 pilot.to_file(ROOT/'data/processed/pilot.gpkg',layer='region',driver='GPKG')
 step=CFG['grid_degrees'];w,s,e,n=pilot.total_bounds;rows=[]
 for iy in range(math.floor(s/step)-1,math.ceil(n/step)+2):
  for ix in range(math.floor(w/step)-1,math.ceil(e/step)+2):
   lon,lat=round(ix*step,5),round(iy*step,5);geom=box(lon-step/2,lat-step/2,lon+step/2,lat+step/2)
   clipped=geom.intersection(pilot.geometry.iloc[0])
   if not clipped.is_empty and clipped.area>0:
    area=gpd.GeoSeries([clipped.segmentize(.01)],crs=4326).to_crs(CFG['area_crs']).area.iloc[0]/1e6
    rows.append(dict(cell_id=f'era5_{lat:.2f}_{lon:.2f}',lon=lon,lat=lat,land_area_km2=float(area),geometry=geom))
 cells=gpd.GeoDataFrame(rows,crs=4326).sort_values(['lat','lon']).reset_index(drop=True)
 excluded=cells[cells.land_area_km2<CFG['minimum_cell_land_area_km2']].copy()
 excluded.drop(columns='geometry').to_csv(ROOT/'metrics/excluded_coastal_cells.csv',index=False)
 cells=cells[cells.land_area_km2>=CFG['minimum_cell_land_area_km2']].reset_index(drop=True)
 # Geographic split is fixed from geometry before inspecting climate outcomes.
 latitudes=np.sort(cells.lat.unique());boundary=latitudes[max(2,int(len(latitudes)*.7))]
 cells['geography']=np.where(cells.lat>=boundary,'north_holdout',np.where(cells.lat>=boundary-step,'buffer','development'))
 cells.to_file(ROOT/'data/processed/cells.gpkg',layer='cells',driver='GPKG')
 record=dict(target_area_km2=float(target.area_km2.sum()),pilot_area_km2=float(pilot.area_km2.sum()),grid_supported_land_km2=float(cells.land_area_km2.sum()),
  cells=len(cells),geography=cells.geography.value_counts().to_dict(),target_regions=CFG['target_nuts'],pilot=CFG['pilot_nuts'],source='GISCO NUTS 2024 1:10M, EPSG:3035 area',
  excluded_coastal_land_km2=float(excluded.land_area_km2.sum()),exclusion_rule='Native cells with <50 km² pilot land excluded; geometry-based criterion, not outcome selection',
  status='Native ERA5 pilot; target-domain climate modelling not executed')
 (ROOT/'metrics/domain.json').write_text(json.dumps(record,indent=2))
 print(json.dumps(record,indent=2),flush=True)
 return p,cells


def main():
 args=argparse.ArgumentParser();args.add_argument('--domains-only',action='store_true');opt=args.parse_args()
 dirs();boundary,cells=domains()
 rows=[lineage(boundary,'GISCO NUTS 2024 level 2','Boundary and land-area denominator; not a historical project boundary',
  'Eurostat/GISCO download provisions and attribution; non-commercial research','Generalised 1:10M vector, EPSG:4326','2024','NUTS 2024')]
 if opt.domains_only:write_manifest(rows);return
 for cell in cells.itertuples():
  params=dict(latitude=cell.lat,longitude=cell.lon,start_date=CFG['source_start'],end_date=CFG['source_end'],models=CFG['source_model'],
   daily='temperature_2m_max,precipitation_sum',timezone='GMT',elevation='nan',cell_selection='nearest',temperature_unit='celsius',precipitation_unit='mm')
  p,fresh=fetch(cell.cell_id+'_1991_2022.json','https://archive-api.open-meteo.com/v1/archive',params)
  payload=json.loads(p.read_text(encoding='utf-8'))
  assert abs(payload['latitude']-cell.lat)<.002 and abs(payload['longitude']-cell.lon)<.002
  assert payload['utc_offset_seconds']==0
  assert payload['daily_units']['temperature_2m_max']=='°C' and payload['daily_units']['precipitation_sum']=='mm'
  index=pd.to_datetime(payload['daily']['time']);expected=pd.date_range(CFG['source_start'],CFG['source_end'])
  assert pd.DatetimeIndex(index).equals(expected)
  assert np.isfinite(payload['daily']['temperature_2m_max']).all() and np.isfinite(payload['daily']['precipitation_sum']).all()
  rows.append(lineage(p,'ERA5 via Open-Meteo historical API','Daily Tmax and precipitation; pilot labels and antecedent predictors',
   'CC BY 4.0; free API non-commercial-only service','Native 0.25° EPSG:4326; provider daily UTC aggregates','1991-2022',
   'ERA5 explicit model; unversioned provider delivery snapshot; raw bytes frozen'))
  write_manifest(rows)
  print(cell.cell_id,len(index),'complete daily records',flush=True)
  if fresh:time.sleep(CFG['download_spacing_seconds'])
 for name,url,product in [
  ('JRC130493_2022_drought.pdf','https://publications.jrc.ec.europa.eu/repository/bitstream/JRC130493/JRC130493_01.pdf','JRC130493 Drought in Europe August 2022'),
  ('JRC128743_2022_western_mediterranean.html','https://publications.jrc.ec.europa.eu/repository/handle/JRC128743','JRC128743 Drought in western Mediterranean February 2022')]:
  p,_=fetch(name,url);rows.append(lineage(p,product,'Independent regional narrative context, not pixel-level validation',
   'European Commission reuse with acknowledgement','Report; regional geographic support','2022','2022 report'))
 write_manifest(rows)


def append_unused_cache():
 manifest=pd.read_csv(ROOT/'data_manifest.csv');used=set(manifest.file)
 extra=[]
 for p in (ROOT/'data/raw').glob('era5_*_1991_2022.json'):
  if p.relative_to(ROOT).as_posix() not in used:
   extra.append(lineage(p,'ERA5 via Open-Meteo; excluded cached response','Downloaded during initial footprint audit; excluded from final cube and every model',
    'CC BY 4.0; non-commercial free API','Native 0.25° daily UTC','1991-2022','Frozen unversioned delivery snapshot'))
 if extra:pd.concat([manifest,pd.DataFrame(extra)],ignore_index=True).to_csv(ROOT/'data_manifest.csv',index=False)


if __name__=='__main__':
 main();append_unused_cache()
