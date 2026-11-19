"""Synthetic arrays in these tests are unit-test fixtures, never research outputs."""
import sys,json
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import numpy as np
import pandas as pd
import pytest
import torch
import rasterio
from rio_cogeo.cogeo import cog_validate
from common import ROOT,CFG
from prepare import calendar_day,compound_rule
from methods import TCN,seed_all,conformal_quantile,conformal_sets,classification_metrics


def test_leap_alignment_retains_actual_dates():
 dates=pd.date_range('2020-02-28','2020-03-02')
 assert calendar_day(dates).tolist()==[59,59,60,61]
 assert len(dates)==4


def test_three_day_rule_and_strict_drought_threshold():
 t=np.array([28,32,33,34,31,32,33],float)[:,None]
 h=np.full_like(t,30);rain=np.array([2,2,2,2,10,2,2],float)[:,None]
 y=compound_rule(t,h,rain,np.full_like(t,10),3).ravel()
 assert y.tolist()==[0,0,0,1,0,1,1]
 changed=t.copy();changed[5:]=99
 assert np.array_equal(compound_rule(changed,h,rain,np.full_like(t,10),3)[:5],y[:5,None])


def test_causal_convolutions_cannot_see_future():
 seed_all(4);net=TCN(channels=4,dropout=.3).eval();x=torch.randn(2,5,56)
 altered=x.clone();altered[:,:,30:]+=100
 with torch.no_grad():
  first=net.network(x);second=net.network(altered)
 assert torch.equal(first[:,:,:30],second[:,:,:30])


def test_fixed_seed_repeats_initialisation_and_predictions():
 seed_all(7);a=TCN(channels=4).eval();x=torch.zeros(2,5,56);y=a(x).detach()
 seed_all(7);b=TCN(channels=4).eval()
 assert torch.equal(y,b(x).detach())


def test_finite_conformal_quantile_and_class_sets():
 q=conformal_quantile(np.arange(9)/10,.1)
 assert q==pytest.approx(.8)
 assert conformal_sets([.05,.5,.9],q).tolist()==[[True,False],[True,True],[False,True]]
 assert np.isinf(conformal_quantile([.1,.2],.1))
 assert conformal_sets([.4],(.3,.5)).tolist()==[[False,False]]


def test_hand_checked_metric_denominators():
 m=classification_metrics([1,0,0,0],[.8,.8,.1,.2],.5)
 assert (m['tn'],m['fp'],m['fn'],m['tp'],m['n'])==(2,1,0,1,4)
 assert m['prevalence']==.25 and m['f1']==pytest.approx(2/3)
 assert m['brier']==pytest.approx(.1825)


def test_actual_histories_and_supervised_geography_are_isolated():
 path=ROOT/'data/processed/samples.csv'
 if not path.exists():pytest.skip('Prepare real source data first')
 samples=pd.read_csv(path)
 previous=-1
 for split in CFG['periods']:
  rows=samples[samples.split.eq(split)]
  assert len(rows)>0
  assert rows.raw_history_start.min()>previous
  previous=rows.target.max()
  assert ((rows.target-rows.anchor)==CFG['lead_days']).all()
  assert ((rows.anchor-rows.raw_history_start)==CFG['lookback_days']+CFG['drought_window_days']-2).all()
  if split!='test':assert rows.geography.eq('development').all()


def test_actual_source_units_and_daily_timestamps():
 files=list((ROOT/'data/raw').glob('era5_*_1991_2022.json'))
 if not files:pytest.skip('Acquire data first')
 for path in files:
  a=json.loads(path.read_text(encoding='utf-8'))
  assert a['utc_offset_seconds']==0
  assert a['daily_units']['temperature_2m_max']=='°C' and a['daily_units']['precipitation_sum']=='mm'
  dates=pd.DatetimeIndex(pd.to_datetime(a['daily']['time']))
  assert dates.equals(pd.date_range(CFG['source_start'],CFG['source_end']))


def test_actual_cog_geometry_and_probability_range():
 files=list((ROOT/'maps').glob('*.tif'))
 if not files:pytest.skip('Export maps first')
 profile=None
 for path in files:
  ok,errors,warnings=cog_validate(path,strict=True,quiet=True);assert ok,(path,errors)
  with rasterio.open(path) as src:
   current=(src.crs,src.transform,src.width,src.height)
   if profile is None:profile=current
   assert current==profile and src.crs.to_epsg()==4326 and src.res==(.25,.25)
   assert src.nodata==-9999
   if 'probability' in path.name or 'coverage' in path.name:
    values=src.read(1,masked=True).compressed();assert ((values>=0)&(values<=1)).all()


def test_area_denominator_is_pilot_support_not_full_target():
 path=ROOT/'metrics/domain.json'
 if not path.exists():pytest.skip('Acquire domain first')
 domain=json.loads(path.read_text())
 assert abs(domain['grid_supported_land_km2']+domain['excluded_coastal_land_km2']-domain['pilot_area_km2'])<.01
 assert domain['grid_supported_land_km2']<domain['target_area_km2']/10
 extent=ROOT/'metrics/case2022_daily_extent.csv'
 if extent.exists():
  data=pd.read_csv(extent)
  assert data.supported_land_denominator_km2.eq(data.supported_land_denominator_km2.iloc[0]).all()
  assert (data.compound_land_km2<=data.supported_land_denominator_km2+1e-6).all()


def test_incomplete_rain_history_has_explicit_unknown_labels():
 import xarray as xr
 path=ROOT/'data/processed/climate_cube.nc'
 if not path.exists():pytest.skip('Prepare data first')
 with xr.open_dataset(path) as data:
  assert data.compound.isel(time=slice(0,89)).isnull().all().item()
 arrays=np.load(ROOT/'data/processed/learning_arrays.npz')
 assert not arrays['label_valid'][:89].any()
