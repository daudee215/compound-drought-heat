from pathlib import Path
import hashlib,json,time
from datetime import datetime,timezone
import pandas as pd
import requests
import yaml
ROOT=Path(__file__).resolve().parents[1]
CFG=yaml.safe_load((ROOT/'config.yaml').read_text())

def dirs():
 for p in ['src','tests','notebooks','data/raw','data/processed','data/download_manifest','models','metrics','maps','figures','qgis']:(ROOT/p).mkdir(parents=True,exist_ok=True)

def fetch(name,url,params=None):
 p=ROOT/'data/raw'/name;meta=p.with_suffix(p.suffix+'.request.json')
 if p.exists():
  assert json.loads(meta.read_text())['params']==params,'Cache parameters changed; use a separate cache'
  return p,False
 r=requests.get(url,params=params,timeout=120)
 if r.status_code==429:
  raise RuntimeError('Public API rate limit: '+r.text[:300]+'. Cached work is retained; resume later, do not bypass the limit.')
 r.raise_for_status()
 if 'json' in r.headers.get('Content-Type','') and isinstance(r.json(),dict) and r.json().get('error'):raise RuntimeError(r.text[:400])
 p.write_bytes(r.content)
 meta.write_text(json.dumps(dict(url=r.url,params=params,retrieved_utc=datetime.now(timezone.utc).isoformat(),last_modified=r.headers.get('Last-Modified')),indent=2))
 return p,True

def lineage(p,product,role,licence,resolution,period,version):
 meta=json.loads(p.with_suffix(p.suffix+'.request.json').read_text())
 return dict(file=p.relative_to(ROOT).as_posix(),retrieval_url=meta['url'],product_id=product,licence=licence,
  coverage=period,resolution=resolution,version=version,processing_role=role,retrieved_utc=meta['retrieved_utc'],
  sha256=hashlib.sha256(p.read_bytes()).hexdigest())

def write_manifest(rows):
 pd.DataFrame(rows).to_csv(ROOT/'data_manifest.csv',index=False)

def utc_now():return datetime.now(timezone.utc).isoformat()
