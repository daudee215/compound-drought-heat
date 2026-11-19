"""Metadata-only NASA access audit. No granules or credentials are represented as acquired."""
import json
import pandas as pd
from common import ROOT,fetch,lineage

def main():
 rows=[];status=[]
 for product in ['MOD11A1','MOD13Q1']:
  try:
   p,_=fetch(product+'_061_collection_metadata.json','https://cmr.earthdata.nasa.gov/search/collections.json',dict(short_name=product,version='061',page_size=1))
   data=json.loads(p.read_text());entry=data['feed']['entry'][0]
   rows.append(lineage(p,product+'.061 collection metadata','Source availability investigation only; no LST/NDVI predictor or cloud-QA analysis',
    'NASA Earth science data policy; granule access/terms require separate verification','MOD11A1 daily ~1km; MOD13Q1 16-day ~250m',str(entry.get('time_start'))+'/'+str(entry.get('time_end')),'Collection 6.1'))
   status.append(dict(product=product,status='metadata_only',title=entry.get('title'),time_start=entry.get('time_start'),time_end=entry.get('time_end'),
    granules_downloaded=0,predictor_included=False))
  except Exception as exc:status.append(dict(product=product,status='access_unavailable',reason=str(exc),granules_downloaded=0,predictor_included=False))
 existing=pd.read_csv(ROOT/'data_manifest.csv')
 if rows:
  existing=existing[~existing.product_id.isin([r['product_id'] for r in rows])]
  pd.concat([existing,pd.DataFrame(rows)],ignore_index=True).to_csv(ROOT/'data_manifest.csv',index=False)
 (ROOT/'metrics/source_access_audit.json').write_text(json.dumps(status,indent=2))
 print(json.dumps(status,indent=2),flush=True)

if __name__=='__main__':main()
