#!/usr/bin/env python3
from pathlib import Path
import gzip,pickle,hashlib,itertools,math
import pandas as pd,numpy as np,lightgbm as lgb
from run_oos import FEATURES,MODEL_SHA,A_THR,S_THR,read_csv,update_state_from_race,build_feature_frame,pl_top4
ROOT=Path(__file__).resolve().parent; SRC=ROOT/'source'; MODEL=ROOT/'boatrace_strength_v1_lgbm.txt'; HISTORY=ROOT/'history.pkl.gz'
h=hashlib.sha256(MODEL.read_bytes()).hexdigest(); print('MODEL_SHA',h,flush=True); assert h==MODEL_SHA
booster=lgb.Booster(model_file=str(MODEL)); assert booster.feature_name()==FEATURES
with gzip.open(HISTORY,'rb') as f: state=pickle.load(f)
state['motorhist']={};state['eboathist']={};state['recent_st']={int(k):list(v) for k,v in state['recent_st'].items()};state['recent_finish']={int(k):list(v) for k,v in state['recent_finish'].items()}
for dt in pd.date_range('2026-01-01','2026-08-27',freq='D'):
 y,m,d=dt.strftime('%Y'),dt.strftime('%m'),dt.strftime('%d'); cards=read_csv(SRC/f'data/programs/race_cards/{y}/{m}/{d}.csv'); results=read_csv(SRC/f'data/results/realtime/{y}/{m}/{d}.csv')
 if cards.empty or results.empty: continue
 cdict={str(r['レースコード']):r for _,r in cards.iterrows()}; rdict={str(r['レースコード']):r for _,r in results.iterrows()}; keys=[]
 for code in set(cdict)&set(rdict):
  if len(code)>=12 and code.isdigit(): keys.append((int(code[-2:]),int(code[-4:-2]),code))
 keys.sort()
 for race_no,venue_code,code in keys: update_state_from_race(cdict[code],rdict[code],state,venue_code)
print('ROLLED_2026_TO_0827',flush=True)
code='202608281010'; cards=read_csv(SRC/'data/programs/race_cards/2026/08/28.csv'); cd={str(r['レースコード']):r for _,r in cards.iterrows()}; card=cd[code]
# user screenshot 三国10R: cloudy, wind 0m, wave 0cm; exhibits by boat 1..6
tkz={f'艇{i}_展示タイム':v for i,v in enumerate([6.64,6.72,6.78,6.73,6.74,6.72],1)}
sui={'風速(m)':0.0,'波の高さ(cm)':0.0,'風向':0,'天候':2}
x,err=build_feature_frame(card,tkz,sui,state,10,10); assert err is None,err
raw=booster.predict(x[FEATURES]); strength,topprob,combos,p4=pl_top4(raw,x.boat.to_numpy()); cls='S' if p4>=S_THR else 'A' if p4>=A_THR else 'skip'
print('P4',repr(p4),flush=True); print('CLASS',cls,flush=True); print('TOP4','|'.join(combos),flush=True); print('TOPPROB','|'.join(f'{v:.10f}' for v in topprob),flush=True); print('EXHIBIT','|'.join(f'{v:.2f}' for v in x.sort_values('boat')['exhibit'].to_numpy()),flush=True)
# odds if current preview row exists
od3=read_csv(SRC/'data/previews/od3/2026/08/28.csv'); od={str(r['レースコード']):r for _,r in od3.iterrows()}
if code in od:
 r=od[code]; perm=['-'.join(map(str,p)) for p in itertools.permutations(range(1,7),3)]; vals=[]
 for c in r.index:
  if c in ('レースコード','レース日','レース場','レース回','締切時刻','取得日時'): continue
  try: vals.append(float(r[c]))
  except: pass
 if len(vals)>=120:
  omap=dict(zip(perm,vals[:120])); odds=[omap.get(c,float('nan')) for c in combos]; print('ODDS','|'.join('nan' if not np.isfinite(v) else f'{v:g}' for v in odds),flush=True)
  if all(np.isfinite(v) and v>0 for v in odds): print('COMBINED',repr(1/sum(1/v for v in odds)),flush=True)
else: print('ODDS_NOT_YET',flush=True)
