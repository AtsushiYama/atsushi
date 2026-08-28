#!/usr/bin/env python3
from pathlib import Path
import gzip,pickle,hashlib
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
code='202608281009'; cards=read_csv(SRC/'data/programs/race_cards/2026/08/28.csv'); cd={str(r['レースコード']):r for _,r in cards.iterrows()}; card=cd[code]
# User screenshot: 三国9R, 12:10 water-surface info: cloudy, wind 1m, wave 1cm; exhibits below.
tkz={f'艇{i}_展示タイム':v for i,v in enumerate([6.78,6.67,6.75,6.77,6.82,6.79],1)}
for wind_code in range(1,9):
 sui={'風速(m)':1.0,'波の高さ(cm)':1.0,'風向':wind_code,'天候':2}
 x,err=build_feature_frame(card,tkz,sui,state,10,9); assert err is None,err
 raw=booster.predict(x[FEATURES]); strength,topprob,combos,p4=pl_top4(raw,x.boat.to_numpy()); cls='S' if p4>=S_THR else 'A' if p4>=A_THR else 'skip'
 print('WIND',wind_code,'P4',repr(p4),'CLASS',cls,'TOP4','|'.join(combos),'TOPPROB','|'.join(f'{v:.10f}' for v in topprob),flush=True)
