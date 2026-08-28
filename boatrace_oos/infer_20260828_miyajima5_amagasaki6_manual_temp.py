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
cards=read_csv(SRC/'data/programs/race_cards/2026/08/28.csv'); cd={str(r['レースコード']):r for _,r in cards.iterrows()}
def calc(name,code,venue,rno,exhs,wind_speed,wave,weather,wind_codes):
 card=cd[code]; tkz={f'艇{i}_展示タイム':v for i,v in enumerate(exhs,1)}
 print('===',name,code,'===',flush=True)
 for wind_code in wind_codes:
  sui={'風速(m)':wind_speed,'波の高さ(cm)':wave,'風向':wind_code,'天候':weather}
  x,err=build_feature_frame(card,tkz,sui,state,venue,rno); assert err is None,err
  raw=booster.predict(x[FEATURES]); strength,topprob,combos,p4=pl_top4(raw,x.boat.to_numpy()); cls='S' if p4>=S_THR else 'A' if p4>=A_THR else 'skip'
  print('WIND',wind_code,'P4',repr(p4),'CLASS',cls,'TOP4','|'.join(combos),'TOPPROB','|'.join(f'{v:.10f}' for v in topprob),flush=True)
# User screenshots: 宮島5R cloudy, wind 2m, wave 2cm. Run all 8 directions for robustness.
calc('宮島5R','202608281705',17,5,[6.71,6.77,6.73,6.75,6.69,6.67],2.0,2.0,2,range(1,9))
# 尼崎6R cloudy, wind 0m, wave 0cm. Official zero-wind code uses 0.
calc('尼崎6R','202608281306',13,6,[6.85,6.90,6.85,6.82,6.92,6.87],0.0,0.0,2,[0])
