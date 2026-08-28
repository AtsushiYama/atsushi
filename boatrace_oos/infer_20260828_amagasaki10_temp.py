#!/usr/bin/env python3
from pathlib import Path
import gzip,pickle,itertools,hashlib
import numpy as np,pandas as pd,lightgbm as lgb
from run_oos import FEATURES,MODEL_SHA,A_THR,S_THR,read_csv,update_state_from_race,build_feature_frame,pl_top4
ROOT=Path(__file__).resolve().parent; SRC=ROOT/'source'; MODEL=ROOT/'boatrace_strength_v1_lgbm.txt'; HISTORY=ROOT/'history.pkl.gz'
h=hashlib.sha256(MODEL.read_bytes()).hexdigest(); print('MODEL_SHA',h,flush=True); assert h==MODEL_SHA
booster=lgb.Booster(model_file=str(MODEL)); assert booster.feature_name()==FEATURES
with gzip.open(HISTORY,'rb') as f: state=pickle.load(f)
state['motorhist']={}; state['eboathist']={}; state['recent_st']={int(k):list(v) for k,v in state['recent_st'].items()}; state['recent_finish']={int(k):list(v) for k,v in state['recent_finish'].items()}
for dt in pd.date_range('2026-01-01','2026-08-27',freq='D'):
 y,m,d=dt.strftime('%Y'),dt.strftime('%m'),dt.strftime('%d'); cards=read_csv(SRC/f'data/programs/race_cards/{y}/{m}/{d}.csv'); results=read_csv(SRC/f'data/results/realtime/{y}/{m}/{d}.csv')
 if cards.empty or results.empty: continue
 cdict={str(r['レースコード']):r for _,r in cards.iterrows()}; rdict={str(r['レースコード']):r for _,r in results.iterrows()}; keys=[]
 for code in set(cdict)&set(rdict):
  if len(code)>=12 and code.isdigit(): keys.append((int(code[-2:]),int(code[-4:-2]),code))
 keys.sort()
 for race_no,venue_code,code in keys: update_state_from_race(cdict[code],rdict[code],state,venue_code)
print('ROLLED_2026_TO_0827',flush=True)
cards=read_csv(SRC/'data/programs/race_cards/2026/08/28.csv'); tkz=read_csv(SRC/'data/previews/tkz/2026/08/28.csv'); sui=read_csv(SRC/'data/previews/sui/2026/08/28.csv'); od3=read_csv(SRC/'data/previews/od3/2026/08/28.csv')
cd={str(r['レースコード']):r for _,r in cards.iterrows()}; td={str(r['レースコード']):r for _,r in tkz.iterrows()}; sd={str(r['レースコード']):r for _,r in sui.iterrows()}; od={str(r['レースコード']):r for _,r in od3.iterrows()}
code='202608281505'; card=cd[code]; exh=[6.68,6.72,6.80,6.90,6.83,6.83]
manual_tkz={f'艇{i}_展示タイム':v for i,v in enumerate(exh,1)}
print('CASE MARUGAME5',code,flush=True); print('PRESENT',code in cd,code in td,code in sd,code in od,flush=True)
if code in td and code in sd:
 raw_tkz=td[code]; raw_sui_list=[sd[code]]; print('USING_OFFICIAL_PREVIEW',flush=True)
else:
 raw_tkz=manual_tkz; raw_sui_list=[{'風速(m)':1.0,'波の高さ(cm)':1.0,'風向':w,'天候':2} for w in range(1,9)]; print('USING_SCREENSHOT_PREVIEW',flush=True)
results=[]
for raw_sui in raw_sui_list:
 x,err=build_feature_frame(card,raw_tkz,raw_sui,state,15,5); assert err is None,err
 raw=booster.predict(x[FEATURES]); strength,topprob,combos,p4=pl_top4(raw,x.boat.to_numpy()); cls='S' if p4>=S_THR else 'A' if p4>=A_THR else 'skip'; results.append((raw_sui,x,topprob,combos,p4,cls))
 print('WIND_INPUT',raw_sui.get('風向'),'P4',repr(p4),'CLASS',cls,'TOP4','|'.join(combos),'TOPPROB','|'.join(f'{v:.10f}' for v in topprob),flush=True)
chosen=results[0] if len(results)==1 else results[4]
raw_sui,x,topprob,combos,p4,cls=chosen
print('ACTUAL_P4',repr(p4),'ACTUAL_CLASS',cls,'TOP4','|'.join(combos),'TOPPROB','|'.join(f'{v:.10f}' for v in topprob),flush=True)
print('EXHIBIT','|'.join(f'{v:.2f}' for v in x.sort_values('boat')['exhibit'].to_numpy()),flush=True)
print('WIND_SPEED',x['wind_speed'].iloc[0],'WAVE',x['wave'].iloc[0],'WEATHER_CODE',x['weather'].iloc[0],'WIND_DIR_CODE',x['wind_dir'].iloc[0],flush=True)
perm=['-'.join(map(str,p)) for p in itertools.permutations(range(1,7),3)]
if code in od:
 r=od[code]; vals=[]
 for c in r.index:
  if c in ('レースコード','レース日','レース場','レース回','締切時刻','取得日時'): continue
  try: vals.append(float(r[c]))
  except: pass
 omap=dict(zip(perm,vals[:120])); odds=[omap.get(c,float('nan')) for c in combos]
 print('ODDS','|'.join('nan' if not np.isfinite(v) else f'{v:g}' for v in odds),flush=True)
 if all(np.isfinite(v) and v>0 for v in odds): print('COMBINED',repr(1/sum(1/v for v in odds)),flush=True)
else: print('ODDS_NOT_YET',flush=True)
