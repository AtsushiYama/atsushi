#!/usr/bin/env python3
from pathlib import Path
import gzip,pickle,hashlib
import pandas as pd, lightgbm as lgb
from run_oos import FEATURES,MODEL_SHA,A_THR,S_THR,read_csv,update_state_from_race,build_feature_frame,pl_top4
ROOT=Path(__file__).resolve().parent; SRC=ROOT/'source'; MODEL=ROOT/'boatrace_strength_v1_lgbm.txt'; HISTORY=ROOT/'history.pkl.gz'
h=hashlib.sha256(MODEL.read_bytes()).hexdigest(); print('MODEL_SHA',h,flush=True); assert h==MODEL_SHA
booster=lgb.Booster(model_file=str(MODEL)); assert booster.feature_name()==FEATURES
with gzip.open(HISTORY,'rb') as f: state=pickle.load(f)
state['motorhist']={}; state['eboathist']={}; state['recent_st']={int(k):list(v) for k,v in state['recent_st'].items()}; state['recent_finish']={int(k):list(v) for k,v in state['recent_finish'].items()}
for dt in pd.date_range('2026-01-01','2026-08-30',freq='D'):
 y,m,d=dt.strftime('%Y'),dt.strftime('%m'),dt.strftime('%d'); cards=read_csv(SRC/f'data/programs/race_cards/{y}/{m}/{d}.csv'); results=read_csv(SRC/f'data/results/realtime/{y}/{m}/{d}.csv')
 if cards.empty or results.empty: continue
 cdict={str(r['レースコード']):r for _,r in cards.iterrows()}; rdict={str(r['レースコード']):r for _,r in results.iterrows()}; keys=[]
 for code in set(cdict)&set(rdict):
  if len(code)>=12 and code.isdigit(): keys.append((int(code[-2:]),int(code[-4:-2]),code))
 keys.sort()
 for race_no,venue_code,code in keys: update_state_from_race(cdict[code],rdict[code],state,venue_code)
cards=read_csv(SRC/'data/programs/race_cards/2026/08/31.csv'); cd={str(r['レースコード']):r for _,r in cards.iterrows()}
code='202608311812'; venue=18; rno=12; card=cd[code]
tkz=pd.Series({'艇1_展示タイム':'6.97','艇2_展示タイム':'6.99','艇3_展示タイム':'7.03','艇4_展示タイム':'6.98','艇5_展示タイム':'7.00','艇6_展示タイム':'7.03'})
sui=pd.Series({'風速(m)':'3','波の高さ(cm)':'3','風向':'6','天候':'1'})
x,err=build_feature_frame(card,tkz,sui,state,venue,rno); assert err is None,err
raw=booster.predict(x[FEATURES]); strength,topprob,combos,p4=pl_top4(raw,x.boat.to_numpy()); cls='S' if p4>=S_THR else 'A' if p4>=A_THR else 'skip'
print('P4',repr(p4),flush=True); print('CLASS',cls,flush=True); print('TOP4','|'.join(combos),flush=True); print('TOPPROB','|'.join(f'{v:.10f}' for v in topprob),flush=True); print('STRENGTHS','|'.join(f'{v:.8f}' for v in strength),flush=True)
print('EXHIBIT','|'.join(f'{v:.2f}' for v in x.sort_values('boat')['exhibit'].to_numpy()),flush=True)
print('WIND_SPEED',x['wind_speed'].iloc[0],'WAVE',x['wave'].iloc[0],'WEATHER_CODE',x['weather'].iloc[0],'WIND_DIR_CODE',x['wind_dir'].iloc[0],flush=True)
print('THRESHOLDS','A',A_THR,'S',S_THR,flush=True)
