#!/usr/bin/env python3
from pathlib import Path
import gzip,pickle,itertools,math
import numpy as np,pandas as pd,lightgbm as lgb
from run_oos import (FEATURES,CAT,MODEL_SHA,T,A_THR,S_THR,VENUE_CODE_TO_NAME,VMAP,WIND_TO_TRAIN,WEATHER_TO_TRAIN,read_csv,update_state_from_race,build_feature_frame,pl_top4)
ROOT=Path(__file__).resolve().parent
SRC=ROOT/'source'
MODEL=ROOT/'boatrace_strength_v1_lgbm.txt'
HISTORY=ROOT/'history.pkl.gz'
import hashlib
h=hashlib.sha256(MODEL.read_bytes()).hexdigest(); print('MODEL_SHA',h,flush=True); assert h==MODEL_SHA
booster=lgb.Booster(model_file=str(MODEL)); assert booster.feature_name()==FEATURES
with gzip.open(HISTORY,'rb') as f: state=pickle.load(f)
state['motorhist']={};state['eboathist']={}
state['recent_st']={int(k):list(v) for k,v in state['recent_st'].items()};state['recent_finish']={int(k):list(v) for k,v in state['recent_finish'].items()}
for dt in pd.date_range('2026-01-01','2026-08-27',freq='D'):
    y,m,d=dt.strftime('%Y'),dt.strftime('%m'),dt.strftime('%d')
    cards=read_csv(SRC/f'data/programs/race_cards/{y}/{m}/{d}.csv');results=read_csv(SRC/f'data/results/realtime/{y}/{m}/{d}.csv')
    if cards.empty or results.empty: continue
    cdict={str(r['レースコード']):r for _,r in cards.iterrows()};rdict={str(r['レースコード']):r for _,r in results.iterrows()}
    keys=[]
    for code in set(cdict)&set(rdict):
        if len(code)>=12 and code.isdigit(): keys.append((int(code[-2:]),int(code[-4:-2]),code))
    keys.sort()
    for race_no,venue_code,code in keys: update_state_from_race(cdict[code],rdict[code],state,venue_code)
print('ROLLED_2026_TO_0827',flush=True)
code='202608282308'; cards=read_csv(SRC/'data/programs/race_cards/2026/08/28.csv');tkz=read_csv(SRC/'data/previews/tkz/2026/08/28.csv');sui=read_csv(SRC/'data/previews/sui/2026/08/28.csv')
cd={str(r['レースコード']):r for _,r in cards.iterrows()};td={str(r['レースコード']):r for _,r in tkz.iterrows()};sd={str(r['レースコード']):r for _,r in sui.iterrows()}
assert code in cd and code in td and code in sd,(code in cd,code in td,code in sd)
x,err=build_feature_frame(cd[code],td[code],sd[code],state,23,8); assert err is None,err
raw=booster.predict(x[FEATURES]);strength,topprob,combos,p4=pl_top4(raw,x.boat.to_numpy());cls='S' if p4>=S_THR else 'A' if p4>=A_THR else 'skip'
print('P4',repr(p4),flush=True);print('CLASS',cls,flush=True);print('TOP4','|'.join(combos),flush=True);print('TOPPROB','|'.join(f'{v:.10f}' for v in topprob),flush=True);print('STRENGTHS','|'.join(f'{v:.8f}' for v in strength),flush=True)
