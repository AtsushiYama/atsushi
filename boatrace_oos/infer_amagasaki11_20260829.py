#!/usr/bin/env python3
import copy,gzip,pickle,hashlib
import pandas as pd, numpy as np, lightgbm as lgb
from run_oos import *
from screen_20260829 import read_csv,D,roll_state,MODEL,SRC,MODEL_SHA,FEATURES,CAT
DATE='2026-08-29'; VENUE=13; RNO=11
EX=[6.80,6.80,6.78,6.74,6.79,6.84]
WIND_SPEED=3.0; WAVE=0.0; WEATHER=2

def advance(state,date):
 c=read_csv(SRC/f'data/programs/race_cards/{date[:4]}/{date[5:7]}/{date[8:10]}.csv'); r=read_csv(SRC/f'data/results/realtime/{date[:4]}/{date[5:7]}/{date[8:10]}.csv'); cd,rd=D(c),D(r)
 for code in sorted(set(cd)&set(rd),key=lambda x:(int(x[-2:]),int(x[-4:-2]))): update_state_from_race(cd[code],rd[code],state,int(code[-4:-2]))

def main():
 assert hashlib.sha256(MODEL.read_bytes()).hexdigest()==MODEL_SHA
 booster=lgb.Booster(model_file=str(MODEL)); assert booster.feature_name()==FEATURES
 with gzip.open('boatrace_oos/history.pkl.gz','rb') as f: base=pickle.load(f)
 state=roll_state(copy.deepcopy(base),'2026-08-26'); advance(state,'2026-08-27'); advance(state,'2026-08-28')
 cards=read_csv(SRC/'data/programs/race_cards/2026/08/29.csv'); card=D(cards)[f'2026082913{RNO:02d}']
 tkz={f'艇{i}_展示タイム':EX[i-1] for i in range(1,7)}
 for wc in range(9):
  sui={'風速(m)':WIND_SPEED,'波の高さ(cm)':WAVE,'風向':wc,'天候':WEATHER}
  x,err=build_feature_frame(card,tkz,sui,state,VENUE,RNO)
  if err: print('ERR',wc,err); continue
  for c in CAT:x[c]=x[c].astype('category')
  raw=booster.predict(x[FEATURES]); top4,_,_,p4=pl_top4(raw,x.boat.to_numpy())
  print('WIND_CODE',wc,'P4',f'{p4:.12f}','CLASS',classify_p4(p4),'TOP4',top4,'RAW',','.join(f'{v:.8f}' for v in raw))
if __name__=='__main__':main()
