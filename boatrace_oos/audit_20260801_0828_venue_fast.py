#!/usr/bin/env python3
from pathlib import Path
import gzip,pickle,hashlib
import pandas as pd, numpy as np, lightgbm as lgb
from run_oos import FEATURES,MODEL_SHA,A_THR,S_THR,read_csv,update_state_from_race,build_feature_frame,pl_top4,VENUE_CODE_TO_NAME
ROOT=Path(__file__).resolve().parent; SRC=ROOT/'source'; MODEL=ROOT/'boatrace_strength_v1_lgbm.txt'; HISTORY=ROOT/'history.pkl.gz'
def D(df): return {str(r['レースコード']):r for _,r in df.iterrows()}
def advance(state,cards,res):
 cd,rd=D(cards),D(res)
 for code in sorted(set(cd)&set(rd),key=lambda x:(int(x[-2:]),int(x[-4:-2]))): update_state_from_race(cd[code],rd[code],state,int(code[-4:-2]))
def main():
 assert hashlib.sha256(MODEL.read_bytes()).hexdigest()==MODEL_SHA
 booster=lgb.Booster(model_file=str(MODEL)); assert booster.feature_name()==FEATURES
 with gzip.open(HISTORY,'rb') as f: state=pickle.load(f)
 state['motorhist']={}; state['eboathist']={}; state['recent_st']={int(k):list(v) for k,v in state['recent_st'].items()}; state['recent_finish']={int(k):list(v) for k,v in state['recent_finish'].items()}
 for dt in pd.date_range('2026-01-01','2026-07-31',freq='D'):
  y,m,d=dt.strftime('%Y'),dt.strftime('%m'),dt.strftime('%d'); c=read_csv(SRC/f'data/programs/race_cards/{y}/{m}/{d}.csv'); r=read_csv(SRC/f'data/results/realtime/{y}/{m}/{d}.csv')
  if not c.empty and not r.empty: advance(state,c,r)
 rows=[]
 for dt in pd.date_range('2026-08-01','2026-08-28',freq='D'):
  y,m,d=dt.strftime('%Y'),dt.strftime('%m'),dt.strftime('%d'); cards=read_csv(SRC/f'data/programs/race_cards/{y}/{m}/{d}.csv'); res=read_csv(SRC/f'data/results/realtime/{y}/{m}/{d}.csv'); tkz=read_csv(SRC/f'data/previews/tkz/{y}/{m}/{d}.csv'); sui=read_csv(SRC/f'data/previews/sui/{y}/{m}/{d}.csv')
  cd,td,sd=D(cards),D(tkz),D(sui); dayn=0
  for code in sorted(set(cd)&set(td)&set(sd)):
   if len(code)<12 or not code.isdigit(): continue
   v=int(code[-4:-2]); rno=int(code[-2:]); x,err=build_feature_frame(cd[code],td[code],sd[code],state,v,rno)
   if err: continue
   raw=booster.predict(x[FEATURES]); _,_,_,p4=pl_top4(raw,x.boat.to_numpy()); cls='S' if p4>=S_THR else 'A' if p4>=A_THR else 'skip'; venue=VENUE_CODE_TO_NAME.get(v,str(v)); rows.append((dt.strftime('%Y-%m-%d'),venue,rno,code,p4,cls)); dayn+=cls in ('A','S')
  print('DAY',dt.strftime('%Y-%m-%d'),'AS',dayn,'PREVIEW',len(set(cd)&set(td)&set(sd)),flush=True)
  if not cards.empty and not res.empty: advance(state,cards,res)
 df=pd.DataFrame(rows,columns=['date','venue','race_no','code','p4','class']); print('TOTAL_ROWS',len(df),'TOTAL_AS',int(df['class'].isin(['A','S']).sum()),flush=True)
 for i,(venue,g) in enumerate(sorted(df.groupby('venue'),key=lambda kv:(-int(kv[1]['class'].isin(['A','S']).sum()),kv[0])),1):
  a=int((g['class']=='A').sum()); s=int((g['class']=='S').sum()); n=a+s; total=len(g); print('VENUE',i,venue,'RACES',total,'A',a,'S',s,'AS',n,'RATE_ALL',f'{n/total*100:.2f}',flush=True)
 d28=df[df.date=='2026-08-28']; print('REG_0828_AS',int(d28['class'].isin(['A','S']).sum()),'EXPECTED',41,flush=True)
if __name__=='__main__': main()
