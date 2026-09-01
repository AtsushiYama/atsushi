#!/usr/bin/env python3
from pathlib import Path
import gzip,pickle,hashlib
import pandas as pd, lightgbm as lgb
from run_oos import FEATURES,MODEL_SHA,A_THR,S_THR,read_csv,update_state_from_race,build_feature_frame,pl_top4,VENUE_CODE_TO_NAME
ROOT=Path(__file__).resolve().parent; SRC=ROOT/'source'; MODEL=ROOT/'boatrace_strength_v1_lgbm.txt'; HISTORY=ROOT/'history.pkl.gz'
KEEP={
21:[1,2,3,5,8,9,11,12],23:[1,2,3,4,8,10,11,12],18:[2,3,4,9,10,12],8:[1,2,6,7,8,10,11,12],
11:[2,3,4,5,6,7,11],16:[1,2,4,5,9,10,11,12],6:[1,2,4,8,9,10,11,12],5:[2,3,5,8,9,10,11,12],
7:[1,2,3,5,6,7,9,10,11,12],20:[5,8,10],19:[5,7,8,10,11],4:[12]
}
h=hashlib.sha256(MODEL.read_bytes()).hexdigest(); assert h==MODEL_SHA,(h,MODEL_SHA)
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
 for rno,venue,code in keys: update_state_from_race(cdict[code],rdict[code],state,venue)

def idx(path):
 d=read_csv(path); return {str(r['レースコード']):r for _,r in d.iterrows()}
cards=idx(SRC/'data/programs/race_cards/2026/08/31.csv')
tkz=idx(SRC/'data/previews/tkz/2026/08/31.csv'); sui=idx(SRC/'data/previews/sui/2026/08/31.csv')
res=idx(SRC/'data/results/realtime/2026/08/31.csv'); pay=idx(SRC/'data/results/payouts/2026/08/31.csv')
rows=[]
for venue,races in KEEP.items():
 for rno in races:
  code=f'20260831{venue:02d}{rno:02d}'
  if code not in cards or code not in tkz or code not in sui or code not in res or code not in pay:
   rows.append(dict(code=code,venue=VENUE_CODE_TO_NAME.get(venue),rno=rno,error='missing_data')); continue
  x,err=build_feature_frame(cards[code],tkz[code],sui[code],state,venue,rno)
  if err:
   rows.append(dict(code=code,venue=VENUE_CODE_TO_NAME.get(venue),rno=rno,error=err)); continue
  raw=booster.predict(x[FEATURES]); strength,topprob,combos,p4=pl_top4(raw,x.boat.to_numpy()); cls='S' if p4>=S_THR else 'A' if p4>=A_THR else 'skip'
  actual=f"{res[code].get('1着_艇番')}-{res[code].get('2着_艇番')}-{res[code].get('3着_艇番')}"
  hit=actual in combos
  payout=float(str(pay[code].get('3連単_払戻金','0')).replace(',','') or 0)
  # 4 points x 100 yen = 400 yen stake; hit return is official 100-yen payout.
  ret=payout if hit else 0.0
  rows.append(dict(code=code,venue=VENUE_CODE_TO_NAME.get(venue),rno=rno,p4=p4,cls=cls,top4='|'.join(combos),actual=actual,hit=hit,payout100=payout,stake_all400=400,ret_all=ret,error=''))
df=pd.DataFrame(rows)
df.to_csv('eval_20260831.csv',index=False)
ok=df[df.error.eq('')].copy(); asdf=ok[ok.cls.isin(['A','S'])].copy()
def summ(label,d):
 n=len(d); h=int(d.hit.sum()); stake=400*n; ret=float(d.ret_all.sum());
 print(label,'N',n,'HITS',h,'HIT_RATE',f'{h/n*100:.3f}' if n else 'NA','STAKE',stake,'RETURN',int(ret),'ROI',f'{ret/stake*100:.3f}' if stake else 'NA','NET',int(ret-stake),flush=True)
print('MODEL_SHA',h,flush=True); print('EXTRACTED_EXPECTED',sum(len(v) for v in KEEP.values()),'VALID',len(ok),'MISSING',len(df)-len(ok),flush=True)
summ('ALL_EXTRACTED_TOP4',ok)
summ('AFTER_STRENGTH_AS',asdf)
print('CLASS_COUNTS',ok.cls.value_counts().to_dict(),flush=True)
print('AS_HITS')
for r in asdf[asdf.hit].sort_values(['venue','rno']).itertuples(): print(r.venue,r.rno,r.cls,f'P4={r.p4:.6f}',r.actual,int(r.payout100),flush=True)
