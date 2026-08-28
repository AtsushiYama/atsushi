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
 y,m,d=dt.strftime('%Y'),dt.strftime('%m'),dt.strftime('%d'); cards=read_csv(SRC/f'data/programs/race_cards/{y}/{m}/{d}.csv'); res=read_csv(SRC/f'data/results/realtime/{y}/{m}/{d}.csv')
 if cards.empty or res.empty: continue
 c={str(r['レースコード']):r for _,r in cards.iterrows()}; rr={str(r['レースコード']):r for _,r in res.iterrows()}; keys=[]
 for code in set(c)&set(rr):
  if len(code)>=12 and code.isdigit(): keys.append((int(code[-2:]),int(code[-4:-2]),code))
 for race_no,venue_code,code in sorted(keys): update_state_from_race(c[code],rr[code],state,venue_code)
print('ROLLED_2026_TO_0827',flush=True)
base=SRC/'data'; cards=read_csv(base/'programs/race_cards/2026/08/28.csv'); tkz=read_csv(base/'previews/tkz/2026/08/28.csv'); sui=read_csv(base/'previews/sui/2026/08/28.csv'); od3=read_csv(base/'previews/od3/2026/08/28.csv'); res=read_csv(base/'results/realtime/2026/08/28.csv'); pay=read_csv(base/'results/payouts/2026/08/28.csv')
D=lambda df:{str(r['レースコード']):r for _,r in df.iterrows()}; cd,td,sd,od,rd,pdct=map(D,[cards,tkz,sui,od3,res,pay])
R=[('唐津',23,8),('尼崎',13,5),('津',9,5),('唐津',23,9),('びわこ',11,5),('三国',10,9),('宮島',17,5),('尼崎',13,6),('びわこ',11,6),('三国',10,10),('びわこ',11,7),('尼崎',13,7),('宮島',17,7),('尼崎',13,8),('津',9,9),('びわこ',11,9),('多摩川',5,7),('尼崎',13,9),('平和島',4,7),('津',9,10),('尼崎',13,10),('宮島',17,10),('津',9,11),('びわこ',11,11),('多摩川',5,9),('蒲郡',7,2),('尼崎',13,11),('桐生',1,3),('宮島',17,11),('津',9,12),('びわこ',11,12),('丸亀',15,5),('蒲郡',7,5),('丸亀',15,6),('蒲郡',7,6),('丸亀',15,8)]
perm=['-'.join(map(str,p)) for p in itertools.permutations(range(1,7),3)]; meta={'レースコード','レース日','レース場','レース回','締切時刻','取得日時'}
rows=[]
for venue,v,rno in R:
 code=f'20260828{v:02d}{rno:02d}'
 if not all(code in z for z in [cd,td,sd,rd,pdct]): print('MISSING',venue,rno,code,[code in z for z in [cd,td,sd,rd,pdct]],flush=True); continue
 x,err=build_feature_frame(cd[code],td[code],[sd[code]],state,v,rno)
 if err: print('ERR',venue,rno,err,flush=True); continue
 raw=booster.predict(x[FEATURES]); strength,probs,combos,p4=pl_top4(raw,x.boat.to_numpy()); cls='S' if p4>=S_THR else 'A' if p4>=A_THR else 'skip'
 rr=rd[code]; actual=f"{int(rr['1着_艇番'])}-{int(rr['2着_艇番'])}-{int(rr['3着_艇番'])}"; hit=actual in combos
 pp=pdct[code]; payout=float(pp['3連単_払戻金']) if pd.notna(pp['3連単_払戻金']) else 0.0
 comb=np.nan
 if code in od:
  oo=od[code]; vals=[]
  for c in oo.index:
   if c in meta: continue
   try: vals.append(float(oo[c]))
   except: pass
  omap=dict(zip(perm,vals[:120])); odds=[omap.get(c,np.nan) for c in combos]
  if all(np.isfinite(z) and z>0 for z in odds): comb=1/sum(1/z for z in odds)
 eligible=cls!='skip' and np.isfinite(comb) and comb>=3.0
 ret=payout if hit else 0.0
 rows.append(dict(venue=venue,race=rno,code=code,p4=p4,cls=cls,top4='|'.join(combos),actual=actual,hit=hit,payout=payout,combined=comb,eligible=eligible,ret=ret))
 print('ROW',venue,rno,f'{p4:.9f}',cls,'|'.join(combos),actual,'HIT' if hit else 'MISS',f'PAY={payout:.0f}',f'COMB={comb:.3f}' if np.isfinite(comb) else 'COMB=NA','BUY' if eligible else 'NO',flush=True)
df=pd.DataFrame(rows)
print('COUNT',len(df),df.cls.value_counts().to_dict(),flush=True)
for label,mask in [('AS_ALL',df.cls!='skip'),('A_ONLY',df.cls=='A'),('S_ONLY',df.cls=='S'),('ODDS3',df.eligible)]:
 q=df[mask]; n=len(q); hits=int(q.hit.sum()); stake=400*n; ret=float(q.ret.sum()); roi=(ret/stake*100 if stake else np.nan); print('SUMMARY',label,'N',n,'HITS',hits,'HITRATE',f'{hits/n*100:.2f}' if n else 'NA','STAKE',stake,'RETURN',f'{ret:.0f}','PROFIT',f'{ret-stake:.0f}','ROI',f'{roi:.2f}' if n else 'NA',flush=True)
print('SKIP_HIT_REFERENCE',int(df[df.cls=='skip'].hit.sum()),'/',len(df[df.cls=='skip']),flush=True)
