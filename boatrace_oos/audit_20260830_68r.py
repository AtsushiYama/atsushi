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
for dt in pd.date_range('2026-01-01','2026-08-29',freq='D'):
 y,m,d=dt.strftime('%Y'),dt.strftime('%m'),dt.strftime('%d'); cards=read_csv(SRC/f'data/programs/race_cards/{y}/{m}/{d}.csv'); res=read_csv(SRC/f'data/results/realtime/{y}/{m}/{d}.csv')
 if cards.empty or res.empty: continue
 c={str(r['レースコード']):r for _,r in cards.iterrows()}; rr={str(r['レースコード']):r for _,r in res.iterrows()}; keys=[]
 for code in set(c)&set(rr):
  if len(code)>=12 and code.isdigit(): keys.append((int(code[-2:]),int(code[-4:-2]),code))
 for race_no,venue_code,code in sorted(keys): update_state_from_race(c[code],rr[code],state,venue_code)
base=SRC/'data'; cards=read_csv(base/'programs/race_cards/2026/08/30.csv'); tkz=read_csv(base/'previews/tkz/2026/08/30.csv'); sui=read_csv(base/'previews/sui/2026/08/30.csv'); od3=read_csv(base/'previews/od3/2026/08/30.csv'); res=read_csv(base/'results/realtime/2026/08/30.csv'); pay=read_csv(base/'results/payouts/2026/08/30.csv')
D=lambda df:{str(r['レースコード']):r for _,r in df.iterrows()}; cd,td,sd,od,rd,pdct=map(D,[cards,tkz,sui,od3,res,pay])
R=[('桐生',1,6),('桐生',1,12),('多摩川',5,3),('多摩川',5,9),('多摩川',5,11),('多摩川',5,12),('浜名湖',6,4),('浜名湖',6,8),('浜名湖',6,9),('浜名湖',6,11),('蒲郡',7,1),('蒲郡',7,6),('蒲郡',7,7),('蒲郡',7,8),('蒲郡',7,9),('蒲郡',7,10),('蒲郡',7,11),('蒲郡',7,12),('常滑',8,1),('常滑',8,2),('常滑',8,3),('常滑',8,6),('常滑',8,7),('常滑',8,8),('常滑',8,12),('津',9,5),('津',9,8),('津',9,11),('津',9,12),('びわこ',11,3),('びわこ',11,4),('びわこ',11,5),('びわこ',11,6),('びわこ',11,10),('びわこ',11,11),('鳴門',14,1),('鳴門',14,11),('児島',16,1),('児島',16,2),('児島',16,8),('児島',16,9),('児島',16,11),('宮島',17,1),('宮島',17,2),('宮島',17,3),('宮島',17,5),('宮島',17,7),('宮島',17,10),('宮島',17,11),('宮島',17,12),('下関',19,4),('下関',19,5),('下関',19,7),('下関',19,8),('下関',19,9),('下関',19,10),('下関',19,11),('芦屋',21,1),('芦屋',21,2),('芦屋',21,3),('芦屋',21,4),('芦屋',21,5),('芦屋',21,11),('唐津',23,2),('唐津',23,3),('唐津',23,10),('唐津',23,11),('唐津',23,12)]
perm=['-'.join(map(str,p)) for p in itertools.permutations(range(1,7),3)]; meta={'レースコード','レース日','レース場','レース回','締切時刻','取得日時'}
rows=[]
for venue,v,rno in R:
 code=f'20260830{v:02d}{rno:02d}'
 if not all(code in z for z in [cd,td,sd,rd,pdct]): print('MISSING',venue,rno,[code in z for z in [cd,td,sd,rd,pdct]],flush=True); continue
 x,err=build_feature_frame(cd[code],td[code],sd[code],state,v,rno)
 if err: print('ERR',venue,rno,err,flush=True); continue
 raw=booster.predict(x[FEATURES]); _,_,combos,p4=pl_top4(raw,x.boat.to_numpy()); cls='S' if p4>=S_THR else 'A' if p4>=A_THR else 'skip'
 rr=rd[code]; actual=f"{int(rr['1着_艇番'])}-{int(rr['2着_艇番'])}-{int(rr['3着_艇番'])}"; hit=actual in combos
 payout=float(pdct[code]['3連単_払戻金'] or 0)
 comb=np.nan
 if code in od:
  oo=od[code]; vals=[]
  for c in oo.index:
   if c in meta: continue
   try: vals.append(float(oo[c]))
   except: pass
  omap=dict(zip(perm,vals[:120])); ovs=[omap.get(c,np.nan) for c in combos]
  if all(np.isfinite(z) and z>0 for z in ovs): comb=1/sum(1/z for z in ovs)
 eligible=cls!='skip' and np.isfinite(comb) and comb>=3.0
 ret=payout if hit else 0.0
 rows.append(dict(venue=venue,race=rno,p4=p4,cls=cls,top4='|'.join(combos),actual=actual,hit=hit,payout=payout,combined=comb,eligible=eligible,ret=ret))
 print('ROW',venue,rno,f'{p4:.6f}',cls,'|'.join(combos),actual,'HIT' if hit else 'MISS',f'PAY={payout:.0f}',f'COMB={comb:.3f}' if np.isfinite(comb) else 'COMB=NA','BUY' if eligible else 'NO',flush=True)
df=pd.DataFrame(rows); print('COUNT',len(df),df.cls.value_counts().to_dict(),flush=True)
for label,mask in [('AS_ALL',df.cls!='skip'),('A_ONLY',df.cls=='A'),('S_ONLY',df.cls=='S'),('ODDS3',df.eligible)]:
 q=df[mask]; n=len(q); hits=int(q.hit.sum()); stake=400*n; ret=float(q.ret.sum()); roi=ret/stake*100 if stake else np.nan
 print('SUMMARY',label,'N',n,'HITS',hits,'HITRATE',f'{hits/n*100:.2f}' if n else 'NA','STAKE',stake,'RETURN',f'{ret:.0f}','PROFIT',f'{ret-stake:.0f}','ROI',f'{roi:.2f}' if n else 'NA',flush=True)
df.to_csv('audit_20260830_68r.csv',index=False)
