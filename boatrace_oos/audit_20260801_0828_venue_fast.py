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
def fnum(x):
 try: return float(str(x).replace(',','').strip())
 except: return np.nan
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
  y,m,d=dt.strftime('%Y'),dt.strftime('%m'),dt.strftime('%d')
  cards=read_csv(SRC/f'data/programs/race_cards/{y}/{m}/{d}.csv'); res=read_csv(SRC/f'data/results/realtime/{y}/{m}/{d}.csv'); tkz=read_csv(SRC/f'data/previews/tkz/{y}/{m}/{d}.csv'); sui=read_csv(SRC/f'data/previews/sui/{y}/{m}/{d}.csv'); od3=read_csv(SRC/f'data/previews/od3/{y}/{m}/{d}.csv'); pay=read_csv(SRC/f'data/results/payouts/{y}/{m}/{d}.csv')
  cd,rd,td,sd,od,pdct=D(cards),D(res),D(tkz),D(sui),D(od3),D(pay); dayn=0; buyday=0
  for code in sorted(set(cd)&set(td)&set(sd)&set(rd)):
   if len(code)<12 or not code.isdigit(): continue
   v=int(code[-4:-2]); rno=int(code[-2:]); x,err=build_feature_frame(cd[code],td[code],sd[code],state,v,rno)
   if err: continue
   raw=booster.predict(x[FEATURES]); _,_,combos,p4=pl_top4(raw,x.boat.to_numpy()); cls='S' if p4>=S_THR else 'A' if p4>=A_THR else 'skip'; venue=VENUE_CODE_TO_NAME.get(v,str(v))
   rr=rd[code]
   try: actual=f"{int(float(rr['1着_艇番']))}-{int(float(rr['2着_艇番']))}-{int(float(rr['3着_艇番']))}"
   except: actual=''
   hit=actual in combos if actual else False
   odds=[]
   if code in od:
    for c in combos: odds.append(fnum(od[code].get('3連単_'+c,np.nan)))
   combined=1.0/sum(1.0/z for z in odds) if len(odds)==4 and all(np.isfinite(z) and z>0 for z in odds) else np.nan
   buy=cls in ('A','S') and np.isfinite(combined) and combined>=3.0
   payout=0.0
   if hit and code in pdct:
    payout=fnum(pdct[code].get('3連単_払戻金',0)); payout=0.0 if not np.isfinite(payout) else payout
   ret=payout if buy and hit else 0.0
   rows.append((dt.strftime('%Y-%m-%d'),venue,rno,code,p4,cls,actual,hit,'|'.join(combos),combined,buy,payout,ret)); dayn+=cls in ('A','S'); buyday+=buy
  print('DAY',dt.strftime('%Y-%m-%d'),'AS',dayn,'BUY3',buyday,'PREVIEW_RESULT',len(set(cd)&set(td)&set(sd)&set(rd)),flush=True)
  if not cards.empty and not res.empty: advance(state,cards,res)
 df=pd.DataFrame(rows,columns=['date','venue','race_no','code','p4','class','actual','hit','top4','combined','buy3','payout','ret']); print('TOTAL_ROWS',len(df),'TOTAL_AS',int(df['class'].isin(['A','S']).sum()),'TOTAL_BUY3',int(df.buy3.sum()),flush=True)
 for venue,g in df.groupby('venue'):
  out=[]
  for cls in ['A','S','AS']:
   q=g[g.buy3 & (g['class'].isin(['A','S']) if cls=='AS' else (g['class']==cls))]
   n=len(q); h=int(q.hit.sum()); stake=400*n; ret=float(q.ret.sum()); roi=100*ret/stake if stake else np.nan; profit=ret-stake
   out.extend([n,h,(100*h/n if n else np.nan),stake,ret,profit,roi])
  print('VENUEROI',venue,
        'A_N',out[0],'A_H',out[1],'A_HR',f'{out[2]:.2f}' if np.isfinite(out[2]) else 'NA','A_STAKE',out[3],'A_RET',f'{out[4]:.0f}','A_PROFIT',f'{out[5]:.0f}','A_ROI',f'{out[6]:.2f}' if np.isfinite(out[6]) else 'NA',
        'S_N',out[7],'S_H',out[8],'S_HR',f'{out[9]:.2f}' if np.isfinite(out[9]) else 'NA','S_STAKE',out[10],'S_RET',f'{out[11]:.0f}','S_PROFIT',f'{out[12]:.0f}','S_ROI',f'{out[13]:.2f}' if np.isfinite(out[13]) else 'NA',
        'AS_N',out[14],'AS_H',out[15],'AS_HR',f'{out[16]:.2f}' if np.isfinite(out[16]) else 'NA','AS_STAKE',out[17],'AS_RET',f'{out[18]:.0f}','AS_PROFIT',f'{out[19]:.0f}','AS_ROI',f'{out[20]:.2f}' if np.isfinite(out[20]) else 'NA',flush=True)
 for cls in ['A','S','AS']:
  q=df[df.buy3 & (df['class'].isin(['A','S']) if cls=='AS' else (df['class']==cls))]; n=len(q); h=int(q.hit.sum()); stake=400*n; ret=float(q.ret.sum()); print('OVERALL_ROI',cls,'N',n,'H',h,'HR',f'{100*h/n:.2f}' if n else 'NA','STAKE',stake,'RET',f'{ret:.0f}','PROFIT',f'{ret-stake:.0f}','ROI',f'{100*ret/stake:.2f}' if stake else 'NA',flush=True)
 d28=df[df.date=='2026-08-28']; print('REG_0828_AS',int(d28['class'].isin(['A','S']).sum()),'EXPECTED',41,'BUY3',int(d28.buy3.sum()),flush=True)
if __name__=='__main__': main()
