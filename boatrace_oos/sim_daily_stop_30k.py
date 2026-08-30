#!/usr/bin/env python3
from pathlib import Path
import pandas as pd, numpy as np
from audit_daily_20260801_0830 import read_csv,D,SRC

df=pd.read_csv('audit_daily_20260801_0830_detail.csv')
out=[]
for ds,g in df[df.eligible==True].groupby('date'):
    dt=pd.Timestamp(ds); y,m,d=dt.strftime('%Y'),dt.strftime('%m'),dt.strftime('%d')
    cards=read_csv(SRC/f'data/programs/race_cards/{y}/{m}/{d}.csv'); cd=D(cards)
    rows=[]
    for r in g.itertuples():
        card=cd.get(str(r.code),{})
        cutoff=str(card.get('締切時刻','99:99'))
        rows.append((cutoff,int(r.code),r))
    rows.sort(key=lambda x:(x[0],x[1]))
    cum=0; stake=0; ret=0; buys=0; hits=0; stopped=False; stop_after=''
    for cutoff,code,r in rows:
        if cum>=30000:
            stopped=True; break
        buys+=1; stake+=4000
        rr=float(r.payout)*10 if bool(r.hit) else 0.0
        if bool(r.hit): hits+=1
        ret+=rr; cum += rr-4000
        stop_after=f'{code}:{cutoff}'
    roi=ret/stake*100 if stake else np.nan
    out.append(dict(date=ds,buys=buys,hits=hits,stake=stake,ret=ret,profit=ret-stake,roi=roi,stopped=stopped,stop_after=stop_after,available=len(rows)))
    print('DAY',ds,'AVAILABLE',len(rows),'BUY',buys,'HITS',hits,'STAKE',stake,'RETURN',int(ret),'PROFIT',int(ret-stake),'ROI',f'{roi:.2f}','STOP',stopped,'AFTER',stop_after,flush=True)
res=pd.DataFrame(out)
res.to_csv('sim_daily_stop_30k.csv',index=False)
print('TOTAL','BUY',int(res.buys.sum()),'HITS',int(res.hits.sum()),'STAKE',int(res.stake.sum()),'RETURN',int(res.ret.sum()),'PROFIT',int(res.profit.sum()),'ROI',f'{res.ret.sum()/res.stake.sum()*100:.2f}',flush=True)
print('STOP_DAYS',int(res.stopped.sum()),','.join(res.loc[res.stopped,'date'].tolist()),flush=True)
