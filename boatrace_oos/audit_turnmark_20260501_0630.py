#!/usr/bin/env python3
from pathlib import Path
import gzip,pickle,hashlib,requests
import numpy as np,pandas as pd,lightgbm as lgb
from run_oos import FEATURES,MODEL_SHA,A_THR,S_THR,read_csv,update_state_from_race,build_feature_frame,pl_top4,VENUE_CODE_TO_NAME
from screen_20260829 import scenario_patterns

ROOT=Path(__file__).resolve().parent
SRC=ROOT/'source'
MODEL=ROOT/'boatrace_strength_v1_lgbm.txt'
HISTORY=ROOT/'history.pkl.gz'

def D(df):
    return {str(r['レースコード']):r for _,r in df.iterrows()}

def roll_one_day(state,cards,res):
    cd,rd=D(cards),D(res); keys=[]
    for code in set(cd)&set(rd):
        if len(code)>=12 and code.isdigit():
            keys.append((int(code[-2:]),int(code[-4:-2]),code))
    for rno,v,code in sorted(keys):
        update_state_from_race(cd[code],rd[code],state,v)

def prescreen_one(card,state,booster,patterns):
    code=str(card['レースコード']); v=int(code[-4:-2]); rno=int(code[-2:])
    best=-1.; n=0
    for _,ex,wsp,wave,weather in patterns:
        tkz={f'艇{b}_展示タイム':ex[b-1] for b in range(1,7)}
        for wind in range(9):
            sui={'風速(m)':wsp,'波の高さ(cm)':wave,'風向':wind,'天候':weather}
            x,err=build_feature_frame(card,tkz,sui,state,v,rno)
            if err: continue
            raw=booster.predict(x[FEATURES]); _,_,_,p4=pl_top4(raw,x.boat.to_numpy())
            n+=1; best=max(best,p4)
            if best>=A_THR: return best,n,True
    return best,n,False

def load_turnmark(dt):
    ds=dt.strftime('%Y%m%d')
    u=f'https://turnmark.github.io/api/v1/2026/{ds}.json'
    r=requests.get(u,timeout=30); r.raise_for_status()
    return r.json()

def tri_odds(day_json, venue, race, combo):
    try:
        tri=day_json['programs']['stadiums'][str(venue)]['races'][str(race)]['odds']['trifecta']
        a,b,c=combo.split('-')
        v=tri[a][b][c]
        return float(v) if v is not None else np.nan
    except Exception:
        return np.nan

def summarize(df, mask, label, month):
    q=df[(df.month==month) & mask(df)].copy()
    n=len(q); hits=int(q.hit.sum()); stake=400*n; ret=float(q.loc[q.hit,'payout'].sum())
    roi=ret/stake*100 if stake else np.nan
    net=ret-stake
    return dict(month=month,label=label,purchases=n,hits=hits,hit_rate=(hits/n*100 if n else np.nan),stake=stake,ret=ret,net=net,roi=roi)

def main():
    h=hashlib.sha256(MODEL.read_bytes()).hexdigest()
    assert h==MODEL_SHA,(h,MODEL_SHA)
    print('MODEL_SHA',h,flush=True)
    booster=lgb.Booster(model_file=str(MODEL)); assert booster.feature_name()==FEATURES
    with gzip.open(HISTORY,'rb') as f: state=pickle.load(f)
    state['motorhist']={}; state['eboathist']={}
    state['recent_st']={int(k):list(v) for k,v in state['recent_st'].items()}
    state['recent_finish']={int(k):list(v) for k,v in state['recent_finish'].items()}

    for dt in pd.date_range('2026-01-01','2026-04-30',freq='D'):
        y,m,d=dt.strftime('%Y'),dt.strftime('%m'),dt.strftime('%d')
        roll_one_day(state,
            read_csv(SRC/f'data/programs/race_cards/{y}/{m}/{d}.csv'),
            read_csv(SRC/f'data/results/realtime/{y}/{m}/{d}.csv'))

    patterns=scenario_patterns(); rows=[]
    for dt in pd.date_range('2026-05-01','2026-06-30',freq='D'):
        ds=dt.strftime('%Y-%m-%d'); y,m,d=dt.strftime('%Y'),dt.strftime('%m'),dt.strftime('%d')
        base=SRC/'data'
        cards=read_csv(base/f'programs/race_cards/{y}/{m}/{d}.csv')
        tkz=read_csv(base/f'previews/tkz/{y}/{m}/{d}.csv')
        sui=read_csv(base/f'previews/sui/{y}/{m}/{d}.csv')
        res=read_csv(base/f'results/realtime/{y}/{m}/{d}.csv')
        pay=read_csv(base/f'results/payouts/{y}/{m}/{d}.csv')
        cd,td,sd,rd,pdct=map(D,[cards,tkz,sui,res,pay])
        tj=load_turnmark(dt)
        day_current=day_cand=0
        for code in sorted(set(cd)&set(td)&set(sd)&set(rd)&set(pdct)):
            if len(code)<12 or not code.isdigit(): continue
            v=int(code[-4:-2]); rno=int(code[-2:]); card=cd[code]
            x,err=build_feature_frame(card,td[code],sd[code],state,v,rno)
            if err: continue
            raw=booster.predict(x[FEATURES]); _,_,combos,p4=pl_top4(raw,x.boat.to_numpy())
            cls='S' if p4>=S_THR else 'A' if p4>=A_THR else 'skip'
            if cls=='skip': continue
            _,_,screen_keep=prescreen_one(card,state,booster,patterns)
            if not screen_keep: continue
            ovs=[tri_odds(tj,v,rno,c) for c in combos]
            if not all(np.isfinite(z) and z>0 for z in ovs): continue
            combined=1/sum(1/z for z in ovs)
            if combined<3.0: continue
            rr=rd[code]
            actual=f"{int(float(rr['1着_艇番']))}-{int(float(rr['2着_艇番']))}-{int(float(rr['3着_艇番']))}"
            hit=actual in combos
            payout=float(pdct[code]['3連単_払戻金']) if pdct[code].get('3連単_払戻金','') not in ('',None) else 0.0
            cand=(rno<=9 and not (3.5<=combined<4.0))
            rows.append(dict(date=ds,month=dt.strftime('%Y-%m'),venue=VENUE_CODE_TO_NAME.get(v,str(v)),race=rno,code=code,cls=cls,p4=p4,combined=combined,hit=hit,payout=payout,candidate=cand,top4='|'.join(combos),actual=actual))
            day_current+=1
            if cand: day_cand+=1
        print('DAY',ds,'CURRENT',day_current,'CANDIDATE',day_cand,flush=True)
        roll_one_day(state,cards,res)

    df=pd.DataFrame(rows)
    df.to_csv('audit_turnmark_20260501_0630_detail.csv',index=False)
    stats=[]
    current=lambda x: pd.Series(True,index=x.index)
    candidate=lambda x: x.candidate.astype(bool)
    for month in ['2026-05','2026-06']:
        stats.append(summarize(df,current,'current',month))
        stats.append(summarize(df,candidate,'candidate',month))
    out=pd.DataFrame(stats)
    out.to_csv('audit_turnmark_20260501_0630_summary.csv',index=False)
    for _,r in out.iterrows():
        print('SUMMARY',r['month'],r['label'],'BUY',int(r['purchases']),'HITS',int(r['hits']),
              'HIT_RATE',f"{r['hit_rate']:.2f}",'STAKE',int(r['stake']),'RETURN',f"{r['ret']:.0f}",
              'NET',f"{r['net']:.0f}",'ROI',f"{r['roi']:.2f}",flush=True)

if __name__=='__main__': main()
