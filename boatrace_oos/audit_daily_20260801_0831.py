#!/usr/bin/env python3
from pathlib import Path
import gzip,pickle,itertools,hashlib
import numpy as np,pandas as pd,lightgbm as lgb
from run_oos import FEATURES,MODEL_SHA,A_THR,S_THR,read_csv,update_state_from_race,build_feature_frame,pl_top4,VENUE_CODE_TO_NAME
from screen_20260829 import scenario_patterns
ROOT=Path(__file__).resolve().parent; SRC=ROOT/'source'; MODEL=ROOT/'boatrace_strength_v1_lgbm.txt'; HISTORY=ROOT/'history.pkl.gz'
META={'レースコード','レース日','レース場','レース回','締切時刻','取得日時'}
PERM=['-'.join(map(str,p)) for p in itertools.permutations(range(1,7),3)]
def D(df): return {str(r['レースコード']):r for _,r in df.iterrows()}
def roll_one_day(state,cards,res):
    cd,rd=D(cards),D(res); keys=[]
    for code in set(cd)&set(rd):
        if len(code)>=12 and code.isdigit(): keys.append((int(code[-2:]),int(code[-4:-2]),code))
    for rno,v,code in sorted(keys): update_state_from_race(cd[code],rd[code],state,v)
def prescreen_one(card,state,booster,patterns):
    code=str(card['レースコード']); v=int(code[-4:-2]); rno=int(code[-2:]); best=-1.; n=0
    for _,ex,wsp,wave,weather in patterns:
        tkz={f'艇{b}_展示タイム':ex[b-1] for b in range(1,7)}
        for wind in range(9):
            sui={'風速(m)':wsp,'波の高さ(cm)':wave,'風向':wind,'天候':weather}
            x,err=build_feature_frame(card,tkz,sui,state,v,rno)
            if err: continue
            raw=booster.predict(x[FEATURES]); _,_,_,p4=pl_top4(raw,x.boat.to_numpy()); n+=1; best=max(best,p4)
            if best>=A_THR: return best,n,True
    return best,n,False
def odds_map(row):
    vals=[]
    for c in row.index:
        if c in META: continue
        try: vals.append(float(row[c]))
        except: pass
    return dict(zip(PERM,vals[:120]))
def main():
    h=hashlib.sha256(MODEL.read_bytes()).hexdigest(); assert h==MODEL_SHA,(h,MODEL_SHA); print('MODEL_SHA',h,flush=True)
    booster=lgb.Booster(model_file=str(MODEL)); assert booster.feature_name()==FEATURES
    with gzip.open(HISTORY,'rb') as f: state=pickle.load(f)
    state['motorhist']={}; state['eboathist']={}; state['recent_st']={int(k):list(v) for k,v in state['recent_st'].items()}; state['recent_finish']={int(k):list(v) for k,v in state['recent_finish'].items()}
    for dt in pd.date_range('2026-01-01','2026-07-31',freq='D'):
        y,m,d=dt.strftime('%Y'),dt.strftime('%m'),dt.strftime('%d'); roll_one_day(state,read_csv(SRC/f'data/programs/race_cards/{y}/{m}/{d}.csv'),read_csv(SRC/f'data/results/realtime/{y}/{m}/{d}.csv'))
    patterns=scenario_patterns(); rows=[]; daily=[]
    for dt in pd.date_range('2026-08-01','2026-09-03',freq='D'):
        ds=dt.strftime('%Y-%m-%d'); y,m,d=dt.strftime('%Y'),dt.strftime('%m'),dt.strftime('%d'); base=SRC/'data'
        cards=read_csv(base/f'programs/race_cards/{y}/{m}/{d}.csv'); tkz=read_csv(base/f'previews/tkz/{y}/{m}/{d}.csv'); sui=read_csv(base/f'previews/sui/{y}/{m}/{d}.csv'); od3=read_csv(base/f'previews/od3/{y}/{m}/{d}.csv'); res=read_csv(base/f'results/realtime/{y}/{m}/{d}.csv'); pay=read_csv(base/f'results/payouts/{y}/{m}/{d}.csv')
        cd,td,sd,od,rd,pdct=map(D,[cards,tkz,sui,od3,res,pay]); dayrows=[]; errors=0
        for code in sorted(set(cd)&set(td)&set(sd)&set(rd)&set(pdct)):
            if len(code)<12 or not code.isdigit(): continue
            v=int(code[-4:-2]); rno=int(code[-2:]); venue=VENUE_CODE_TO_NAME.get(v,str(v)); card=cd[code]
            x,err=build_feature_frame(card,td[code],sd[code],state,v,rno)
            if err: errors+=1; continue
            raw=booster.predict(x[FEATURES]); _,probs,combos,p4=pl_top4(raw,x.boat.to_numpy()); cls='S' if p4>=S_THR else 'A' if p4>=A_THR else 'skip'
            if cls=='skip': continue
            proxy,n,screen_keep=prescreen_one(card,state,booster,patterns)
            rr=rd[code]; actual=f"{int(float(rr['1着_艇番']))}-{int(float(rr['2着_艇番']))}-{int(float(rr['3着_艇番']))}"; hit=actual in combos
            payout=float(pdct[code]['3連単_払戻金']) if pdct[code].get('3連単_払戻金','') not in ('',None) else 0.0
            comb=np.nan
            if code in od:
                om=odds_map(od[code]); ovs=[om.get(c,np.nan) for c in combos]
                if all(np.isfinite(z) and z>0 for z in ovs): comb=1/sum(1/z for z in ovs)
            eligible=screen_keep and np.isfinite(comb) and comb>=3.0
            ret=payout if eligible and hit else 0.0; stake=400 if eligible else 0
            row=dict(date=ds,venue=venue,race=rno,code=code,p4=p4,cls=cls,proxy=proxy,screen_keep=screen_keep,top4='|'.join(combos),actual=actual,hit=hit,payout=payout,combined=comb,eligible=eligible,stake=stake,ret=ret)
            rows.append(row); dayrows.append(row)
        q=pd.DataFrame(dayrows)
        if len(q):
            elig=q[q.eligible]; n=len(elig); hits=int(elig.hit.sum()); stake=int(elig.stake.sum()); ret=float(elig.ret.sum()); roi=ret/stake*100 if stake else np.nan; sa=len(q); screened=int(q.screen_keep.sum()); fn=sa-screened
        else: n=hits=stake=0; ret=0.; roi=np.nan; sa=screened=fn=0
        daily.append(dict(date=ds,sa=sa,screened_sa=screened,screen_false_negatives=fn,purchases=n,hits=hits,stake=stake,ret=ret,roi=roi,errors=errors))
        print('DAY',ds,'SA',sa,'SCREENED_SA',screened,'FN',fn,'BUY',n,'HITS',hits,'STAKE',stake,'RETURN',f'{ret:.0f}','ROI',f'{roi:.2f}' if np.isfinite(roi) else 'NA','ERR',errors,flush=True)
        roll_one_day(state,cards,res)
    df=pd.DataFrame(rows); dd=pd.DataFrame(daily)
    df.to_csv('audit_daily_20260801_0903_detail.csv',index=False); dd.to_csv('audit_daily_20260801_0903_summary.csv',index=False)
    active=dd[dd.stake>0].copy(); loss=active[active.roi<100]
    print('ACTIVE_DAYS',len(active),'LOSS_DAYS',len(loss),flush=True)
    print('TOTAL','BUY',int(active.purchases.sum()),'HITS',int(active.hits.sum()),'HIT_RATE',f'{active.hits.sum()/active.purchases.sum()*100:.2f}' if active.purchases.sum() else 'NA','STAKE',int(active.stake.sum()),'RETURN',f'{active.ret.sum():.0f}','ROI',f'{active.ret.sum()/active.stake.sum()*100:.2f}' if active.stake.sum() else 'NA','NET',f'{active.ret.sum()-active.stake.sum():.0f}',flush=True)
if __name__=='__main__': main()
