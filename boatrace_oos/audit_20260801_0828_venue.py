#!/usr/bin/env python3
from pathlib import Path
import gzip,pickle,hashlib,copy
import numpy as np,pandas as pd,lightgbm as lgb
from run_oos import FEATURES,MODEL_SHA,A_THR,S_THR,read_csv,update_state_from_race,build_feature_frame,pl_top4,VENUE_CODE_TO_NAME,CAT,WIND_TO_TRAIN,WEATHER_TO_TRAIN
ROOT=Path(__file__).resolve().parent; SRC=ROOT/'source'; MODEL=ROOT/'boatrace_strength_v1_lgbm.txt'; HISTORY=ROOT/'history.pkl.gz'; SEED=20260826

def D(df): return {str(r['レースコード']):r for _,r in df.iterrows()}

def advance_day(state,cards,res):
    cd,rd=D(cards),D(res)
    for code in sorted(set(cd)&set(rd),key=lambda x:(int(x[-2:]),int(x[-4:-2]))): update_state_from_race(cd[code],rd[code],state,int(code[-4:-2]))

def roll_to_jul31(state):
    state['motorhist']={}; state['eboathist']={}
    state['recent_st']={int(k):list(v) for k,v in state['recent_st'].items()}; state['recent_finish']={int(k):list(v) for k,v in state['recent_finish'].items()}
    for dt in pd.date_range('2026-01-01','2026-07-31',freq='D'):
        y,m,d=dt.strftime('%Y'),dt.strftime('%m'),dt.strftime('%d'); c=read_csv(SRC/f'data/programs/race_cards/{y}/{m}/{d}.csv'); r=read_csv(SRC/f'data/results/realtime/{y}/{m}/{d}.csv')
        if not c.empty and not r.empty: advance_day(state,c,r)
    return state

def scenario_patterns():
    pairs=[]
    for tkzp in sorted((SRC/'data/previews/tkz/2025').glob('*/*.csv')):
        rel=tkzp.relative_to(SRC/'data/previews/tkz/2025'); suip=SRC/'data/previews/sui/2025'/rel
        if not suip.exists(): continue
        tkz,sui=read_csv(tkzp),read_csv(suip)
        if tkz.empty or sui.empty: continue
        td,sd=D(tkz),D(sui)
        for code in sorted(set(td)&set(sd)):
            tr,sr=td[code],sd[code]
            try: ex=[float(tr[f'艇{b}_展示タイム']) for b in range(1,7)]; wsp=float(sr['風速(m)']); wave=float(sr['波の高さ(cm)']); weather=int(float(sr['天候']))
            except: continue
            if all(np.isfinite(ex)) and np.isfinite(wsp) and np.isfinite(wave): pairs.append((code,ex,wsp,wave,weather))
    rng=np.random.default_rng(SEED); idx=rng.choice(len(pairs),size=30,replace=False); out=[pairs[int(i)] for i in idx]
    print('PATTERN_CODES',','.join(x[0] for x in out),flush=True); return out

def screen_ultra(cards,state,booster,pats):
    scenario=[]
    for _,ex,wspd,wave,weather in pats:
        for wc in range(9): scenario.append((np.array(ex,float),float(wspd),float(wave),int(weather),wc))
    ans=[]
    for _,card in cards.iterrows():
        code=str(card['レースコード']); v=int(code[-4:-2]); rno=int(code[-2:]); venue=VENUE_CODE_TO_NAME.get(v,str(v)); ex0,wspd0,wave0,wea0,wc0=scenario[0]
        tkz0={f'艇{b}_展示タイム':ex0[b-1] for b in range(1,7)}; sui0={'風速(m)':wspd0,'波の高さ(cm)':wave0,'風向':wc0,'天候':wea0}
        base,err=build_feature_frame(card,tkz0,sui0,state,v,rno)
        if err: ans.append((venue,rno,code,-1.,False)); continue
        frames=[]
        for ex,wspd,wave,wea,wc in scenario:
            x=base.copy(); ranks=pd.Series(ex).rank(method='average',ascending=True).to_numpy(); x['weather']=WEATHER_TO_TRAIN.get(wea,999); x['wind_dir']=WIND_TO_TRAIN.get(wc,999); x['wind_speed']=wspd; x['wave']=wave; x['exhibit']=ex; x['exhibit_delta']=ex-ex.min(); x['exhibit_center']=ex-ex.mean(); x['exhibit_rank']=ranks; x['wind_boat']=x['wind_dir'].astype(int)*10+x['boat'].astype(int); x['venue_wind_boat']=x['venue'].astype(int)*1000+x['wind_dir'].astype(int)*10+x['boat'].astype(int); frames.append(x)
        z=pd.concat(frames,ignore_index=True)
        for c in CAT: z[c]=z[c].astype('category')
        raw=booster.predict(z[FEATURES]); best=-1.
        for j in range(0,len(raw),6):
            _,_,_,p4=pl_top4(raw[j:j+6],z.iloc[j:j+6].boat.to_numpy()); best=max(best,p4)
        ans.append((venue,rno,code,best,best>=A_THR))
    return ans

def actual_class(card,tkz,sui,state,booster,v,rno):
    x,err=build_feature_frame(card,tkz,sui,state,v,rno)
    if err: return None,np.nan
    raw=booster.predict(x[FEATURES]); _,_,_,p4=pl_top4(raw,x.boat.to_numpy()); cls='S' if p4>=S_THR else 'A' if p4>=A_THR else 'skip'; return cls,p4

def main():
    assert hashlib.sha256(MODEL.read_bytes()).hexdigest()==MODEL_SHA
    booster=lgb.Booster(model_file=str(MODEL)); assert booster.feature_name()==FEATURES
    with gzip.open(HISTORY,'rb') as f: state=pickle.load(f)
    state=roll_to_jul31(state); pats=scenario_patterns(); rows=[]; daily=[]
    for dt in pd.date_range('2026-08-01','2026-08-28',freq='D'):
        y,m,d=dt.strftime('%Y'),dt.strftime('%m'),dt.strftime('%d'); cards=read_csv(SRC/f'data/programs/race_cards/{y}/{m}/{d}.csv'); res=read_csv(SRC/f'data/results/realtime/{y}/{m}/{d}.csv'); tkz=read_csv(SRC/f'data/previews/tkz/{y}/{m}/{d}.csv'); sui=read_csv(SRC/f'data/previews/sui/{y}/{m}/{d}.csv')
        if cards.empty: continue
        scr=screen_ultra(cards,state,booster,pats); cd,td,sd=D(cards),D(tkz),D(sui); keep=[x for x in scr if x[4]]; asn=0
        for venue,rno,code,best,_ in keep:
            cls=None; p4=np.nan
            if code in td and code in sd:
                v=int(code[-4:-2]); cls,p4=actual_class(cd[code],td[code],sd[code],state,booster,v,rno)
            rows.append({'date':dt.strftime('%Y-%m-%d'),'venue':venue,'race_no':rno,'code':code,'max_p4proxy':best,'class':cls,'p4':p4})
            if cls in ('A','S'): asn+=1
        print('DAY',dt.strftime('%Y-%m-%d'),'TOTAL',len(cards),'SCREEN',len(keep),'AS',asn,flush=True); daily.append((dt.strftime('%Y-%m-%d'),len(cards),len(keep),asn))
        if not res.empty: advance_day(state,cards,res)
    df=pd.DataFrame(rows); df.to_csv('audit_20260801_0828_venue_rows.csv',index=False)
    valid=df[df['class'].isin(['A','S'])].copy(); agg=[]
    for venue,g in df.groupby('venue'):
        a=int((g['class']=='A').sum()); s=int((g['class']=='S').sum()); screened=len(g); asn=a+s; agg.append((venue,screened,a,s,asn,asn/screened*100 if screened else np.nan))
    agg=sorted(agg,key=lambda x:(-x[4],-x[5],x[0])); print('TOTAL_SCREENED',len(df),'TOTAL_AS',len(valid),flush=True)
    for i,(venue,screened,a,s,asn,rate) in enumerate(agg,1): print('VENUE',i,venue,'SCREEN',screened,'A',a,'S',s,'AS',asn,'RATE',f'{rate:.2f}',flush=True)
    # regression anchors
    d27=df[df.date=='2026-08-27']; d28=df[df.date=='2026-08-28']; print('REG_0827_SCREEN',len(d27),'EXPECTED',61,'MATCH',len(d27)==61,flush=True); print('REG_0828_SCREEN',len(d28),'EXPECTED',74,'MATCH',len(d28)==74,flush=True)
if __name__=='__main__': main()
