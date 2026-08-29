#!/usr/bin/env python3
import copy,gzip,pickle,hashlib
import numpy as np,pandas as pd, lightgbm as lgb
from screen_20260829 import *
from run_oos import CAT,WIND_TO_TRAIN,WEATHER_TO_TRAIN

def screen_ultra(cards,state,booster,pats):
    ans=[]
    scenario=[]
    for _,ex,wspd,wave,weather in pats:
        for wc in range(9): scenario.append((np.array(ex,float),float(wspd),float(wave),int(weather),wc))
    for ii,(_,card) in enumerate(cards.iterrows(),1):
        code=str(card['レースコード']); v=int(code[-4:-2]); rno=int(code[-2:]); venue=VENUE_CODE_TO_NAME.get(v,str(v))
        ex0,wspd0,wave0,wea0,wc0=scenario[0]
        tkz0={f'艇{b}_展示タイム':ex0[b-1] for b in range(1,7)}; sui0={'風速(m)':wspd0,'波の高さ(cm)':wave0,'風向':wc0,'天候':wea0}
        base,err=build_feature_frame(card,tkz0,sui0,state,v,rno)
        if err:
            ans.append((venue,rno,code,-1.,False,0)); continue
        frames=[]
        for ex,wspd,wave,wea,wc in scenario:
            x=base.copy()
            ranks=pd.Series(ex).rank(method='average',ascending=True).to_numpy()
            x['weather']=WEATHER_TO_TRAIN.get(wea,999); x['wind_dir']=WIND_TO_TRAIN.get(wc,999); x['wind_speed']=wspd; x['wave']=wave
            x['exhibit']=ex; x['exhibit_delta']=ex-ex.min(); x['exhibit_center']=ex-ex.mean(); x['exhibit_rank']=ranks
            x['wind_boat']=x['wind_dir'].astype(int)*10+x['boat'].astype(int)
            x['venue_wind_boat']=x['venue'].astype(int)*1000+x['wind_dir'].astype(int)*10+x['boat'].astype(int)
            frames.append(x)
        z=pd.concat(frames,ignore_index=True)
        for c in CAT: z[c]=z[c].astype('category')
        raw=booster.predict(z[FEATURES]); best=-1.
        for j in range(0,len(raw),6):
            _,_,_,p4=pl_top4(raw[j:j+6],z.iloc[j:j+6].boat.to_numpy()); best=max(best,p4)
        keep=best>=A_THR; ans.append((venue,rno,code,best,keep,len(scenario)))
        if ii%24==0: print('SCREENED',ii,'/',len(cards),flush=True)
    return ans

def advance_day(state,cards,res):
    cd,rd=D(cards),D(res)
    for code in sorted(set(cd)&set(rd),key=lambda x:(int(x[-2:]),int(x[-4:-2]))): update_state_from_race(cd[code],rd[code],state,int(code[-4:-2]))

def main():
    assert hashlib.sha256(MODEL.read_bytes()).hexdigest()==MODEL_SHA
    booster=lgb.Booster(model_file=str(MODEL)); assert booster.feature_name()==FEATURES
    with gzip.open(HISTORY,'rb') as f: base=pickle.load(f)
    pats=scenario_patterns(); s26=roll_state(copy.deepcopy(base),'2026-08-26')
    c27=read_csv(SRC/'data/programs/race_cards/2026/08/27.csv'); a27=screen_ultra(c27,s26,booster,pats); k27=sum(x[4] for x in a27); print('REGRESSION_0827',len(c27),k27,'EXPECTED_KEEP',61,'MATCH',k27==61,flush=True)
    s27=copy.deepcopy(s26); advance_day(s27,c27,read_csv(SRC/'data/results/realtime/2026/08/27.csv'))
    c28=read_csv(SRC/'data/programs/race_cards/2026/08/28.csv'); a28=screen_ultra(c28,s27,booster,pats); got28={(v,r) for v,r,_,_,k,_ in a28 if k}; print('REGRESSION_0828',len(c28),len(got28),'EXPECTED',len(EXPECTED_0828),'MATCH',got28==EXPECTED_0828,flush=True); print('MISSING_0828',sorted(EXPECTED_0828-got28),flush=True); print('EXTRA_0828',sorted(got28-EXPECTED_0828),flush=True)
    s28=copy.deepcopy(s27); advance_day(s28,c28,read_csv(SRC/'data/results/realtime/2026/08/28.csv'))
    c29=official_b_cards(TARGET); a29=screen_ultra(c29,s28,booster,pats); kept=[x for x in a29 if x[4]]; print('RESULT_0829 TOTAL',len(a29),'KEEP',len(kept),'EXCLUDE',len(a29)-len(kept),'RATE',f'{len(kept)/len(a29):.6f}',flush=True)
    for v,r,code,b,k,n in sorted(kept,key=lambda x:(int(x[2][-4:-2]),x[1])): print('KEEP',v,r,f'{b:.12f}',n,flush=True)
    pd.DataFrame(a29,columns=['venue','race_no','code','max_p4proxy','keep','scenario_count']).to_csv('screen_20260829.csv',index=False)
if __name__=='__main__': main()
