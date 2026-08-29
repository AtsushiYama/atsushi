#!/usr/bin/env python3
import copy,gzip,pickle,hashlib
import numpy as np,pandas as pd, lightgbm as lgb
from screen_20260829 import *
from run_oos import CAT

def screen_batch(cards,state,booster,pats):
    ans=[]
    for ii,(_,card) in enumerate(cards.iterrows(),1):
        code=str(card['レースコード']); v=int(code[-4:-2]); rno=int(code[-2:]); venue=VENUE_CODE_TO_NAME.get(v,str(v)); frames=[]
        for _,ex,wspd,wave,weather in pats:
            tkz={f'艇{b}_展示タイム':ex[b-1] for b in range(1,7)}
            for wind in range(9):
                sui={'風速(m)':wspd,'波の高さ(cm)':wave,'風向':wind,'天候':weather}
                x,err=build_feature_frame(card,tkz,sui,state,v,rno)
                if not err: frames.append(x)
        if not frames:
            ans.append((venue,rno,code,-1.,False,0)); continue
        z=pd.concat(frames,ignore_index=True)
        for c in CAT: z[c]=z[c].astype('category')
        raw=booster.predict(z[FEATURES]); best=-1.
        for j in range(0,len(raw),6):
            _,_,_,p4=pl_top4(raw[j:j+6],z.iloc[j:j+6].boat.to_numpy()); best=max(best,p4)
        keep=best>=A_THR; ans.append((venue,rno,code,best,keep,len(frames)))
        if ii%12==0: print('SCREENED',ii,'/',len(cards),flush=True)
    return ans

def main():
    h=hashlib.sha256(MODEL.read_bytes()).hexdigest(); assert h==MODEL_SHA
    booster=lgb.Booster(model_file=str(MODEL)); assert booster.feature_name()==FEATURES
    with gzip.open(HISTORY,'rb') as f: base=pickle.load(f)
    pats=scenario_patterns()
    # Build state once to 8/26, then advance copies one day at a time.
    s26=roll_state(copy.deepcopy(base),'2026-08-26')
    c27=read_csv(SRC/'data/programs/race_cards/2026/08/27.csv'); a27=screen_batch(c27,copy.deepcopy(s26),booster,pats); k27=sum(x[4] for x in a27)
    print('REGRESSION_0827',len(c27),k27,'EXPECTED_KEEP',61,'MATCH',k27==61,flush=True)
    s27=copy.deepcopy(s26); r27=read_csv(SRC/'data/results/realtime/2026/08/27.csv'); cd27,rd27=D(c27),D(r27)
    for code in sorted(set(cd27)&set(rd27),key=lambda x:(int(x[-2:]),int(x[-4:-2]))): update_state_from_race(cd27[code],rd27[code],s27,int(code[-4:-2]))
    c28=read_csv(SRC/'data/programs/race_cards/2026/08/28.csv'); a28=screen_batch(c28,copy.deepcopy(s27),booster,pats); got28={(v,r) for v,r,_,_,k,_ in a28 if k}
    print('REGRESSION_0828',len(c28),len(got28),'EXPECTED',len(EXPECTED_0828),'MATCH',got28==EXPECTED_0828,flush=True)
    print('MISSING_0828',sorted(EXPECTED_0828-got28),flush=True); print('EXTRA_0828',sorted(got28-EXPECTED_0828),flush=True)
    s28=copy.deepcopy(s27); r28=read_csv(SRC/'data/results/realtime/2026/08/28.csv'); cd28,rd28=D(c28),D(r28)
    for code in sorted(set(cd28)&set(rd28),key=lambda x:(int(x[-2:]),int(x[-4:-2]))): update_state_from_race(cd28[code],rd28[code],s28,int(code[-4:-2]))
    c29=official_b_cards(TARGET); a29=screen_batch(c29,s28,booster,pats); kept=[x for x in a29 if x[4]]
    print('RESULT_0829 TOTAL',len(a29),'KEEP',len(kept),'EXCLUDE',len(a29)-len(kept),'RATE',f'{len(kept)/len(a29):.6f}',flush=True)
    for v,r,code,b,k,n in sorted(kept,key=lambda x:(int(x[2][-4:-2]),x[1])): print('KEEP',v,r,f'{b:.12f}',n,flush=True)
    pd.DataFrame(a29,columns=['venue','race_no','code','max_p4proxy','keep','scenario_count']).to_csv('screen_20260829.csv',index=False)
if __name__=='__main__': main()
