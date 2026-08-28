#!/usr/bin/env python3
import copy,gzip,pickle,hashlib
import pandas as pd
import lightgbm as lgb
from screen_20260829 import ROOT,SRC,MODEL,HISTORY,A_THR,MODEL_SHA,FEATURES,EXPECTED_0828,TARGET,D,roll_state,official_b_cards,scenario_patterns,build_feature_frame,pl_top4,VENUE_CODE_TO_NAME

def screen_fast(cards,state,booster,patterns):
    ans=[]
    for _,card in cards.iterrows():
        code=str(card['レースコード']); v=int(code[-4:-2]); rno=int(code[-2:]); venue=VENUE_CODE_TO_NAME.get(v,str(v)); best=-1.; n=0; keep=False
        for _,ex,wspd,wave,weather in patterns:
            tkz={f'艇{b}_展示タイム':ex[b-1] for b in range(1,7)}
            for wind in range(9):
                sui={'風速(m)':wspd,'波の高さ(cm)':wave,'風向':wind,'天候':weather}
                x,err=build_feature_frame(card,tkz,sui,state,v,rno)
                if err: continue
                raw=booster.predict(x[FEATURES]); _,_,_,p4=pl_top4(raw,x.boat.to_numpy()); n+=1; best=max(best,p4)
                if p4>=A_THR:
                    keep=True; break
            if keep: break
        ans.append((venue,rno,code,best,keep,n))
    return ans

def main():
    h=hashlib.sha256(MODEL.read_bytes()).hexdigest(); assert h==MODEL_SHA,(h,MODEL_SHA)
    booster=lgb.Booster(model_file=str(MODEL)); assert booster.feature_name()==FEATURES
    with gzip.open(HISTORY,'rb') as f: base=pickle.load(f)
    pats=scenario_patterns()
    s28=roll_state(copy.deepcopy(base),'2026-08-27'); c28=pd.read_csv(SRC/'data/programs/race_cards/2026/08/28.csv',dtype=str,keep_default_na=False)
    a28=screen_fast(c28,s28,booster,pats); got28={(v,r) for v,r,_,_,k,_ in a28 if k}
    print('REGRESSION_0828',len(c28),len(got28),'EXPECTED',len(EXPECTED_0828),'MATCH',got28==EXPECTED_0828,flush=True)
    print('MISSING_0828',sorted(EXPECTED_0828-got28),flush=True); print('EXTRA_0828',sorted(got28-EXPECTED_0828),flush=True)
    s27=roll_state(copy.deepcopy(base),'2026-08-26'); c27=pd.read_csv(SRC/'data/programs/race_cards/2026/08/27.csv',dtype=str,keep_default_na=False)
    a27=screen_fast(c27,s27,booster,pats); k27=sum(x[4] for x in a27); print('REGRESSION_0827',len(c27),k27,'EXPECTED_KEEP',61,'MATCH',k27==61,flush=True)
    s29=roll_state(copy.deepcopy(base),'2026-08-28'); c29=official_b_cards(TARGET); a29=screen_fast(c29,s29,booster,pats); kept=[x for x in a29 if x[4]]
    print('RESULT_0829 TOTAL',len(a29),'KEEP',len(kept),'EXCLUDE',len(a29)-len(kept),'RATE',f'{len(kept)/len(a29):.6f}',flush=True)
    for v,r,code,b,k,n in sorted(kept,key=lambda x:(int(x[2][-4:-2]),x[1])): print('KEEP',v,r,f'{b:.12f}','SCENARIOS_TO_HIT',n,flush=True)
    pd.DataFrame(a29,columns=['venue','race_no','code','first_hit_p4proxy','keep','scenarios_tested']).to_csv('screen_20260829.csv',index=False)
if __name__=='__main__': main()
