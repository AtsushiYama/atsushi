#!/usr/bin/env python3
import audit_daily_20260801_0830 as a
from run_oos import FEATURES,A_THR,build_feature_frame,pl_top4

def prescreen_one_fast(card,state,booster,patterns):
    code=str(card['レースコード']); v=int(code[-4:-2]); rno=int(code[-2:]); best=-1.; n=0
    for _,ex,wsp,wave,weather in patterns:
        tkz={f'艇{b}_展示タイム':ex[b-1] for b in range(1,7)}
        for wind in range(9):
            sui={'風速(m)':wsp,'波の高さ(cm)':wave,'風向':wind,'天候':weather}
            x,err=build_feature_frame(card,tkz,sui,state,v,rno)
            if err: continue
            raw=booster.predict(x[FEATURES]); _,_,_,p4=pl_top4(raw,x.boat.to_numpy()); n+=1; best=max(best,p4)
            # Exact boolean-equivalent shortcut: once any scenario reaches A threshold,
            # the original max-over-270 prescreen is guaranteed to keep the race.
            if best>=A_THR: return best,n,True
    return best,n,False

a.prescreen_one=prescreen_one_fast
if __name__=='__main__': a.main()
