#!/usr/bin/env python3
from __future__ import annotations
from datetime import date
import copy,gzip,pickle,hashlib
import pandas as pd
import lightgbm as lgb
from screen_20260829 import ROOT,SRC,MODEL,HISTORY,MODEL_SHA,FEATURES,A_THR,scenario_patterns,roll_state,official_b_cards,screen

TARGET=date(2026,8,30)

def main():
    h=hashlib.sha256(MODEL.read_bytes()).hexdigest()
    assert h==MODEL_SHA,(h,MODEL_SHA)
    booster=lgb.Booster(model_file=str(MODEL))
    assert booster.feature_name()==FEATURES
    with gzip.open(HISTORY,'rb') as f:
        base=pickle.load(f)
    patterns=scenario_patterns()
    state=roll_state(copy.deepcopy(base),'2026-08-29')
    cards=official_b_cards(TARGET)
    ans=screen(cards,state,booster,patterns)
    kept=[x for x in ans if x[4]]
    print('MODEL_SHA',h,flush=True)
    print('A_THR',f'{A_THR:.17f}',flush=True)
    print('RESULT_0830 TOTAL',len(ans),'KEEP',len(kept),'EXCLUDE',len(ans)-len(kept),'RATE',f'{len(kept)/len(ans):.6f}',flush=True)
    for v,r,code,b,k,n in sorted(kept,key=lambda x:(int(x[2][-4:-2]),x[1])):
        print('KEEP',v,r,f'{b:.12f}',n,flush=True)
    pd.DataFrame(ans,columns=['venue','race_no','code','max_p4proxy','keep','scenarios']).to_csv('screen_20260830.csv',index=False)

if __name__=='__main__':
    main()
