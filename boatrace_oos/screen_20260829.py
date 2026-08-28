#!/usr/bin/env python3
from __future__ import annotations
from pathlib import Path
from datetime import date
import gzip,pickle,hashlib,re
import numpy as np,pandas as pd,lightgbm as lgb
from boatrace_lzh import LzhDownloader
from run_oos import FEATURES,MODEL_SHA,A_THR,read_csv,update_state_from_race,build_feature_frame,pl_top4,VENUE_CODE_TO_NAME

ROOT=Path(__file__).resolve().parent
SRC=ROOT/'source'
MODEL=ROOT/'boatrace_strength_v1_lgbm.txt'
HISTORY=ROOT/'history.pkl.gz'
SEED=20260826
TARGET=date(2026,8,29)
EXPECTED_0828={
 ('桐生',3),('桐生',10),('桐生',11),('平和島',7),('平和島',8),('多摩川',1),('多摩川',4),('多摩川',7),('多摩川',9),
 ('蒲郡',1),('蒲郡',2),('蒲郡',5),('蒲郡',6),('蒲郡',7),('蒲郡',8),('蒲郡',9),('蒲郡',10),('蒲郡',11),('蒲郡',12),
 ('津',1),('津',5),('津',9),('津',10),('津',11),('津',12),
 ('三国',1),('三国',2),('三国',3),('三国',5),('三国',6),('三国',7),('三国',9),('三国',10),
 ('びわこ',3),('びわこ',4),('びわこ',5),('びわこ',6),('びわこ',7),('びわこ',9),('びわこ',11),('びわこ',12),
 ('尼崎',1),('尼崎',4),('尼崎',5),('尼崎',6),('尼崎',7),('尼崎',8),('尼崎',9),('尼崎',10),('尼崎',11),
 ('鳴門',1),('鳴門',2),('鳴門',3),('鳴門',8),
 ('丸亀',5),('丸亀',6),('丸亀',8),('丸亀',9),('丸亀',11),
 ('宮島',1),('宮島',2),('宮島',3),('宮島',4),('宮島',5),('宮島',7),('宮島',10),('宮島',11),
 ('唐津',2),('唐津',3),('唐津',5),('唐津',6),('唐津',7),('唐津',8),('唐津',9)
}

def D(df): return {str(r['レースコード']):r for _,r in df.iterrows()}

def roll_state(state,end='2026-08-28'):
    state['motorhist']={}; state['eboathist']={}
    state['recent_st']={int(k):list(v) for k,v in state['recent_st'].items()}; state['recent_finish']={int(k):list(v) for k,v in state['recent_finish'].items()}
    for dt in pd.date_range('2026-01-01',end,freq='D'):
        y,m,d=dt.strftime('%Y'),dt.strftime('%m'),dt.strftime('%d')
        cards=read_csv(SRC/f'data/programs/race_cards/{y}/{m}/{d}.csv'); res=read_csv(SRC/f'data/results/realtime/{y}/{m}/{d}.csv')
        if cards.empty or res.empty: continue
        cd,rd=D(cards),D(res); keys=[]
        for code in set(cd)&set(rd):
            if len(code)>=12 and code.isdigit(): keys.append((int(code[-2:]),int(code[-4:-2]),code))
        for rno,v,code in sorted(keys): update_state_from_race(cd[code],rd[code],state,v)
    return state

def official_b_cards(dt):
    dl=LzhDownloader(cache_dir='./cache'); fs=dl.download(dt,'schedule')
    if not fs: raise RuntimeError('B file unavailable')
    txt=next(iter(fs.values())); FW=str.maketrans('０１２３４５６７８９Ｒ','0123456789R')
    rows={}; venue=None; rno=None
    for raw in txt.splitlines():
        m=re.match(r'^(\d{2})BBGN\s*$',raw)
        if m: venue=int(m.group(1)); rno=None; continue
        if re.match(r'^\d{2}BEND\s*$',raw): venue=None; rno=None; continue
        mh=re.match(r'^\s*(\d{1,2})R\s',raw.translate(FW))
        if mh and venue is not None: rno=int(mh.group(1)); continue
        if venue is None or rno is None or not re.match(r'^[1-6] \d{4}',raw) or len(raw)<58: continue
        try:
            lane=int(raw[0]); racer=int(raw[2:6]); motor=int(raw[41:43].strip()); equip=int(raw[49:52].strip())
        except: continue
        code=f'{dt.strftime("%Y%m%d")}{venue:02d}{rno:02d}'
        rec=rows.setdefault(code,{'レースコード':code,'レース日':dt.isoformat(),'レース場コード':f'{venue:02d}','レース回':f'{rno:02d}R'})
        rec[f'艇{lane}_登録番号']=racer; rec[f'艇{lane}_モーター番号']=motor; rec[f'艇{lane}_ボート番号']=equip
    out=[]
    for code,r in rows.items():
        if all(r.get(f'艇{b}_登録番号') and r.get(f'艇{b}_モーター番号') and r.get(f'艇{b}_ボート番号') for b in range(1,7)): out.append(r)
    return pd.DataFrame(out)

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
            try: ex=[float(tr[f'艇{b}_展示タイム']) for b in range(1,7)]
            except: continue
            try:
                wsp=float(sr['風速(m)']); wave=float(sr['波の高さ(cm)']); weather=int(float(sr['天候']))
            except: continue
            if not all(np.isfinite(ex)) or not np.isfinite(wsp) or not np.isfinite(wave): continue
            pairs.append((code,ex,wsp,wave,weather))
    if len(pairs)<30: raise RuntimeError(f'only {len(pairs)} 2025 preview pairs')
    rng=np.random.default_rng(SEED); idx=rng.choice(len(pairs),size=30,replace=False)
    out=[pairs[int(i)] for i in idx]
    print('PATTERN_CODES',','.join(x[0] for x in out),flush=True)
    return out

def screen(cards,state,booster,patterns):
    ans=[]
    for _,card in cards.iterrows():
        code=str(card['レースコード']); v=int(code[-4:-2]); rno=int(code[-2:]); venue=VENUE_CODE_TO_NAME.get(v,str(v)); best=-1.; n=0
        for _,ex,wsp,wave,weather in patterns:
            tkz={f'艇{b}_展示タイム':ex[b-1] for b in range(1,7)}
            for wind in range(9):
                sui={'風速(m)':wsp,'波の高さ(cm)':wave,'風向':wind,'天候':weather}
                x,err=build_feature_frame(card,tkz,sui,state,v,rno)
                if err: continue
                raw=booster.predict(x[FEATURES]); _,_,_,p4=pl_top4(raw,x.boat.to_numpy()); n+=1; best=max(best,p4)
        keep=best>=A_THR
        ans.append((venue,rno,code,best,keep,n))
    return ans

def main():
    h=hashlib.sha256(MODEL.read_bytes()).hexdigest(); assert h==MODEL_SHA,(h,MODEL_SHA)
    booster=lgb.Booster(model_file=str(MODEL)); assert booster.feature_name()==FEATURES
    with gzip.open(HISTORY,'rb') as f: base=pickle.load(f)
    patterns=scenario_patterns()
    import copy
    s28=roll_state(copy.deepcopy(base),'2026-08-27')
    c28=read_csv(SRC/'data/programs/race_cards/2026/08/28.csv')
    a28=screen(c28,s28,booster,patterns); got28={(v,r) for v,r,_,_,k,_ in a28 if k}
    print('REGRESSION_0828',len(c28),len(got28),'EXPECTED',len(EXPECTED_0828),'MATCH',got28==EXPECTED_0828,flush=True)
    print('MISSING_0828',sorted(EXPECTED_0828-got28),flush=True); print('EXTRA_0828',sorted(got28-EXPECTED_0828),flush=True)
    s29=roll_state(copy.deepcopy(base),'2026-08-28'); c29=official_b_cards(TARGET); a29=screen(c29,s29,booster,patterns)
    kept=[x for x in a29 if x[4]]
    print('RESULT_0829 TOTAL',len(a29),'KEEP',len(kept),'EXCLUDE',len(a29)-len(kept),'RATE',f'{len(kept)/len(a29):.6f}',flush=True)
    for v,r,code,b,k,n in sorted(kept,key=lambda x:(int(x[2][-4:-2]),x[1])): print('KEEP',v,r,f'{b:.12f}',n,flush=True)
    pd.DataFrame(a29,columns=['venue','race_no','code','max_p4proxy','keep','scenarios']).to_csv('screen_20260829.csv',index=False)
if __name__=='__main__': main()
