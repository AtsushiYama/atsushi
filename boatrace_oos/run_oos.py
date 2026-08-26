#!/usr/bin/env python3
from __future__ import annotations
import csv, gzip, itertools, json, math, os, pickle, re
from collections import defaultdict, deque
from pathlib import Path
import numpy as np
import pandas as pd
import lightgbm as lgb

ROOT = Path(__file__).resolve().parent
SRC = ROOT / 'source'
OUT = ROOT / 'output'
OUT.mkdir(parents=True, exist_ok=True)
MODEL = ROOT / 'boatrace_strength_v1_lgbm.txt'
HISTORY = ROOT / 'history.pkl.gz'

MODEL_SHA = '334b3a54a482a957f50c51b40614797821ae765d221734b92b95f1d4fb96cde0'
T = 1.2184870199794324
A_THR = 0.31242984023268894
S_THR = 0.37995532313528696
SEED = 20260826

VENUES = ['びわこ','三国','下関','丸亀','住之江','児島','唐津','多摩川','大村','宮島','尼崎','常滑','平和島','徳山','戸田','桐生','江戸川','津','浜名湖','芦屋','若松','蒲郡','鳴門']
VMAP = {v:i for i,v in enumerate(VENUES)}
VENUE_CODE_TO_NAME = {
 1:'桐生',2:'戸田',3:'江戸川',4:'平和島',5:'多摩川',6:'浜名湖',7:'蒲郡',8:'常滑',9:'津',10:'三国',11:'びわこ',12:'住之江',13:'尼崎',14:'鳴門',15:'丸亀',16:'児島',17:'宮島',18:'徳山',19:'下関',20:'若松',21:'芦屋',22:'福岡',23:'唐津',24:'大村'
}
WIND_TO_TRAIN = {1:0,2:1,3:6,4:4,5:3,6:5,7:8,8:2,0:7}
WEATHER_TO_TRAIN = {1:0,2:1,3:2,4:3,6:4}

FEATURES = [
'venue','race_no','distance','weather','wind_dir','wind_speed','wave','boat','racer_id','exhibit','exhibit_delta','exhibit_center','exhibit_rank','recent5_st','recent10_finish','racer_n_log','racer_win','racer_top2','racer_top3','racer_avg_finish','racer_avg_st','racer_f_rate','lane_n_log','lane_win','lane_top2','lane_top3','lane_avg_finish','lane_avg_st','lane_f_rate','venuehist_n_log','venuehist_win','venuehist_top2','venuehist_top3','venuehist_avg_finish','venuehist_avg_st','venuehist_f_rate','motorhist_n_log','motorhist_win','motorhist_top2','motorhist_top3','motorhist_avg_finish','motorhist_avg_st','motorhist_f_rate','eboathist_n_log','eboathist_win','eboathist_top2','eboathist_top3','eboathist_avg_finish','eboathist_avg_st','eboathist_f_rate','racer_win_rel','racer_top3_rel','racer_avg_finish_rel','racer_avg_st_rel','recent5_st_rel','recent10_finish_rel','lane_top3_rel','venuehist_top3_rel','motorhist_top3_rel','eboathist_top3_rel','venue_boat','wind_boat','venue_wind_boat','race_boat']
CAT = ['venue','boat','race_no','wind_dir','weather','venue_boat','wind_boat','venue_wind_boat','race_boat','racer_id']
PERMS = np.array(list(itertools.permutations(range(6),3)), dtype=int)


def num(x, default=np.nan):
    if x is None: return default
    s=str(x).strip()
    if s in ('','None','nan','NaN','-','--'): return default
    try: return float(s.replace(',',''))
    except Exception: return default

def integer(x, default=None):
    v=num(x, np.nan)
    return default if np.isnan(v) else int(v)

def read_csv(path: Path):
    if not path.exists(): return pd.DataFrame()
    try: return pd.read_csv(path, dtype=str, keep_default_na=False)
    except Exception as e:
        print('READ_FAIL', path, repr(e), flush=True); return pd.DataFrame()

def agg_feats(a):
    if a is None: a=[0,0,0,0,0.0,0.0,0,0]
    n=a[0]
    return dict(
        n_log=math.log1p(n),
        win=a[1]/n if n else 1/6,
        top2=a[2]/n if n else 2/6,
        top3=a[3]/n if n else 3/6,
        avg_finish=a[4]/n if n else 3.5,
        avg_st=a[5]/a[6] if a[6] else 0.18,
        f_rate=a[7]/n if n else 0.0,
    )

def upd(d,key,rank,st,f):
    a=d.get(key)
    if a is None: a=[0,0,0,0,0.0,0.0,0,0]; d[key]=a
    a[0]+=1; rr=rank if rank is not None else 7
    a[4]+=rr; a[1]+=int(rank==1); a[2]+=int(rank is not None and rank<=2); a[3]+=int(rank is not None and rank<=3)
    if st is not None and np.isfinite(st): a[5]+=float(st); a[6]+=1
    a[7]+=int(f)

def result_maps(rr):
    rank_by_boat={}
    for rank in range(1,7):
        b=integer(rr.get(f'{rank}着_艇番'))
        if b in range(1,7): rank_by_boat[b]=rank
    st_by_boat={}; f_by_boat={}
    for course in range(1,7):
        b=integer(rr.get(f'{course}コース_艇番'))
        if b in range(1,7):
            st=num(rr.get(f'{course}コース_スタートタイミング'), np.nan)
            st_by_boat[b]=None if np.isnan(st) else float(st)
            f_by_boat[b]=1 if str(rr.get(f'{course}コース_F','')).strip() else 0
    return rank_by_boat, st_by_boat, f_by_boat

def classify_p4(p):
    if p >= S_THR: return 'S'
    if p >= A_THR: return 'A'
    return 'skip'

def pl_top4(raw, boats):
    strength=np.asarray(raw,float)/T
    z=strength-strength.max(); w=np.exp(z); sw=w.sum()
    probs=np.empty(120,float)
    for q,(i,j,k) in enumerate(PERMS):
        probs[q]=(w[i]/sw)*(w[j]/(sw-w[i]))*(w[k]/(sw-w[i]-w[j]))
    top=np.argsort(-probs)[:4]
    combos=['-'.join(str(int(boats[x])) for x in PERMS[t]) for t in top]
    return strength, probs[top], combos, float(probs[top].sum())

def build_feature_frame(card, tkz, sui, state, venue_code, race_no):
    venue_name=VENUE_CODE_TO_NAME.get(venue_code)
    if not venue_name: return None, 'venue_unknown'
    venue_cat=VMAP.get(venue_name, 23)
    wind_speed=num(sui.get('風速(m)'), np.nan); wave=num(sui.get('波の高さ(cm)'), np.nan)
    wc=integer(sui.get('風向'),0); weather_code=integer(sui.get('天候'))
    wind_cat=WIND_TO_TRAIN.get(wc, 999)
    weather_cat=WEATHER_TO_TRAIN.get(weather_code,999)
    exhs=[]; boats=[]
    for b in range(1,7):
        rid=integer(card.get(f'艇{b}_登録番号')); motor=integer(card.get(f'艇{b}_モーター番号')); eq=integer(card.get(f'艇{b}_ボート番号'))
        ex=num(tkz.get(f'艇{b}_展示タイム'), np.nan)
        if rid is None or motor is None or eq is None or not np.isfinite(ex): return None, f'missing_static_b{b}'
        boats.append((b,rid,motor,eq)); exhs.append(float(ex))
    if not np.isfinite(wind_speed) or not np.isfinite(wave): return None,'missing_weather'
    exhs=np.array(exhs,float)
    ranks=pd.Series(exhs).rank(method='average',ascending=True).to_numpy()
    rows=[]
    for j,(b,rid,motor,eq) in enumerate(boats):
        rf=agg_feats(state['racer'].get(rid)); lf=agg_feats(state['lane'].get((rid,b))); vf=agg_feats(state['venuehist'].get((rid,venue_name)))
        mf=agg_feats(state['motorhist'].get((venue_name,motor))); bf=agg_feats(state['eboathist'].get((venue_name,eq)))
        rs=state['recent_st'].get(rid,[]); rfin=state['recent_finish'].get(rid,[])
        recst=float(np.mean(rs)) if rs else 0.18; recfin=float(np.mean(rfin)) if rfin else 3.5
        r={
          'venue':venue_cat,'race_no':race_no,'distance':1800,'weather':weather_cat,'wind_dir':wind_cat,'wind_speed':float(wind_speed),'wave':float(wave),'boat':b,'racer_id':rid,
          'exhibit':exhs[j],'exhibit_delta':exhs[j]-exhs.min(),'exhibit_center':exhs[j]-exhs.mean(),'exhibit_rank':float(ranks[j]),
          'recent5_st':recst,'recent10_finish':recfin,
        }
        for prefix,f in [('racer',rf),('lane',lf),('venuehist',vf),('motorhist',mf),('eboathist',bf)]:
            for k,v in f.items(): r[f'{prefix}_{k}']=v
        rows.append(r)
    for c in ['racer_win','racer_top3','racer_avg_finish','racer_avg_st','recent5_st','recent10_finish','lane_top3','venuehist_top3','motorhist_top3','eboathist_top3']:
        vals=np.array([r[c] for r in rows],float); mu=vals.mean()
        for r,v in zip(rows,vals): r[c+'_rel']=float(v-mu)
    for r in rows:
        r['venue_boat']=int(r['venue']*10+r['boat'])
        r['wind_boat']=int(r['wind_dir']*10+r['boat'])
        r['venue_wind_boat']=int(r['venue']*1000+r['wind_dir']*10+r['boat'])
        r['race_boat']=int(r['race_no']*10+r['boat'])
    df=pd.DataFrame(rows)
    for c in CAT: df[c]=df[c].astype('category')
    return df, None

def update_state_from_race(card, result, state, venue_code):
    venue_name=VENUE_CODE_TO_NAME.get(venue_code)
    if not venue_name: return False
    rank, st, fl=result_maps(result)
    if len(rank)<3: return False
    for b in range(1,7):
        rid=integer(card.get(f'艇{b}_登録番号')); motor=integer(card.get(f'艇{b}_モーター番号')); eq=integer(card.get(f'艇{b}_ボート番号'))
        if rid is None or motor is None or eq is None: continue
        rr=rank.get(b); ss=st.get(b); ff=fl.get(b,0)
        upd(state['racer'],rid,rr,ss,ff); upd(state['lane'],(rid,b),rr,ss,ff); upd(state['venuehist'],(rid,venue_name),rr,ss,ff)
        upd(state['motorhist'],(venue_name,motor),rr,ss,ff); upd(state['eboathist'],(venue_name,eq),rr,ss,ff)
        if ss is not None and np.isfinite(ss):
            q=state['recent_st'].setdefault(rid,[]); q.append(float(ss)); del q[:-5]
        q=state['recent_finish'].setdefault(rid,[]); q.append(rr if rr is not None else 7); del q[:-10]
    return True

def bootstrap_roi(rows, B=10000):
    if not rows: return [None,None]
    d=pd.DataFrame(rows)
    days=sorted(d.date.unique()); by={day:d[d.date==day] for day in days}; rng=np.random.default_rng(SEED)
    vals=[]
    for _ in range(B):
        samp=rng.choice(days,size=len(days),replace=True)
        stake=sum(float(by[x].stake.sum()) for x in samp); ret=sum(float(by[x].ret.sum()) for x in samp)
        if stake>0: vals.append(ret/stake*100)
    return [float(np.percentile(vals,2.5)),float(np.percentile(vals,97.5))] if vals else [None,None]

def max_losing_streak(rs):
    best=cur=0
    for x in rs:
        if x<=0: cur+=1; best=max(best,cur)
        else: cur=0
    return best

def max_drawdown(profits):
    cum=0.0; peak=0.0; dd=0.0
    for x in profits:
        cum+=x; peak=max(peak,cum); dd=max(dd,peak-cum)
    return dd

def summarize(d, mask, label):
    x=d[mask].sort_values(['date','deadline','race_code']).copy()
    n=len(x); stake=float(x.stake.sum()) if n else 0.; ret=float(x.ret.sum()) if n else 0.; hits=int(x.hit.sum()) if n else 0
    active=int(x.date.nunique()) if n else 0
    daily=x.groupby('date',as_index=False).agg(stake=('stake','sum'),ret=('ret','sum')) if n else pd.DataFrame(columns=['stake','ret'])
    pos=int((daily.ret>daily.stake).sum()) if n else 0
    top_removed=None; top3_removed=None
    if n:
        sx=x.sort_values('ret',ascending=False)
        if n>1:
            y=sx.iloc[1:]; top_removed=float(y.ret.sum()/y.stake.sum()*100) if y.stake.sum()>0 else None
        if n>3:
            y=sx.iloc[3:]; top3_removed=float(y.ret.sum()/y.stake.sum()*100) if y.stake.sum()>0 else None
    rows=[type('R',(),{'date':r.date,'stake':r.stake,'ret':r.ret}) for r in x.itertuples()]
    ci=bootstrap_roi(rows)
    return {
      'condition':label,'purchases':n,'hits':hits,'hit_rate':hits/n if n else None,'stake_yen':stake,'return_yen':ret,'roi_pct':ret/stake*100 if stake else None,'net_yen':ret-stake,
      'active_days':active,'positive_days':pos,'positive_day_rate_active':pos/active if active else None,'positive_day_rate_all23':pos/23,
      'max_losing_streak':max_losing_streak(x.ret.tolist()) if n else 0,'max_drawdown_yen':max_drawdown((x.ret-x.stake).tolist()) if n else 0,
      'roi_excluding_largest_return_pct':top_removed,'roi_excluding_top3_returns_pct':top3_removed,
      'roi_if_returns_haircut_5pct':ret*0.95/stake*100 if stake else None,'roi_if_returns_haircut_10pct':ret*0.90/stake*100 if stake else None,
      'day_cluster_bootstrap_roi_95pct':ci,
    }

def main():
    import hashlib
    h=hashlib.sha256(MODEL.read_bytes()).hexdigest()
    if h!=MODEL_SHA: raise SystemExit(f'MODEL SHA MISMATCH {h}')
    booster=lgb.Booster(model_file=str(MODEL))
    if booster.feature_name()!=FEATURES: raise SystemExit('FEATURE CONTRACT MISMATCH')
    with gzip.open(HISTORY,'rb') as f: state=pickle.load(f)
    state['motorhist']={}; state['eboathist']={}
    state['recent_st']={int(k):list(v) for k,v in state['recent_st'].items()}; state['recent_finish']={int(k):list(v) for k,v in state['recent_finish'].items()}
    details=[]; coverage=[]
    dates=pd.date_range('2026-01-01','2026-08-23',freq='D')
    for dt in dates:
        y,m,dd=dt.strftime('%Y'),dt.strftime('%m'),dt.strftime('%d')
        cards=read_csv(SRC/f'data/programs/race_cards/{y}/{m}/{dd}.csv')
        results=read_csv(SRC/f'data/results/realtime/{y}/{m}/{dd}.csv')
        if cards.empty or results.empty:
            coverage.append({'date':dt.strftime('%Y-%m-%d'),'cards':len(cards),'results':len(results),'predicted':0,'note':'missing cards/results'})
            continue
        cdict={str(r['レースコード']):r for _,r in cards.iterrows()}; rdict={str(r['レースコード']):r for _,r in results.iterrows()}
        aug = dt>=pd.Timestamp('2026-08-01')
        tkzd=suid=od3d=payoutd={}
        if aug:
            tkz=read_csv(SRC/f'data/previews/tkz/2026/08/{dd}.csv'); sui=read_csv(SRC/f'data/previews/sui/2026/08/{dd}.csv'); od3=read_csv(SRC/f'data/previews/od3/2026/08/{dd}.csv'); pay=read_csv(SRC/f'data/results/payouts/2026/08/{dd}.csv')
            tkzd={str(r['レースコード']):r for _,r in tkz.iterrows()} if not tkz.empty else {}
            suid={str(r['レースコード']):r for _,r in sui.iterrows()} if not sui.empty else {}
            od3d={str(r['レースコード']):r for _,r in od3.iterrows()} if not od3.empty else {}
            payoutd={str(r['レースコード']):r for _,r in pay.iterrows()} if not pay.empty else {}
        keys=[]
        for code in set(cdict)&set(rdict):
            if len(code)>=12 and code.isdigit(): keys.append((int(code[-2:]),int(code[-4:-2]),code))
        keys.sort()
        predn=0
        for race_no, venue_code, code in keys:
            card=cdict[code]; rr=rdict[code]
            if aug:
                rec={'date':dt.strftime('%Y-%m-%d'),'race_code':code,'venue_code':venue_code,'venue':VENUE_CODE_TO_NAME.get(venue_code,''),'race_no':race_no,
                     'deadline':str(rr.get('締切時刻','')),'status':'','S_A':'','P4':np.nan,'combined_odds':np.nan,'hit':0,'stake':0.0,'ret':0.0,'actual_combo':'','payout':np.nan}
                if code not in tkzd or code not in suid or code not in od3d:
                    rec['status']='missing_preclose_csv'; details.append(rec)
                else:
                    x,err=build_feature_frame(card,tkzd[code],suid[code],state,venue_code,race_no)
                    if err:
                        rec['status']=err; details.append(rec)
                    else:
                        raw=booster.predict(x[FEATURES]); strength,topprob,combos,p4=pl_top4(raw,x.boat.to_numpy())
                        sa=classify_p4(p4); rec['P4']=p4; rec['S_A']=sa; rec['top4']='|'.join(combos); rec['top4_probs']='|'.join(f'{z:.10f}' for z in topprob); rec['strengths']='|'.join(f'{z:.8f}' for z in strength)
                        odds=[]; odrow=od3d[code]
                        for combo in combos:
                            o=num(odrow.get(f'3連単_{combo}'),np.nan); odds.append(o)
                        if any(not np.isfinite(o) or o<=0 for o in odds):
                            rec['status']='missing_top4_odds'
                        else:
                            C=1.0/sum(1.0/o for o in odds); rec['combined_odds']=C; rec['top4_odds']='|'.join(str(float(o)) for o in odds)
                            rankmap,_,_=result_maps(rr)
                            actual='-'.join(str(next((b for b,rk in rankmap.items() if rk==k),0)) for k in (1,2,3))
                            rec['actual_combo']=actual; rec['hit']=int(actual in combos)
                            prow=payoutd.get(code)
                            pcombo=str(prow.get('3連単_組番','')).strip() if prow is not None else ''
                            payout=num(prow.get('3連単_払戻金'),np.nan) if prow is not None else np.nan
                            rec['payout']=payout; rec['payout_combo']=pcombo
                            if sa in ('S','A') and np.isfinite(payout) and re.fullmatch(r'[1-6]-[1-6]-[1-6]',pcombo or ''):
                                rec['stake']=400.0; rec['ret']=float(payout) if rec['hit'] else 0.0; rec['status']='settled'
                            elif sa=='skip': rec['status']='p4_skip'
                            else: rec['status']='no_settlement'
                            predn+=1
                        details.append(rec)
            update_state_from_race(card,rr,state,venue_code)
        coverage.append({'date':dt.strftime('%Y-%m-%d'),'cards':len(cards),'results':len(results),'joined':len(keys),'predicted':predn,'note':''})
        if dt.day==1 or aug: print('DAY',dt.strftime('%Y-%m-%d'),'joined',len(keys),'pred',predn, flush=True)
    d=pd.DataFrame(details)
    d.to_csv(OUT/'race_detail_20260801_0823.csv',index=False)
    pd.DataFrame(coverage).to_csv(OUT/'data_coverage_20260101_0823.csv',index=False)
    ev=d[(d.status=='settled') & (d.S_A.isin(['S','A']))].copy()
    conds=[('S+A all',np.ones(len(ev),dtype=bool)),('combined>=3.0',ev.combined_odds>=3.0),('combined>=3.5',ev.combined_odds>=3.5),('combined>=4.0',ev.combined_odds>=4.0),('3.0<=combined<3.5',(ev.combined_odds>=3.0)&(ev.combined_odds<3.5)),('3.5<=combined<4.0',(ev.combined_odds>=3.5)&(ev.combined_odds<4.0)),('A & combined>=3.0',(ev.S_A=='A')&(ev.combined_odds>=3.0)),('S & combined>=3.0',(ev.S_A=='S')&(ev.combined_odds>=3.0))]
    summaries=[summarize(ev,mask,label) for label,mask in conds]
    pd.DataFrame(summaries).to_csv(OUT/'summary.csv',index=False)
    model_rows=d[d.P4.notna()].copy(); model_hit=float(model_rows.hit.mean()) if len(model_rows) else None
    sa_counts=model_rows.S_A.value_counts().to_dict()
    main=ev[ev.combined_odds>=3.0].copy()
    venue_tbl=main.groupby(['venue_code','venue'],as_index=False).agg(purchases=('stake','size'),hits=('hit','sum'),stake_yen=('stake','sum'),return_yen=('ret','sum')) if len(main) else pd.DataFrame()
    if len(venue_tbl):
        venue_tbl['roi_pct']=venue_tbl.return_yen/venue_tbl.stake_yen*100; venue_tbl['net_yen']=venue_tbl.return_yen-venue_tbl.stake_yen
    venue_tbl.to_csv(OUT/'venue_combined_ge3.csv',index=False)
    result={'model_sha256':h,'temperature':T,'thresholds':{'A_lower':A_THR,'S_lower':S_THR},'evaluation_period':'2026-08-01..2026-08-23','history_rollforward':'2026-01-01..2026-07-31 using only prior settled results before each race','distance_note':'distance fixed to 1800 in inference because frozen model has exactly zero gain/splits on distance; does not change any prediction','model_evaluable_races':int(len(model_rows)),'model_top4_hit_rate':model_hit,'sa_counts':sa_counts,'summaries':summaries,'coverage_august':pd.DataFrame(coverage).query("date >= '2026-08-01'").to_dict('records')}
    (OUT/'result.json').write_text(json.dumps(result,ensure_ascii=False,indent=2,default=str),encoding='utf-8')
    lines=['# Strength-v1.0 frozen OOS validation','',f'- Model SHA-256: `{h}`',f'- Model-evaluable races: {len(model_rows)}',f'- Model top4 exact hit rate: {model_hit:.4%}' if model_hit is not None else '- Model top4 exact hit rate: n/a','', '|Condition|Races|Hits|Hit rate|ROI|Net|Max DD|Max L streak|ROI excl max|ROI 10% haircut|','|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for s in summaries:
        fmt=lambda x: 'n/a' if x is None else f'{x:.2f}'
        hr='n/a' if s['hit_rate'] is None else f"{s['hit_rate']*100:.2f}%"
        lines.append(f"|{s['condition']}|{s['purchases']}|{s['hits']}|{hr}|{fmt(s['roi_pct'])}%|{s['net_yen']:.0f}|{s['max_drawdown_yen']:.0f}|{s['max_losing_streak']}|{fmt(s['roi_excluding_largest_return_pct'])}%|{fmt(s['roi_if_returns_haircut_10pct'])}%|")
    (OUT/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps(result,ensure_ascii=False,indent=2,default=str),flush=True)

if __name__=='__main__': main()
