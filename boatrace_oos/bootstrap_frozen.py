#!/usr/bin/env python3
from __future__ import annotations
import os,re,math,pickle,gzip,subprocess,tempfile,concurrent.futures
from collections import defaultdict,deque
from datetime import date,timedelta,datetime
from pathlib import Path
import numpy as np, pandas as pd, requests, zipfile
from numba import njit
import lightgbm as lgb

ROOT=Path(__file__).resolve().parent
EXPECTED_SHA='334b3a54a482a957f50c51b40614797821ae765d221734b92b95f1d4fb96cde0'
VEN_RE=re.compile(r'^(.+?)［成績］')
RACE_RE=re.compile(r'^\s*(\d{1,2})R\s+.*?H(\d+)m\s+(.*?)\s+風\s+(.*?)\s+(\d+)m\s+波\s+(\d+)cm')
ENT_RE=re.compile(r'^\s*(\d{2}|F|L|K\d?|S\d?|転|失|欠)\s+([1-6])\s+(\d{4})\s+(.+?)\s+(\d{1,2})\s+(\d{1,2})\s+(\d\.\d{2})\s+([1-6])\s+([FL]?\d\.\d{2})')
PAY_RE=re.compile(r'^\s*３連単\s+([1-6]-[1-6]-[1-6])\s+(\d+)')
DATE_RE=re.compile(r'K(\d{6})\.TXT$',re.I)

def clean_venue(s): return re.sub(r'[\s\u3000]+','',s)
def parse_st(s):
    if s.startswith('F'): return np.nan,1
    if s.startswith('L'): return np.nan,0
    try:return float(s),0
    except:return np.nan,0

def parse_text(text,dt):
    races=[];venue=None;cur=None
    for line in text.splitlines():
        vm=VEN_RE.match(line)
        if vm: venue=clean_venue(vm.group(1));cur=None;continue
        rm=RACE_RE.match(line)
        if rm and venue:
            if cur is not None and len(cur['entrants'])>=3:races.append(cur)
            rno,dist,weather,wdir,wspd,wave=rm.groups()
            cur=dict(date=pd.Timestamp(dt),venue=venue,race_no=int(rno),distance=int(dist),weather=weather.strip(),wind_dir=wdir.strip(),wind_speed=int(wspd),wave=int(wave),entrants=[],payout=None)
            continue
        if cur is None:continue
        em=ENT_RE.match(line)
        if em:
            rank_s,boat,rid,name,motor,bno,exh,course,st_s=em.groups();st,f=parse_st(st_s)
            rank=int(rank_s) if rank_s.isdigit() else None
            cur['entrants'].append(dict(rank=rank,boat=int(boat),racer_id=int(rid),motor=int(motor),equip_boat=int(bno),exhibit=float(exh),st=st,f=f));continue
        pm=PAY_RE.match(line)
        if pm:cur['payout']=int(pm.group(2))
    if cur is not None and len(cur['entrants'])>=3:races.append(cur)
    return races

def from_local_zips(paths):
    races=[]
    for zp in paths:
        with zipfile.ZipFile(zp) as z:
            for n in z.namelist():
                m=DATE_RE.search(n.split('/')[-1])
                if not m:continue
                dt=datetime.strptime('20'+m.group(1),'%Y%m%d').date()
                races.extend(parse_text(z.read(n).decode('cp932','replace'),dt))
    return races

def fetch_day(dt):
    ym=dt.strftime('%Y%m'); fn='k'+dt.strftime('%y%m%d')+'.lzh';url=f'https://www1.mbrace.or.jp/od2/K/{ym}/{fn}'
    for attempt in range(3):
        try:
            r=requests.get(url,timeout=30,headers={'User-Agent':'Mozilla/5.0 BoatRace OOS validation'})
            if r.status_code==404:return []
            r.raise_for_status()
            with tempfile.NamedTemporaryFile(suffix='.lzh') as f:
                f.write(r.content);f.flush()
                p=subprocess.run(['7z','e','-so',f.name],stdout=subprocess.PIPE,stderr=subprocess.DEVNULL,check=True)
            return parse_text(p.stdout.decode('cp932','replace'),dt)
        except Exception:
            if attempt==2:raise
    return []

def download_official():
    dates=[];d=date(2023,1,1);end=date(2025,12,31)
    while d<=end:dates.append(d);d+=timedelta(days=1)
    races=[]
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as ex:
        fut={ex.submit(fetch_day,d):d for d in dates}
        for i,f in enumerate(concurrent.futures.as_completed(fut),1):
            try:races.extend(f.result())
            except Exception as e:print('DOWNLOAD_FAIL',fut[f],repr(e),flush=True);raise
            if i%100==0:print('downloaded',i,'days races',len(races),flush=True)
    return races

def normalize(races):
    d={}
    for r in races:d[(r['date'],r['venue'],r['race_no'])]=r
    races=list(d.values());races.sort(key=lambda r:(r['date'],r['race_no'],r['venue']))
    return races

def fact(arr,keys):
    if len(keys)==1:codes,_=pd.factorize(arr[keys[0]],sort=False)
    else:codes,_=pd.factorize(pd.MultiIndex.from_arrays([arr[k] for k in keys]),sort=False)
    return codes.astype(np.int32),int(codes.max()+1)

@njit(cache=True)
def expanding(gids,finish,st,ff,ng):
    nr=len(gids);out=np.empty((nr,7),np.float32);n=np.zeros(ng,np.int32);win=np.zeros(ng,np.int32);top2=np.zeros(ng,np.int32);top3=np.zeros(ng,np.int32);fins=np.zeros(ng,np.float64);sts=np.zeros(ng,np.float64);stn=np.zeros(ng,np.int32);fs=np.zeros(ng,np.int32)
    for i in range(nr):
        g=gids[i];nn=n[g];out[i,0]=math.log1p(nn)
        if nn==0:out[i,1]=1/6;out[i,2]=2/6;out[i,3]=3/6;out[i,4]=3.5;out[i,6]=0
        else:out[i,1]=win[g]/nn;out[i,2]=top2[g]/nn;out[i,3]=top3[g]/nn;out[i,4]=fins[g]/nn;out[i,6]=fs[g]/nn
        out[i,5]=sts[g]/stn[g] if stn[g]>0 else .18
        rr=finish[i];n[g]+=1;fins[g]+=rr
        if rr==1:win[g]+=1
        if rr<=2:top2[g]+=1
        if rr<=3:top3[g]+=1
        if not np.isnan(st[i]):sts[g]+=st[i];stn[g]+=1
        fs[g]+=ff[i]
    return out

@njit(cache=True)
def recent(rg,finish,st,ng):
    nr=len(rg);out=np.empty((nr,2),np.float32);sb=np.full((ng,5),np.nan,np.float32);sp=np.zeros(ng,np.int32);sc=np.zeros(ng,np.int32);fb=np.full((ng,10),3.5,np.float32);fp=np.zeros(ng,np.int32);fc=np.zeros(ng,np.int32)
    for i in range(nr):
        g=rg[i]
        if sc[g]>0:
            s=0.;
            for j in range(sc[g]):s+=sb[g,j]
            out[i,0]=s/sc[g]
        else:out[i,0]=.18
        if fc[g]>0:
            s=0.;
            for j in range(fc[g]):s+=fb[g,j]
            out[i,1]=s/fc[g]
        else:out[i,1]=3.5
        if not np.isnan(st[i]):sb[g,sp[g]]=st[i];sp[g]=(sp[g]+1)%5;sc[g]=min(5,sc[g]+1)
        fb[g,fp[g]]=finish[i];fp[g]=(fp[g]+1)%10;fc[g]=min(10,fc[g]+1)
    return out

def build_features(races):
    venues=sorted({r['venue'] for r in races});vmap={v:i for i,v in enumerate(venues)};wdirs=sorted({r['wind_dir'] for r in races});wdmap={v:i for i,v in enumerate(wdirs)};weathers=sorted({r['weather'] for r in races});wmap={v:i for i,v in enumerate(weathers)}
    names=['date_int','year','venue','race_no','distance','weather','wind_dir','wind_speed','wave','boat','racer_id','motor','equip_boat','exhibit','finish','st','f','valid','race_seq']
    cols={k:[] for k in names};seq=0;validr=0
    for r in races:
        ents=r['entrants'];boats={e['boat'] for e in ents};top={e['rank']:e['boat'] for e in ents if e['rank'] in (1,2,3)};valid=(boats==set(range(1,7)) and set(top)=={1,2,3});validr+=int(valid)
        for e in sorted(ents,key=lambda x:x['boat']):
            if e['boat'] not in range(1,7):continue
            di=int((r['date']-pd.Timestamp('2023-01-01')).days)
            vals=dict(date_int=di,year=int(r['date'].year),venue=vmap[r['venue']],race_no=r['race_no'],distance=r['distance'],weather=wmap[r['weather']],wind_dir=wdmap[r['wind_dir']],wind_speed=r['wind_speed'],wave=r['wave'],boat=e['boat'],racer_id=e['racer_id'],motor=e['motor'],equip_boat=e['equip_boat'],exhibit=e['exhibit'],finish=e['rank'] if e['rank'] is not None else 7,st=e['st'] if not np.isnan(e['st']) else np.nan,f=e['f'],valid=int(valid),race_seq=seq)
            for k,v in vals.items():cols[k].append(v)
        seq+=1
    arr={k:np.asarray(v,dtype=np.float32 if k in ('exhibit','st') else np.int32) for k,v in cols.items()}
    groups={}
    for name,keys in [('racer',['racer_id']),('lane',['racer_id','boat']),('venuehist',['racer_id','venue']),('motorhist',['year','venue','motor']),('eboathist',['year','venue','equip_boat'])]:groups[name]=fact(arr,keys)
    blocks={name:expanding(g,arr['finish'],arr['st'],arr['f'],ng) for name,(g,ng) in groups.items()};rec=recent(groups['racer'][0],arr['finish'],arr['st'],groups['racer'][1])
    idx=np.flatnonzero(arr['valid']==1);vseq=arr['race_seq'][idx];_,counts=np.unique(vseq,return_counts=True);assert np.all(counts==6)
    N=len(idx)//6;exh=arr['exhibit'][idx].reshape(N,6);delta=(exh-exh.min(axis=1,keepdims=True)).reshape(-1).astype(np.float32);center=(exh-exh.mean(axis=1,keepdims=True)).reshape(-1).astype(np.float32)
    ranks=np.empty_like(exh,dtype=np.float32)
    for i in range(N):ranks[i]=pd.Series(exh[i]).rank(method='average',ascending=True).to_numpy(np.float32)
    D={'date_int':arr['date_int'][idx],'year':arr['year'][idx],'venue':arr['venue'][idx],'race_no':arr['race_no'][idx],'distance':arr['distance'][idx],'weather':arr['weather'][idx],'wind_dir':arr['wind_dir'][idx],'wind_speed':arr['wind_speed'][idx],'wave':arr['wave'][idx],'boat':arr['boat'][idx],'racer_id':arr['racer_id'][idx],'motor':arr['motor'][idx],'equip_boat':arr['equip_boat'][idx],'exhibit':arr['exhibit'][idx],'finish':arr['finish'][idx],'label':np.select([arr['finish'][idx]==1,arr['finish'][idx]==2,arr['finish'][idx]==3],[3,2,1],default=0).astype(np.int8),'race_seq':arr['race_seq'][idx],'exhibit_delta':delta,'exhibit_center':center,'exhibit_rank':ranks.reshape(-1),'recent5_st':rec[idx,0],'recent10_finish':rec[idx,1]}
    fn=['n_log','win','top2','top3','avg_finish','avg_st','f_rate']
    for name,b in blocks.items():
        for j,fname in enumerate(fn):D[f'{name}_{fname}']=b[idx,j]
    df=pd.DataFrame(D)
    for c in ['racer_win','racer_top3','racer_avg_finish','racer_avg_st','recent5_st','recent10_finish','lane_top3','venuehist_top3','motorhist_top3','eboathist_top3']:
        a=df[c].to_numpy(np.float32).reshape(N,6);df[c+'_rel']=(a-a.mean(axis=1,keepdims=True)).reshape(-1)
    df['venue_boat']=(df.venue*10+df.boat).astype('int32');df['wind_boat']=(df.wind_dir*10+df.boat).astype('int32');df['venue_wind_boat']=(df.venue*1000+df.wind_dir*10+df.boat).astype('int32');df['race_boat']=(df.race_no*10+df.boat).astype('int32')
    return df,validr,venues,wdirs,weathers

def make_history(races,out):
    def upd(d,k,rank,st,ff):
        a=d.get(k)
        if a is None:a=[0,0,0,0,0.,0.,0,0];d[k]=a
        rr=rank if rank is not None else 7;a[0]+=1;a[4]+=rr;a[1]+=int(rank==1);a[2]+=int(rank is not None and rank<=2);a[3]+=int(rank is not None and rank<=3)
        if st is not None and not np.isnan(st):a[5]+=float(st);a[6]+=1
        a[7]+=int(ff)
    racer={};lane={};vh={};rs=defaultdict(lambda:deque(maxlen=5));rf=defaultdict(lambda:deque(maxlen=10))
    for r in races:
        for e in r['entrants']:
            if e['boat'] not in range(1,7):continue
            rid=e['racer_id'];upd(racer,rid,e['rank'],e['st'],e['f']);upd(lane,(rid,e['boat']),e['rank'],e['st'],e['f']);upd(vh,(rid,r['venue']),e['rank'],e['st'],e['f'])
            if not np.isnan(e['st']):rs[rid].append(float(e['st']))
            rf[rid].append(e['rank'] if e['rank'] is not None else 7)
    snap={'racer':racer,'lane':lane,'venuehist':vh,'recent_st':{k:list(v) for k,v in rs.items()},'recent_finish':{k:list(v) for k,v in rf.items()}}
    with gzip.open(out,'wb',compresslevel=9) as f:pickle.dump(snap,f,protocol=pickle.HIGHEST_PROTOCOL)

def train(df,out):
    cat=['venue','boat','race_no','wind_dir','weather','venue_boat','wind_boat','venue_wind_boat','race_boat','racer_id'];exclude={'finish','label','race_seq','date_int','year','motor','equip_boat'};features=[c for c in df.columns if c not in exclude]
    for c in cat:df[c]=df[c].astype('category')
    tr=df[df.year<=2024];g=np.full(len(tr)//6,6,np.int32)
    params=dict(objective='lambdarank',metric='ndcg',ndcg_eval_at=[1,3],label_gain=[0,1,3,7],learning_rate=.03,n_estimators=223,num_leaves=31,min_child_samples=200,subsample=.9,subsample_freq=1,colsample_bytree=.85,reg_lambda=2.,reg_alpha=.1,max_bin=127,random_state=20260826,n_jobs=4,verbosity=-1,deterministic=True,force_col_wise=True,lambdarank_truncation_level=3)
    m=lgb.LGBMRanker(**params).fit(tr[features],tr.label,group=g,categorical_feature=cat,callbacks=[lgb.log_evaluation(0)]);m.booster_.save_model(str(out))
    import hashlib;sha=hashlib.sha256(Path(out).read_bytes()).hexdigest();print('MODEL_SHA',sha,flush=True)
    if sha!=EXPECTED_SHA:raise SystemExit(f'FROZEN MODEL SHA MISMATCH {sha}')
    return len(tr)//6

def main():
    local=os.environ.get('LOCAL_ZIPS','').strip()
    races=from_local_zips(local.split(os.pathsep)) if local else download_official();races=normalize(races)
    print('SOURCE_RACES',len(races),flush=True)
    df,validr,venues,wdirs,weathers=build_features(races);print('VALID_RACES',validr,'FEATURE_RACES',len(df)//6,flush=True);print('MAPS',venues,wdirs,weathers,flush=True)
    assert len(races)==145219 and len(df)//6==143270
    train_n=train(df,ROOT/'boatrace_strength_v1_lgbm.txt');assert train_n==96494
    make_history(races,ROOT/'history.pkl.gz')
    print('BOOTSTRAP_OK',flush=True)
if __name__=='__main__':main()
