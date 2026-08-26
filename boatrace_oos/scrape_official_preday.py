#!/usr/bin/env python3
import re,time,requests,pandas as pd
from bs4 import BeautifulSoup
from concurrent.futures import ThreadPoolExecutor,as_completed

DATE='20260827'
BASE='https://www.boatrace.jp/owpc/pc/race/racelist'
HEADERS={'User-Agent':'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/122.0.0.0 Safari/537.36'}
S=requests.Session();S.headers.update(HEADERS)

def first_int(text, lo, hi):
    for m in re.findall(r'\d+', text):
        v=int(m)
        if lo<=v<=hi:return v
    return None

def scrape(jcd,rno):
    url=f'{BASE}?jcd={jcd:02d}&hd={DATE}&rno={rno}'
    for k in range(3):
        try:
            time.sleep(.25)
            resp=S.get(url,timeout=20);resp.raise_for_status()
            soup=BeautifulSoup(resp.text,'html.parser')
            out=[]
            rows=soup.select('tbody.is-fs12 tr.is-fs12')
            for tr in rows:
                cells=tr.find_all('td')
                if len(cells)<10: continue
                lane=first_int(cells[0].get_text(' ',strip=True),1,6)
                racer=first_int(cells[1].get_text(' ',strip=True),1000,9999)
                motor=first_int(cells[8].get_text(' ',strip=True),1,99)
                boat=first_int(cells[9].get_text(' ',strip=True),1,99)
                name=' '.join(cells[1].stripped_strings)
                if lane and racer and motor and boat:
                    out.append((lane,racer,motor,boat,name))
            # de-dup by lane
            d={x[0]:x for x in out}
            out=[d[x] for x in sorted(d)]
            if len(out)==6:return out
            return []
        except Exception:
            if k==2:return []
            time.sleep(1)
    return []

# detect active venues from R1
active=[]
with ThreadPoolExecutor(max_workers=8) as ex:
    fut={ex.submit(scrape,j,1):j for j in range(1,25)}
    for f in as_completed(fut):
        j=fut[f];x=f.result()
        if len(x)==6:active.append(j)
active=sorted(active)
print('ACTIVE',active,flush=True)

jobs=[(j,r) for j in active for r in range(1,13)]
raw={}
with ThreadPoolExecutor(max_workers=8) as ex:
    fut={ex.submit(scrape,j,r):(j,r) for j,r in jobs}
    for f in as_completed(fut):
        key=fut[f];raw[key]=f.result()
        print(key,len(raw[key]),flush=True)

rows=[]
for j,r in jobs:
    boats=raw.get((j,r),[])
    if len(boats)!=6:
        print('MISSING',j,r,flush=True);continue
    rec={'レースコード':int(f'{DATE}{j:02d}{r:02d}'),'レース日':'2026-08-27','レース場コード':f'{j:02d}','レース回':f'{r:02d}R'}
    for lane,racer,motor,boat,name in boats:
        rec[f'艇{lane}_登録番号']=racer;rec[f'艇{lane}_モーター番号']=motor;rec[f'艇{lane}_ボート番号']=boat;rec[f'艇{lane}_選手名']=name
    rows.append(rec)
df=pd.DataFrame(rows).sort_values(['レース場コード','レース回'])
df.to_csv('official_preday_20260827.csv',index=False,encoding='utf-8-sig')
print('RACES',len(df),'VENUES',df['レース場コード'].nunique(),flush=True)
print(df[['レース場コード','レース回']].groupby('レース場コード').size().to_string(),flush=True)
