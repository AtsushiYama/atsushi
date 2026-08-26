#!/usr/bin/env python3
from datetime import date
import re
import pandas as pd
from boatrace_lzh import LzhDownloader

TARGET=date(2026,8,27)
FW=str.maketrans('０１２３４５６７８９','0123456789')
dl=LzhDownloader(cache_dir='./cache')
files=dl.download(TARGET,'schedule')
if not files: raise SystemExit('B file not downloaded')
txt=next(iter(files.values()))

rows_by_race={}
venue=None; race_no=None
for raw in txt.splitlines():
    m=re.match(r'^(\d{2})BBGN\s*$',raw)
    if m:
        venue=int(m.group(1)); race_no=None; continue
    if re.match(r'^\d{2}BEND\s*$',raw):
        venue=None; race_no=None; continue
    norm=raw.translate(FW)
    mh=re.match(r'^\s*(\d{1,2})R\s',norm)
    if mh and venue is not None:
        race_no=int(mh.group(1)); continue
    if venue is None or race_no is None: continue
    if not re.match(r'^[1-6] \d{4}',raw): continue
    if len(raw)<58:
        print('SHORT',venue,race_no,repr(raw),flush=True); continue
    try:
        lane=int(raw[0:1]); racer=int(raw[2:6])
        name=raw[6:10].strip().replace('\u3000',' ')
        motor=int(raw[41:43].strip())
        equip=int(raw[49:52].strip())
    except Exception as e:
        print('PARSE_FAIL',venue,race_no,repr(raw),repr(e),flush=True); continue
    key=(venue,race_no)
    rec=rows_by_race.setdefault(key,{
        'レースコード':int(f'20260827{venue:02d}{race_no:02d}'),
        'レース日':'2026-08-27','レース場コード':f'{venue:02d}','レース回':f'{race_no:02d}R'})
    rec[f'艇{lane}_登録番号']=racer
    rec[f'艇{lane}_モーター番号']=motor
    rec[f'艇{lane}_ボート番号']=equip
    rec[f'艇{lane}_選手名']=name

rows=[]
for key,rec in sorted(rows_by_race.items()):
    ok=all(rec.get(c) not in (None,'') for b in range(1,7) for c in (f'艇{b}_登録番号',f'艇{b}_モーター番号',f'艇{b}_ボート番号'))
    if ok: rows.append(rec)
    else: print('INCOMPLETE',key,sorted(rec),flush=True)

df=pd.DataFrame(rows)
if df.empty: raise SystemExit('No complete races parsed')
df=df.sort_values(['レース場コード','レース回'])
df.to_csv('official_preday_20260827.csv',index=False,encoding='utf-8-sig')
print('RACES',len(df),'VENUES',df['レース場コード'].nunique(),flush=True)
print(df.groupby('レース場コード').size().to_string(),flush=True)
if len(df)!=156 or df['レース場コード'].nunique()!=13:
    raise SystemExit(f'Expected 156 races/13 venues, got {len(df)}/{df["レース場コード"].nunique()}')
print('PARSE_OK',flush=True)
