#!/usr/bin/env python3
from datetime import date
import pandas as pd
from boatrace_lzh import LzhDownloader, ScheduleParser

TARGET=date(2026,8,27)
dl=LzhDownloader(cache_dir='./cache')
files=dl.download(TARGET,'schedule')
print('FILES',files,flush=True)
parser=ScheduleParser()
res=parser.parse(files)
print('PARSED races',len(res.races),'entries',len(res.entries),'racers',len(res.racers),flush=True)

name_by_id={r.racer_number:r.name for r in res.racers}
rows_by_race={}
for e in res.entries:
    try:
        j=int(e.venue_code); rno=int(e.race_number); lane=int(e.boat_number)
    except Exception:
        continue
    if lane not in range(1,7): continue
    key=(j,rno)
    rec=rows_by_race.setdefault(key,{
        'レースコード':int(f'20260827{j:02d}{rno:02d}'),
        'レース日':'2026-08-27','レース場コード':f'{j:02d}','レース回':f'{rno:02d}R'})
    rec[f'艇{lane}_登録番号']=int(e.racer_number)
    rec[f'艇{lane}_モーター番号']=e.motor_number
    rec[f'艇{lane}_ボート番号']=e.boat_part
    rec[f'艇{lane}_選手名']=name_by_id.get(e.racer_number,'')

rows=[]
for key,rec in sorted(rows_by_race.items()):
    ok=True
    for b in range(1,7):
        for c in (f'艇{b}_登録番号',f'艇{b}_モーター番号',f'艇{b}_ボート番号'):
            if rec.get(c) in (None,''):
                ok=False
    if ok: rows.append(rec)
    else: print('INCOMPLETE',key,rec,flush=True)

df=pd.DataFrame(rows)
if df.empty:
    raise SystemExit('No complete races parsed from official B file')
df=df.sort_values(['レース場コード','レース回'])
df.to_csv('official_preday_20260827.csv',index=False,encoding='utf-8-sig')
print('RACES',len(df),'VENUES',df['レース場コード'].nunique(),flush=True)
print(df.groupby('レース場コード').size().to_string(),flush=True)
