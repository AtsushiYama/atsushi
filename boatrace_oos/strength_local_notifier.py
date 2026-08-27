#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import os
import pickle
import sqlite3
import sys
import tempfile
import time
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple
from zoneinfo import ZoneInfo

import lightgbm as lgb
import pandas as pd
import requests

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
import run_oos as core

JST = ZoneInfo("Asia/Tokyo")
RUNTIME = ROOT / "runtime"
CACHE = RUNTIME / "cache"
DB_PATH = RUNTIME / "notifier.db"
CHECKPOINT_PATH = RUNTIME / "state_checkpoint.pkl.gz"
LOG_PATH = RUNTIME / "strength_notifier.log"
ENV_PATH = ROOT / ".env"
BASE_URL = "https://boatracecsv.github.io"
BASE_HISTORY_THROUGH = date(2025, 12, 31)

@dataclass(frozen=True)
class Config:
    user_key: str
    app_token: str
    priority: int = 1
    eval_from_min: float = 10.0
    eval_until_min: float = 3.0
    timeout: float = 20.0
    dry_run: bool = False

def log(msg: str) -> None:
    RUNTIME.mkdir(parents=True, exist_ok=True)
    line = f"{datetime.now(JST).isoformat(timespec='seconds')} {msg}"
    print(line, flush=True)
    with LOG_PATH.open("a", encoding="utf-8") as f:
        f.write(line + "\n")

def load_env() -> None:
    if not ENV_PATH.exists(): return
    for raw in ENV_PATH.read_text(encoding="utf-8-sig").splitlines():
        s = raw.strip()
        if not s or s.startswith("#") or "=" not in s: continue
        k, v = s.split("=", 1)
        v = v.strip().strip('"').strip("'")
        os.environ.setdefault(k.strip(), v)

def config() -> Config:
    load_env()
    return Config(
        os.getenv("PUSHOVER_USER_KEY", "").strip(),
        os.getenv("PUSHOVER_APP_TOKEN", "").strip(),
        int(os.getenv("PUSHOVER_PRIORITY", "1")),
        float(os.getenv("EVAL_FROM_MIN", "10")),
        float(os.getenv("EVAL_UNTIL_MIN", "3")),
        float(os.getenv("REQUEST_TIMEOUT_SEC", "20")),
        os.getenv("DRY_RUN", "0").lower() in {"1","true","yes","on"},
    )

def ensure_runtime() -> None:
    RUNTIME.mkdir(parents=True, exist_ok=True); CACHE.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(DB_PATH) as con:
        con.execute("""CREATE TABLE IF NOT EXISTS evaluations(
          race_code TEXT PRIMARY KEY,evaluated_at TEXT NOT NULL,status TEXT NOT NULL,
          rating TEXT,p4 REAL,top4 TEXT,deadline TEXT,data_timestamp TEXT,error TEXT)""")
        con.commit()

def self_check() -> lgb.Booster:
    if not core.MODEL.exists(): raise FileNotFoundError(f"model missing: {core.MODEL}")
    if not core.HISTORY.exists(): raise FileNotFoundError(f"history missing: {core.HISTORY}")
    h = hashlib.sha256(core.MODEL.read_bytes()).hexdigest()
    if h != core.MODEL_SHA: raise RuntimeError(f"MODEL SHA MISMATCH {h}")
    if core.T != 1.2184870199794324: raise RuntimeError("temperature mismatch")
    if core.A_THR != 0.31242984023268894: raise RuntimeError("A threshold mismatch")
    if core.S_THR != 0.37995532313528696: raise RuntimeError("S threshold mismatch")
    booster = lgb.Booster(model_file=str(core.MODEL))
    if booster.feature_name() != core.FEATURES: raise RuntimeError("FEATURE CONTRACT MISMATCH")
    return booster

def url(kind: str, d: date) -> str:
    ds=f"{d:%Y/%m/%d}.csv"
    paths={
      "cards":f"data/programs/race_cards/{ds}",
      "tkz":f"data/previews/tkz/{ds}",
      "sui":f"data/previews/sui/{ds}",
      "results":f"data/results/realtime/{ds}",
    }
    return f"{BASE_URL}/{paths[kind]}"

def fetch_csv(session: requests.Session, kind: str, d: date, cfg: Config, immutable=False) -> pd.DataFrame:
    cp=CACHE/f"{d:%Y%m%d}_{kind}.csv"
    if immutable and cp.exists() and cp.stat().st_size:
        return pd.read_csv(cp,dtype=str,keep_default_na=False)
    r=session.get(url(kind,d),params=None if immutable else {"_":str(int(time.time()))},
      headers={"User-Agent":"Strength-v1.0-local-notifier/1.0","Cache-Control":"no-cache"},timeout=cfg.timeout)
    if r.status_code==404: return pd.DataFrame()
    r.raise_for_status()
    text=r.content.decode("utf-8-sig")
    if not text.strip(): return pd.DataFrame()
    df=pd.read_csv(io.StringIO(text),dtype=str,keep_default_na=False)
    if immutable and not df.empty:
        tmp=cp.with_suffix(".tmp"); tmp.write_bytes(r.content); tmp.replace(cp)
    return df

def by_code(df: pd.DataFrame) -> Dict[str,pd.Series]:
    if df.empty or "レースコード" not in df.columns: return {}
    return {str(r["レースコード"]).strip():r for _,r in df.iterrows() if str(r["レースコード"]).strip()}

def parts(code: str) -> Tuple[int,int]:
    if len(code)<12 or not code.isdigit(): raise ValueError(code)
    return int(code[-4:-2]),int(code[-2:])

def apply_results(state: dict,cards: pd.DataFrame,results: pd.DataFrame) -> int:
    c,r=by_code(cards),by_code(results); keys=[]
    for code in set(c)&set(r):
        try: venue,race=parts(code)
        except ValueError: continue
        keys.append((race,venue,code))
    keys.sort(); n=0
    for race,venue,code in keys:
        n += int(core.update_state_from_race(c[code],r[code],state,venue))
    return n

def base_state() -> dict:
    with gzip.open(core.HISTORY,"rb") as f: s=pickle.load(f)
    s["motorhist"]={}; s["eboathist"]={}
    s["recent_st"]={int(k):list(v) for k,v in s["recent_st"].items()}
    s["recent_finish"]={int(k):list(v) for k,v in s["recent_finish"].items()}
    return s

def load_checkpoint():
    if not CHECKPOINT_PATH.exists(): return None
    try:
        with gzip.open(CHECKPOINT_PATH,"rb") as f: x=pickle.load(f)
        if x.get("model_sha")!=core.MODEL_SHA: return None
        return date.fromisoformat(x["through"]),x["state"]
    except Exception as e:
        log(f"CHECKPOINT_READ_FAIL {e!r}"); return None

def save_checkpoint(state: dict,through: date) -> None:
    fd,name=tempfile.mkstemp(prefix="state_",suffix=".pkl.gz",dir=RUNTIME); os.close(fd); p=Path(name)
    try:
        with gzip.open(p,"wb",compresslevel=6) as f:
            pickle.dump({"through":through.isoformat(),"model_sha":core.MODEL_SHA,"state":state},f,pickle.HIGHEST_PROTOCOL)
        p.replace(CHECKPOINT_PATH)
    finally: p.unlink(missing_ok=True)

def state_yesterday(session: requests.Session,cfg: Config,today: date) -> dict:
    target=today-timedelta(days=1); cp=load_checkpoint()
    if cp is None or cp[0]<BASE_HISTORY_THROUGH or cp[0]>target:
        through,state=BASE_HISTORY_THROUGH,base_state()
    else: through,state=cp
    d=through+timedelta(days=1); unsaved=0
    while d<=target:
        if d.month==1 and d.day==1: state["motorhist"]={}; state["eboathist"]={}
        cards=fetch_csv(session,"cards",d,cfg,True); results=fetch_csv(session,"results",d,cfg,True)
        if cards.empty or results.empty:
            raise RuntimeError(f"historical source incomplete {d}: cards={len(cards)} results={len(results)}")
        n=apply_results(state,cards,results); log(f"ROLL_FORWARD {d} races={n}")
        through=d; unsaved+=1
        if unsaved>=7 or d==target: save_checkpoint(state,through); unsaved=0
        d+=timedelta(days=1)
    return state

def state_now(session: requests.Session,cfg: Config,today: date,cards: pd.DataFrame) -> dict:
    state=state_yesterday(session,cfg,today)
    results=fetch_csv(session,"results",today,cfg,False)
    if not results.empty: log(f"TODAY_ROLL_FORWARD completed_races={apply_results(state,cards,results)}")
    return state

def deadline(today: date,*rows: pd.Series) -> Optional[datetime]:
    for row in rows:
        for key in ("締切時刻","締切予定時刻"):
            s=str(row.get(key,"")).strip()
            for fmt in ("%H:%M","%H:%M:%S"):
                try: return datetime.combine(today,datetime.strptime(s,fmt).time(),tzinfo=JST)
                except ValueError: pass
    return None

def data_ts(*rows: pd.Series) -> str:
    a=[str(r.get("取得日時","")).strip() for r in rows if str(r.get("取得日時","")).strip()]
    return max(a) if a else ""

def evaluated(code: str) -> bool:
    with sqlite3.connect(DB_PATH) as con:
        return con.execute("SELECT 1 FROM evaluations WHERE race_code=?",(code,)).fetchone() is not None

def reserve(code,rating,p4,combos,dl,ts,status) -> bool:
    try:
        with sqlite3.connect(DB_PATH) as con:
            con.execute("INSERT INTO evaluations VALUES(?,?,?,?,?,?,?,?,NULL)",(
              code,datetime.now(JST).isoformat(timespec="seconds"),status,rating,float(p4),"|".join(combos),dl.isoformat(timespec="minutes"),ts))
            con.commit(); return True
    except sqlite3.IntegrityError: return False

def set_status(code,status,error=""):
    with sqlite3.connect(DB_PATH) as con:
        con.execute("UPDATE evaluations SET status=?,error=? WHERE race_code=?",(status,error,code)); con.commit()

def remove(code):
    with sqlite3.connect(DB_PATH) as con: con.execute("DELETE FROM evaluations WHERE race_code=?",(code,)); con.commit()

def mark_late(code,dl,ts):
    with sqlite3.connect(DB_PATH) as con:
        con.execute("INSERT OR IGNORE INTO evaluations VALUES(?,?,?,?,?,?,?,?,?)",(
          code,datetime.now(JST).isoformat(timespec="seconds"),"too_late","",None,"",dl.isoformat(timespec="minutes"),ts,"preview too late")); con.commit()

def push(cfg: Config,title: str,message: str):
    if cfg.dry_run: log("DRY_RUN "+title+" | "+message.replace("\n"," | ")); return
    if not cfg.user_key or not cfg.app_token: raise RuntimeError("Pushover keys missing in .env")
    r=requests.post("https://api.pushover.net/1/messages.json",data={
      "token":cfg.app_token,"user":cfg.user_key,"title":title,"message":message,"priority":cfg.priority},timeout=cfg.timeout)
    r.raise_for_status(); body=r.json()
    if body.get("status")!=1: raise RuntimeError(str(body))

def message(venue,race,rating,p4,combos,dl,mins,ts):
    title=f"Strength-v1.0 {rating} | {venue}{race}R"
    body="\n".join([venue+f" {race}R",f"締切 {dl:%H:%M}（残り約{mins:.1f}分）",f"判定 {rating} / P4 {p4*100:.2f}%","","買い目4点",*combos,"","オッズ未参照 / 自動投票なし"]+([f"直前データ {ts}"] if ts else []))
    return title,body

def run_once(cfg: Config) -> int:
    ensure_runtime(); booster=self_check(); now=datetime.now(JST); today=now.date()
    with requests.Session() as session:
        cards=fetch_csv(session,"cards",today,cfg,False); tkz=fetch_csv(session,"tkz",today,cfg,False); sui=fetch_csv(session,"sui",today,cfg,False)
        if cards.empty: log("NO_CARDS"); return 0
        if tkz.empty or sui.empty: log(f"NO_PREVIEW tkz={len(tkz)} sui={len(sui)}"); return 0
        state=state_now(session,cfg,today,cards); c,t,s=by_code(cards),by_code(tkz),by_code(sui)
        q=[]
        for code in set(c)&set(t)&set(s):
            if evaluated(code): continue
            try: venue,race=parts(code)
            except ValueError: continue
            dl=deadline(today,t[code],s[code])
            if dl: q.append((dl,code,venue,race,(dl-now).total_seconds()/60))
        sent=0; q.sort()
        for dl,code,venue,race,mins in q:
            ts=data_ts(t[code],s[code])
            if mins>cfg.eval_from_min: continue
            if mins<cfg.eval_until_min:
                if mins>0: mark_late(code,dl,ts); log(f"TOO_LATE {code} {mins:.2f}m")
                continue
            x,err=core.build_feature_frame(c[code],t[code],s[code],state,venue,race)
            if err: log(f"FEATURE_WAIT {code} {err}"); continue
            raw=booster.predict(x[core.FEATURES]); _,_,combos,p4=core.pl_top4(raw,x.boat.to_numpy()); rating=core.classify_p4(p4)
            name=core.VENUE_CODE_TO_NAME.get(venue,f"場{venue:02d}")
            if rating not in ("S","A"):
                if reserve(code,rating,p4,combos,dl,ts,"skipped"): log(f"SKIP {name}{race}R P4={p4:.6f}")
                continue
            if not reserve(code,rating,p4,combos,dl,ts,"reserved"): continue
            title,body=message(name,race,rating,p4,combos,dl,mins,ts)
            try:
                push(cfg,title,body); set_status(code,"sent"); sent+=1
                log(f"SENT {name}{race}R {rating} P4={p4:.6f} top4={'|'.join(combos)}")
            except Exception as e:
                remove(code); log(f"NOTIFY_FAIL {code} {e!r}")
        log(f"DONE sent={sent}"); return sent

def show_status():
    ensure_runtime()
    with sqlite3.connect(DB_PATH) as con:
        rows=con.execute("SELECT race_code,status,rating,p4,top4,deadline,evaluated_at FROM evaluations ORDER BY evaluated_at DESC LIMIT 20").fetchall()
    for r in rows: print(",".join("" if x is None else str(x) for x in r))

def main():
    p=argparse.ArgumentParser(); p.add_argument("--self-check",action="store_true"); p.add_argument("--test-notification",action="store_true"); p.add_argument("--status",action="store_true"); a=p.parse_args(); cfg=config(); ensure_runtime()
    if a.self_check:
        b=self_check(); print("SELF_CHECK_OK"); print("MODEL_SHA",core.MODEL_SHA); print("FEATURES",len(core.FEATURES)); print("T",core.T); print("A_THR",core.A_THR); print("S_THR",core.S_THR); print("TREES",b.num_trees()); return
    if a.test_notification: push(cfg,"Strength-v1.0 通知テスト","iPhone通知テスト\nオッズ未参照 / 自動投票なし"); print("TEST_NOTIFICATION_SENT"); return
    if a.status: show_status(); return
    run_once(cfg)

if __name__=="__main__": main()
