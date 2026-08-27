# Strength-v1.0 Local Notifier / Phase 1

## Scope

WindowsローカルPCで当日レースを監視し、凍結済みStrength-v1.0の **S/Aだけ** をPushoverでiPhoneへ通知する。

通知内容:

- 場 / R
- 締切時刻
- S/A
- P4
- 3連単上位4点

Phase 1に存在しない処理:

- オッズ取得
- 合成オッズ判定
- 自動投票
- 投票アカウント情報の保存

## 1. ブランチ取得

```powershell
git fetch origin
git switch strength-v1-local-notifier
cd boatrace_oos
```

## 2. Python環境

```powershell
Set-ExecutionPolicy -Scope Process Bypass
.\setup_windows.ps1
```

## 3. Frozen model / history

以下が既にあれば再生成不要。

```text
boatrace_strength_v1_lgbm.txt
history.pkl.gz
```

ない場合は次の3ファイルを `boatrace_oos/data_zips/` に置く。

```text
race_2023.zip
race_2024.zip
race.zip
```

その後:

```powershell
.\bootstrap_from_zips.ps1
```

既存 `bootstrap_frozen.py` がSHA-256を照合する。Strength-v1.0の固定SHAと一致しないモデルは使用しない。

## 4. Pushover

`.env.example` を `.env` にコピーし、以下を設定する。

```text
PUSHOVER_USER_KEY=...
PUSHOVER_APP_TOKEN=...
```

`.env` はGitへコミットしない。

通知テスト:

```powershell
.\.venv\Scripts\python.exe .\strength_local_notifier.py --test-notification
```

## 5. Strength契約セルフチェック

```powershell
.\.venv\Scripts\python.exe .\strength_local_notifier.py --self-check
```

期待値:

```text
MODEL_SHA 334b3a54a482a957f50c51b40614797821ae765d221734b92b95f1d4fb96cde0
FEATURES 64
T 1.2184870199794324
A_THR 0.31242984023268894
S_THR 0.37995532313528696
```

## 6. 1回だけ手動実行

```powershell
.\.venv\Scripts\python.exe .\strength_local_notifier.py
```

初回は `history.pkl.gz` を起点に2026-01-01から前日までをロールフォワードし、`runtime/state_checkpoint.pkl.gz` を作る。以降は不足日だけ追加する。

当日は `data/results/realtime` の確定済み結果を状態へ追加してから推論する。

## 7. データ

現在の入力はBoatraceCSVの次のCSVのみ。

```text
data/programs/race_cards/YYYY/MM/DD.csv
data/previews/tkz/YYYY/MM/DD.csv
data/previews/sui/YYYY/MM/DD.csv
data/results/realtime/YYYY/MM/DD.csv
```

オッズ (`od3`) はコードから参照していない。

## 8. 評価タイミング

`.env` 既定値:

```text
EVAL_FROM_MIN=10
EVAL_UNTIL_MIN=3
```

締切10分前を切ってから1回だけ推論する。

- S/A: 通知
- skip: SQLiteへ記録のみ
- 展示/気象が不足: 評価済みにせず次回再試行
- 3分前を切って初めて推論可能: `too_late` として通知しない

## 9. 二重通知防止

`runtime/notifier.db` の `race_code` をPRIMARY KEYにしている。

Pushover送信成功後は `sent`、対象外は `skipped`。送信エラー時のみ予約行を削除し、次回タスクで再送を試す。

## 10. Task Scheduler

推奨:

- タスク名: `Strength-v1 Local Notifier`
- トリガー: 毎日 08:00
- 繰り返し: 2分間隔
- 継続時間: 15時間
- 「スケジュールされた時刻に開始できなかった場合、すぐに実行」: ON
- 既に実行中の場合: **新しいインスタンスを開始しない**
- 必要なら「タスクを実行するためにスリープを解除」: ON

操作:

```text
プログラム/スクリプト:
C:\Windows\System32\cmd.exe

引数の追加:
/c "C:\YOUR_PATH\atsushi\boatrace_oos\run_notifier.bat"

開始:
C:\YOUR_PATH\atsushi\boatrace_oos
```

## 11. 状態確認

直近20件:

```powershell
.\.venv\Scripts\python.exe .\strength_local_notifier.py --status
```

ログ:

```text
runtime/strength_notifier.log
runtime/task.log
```

## Phase 1 completion criteria

1. `--self-check` が通る。
2. 同一レースを既存手動Strength-v1.0で計算したとき、S/A・P4・4点が一致する。
3. 同一レースの二重通知がない。
4. skipはiPhoneへ届かない。
5. S/Aだけ締切時刻付きで届く。
6. PC再起動後もTask Schedulerから起動する。
7. オッズ・自動投票処理がない。
