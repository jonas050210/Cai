# CodePlan-LM

Ein **privates Lern- und Forschungsprojekt**: ein kleines Sprachmodell, das **von Grund auf** trainiert wird — ohne fremde Modellgewichte.

Ziel:
- natürliche Sprache: **Deutsch / Englisch**
- Fokus: **Coding + technische Planung**
- Datenmischung: ca. **99 % Code + Planung**, 1 % allgemeine DE/EN-Sprache
- Reasoning-ähnliches Verhalten über strukturierte SFT-Beispiele: `<plan>`, `<code>`, `<tests>`, `<answer>`

> Wichtig: Ein 50M-Modell von Grund auf ist **kein ChatGPT-Ersatz**. Es wird anfangs schlecht sein. Dieses Repo ist ein sauberer Startpunkt, um Daten, Tokenizer, Training, SFT und Evaluierung selbst aufzubauen.

## Hardware-Ziel

Getestet als Smoke-Test auf CPU. Vorgesehen für:
- Ubuntu
- NVIDIA RTX 4060 Ti, 8 GB VRAM
- 32 GB RAM
- Python 3.10+
- PyTorch mit CUDA

Die 50M-Konfiguration nutzt:
- `bf16`
- Gradient Checkpointing
- Batch Size 1
- Gradient Accumulation
- packed binary datasets

## Schnellstart: Smoke-Test

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
bash scripts/smoke_test.sh
```

Der Smoke-Test nutzt winzige Seed-Daten und ein 0.12M-Modell. Er beweist nur, dass die Pipeline läuft. Er erzeugt **kein nützliches Modell**.

## Echte Pipeline

```bash
# 0. (Optional) Permissiv lizenzierte Repos von GitHub sammeln (MIT/Apache/BSD/...)
codeplan-collect --out data/raw/github.jsonl --max-repos 200 --languages python,typescript

# 1. Eigene lokale Dateien einlesen
codeplan-ingest --input /pfad/zu/deinen/dateien --out data/raw/my_data.jsonl --license "MIT" --source "my-repos"

# 2. Filtern: DE/EN, exakte + Near-Duplikate, Split
codeplan-prepare --input data/raw/my_data.jsonl --out-dir data/filtered --require-license --near-dedup

# 2b. Prüfen, ob die Datenmenge für den geplanten Lauf reicht
codeplan-stats --input data/filtered/train.jsonl --target-tokens 1000000000

# 3. Vokabular-Größe belegen (Tokens/Zeichen pro Domäne und Sprache)
codeplan-fertility --train data/filtered/train.jsonl --vocab-sizes 16000,32000 --out artifacts/fertility.json

# 3b. Tokenizer von Grund auf trainieren (Vokabular aus Schritt 3)
codeplan-tokenizer --train data/filtered/train.jsonl --out artifacts/tokenizer.json --vocab-size 16000

# 4. Daten packen
codeplan-pack --filtered-dir data/filtered --tokenizer artifacts/tokenizer.json --out-dir data/packed

# 5. Vortest: Umgebung, Daten, VRAM und Tempo prüfen (vor jedem langen Lauf)
codeplan-doctor --config configs/pretrain_50m_4060ti.json --bench-steps 20

# 6. Pretraining (ein Lauf oder in kurzen Sessions mit Thermal-Schutz)
codeplan-pretrain --config configs/pretrain_50m_4060ti.json
# oder: bash scripts/train_session.sh --session-steps 800 --max-total 20000

# 6. SFT für strukturierte Coding-/Planungsantworten
codeplan-sft --config configs/sft_50m_4060ti.json --data data/sft/train.jsonl

# 7. Evaluieren: Perplexity pro Domäne + Coding-Aufgaben mit Testausführung
codeplan-eval --checkpoint checkpoints/sft_50m/last.pt --data-dir data/packed --split test --seq-len 2048 --per-domain
codeplan-bench --checkpoint checkpoints/sft_50m/last.pt --tasks data/bench/code_tasks.jsonl --allow-exec

# 8. Chat im Browser
codeplan-serve --checkpoint checkpoints/sft_50m/last.pt --port 8000
# -> http://localhost:8000

# 8b. Chat im Terminalten
codeplan-chat --checkpoint checkpoints/sft_50m/last.pt

# 9. Exportieren und lokal servieren
codeplan-export --checkpoint checkpoints/sft_50m/last.pt --out artifacts/model.safetensors --dtype bf16
codeplan-serve --checkpoint checkpoints/sft_50m/last.pt --host 0.0.0.0 --port 8000
```

## Werkzeuge

| Befehl | Zweck |
|---|---|
| `codeplan-collect` | GitHub-Repos nach Lizenz-Whitelist laden (+ Manifest mit Checksums) |
| `codeplan-stats` | Korpusgröße, Domänen-/Sprachmix, Duplikate, Token-Budget |
| `codeplan-doctor` | Preflight: Umgebung, Config, Daten, VRAM, Durchsatz, ETA |
| `codeplan-sft-data` | SFT-Paare aus Code minen (implement/explain/test aus Docstrings, JSDoc, Go-/Rust-Kommentaren) |
| `codeplan-ingest` | Lokale Dateien zu JSONL |
| `codeplan-prepare` | DE/EN-Filter, Code-Kommentar-Filter, Dedup, Split |
| `codeplan-fertility` | Tokenizer-Kennzahlen, Vokabular-Empfehlung |
| `codeplan-tokenizer` / `codeplan-pack` | BPE-Training, binäre Datasets |
| `codeplan-pretrain` | Pretraining (resumierbar, Early Stopping) |
| `codeplan-sft` | SFT mit Assistant-only-Loss und Validierungssplit |
| `codeplan-eval` | Perplexity gesamt oder pro Domäne |
| `codeplan-bench` | Coding-Aufgaben, Tests im Subprozess, pass@k, Sprach-Compliance |
| `codeplan-export` | safetensors + Config (`load_exported()` zum Zurückladen) |
| `codeplan-serve` | FastAPI-Server (`/health`, `/generate`, Web-UI unter `/`) |

> `codeplan-bench` führt generierten Code als deinen Benutzer aus. Es begrenzt
> CPU-Zeit, Speicher und Laufzeit, ist aber **kein** echter Sandbox-Ersatz.

## Wichtige Dateien

```text
configs/pretrain_50m_4060ti.json   # 50M-Modell für 8 GB VRAM
configs/sft_50m_4060ti.json        # SFT-Konfiguration
configs/tiny_smoke.json            # Mini-Pipeline-Test
src/codeplan/model.py              # Transformer von Grund auf
src/codeplan/tokenizer.py          # BPE-Tokenizer + Chat-Template
src/codeplan/prepare.py            # DE/EN-Filter, Dedup, Split
src/codeplan/pack.py               # Binary-Datasets
src/codeplan/trainer.py            # Training-Loop, Checkpoint-Retention, Early Stopping
src/codeplan/collect.py            # GitHub-Sammler mit Lizenz-Whitelist
src/codeplan/tokenizer_eval.py     # Tokenizer-Fertility / Vokabular-Entscheidung
src/codeplan/bench.py              # Coding-Benchmark mit Testausführung
src/codeplan/export.py             # safetensors-Export
src/codeplan/server.py             # FastAPI-Inferenzserver
src/codeplan/dedup.py              # MinHash + LSH Near-Duplikate
src/codeplan/stats.py              # Korpus-Statistik, Token-Budget
src/codeplan/doctor.py             # Preflight-Checks + Durchsatzmessung
scripts/pilot.sh                   # kompletter Pilot-Lauf, reproduzierbar
scripts/train_session.sh           # Training in Sessions + Thermal-Guard
data/seed/                         # winzige synthetische Testdaten
AI_HANDOFF.md                      # Übergabe für andere KIs / Entwickler
```

## Token einmal setzen

```bash
bash scripts/install_token.sh
```
Fragt den Token versteckt ab, prüft ihn gegen die GitHub-API, legt ihn in
`~/.config/codeplan/env` (mode 600) ab und hängt eine `source`-Zeile in
`~/.bashrc`. Danach nutzen ihn alle Terminals und alle Werkzeuge — auch ohne
Neustart, weil der Sammler die Datei selbst liest.

## Was das Modell nicht kann

`codeplan-eval --token-classes` teilt den Val-Loss nach Token-Art auf, statt
eine Zahl zu zeigen:

```
class             tokens   share     loss       ppl
identifier           225   44.1%    4.515     91.41
operator             138   27.1%    0.922      2.51
keyword               55   10.8%    0.897      2.45
```

Bezeichner sind der Engpass — Operatoren und Keywords sitzen. Genau das, was
„Perplexity 15" verdeckt.

## Training unterbrechen

`Ctrl+C` rechnet den laufenden Step zu Ende, speichert `last.pt` (inkl. Optimizer
und RNG) und beendet sich sauber. Derselbe Befehl nochmal gestartet macht weiter:

```bash
codeplan-pretrain --config configs/pretrain_100m_4060ti.json   # erster Start
# ... Ctrl+C ...
codeplan-pretrain --config configs/pretrain_100m_4060ti.json   # läuft weiter
```

`--fresh` startet bewusst von vorn. Gilt auch für `codeplan-sft`.

## Checkpoints mitteln

```bash
codeplan-soup --dir checkpoints/pretrain_100m --last 5 --out checkpoints/pretrain_100m/soup.pt
codeplan-eval --checkpoint checkpoints/pretrain_100m/soup.pt --data-dir data/packed
```

Gratis-Qualität: kein Training, kein zusätzliches Datum, meist 2–5 % besserer
Val-Loss.

## Kuratierte Repos

`data/repos/core.txt` — **53 Repositories**, jede Lizenz einzeln über die
GitHub-API geprüft (MIT, Apache-2.0, BSD, ISC, Unlicense), alle unter 300 MB
History. `data/repos/heavy.txt` enthält die großen (Storybook, TypeScript, …).

```bash
bash scripts/build_corpus.sh          # sammeln -> filtern -> stats -> Tokenizer -> packen -> doctor
```

Ein Befehl für die ganze Datenkette; jeder Schritt wird übersprungen, wenn sein
Ergebnis schon existiert. `--core-only` nimmt nur die 53 Repos, `--stop-after stats`
hält nach dem Token-Budget an.

## Modelle

| Config | Params | Einsatz |
|---|---|---|
| `pretrain_pilot.json` | 40M | Pipeline-Test, 1–2 h |
| `pretrain_50m_4060ti.json` | 40M | Baseline, 20k Steps |
| `pretrain_100m_4060ti.json` | **99M** | Zielmodell, 983M Tokens, ~8–14 h |

Details, VRAM-Rechnung und Datenanforderungen: `docs/100M_PLAN.md`.

## Chat-GUI

`codeplan-serve` startet eine lokale Web-App (kein Build, keine npm-Abhängigkeit):
Streaming-Antworten, Checkpoint-Umschalter, Regler für temperature/top_k/top_p,
Markdown- und Codeblöcke mit Copy-Button, Verlauf im Browser, Export als `.md`.

```bash
codeplan-serve --checkpoint checkpoints/sft_50m/last.pt --port 8000
```

Endpunkte: `GET /` (UI), `GET /api/health`, `GET /api/checkpoints`,
`POST /api/load`, `POST /api/chat` (SSE-Streaming), `POST /generate` (JSON).

## Performance

Alles, was pro Dokument rechnet, läuft mehrkernig:

| Schritt | Beschleunigung | Wo |
|---|---|---|
| `prepare` (Filter, Sprache, MinHash) | `--workers N`, Default alle Kerne außer einem | Prozess-Pool, Ergebnis byte-identisch zu seriell |
| `pack` | `encode_batch` statt Einzelaufrufe (~1,6x auf 2 Kernen, mehr auf vielen) | Rust-Backend von `tokenizers` parallelisiert selbst |
| Training (CPU) | `torch.set_num_threads(alle Kerne)` | automatisch bei `device=cpu` |
| Datenlader | `persistent_workers` + `prefetch_factor 6` | hält die GPU gesättigt |
| GPU | `"torch_compile": true` | +30–80 % auf Ada |

Gemessen auf 2 Kernen: `prepare` 5,9 s → 3,6 s für 2.000 Dokumente (18,7 MB);
`encode_batch` 4,7 s → 2,9 s. Auf 20 Threads skaliert es weiter, weil beide
Stufen reine CPU-Arbeit sind.

## Erster echter Test

Zwei Stufen, Details in `docs/RUNBOOK.md`:

```bash
export GITHUB_TOKEN=...     # 60 -> 5000 Requests/Stunde
bash scripts/pilot.sh       # ~12M Tokens echte Daten, 16k Tokenizer, Preflight
codeplan-pretrain --config configs/pretrain_pilot.json
```

Das ist ein **Pipeline-Test** (Loss fällt, Checkpoints, Resume, Eval), kein
Qualitätslauf. Für ein brauchbares Modell fehlen ~60x mehr Daten — siehe
`docs/DATA_REPORT.md` und `docs/DECISIONS.md`.

## Weitere Dokumente

```text
docs/RUNBOOK.md     # Schritt-für-Schritt bis zum ersten echten Lauf
docs/DECISIONS.md   # Vokabular, Token-Budget, Dedup, LR - mit Messwerten
docs/DATA_REPORT.md # Pilot-Korpus: Zahlen, Lizenzmix, Loss-Kurve
```

## Grenzen

- Die Sprachbeschränkung wird über Datenfilter, System-Prompt und Evaluierung umgesetzt, nicht über eine harte mathematische Garantie.
- „Reasoning“ wird über Trainingsbeispiele gelernt, nicht durch ein magisches Zusatzmodul.
- Gute Ergebnisse brauchen sehr viele saubere, lizenzierte DE/EN-Code- und Planungsdaten.
- Keine fremden Modellgewichte, keine proprietären API-Modelle als Basis.

## Lizenz

Code: MIT. Daten und trainierte Gewichte haben ihre eigenen Lizenzen. Beachte `docs/DATA_POLICY.md`.
