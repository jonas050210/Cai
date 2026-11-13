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
# 1. Eigene lokale Dateien einlesen
codeplan-ingest --input /pfad/zu/deinen/dateien --out data/raw/my_data.jsonl --license "MIT" --source "my-repos"

# 2. Filtern: DE/EN, Deduplizierung, Split
codeplan-prepare --input data/raw/my_data.jsonl --out-dir data/filtered --require-license

# 3. Tokenizer von Grund auf trainieren
codeplan-tokenizer --train data/filtered/train.jsonl --out artifacts/tokenizer.json --vocab-size 32000

# 4. Daten packen
codeplan-pack --filtered-dir data/filtered --tokenizer artifacts/tokenizer.json --out-dir data/packed

# 5. Pretraining
codeplan-pretrain --config configs/pretrain_50m_4060ti.json

# 6. SFT für strukturierte Coding-/Planungsantworten
codeplan-sft --config configs/sft_50m_4060ti.json --data data/sft/train.jsonl

# 7. Evaluieren
codeplan-eval --checkpoint checkpoints/sft_50m/last.pt --data-dir data/packed --split test --seq-len 2048

# 8. Chatten
codeplan-chat --checkpoint checkpoints/sft_50m/last.pt
```

## Wichtige Dateien

```text
configs/pretrain_50m_4060ti.json   # 50M-Modell für 8 GB VRAM
configs/sft_50m_4060ti.json        # SFT-Konfiguration
configs/tiny_smoke.json            # Mini-Pipeline-Test
src/codeplan/model.py              # Transformer von Grund auf
src/codeplan/tokenizer.py          # BPE-Tokenizer + Chat-Template
src/codeplan/prepare.py            # DE/EN-Filter, Dedup, Split
src/codeplan/pack.py               # Binary-Datasets
src/codeplan/trainer.py            # Training-Loop
data/seed/                         # winzige synthetische Testdaten
AI_HANDOFF.md                      # Übergabe für andere KIs / Entwickler
```

## Grenzen

- Die Sprachbeschränkung wird über Datenfilter, System-Prompt und Evaluierung umgesetzt, nicht über eine harte mathematische Garantie.
- „Reasoning“ wird über Trainingsbeispiele gelernt, nicht durch ein magisches Zusatzmodul.
- Gute Ergebnisse brauchen sehr viele saubere, lizenzierte DE/EN-Code- und Planungsdaten.
- Keine fremden Modellgewichte, keine proprietären API-Modelle als Basis.

## Lizenz

Code: MIT. Daten und trainierte Gewichte haben ihre eigenen Lizenzen. Beachte `docs/DATA_POLICY.md`.
