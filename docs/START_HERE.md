# Start hier — alles kopierbar

Von null bis Chat-GUI. Jeder Block ist copy-paste-fähig.

## 1. Repo auf den Desktop

```bash
cd ~/Desktop
git clone https://github.com/jonas050210/Cai.git
cd Cai
```

## 2. Python-Umgebung

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e ".[dev,serve]"
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

Erwartet: `2.x+cuXXX True NVIDIA GeForce RTX 4060`.

**Nach jedem `git pull`** einmal neu installieren (neue Befehle entstehen nur dabei):

```bash
cd ~/Desktop/Cai && source .venv/bin/activate && pip install -e ".[dev,serve]"
```

## 3. GitHub-Token — einmalig

```bash
bash scripts/install_token.sh
```

Fragt versteckt ab, prüft live gegen GitHub, speichert in
`~/.config/codeplan/env` (mode 600) und hängt sich in `~/.bashrc`.
Danach benutzen ihn `collect`, `git` und `gh` automatisch. Prüfen:

```bash
echo "${GITHUB_TOKEN:0:7}"
```

## 4a. Nur herunterladen (Laptop, ohne GPU)

Der Download braucht weder GPU noch torch. Auf einem leisen Laptop:

```bash
bash scripts/install_token.sh
bash scripts/download_only.sh          # ~30-40 min, ~2 GB in data/raw/
```

Danach `data/raw/` per USB-Stick in denselben Ordner auf den Trainings-PC
kopieren und dort `bash scripts/build_corpus.sh` ausführen (überspringt den
Download automatisch).

**Bildschirm aus ist egal, Deckel zu nicht.** Vorher sperren:

```bash
# Ubuntu: Suspend komplett aus
sudo systemctl mask sleep.target suspend.target hibernate.target hybrid-sleep.target
# Windows (PowerShell): Deckel und Ruhezustand aus
#   powercfg /change standby-timeout-ac 0
#   powercfg /setacvalueindex SCHEME_CURRENT SUB_BUTTONS LIDACTION 0
#   powercfg -setactive SCHEME_CURRENT
```

## 4b. Korpus bauen

Über Nacht, läuft im Hintergrund weiter wenn du das Terminal schließt:

```bash
bash scripts/overnight.sh
tail -f logs/overnight-*.log
```

**~30–40 min** für 473 Repos (12 parallele Downloads), danach Filtern, Tokenizer,
Packen, `doctor`. Ohne Hintergrund:

```bash
bash scripts/build_corpus.sh          # gleiche Kette, im Vordergrund
```

Gemessen: 40 Repos mit 12 Workern ≈ 2,5 min, also ~4 s pro Repo.

Jeder Schritt wird übersprungen, wenn sein Ergebnis existiert — abbrechen ist
unschädlich, derselbe Befehl macht weiter. Varianten:

```bash
bash scripts/build_corpus.sh --core-only         # nur 53 Repos, ~45 min
bash scripts/build_corpus.sh --stop-after stats  # nach dem Token-Budget anhalten
```

Zwischenstand ansehen:

```bash
codeplan-stats --input data/filtered/train.jsonl --target-tokens 983000000
```

## 5. Training

```bash
tmux new -s train
bash scripts/curriculum.sh --config configs/pretrain_100m_4060ti.json
```

- `Strg+B`, dann `D` = detach (Training läuft weiter), `tmux attach -t train` = zurück.
- `Strg+C` = Step zu Ende rechnen, speichern, beenden.
- Derselbe Befehl nochmal = **macht weiter** (`--fresh` startet von vorn).

Zeit für das 99M-Modell (RTX 4060, ~2,1 GB VRAM):

| Steps | Tokens | Zeit |
|---|---|---|
| 30.000 | 983M | ~10–13 h |
| 12.000 | 393M | ~4–5 h |
| 8.000 | 262M | ~3 h |

## 6. Prüfen

```bash
codeplan-eval --checkpoint checkpoints/pretrain_100m/last.pt --data-dir data/packed --token-classes
codeplan-soup --dir checkpoints/pretrain_100m --last 5 --out checkpoints/pretrain_100m/soup.pt
```

`--token-classes` zeigt, *was* das Modell nicht kann (meist Bezeichner).

## 7. SFT und Chat-GUI

```bash
codeplan-sft-data --input data/filtered/train.jsonl --out-dir data/sft
codeplan-sft --config configs/sft_100m_4060ti.json --data data/sft/train.jsonl
codeplan-serve --checkpoint checkpoints/sft_100m/last.pt --host 0.0.0.0 --port 8000
```

→ <http://localhost:8000>

## 8. Offline testen (ohne Downloads)

```bash
bash scripts/quick_test.sh     # Mini-Korpus aus data/seed, 38 s
bash scripts/smoke_test.sh     # ganze Pipeline, CPU
```
