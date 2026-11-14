# Mehr Durchsatz

Gemessen auf der Zielhardware (RTX 4060 8 GB, i7-12700F, 32 GB) bzw. rechnerisch
abgeleitet, wo Messen nicht ging.

## Daten sammeln (netzwerk- und CPU-gebunden)

| Knopf | Standard | Wirkung |
|---|---|---|
| `--workers` | 16 | parallele Downloads; 8 Repos: 22 s → 10 s |
| `--max-repo-mb` | 50 | größere Repos werden mitgenommen (Speicher pro Worker!) |
| `--sleep` | 0.2 | Pause zwischen API-Calls; bei Token-Limit 5000/h unkritisch |

`prepare --workers 20` nutzt alle 20 Threads beim Filtern.

## Training

`configs/pretrain_100m_fast.json` gegenüber `pretrain_100m_4060ti.json`:

| Änderung | Warum |
|---|---|
| micro-batch 2 statt 1 | halb so viele Kernel-Starts pro Token |
| `gradient_checkpointing: false` | ~30 % weniger Rechenzeit, dafür mehr VRAM |
| `torch_compile: true` | 10–20 % schneller ab dem zweiten Step |
| `num_workers: 8` | Datenblockade fällt weg (12C/20T) |

Tokens pro Optimizer-Step bleiben **32.768**, die Lernkurve ist also vergleichbar.

```bash
bash scripts/train_fast.sh --steps 12000
```

**Zu wenig VRAM (OOM)?** Zurück auf die sichere Variante — gleiche Steps, gleiche
Tokens, nur langsamer:

```bash
bash scripts/train_fast.sh --config configs/pretrain_100m_4060ti.json --steps 12000
```

Oder einzelne Knöpfe über die CLI:

```bash
codeplan-pretrain --config configs/pretrain_100m_fast.json \
  --batch-size 1 --grad-accum 16 --no-compile
```

## Messen statt raten

```bash
codeplan-doctor --config configs/pretrain_100m_fast.json --bench-steps 20
```

Zeigt tok/s und die geschätzte Wanduhrzeit für die konfigurierten Steps.
Faustregel: bleibt die GPU unter 90 % Auslastung, ist der Datenloader der
Flaschenhals → `--num-workers` erhöhen. Pendelt sie zwischen 0 und 100 %, ist
der Micro-Batch zu klein.

## Was nichts bringt

- `precision: fp32` (doppelt so langsam, kein Qualitätsgewinn)
- größere Checkpoints behalten (`keep_last_checkpoints: 1` reicht, 1,05 GB pro Stück)
- `eval_iters` hochdrehen (10 reichen für eine Trendlinie)
