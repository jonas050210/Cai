# Agent und Sandbox: was hier wie funktioniert

## Analyse: warum ein Agent-Harness funktioniert

Ein Agent ist kein Modell mit Werkzeugen, sondern eine Schleife mit fünf
Eigenschaften. Fehlt eine, bricht das Verhalten sichtbar:

| Eigenschaft | Warum sie nötig ist | Umsetzung hier |
|---|---|---|
| **Werkzeuge mit klarer Signatur** | Das Modell muss wissen, welche Argumente es geben darf | `TOOLS`-Tabelle, Prompt wird daraus erzeugt |
| **Kurze Beobachtungen** | Ein 200-Zeilen-Log sprengt das Kontextfenster | jede Ausgabe auf 4000 Zeichen gekürzt |
| **Fehler als Beobachtung** | Ausnahmen beenden den Lauf statt zu lehren | `(ok, text)` statt Exception |
| **Gedächtnis mit Budget** | Nach 10 Schritten passt nichts mehr rein | alte Schritte auf eine Zeile verdichtet |
| **Nachvollziehbarkeit** | Ohne Log ist kein Fehler zu finden | `session.json`, ein Eintrag pro Schritt |

Die Reihenfolge ist wichtig: erst Werkzeuge und Grenzen, dann Modell. Ein
schlechtes Modell in einem guten Harness läuft weiter und wird besser; ein gutes
Modell in einem schlechten Harness produziert Müll.

## Aufbau

```
codeplan-agent --task "..." --checkpoint checkpoints/sft/last.pt
      │
      ├── Policy       Modell oder Skript liefert ein JSON-Objekt
      ├── Parser       balanced-brace, verschachtelte args erlaubt
      ├── Toolbox      run / read / write / edit / grep / glob / list / done
      └── Sandbox      bubblewrap (Dateibaum, Netzwerk) + prlimit (CPU, RAM, Zeit)
```

**Sandbox-Schichten**

| Schicht | Flag | Wirkung |
|---|---|---|
| Dateisystem | `--bind workspace /workspace` | nur der Session-Ordner ist schreibbar |
| Systemdateien | `--ro-bind /usr /usr` | Bibliotheken lesbar, nicht änderbar |
| Netzwerk | `--unshare-net` | kein Socket verlässt die Maschine |
| Prozesse | `--unshare-all --die-with-parent` | kein Zugriff auf andere Sessions |
| Zeit | `prlimit --cpu=30` | Endlosschleifen enden |
| Speicher | `prlimit --as=2GB` | kein RAM-Fresser |
| Dateigröße | `prlimit --fsize=16MB` | kein vollgeschriebener Datenträger |

**Ohne `bwrap` wird nichts ausgeführt.** Ein Fallback auf `subprocess.run` wäre
eine Sandbox, die keine ist.

## Was der Agent heute kann

- Mehrere Schritte: lesen → ändern → verifizieren, solange das Step-Budget reicht
- Fehler korrigieren: ein fehlgeschlagener Befehl ist Text, den das Modell sieht
- Grenzen respektieren: `../` wird abgelehnt, Netzwerk ist aus

## Was er nicht kann (ehrlich)

- **Planen über mehr als 2–3 Schritte.** Dafür bräuchte es Tool-Call-Training mit
  Trajektorien; ein 96M-Modell bleibt bei kurzen Ketten.
- **Sich selbst verbessern.** Es gibt keinen Reflexionsschritt, keinen Plan-Baum.
- **Web-GUI-Feedback.** Es gibt kein Browser-Preview wie in einer echten
  Agenten-Umgebung; Tests sind der einzige Verifikationskanal.

## Nächste Schritte, in dieser Reihenfolge

1. Trajektorien erzeugen: Skripte laufen lassen, Erfolg/Misserfolg protokollieren
2. Daraus SFT-Paare bauen (Prompt → nächster Tool-Call)
3. Erst dann das Modell als Policy einsetzen
4. Vergleich gegen die Skript-Policy: wie viele Aufgaben löst es in ≤ 8 Schritten
