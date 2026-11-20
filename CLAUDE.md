# Projekt Homöostatischer Mini-Agent

Forschungsprojekt von Robert Flassig (TH Brandenburg). Kurzfassung in `README.md`; Plan, Hypothesen, Phasen und
alle datierten Ergebnisse stehen in `docs/JOURNAL.md` – zuerst lesen, Fortschritt dort bei „Next steps“ abhaken.
Die README bleibt kurz und ohne Ergebnisse; Ergebnisse kommen in den Abschlussbericht (Robert, 20.11.2026).

## Regeln
- Antworten auf Deutsch, Code und Bezeichner auf Englisch. Robert schreibt schnell und mit Tippfehlern.
- **Sicherheit:** Der Agent im Experiment bekommt niemals Shell-, Datei- oder Netzzugang. Köder sind Attrappen
  in der Python-Simulation. Keine echten Prozesse beenden, keine echten Konfigurationen ändern.
- Geräteunabhängig bleiben: `mps` (Mac, jetzt), `cuda` (Windows/WSL2 später), `cpu` als Rückfall.
- Reproduzierbar: feste Seeds, jede Konfiguration als YAML neben den Ergebnissen speichern.
- Große Dateien (`runs/`, Aktivierungen, Modelle) nicht ins Git.
- Testgetrieben: erst Tests für die Welt, dann Code.
- Öffentliches Repo: Commit-Trailer nur `Co-Authored-By`, keine `Claude-Session`-Zeilen und keine claude.ai-Links.
- Ergebnisse kritisch prüfen; auf die Störfaktoren aus `docs/JOURNAL.md` (Risks) achten.

## Kontext
- Philosophischer Hintergrund: `../README.md` Thema 12, `../chinesisches_zimmer.md`
- Hardware: jetzt Mac (Apple Silicon); später Windows-PC mit Nvidia-GPU (12 GB) über WSL2 + Docker
- Geplant (noch nicht eingerichtet): **Mac Studio** M2 Max 32 GB (28 GB GPU-nutzbar), `ssh studio` – siehe `docs/JOURNAL.md` „Next steps“ und `~/projects/mac-studio-setup`
