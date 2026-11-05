# Startanleitung für den Workshop

Bitte arbeitet diese Anleitung gemeinsam als Zweiergruppe durch. Wenn ein
Schritt nicht funktioniert, haltet an und gebt der Workshopleitung Bescheid.
Bitte installiert keine zusätzlichen Pakete und verändert keine Dateien
außerhalb der Notebooks.

## Vor dem Workshop: aktuellen Stand laden

Führt diese Schritte **bevor** ihr in einem Notebook arbeitet aus.

> Diese Schritte gelten für die GitHub-Version. Falls die Workshopleitung einen
> ZIP-Ordner über Nextcloud bereitstellt, überspringt diesen Abschnitt.

1. Öffnet den Ordner `science-college-ki-medizin-students` in VS Code.
2. Öffnet im Menü **Terminal → Neues Terminal**. Es sollte sich PowerShell
   öffnen.
3. Prüft mit folgendem Befehl, ob der Ordner unverändert ist:

   ```powershell
   git status
   ```

   Erwartet wird die Meldung `working tree clean`. Falls dort veränderte Dateien
   stehen, führt die nächsten Schritte nicht aus und meldet euch bei der
   Workshopleitung.

4. Ladet den aktuellen Stand:

   ```powershell
   git pull --ff-only
   ```

5. Schließt das Terminal nicht; falls eine Fehlermeldung erscheint, zeigt sie
   der Workshopleitung.

## Einmalig pro Laptop: Python-Umgebung einrichten

Dieser Abschnitt wird normalerweise von der IT oder Workshopleitung erledigt.
Führt ihn nur aus, wenn ihr dazu aufgefordert werdet.

```powershell
py -3.11 -m venv .venv
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m ipykernel install --user --name science-college-ki-medizin --display-name "Python 3.11 (Science College KI)"
```

Die Zeile mit `Set-ExecutionPolicy` gilt nur für das gerade geöffnete Terminal
und ändert keine dauerhafte Windows-Einstellung.

## Notebooks starten

1. Öffnet `notebooks/00_grundlagen_python_ml.ipynb`.
2. Wählt oben rechts den Kernel **Python 3.11 (Science College KI)**.
3. Führt die erste vorbereitete Codezelle aus. Erscheint `✓ Bereit`, kann es
   losgehen.
4. Arbeitet von oben nach unten und speichert regelmäßig mit **Strg + S**.
5. Öffnet den Hauptversuch erst, wenn die Workshopleitung dazu auffordert:
   `notebooks/01_hauptversuch_ki_medizin.ipynb`.

## Wichtige Regeln

- `git pull` nur vor dem ersten Bearbeiten ausführen. Danach verändert ihr die
  Notebooks; ein weiterer Pull kann eure Arbeit überschreiben oder blockieren.
- Keine Befehle mit `pip install` ausführen, außer die Workshopleitung fordert
  dazu auf.
- Änderungen nur in den dafür vorgesehenen Notebook-Zellen vornehmen.
- Bei roten Fehlermeldungen: Screenshot machen oder die Meldung stehen lassen
  und die Workshopleitung rufen.
