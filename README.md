# KI in der Medizin – Schülermaterial

Dieses Repository enthält ausschließlich die Materialien für die
Teilnehmenden des Science-College-Workshops. Es ist für die bereitgestellten
Windows-Laptops mit VS Code vorbereitet.

Die verwendeten Bilder und Modelle sind Lehrmaterial. Sie sind kein
medizinisches Diagnosewerkzeug und ihre Ergebnisse sind keine klinische
Evidenz.

## Einmalige Einrichtung durch die Workshopleitung oder IT

Voraussetzungen:

- Windows 10 oder 11 (64 Bit)
- Python 3.11 (64 Bit)
- Visual Studio Code mit den Erweiterungen **Python** und **Jupyter**
- Microsoft Visual C++ Redistributable (aktuelles x64-Paket)

Im Projektordner in PowerShell ausführen:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m ipykernel install --user --name science-college-ki-medizin --display-name "Python 3.11 (Science College KI)"
```

Die Einrichtung muss nur einmal pro Laptop erfolgen. Die Teilnehmenden sollen
während des Workshops keine Pakete installieren.

## Start in VS Code

1. Den gesamten Projektordner in VS Code öffnen.
2. In einer Notebook-Datei oben rechts den Kernel
   **Python 3.11 (Science College KI)** auswählen.
3. Zuerst `notebooks/00_grundlagen_python_ml.ipynb` bearbeiten.
4. Danach `notebooks/01_hauptversuch_ki_medizin.ipynb` öffnen.

Die Dateien unter `data/` und `assets/` dürfen nicht verschoben oder gelöscht
werden. Sie ermöglichen den Grundlagenlauf und dessen lokale Referenzresultate
ohne Download.

## Wichtig zum Hauptversuch

Der Hauptversuch verwendet TensorFlow Datasets und vortrainierte
MobileNetV2-Gewichte. Beim ersten Start benötigt er deshalb Internetzugang und
kann auf CPU-Laptops deutlich länger dauern. Er wird erst nach einem Probelauf
auf einem der Workshop-Laptops freigegeben. Die grundlegende Notebook-Übung ist
davon unabhängig vollständig lokal lauffähig.

## Projektstruktur

```text
notebooks/  # Arbeitsnotebooks und Python-/ML-Kompass
src/        # bereitgestellte Hilfsfunktionen
data/       # lokales Lern-Dataset
assets/     # geprüfte Referenzresultate
```
