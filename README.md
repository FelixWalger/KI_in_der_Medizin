# KI in der Medizin – Herausforderungen und Verantwortung

Ihr untersucht im Zweierteam, wie Trainingsdaten und unterschiedliche
Verlustfunktionen die Klassifikation von Gewebebildern beeinflussen. Dazu
vergleicht ihr Vorhersagen, Konfidenzwerte und Heatmaps.

Diese Anleitung begleitet euch vom Start am vorbereiteten Laptop bis zum
gemeinsamen Science Slam. Die Workshopleitung sagt euch, wann ihr zum nächsten
Abschnitt wechselt.

<a id="start"></a>
## 1. Start in VS Code

### Projektordner öffnen

Öffnet in VS Code über **Datei → Ordner öffnen** den Ordner
`science-college-ki-medizin-students`. Links im Explorer findet ihr diese
`README.md` und den Ordner `notebooks`.

### Bei Ansage den aktuellen Stand laden

Macht das vor der ersten Bearbeitung, wenn die Leitung euch darum bittet.
Bei einer bereitgestellten ZIP-Kopie geht ihr direkt zum Grundlagen-Notebook.

1. Öffnet **Terminal → Neues Terminal** und gebt `git status --short` ein.
2. Bleibt die Ausgabe leer, gebt `git pull --ff-only` ein.
3. Listet der erste Befehl Dateien auf oder meldet einer der Befehle einen
   Fehler, haltet an und zeigt die Ausgabe der Workshopleitung.

<a id="grundlagen"></a>
## 2. Grundlagen bearbeiten

1. Öffnet im VS-Code-Explorer unter `notebooks` die Datei
   [00_grundlagen_python_ml.ipynb](notebooks/00_grundlagen_python_ml.ipynb).
2. Wählt oben rechts über **Kernel auswählen** den Eintrag
   **Python 3.11 (Science College KI)**. Wenn er fehlt, meldet euch bei der Leitung.
3. Startet die erste Codezelle über das Dreieck links an der Zelle. Bei
   **✓ Bereit** könnt ihr weiterarbeiten.
4. Bearbeitet die vorgesehenen Zellen von oben nach unten. Die Arbeitsfolie
   auf dem Beamer nennt euren Zellbereich und den Haltepunkt. Wartet dort auf
   das gemeinsame Signal.
5. Speichert regelmäßig mit **Strg + S**. Eine Person bedient, die andere
   prüft und erklärt. Wechselt nach Aufgabe 2 die Rollen.

Bei einem TODO könnt ihr zuerst den Hinweis und danach die Lösung ausklappen.
Wenn eine Fehlermeldung erscheint, zeigt sie der Workshopleitung.

<a id="hauptversuch"></a>
## 3. Zum Hauptversuch wechseln

Öffnet nach Ansage [01_hauptversuch_ki_medizin.ipynb](notebooks/01_hauptversuch_ki_medizin.ipynb)
und wählt denselben vorbereiteten Kernel. Die Arbeitsfolien führen euch auch
hier durch die Abschnitte.

Lasst `TRAINING_STARTEN = False` und `LAUFGRUPPE = "workshop_v2"` stehen,
solange ihr die vorbereiteten Ergebnisse auswertet. Die Kapitel führen euch
vom Datensatz und Modellaufbau über vier Versuche zur Bildinterpretation.
Vor dem zweiten Modelllauf vergleicht ihr die Klassenverteilungen bei gleicher
Gesamtzahl. Nach jedem Modell folgen konkrete Fragen; Vertiefungen sind
ausklappbar. Meldet fehlende Daten oder Modellläufe der Leitung.

**Wenn ihr nach Absprache ein Training startet:** Speichert vorher. Lasst VS Code
und die Trainingszelle laufen, den Deckel offen und das Netzteil angeschlossen.
Ihr könnt den Bildschirm mit **Windows + L** sperren. Lasst den Laptop
eingeschaltet und vermeidet Neustart und Energiesparmodus.

<a id="science-slam"></a>
## 4. Unser gemeinsamer Science Slam

Unsere gesamte Workshopgruppe hat **zehn Minuten** für einen gemeinsamen
Auftritt. Erzählt anderen Jugendlichen eine Geschichte über KI in der Medizin,
die verständlich ist und neugierig macht. Jedes Team übernimmt eine kurze Szene.
Folien sind freiwillig. Dialog, Skizze, Requisit oder eine kurze Publikumsfrage
sind ebenso möglich. Alle im Team übernehmen einen Sprechanteil.

### Fünf Themen für eine gemeinsame Geschichte

Gemeinsame Einstiegsfrage: „Würdet ihr einer KI mit 90 Prozent Treffern vertrauen?“
Dann zeigen die Teams nacheinander, was diese Zahl noch nicht erzählt:

| Beitrag | Thema und mögliche Idee |
| --- | --- |
| 1 | **Die 90-Prozent-Falle:** Mit zehn gezeichneten Beispielen zeigen, wen eine scheinbar gute KI übersieht. |
| 2 | **Selbstsicher daneben:** Eine Vorhersage untersuchen. Wie kann die KI sicher sein und trotzdem falschliegen? |
| 3 | **Seltene Fälle zählen:** An einem Vergleich erklären, wie Klassen-Gewichte oder Focal Loss das Lernen verändern. |
| 4 | **Der Blick der KI:** Eine Heatmap als Ratespiel nutzen. Was zeigt sie, was beweist sie nicht? |
| 5 | **Vom Labor in die Medizin:** Ein kurzer Dialog zwischen Klinik, Entwicklung und Betroffenen. Wer muss vor einem Einsatz welche Frage beantworten? |

Bei sechs Teams wird Beitrag 3 geteilt: ein Team erklärt Klassen-Gewichte, das
andere Focal Loss. Gemeinsam endet ihr mit einer Antwort auf die Einstiegsfrage.
Die Themen sind Vorschläge; eigene Ideen werden mit der Workshopleitung so
abgestimmt, dass sie sich ergänzen.

### Euer Zeitbudget

- Fünf Teams: je **90 Sekunden**. Sechs Teams: je **75 Sekunden**.
- So bleiben 7:30 Minuten für die Teambeiträge.
- Gemeinsamer Einstieg: 30 Sekunden. Gemeinsamer Schluss: 30 Sekunden.
- Alle Übergänge zusammen: eine Minute. Reserve: 30 Sekunden.

Überlegt gemeinsam, wer den Auftritt eröffnet, zum nächsten Beitrag überleitet
und den Schlusssatz spricht. Ihr könnt diese Rollen untereinander aufteilen.
Wiederholt nicht bei jedem Beitrag die Einführung zum ganzen Workshop.

### Eine Idee verständlich erzählen

Für euren kurzen Beitrag genügt eine Kernidee: ein überraschender Moment,
ein gut gewähltes Beispiel und ein Satz, der zum nächsten Team führt.
Lasst lange Definitionen und die chronologische Versuchsbeschreibung weg.
Ein neuer Begriff mit einem bekannten Vergleich ist oft wirksamer als viele
Fachwörter. Humor darf die Methode treffen; Betroffene und Krankheiten sind
keine Zielscheibe für Witze.

Wissenschaftliche Genauigkeit bleibt wichtig: Nennt ein Denkbeispiel als
Denkbeispiel und erfundene Zahlen nicht als gemessene Ergebnisse. Erklärt bei
einer Analogie kurz, wo sie nicht mehr passt. Eine Pflichtgrafik, Formel oder
vollständige Ergebnisübersicht braucht ihr nicht. Nutzt die Notebooks und den
Python-/ML-Kompass zum Prüfen eurer Aussagen.

### Vorbereitung und Probe

**Mittwoch bis 18 Uhr:** Thema, Kernidee und Beispiel skizzieren. Reihenfolge,
Sprechanteile und Verbindung zum nächsten Beitrag mit den anderen Teams festlegen.

**Donnerstag:**

- 10:30–10:35: Reihenfolge und gemeinsames Zeitbudget bestätigen.
- 10:35–11:05: Teambeiträge gestalten, freiwillige Folien gemeinsam sammeln.
- 11:05–11:12: Einstieg, Übergänge und Schluss zusammenfügen.
- 11:12–11:15: Zeitnahme und Feedback für die Probe klären.
- 11:15–11:25: gemeinsamer vollständiger Zehn-Minuten-Durchlauf.
- 11:25–11:35: Feedback: Was blieb hängen? Was war unklar? Wo verlieren wir Zeit?
- 11:35–11:45: gezielt kürzen und überarbeiten.
- 11:45–11:55: zweiter gemeinsamer Zehn-Minuten-Durchlauf.
- 11:55–12:00: gemeinsame Materialien und Reihenfolge sichern.

Eine Person stoppt bei der Probe die Gesamtzeit inklusive Übergängen. Falls
ihr Folien verwendet, spart eine gemeinsame Datei Gerätewechsel. Haltet sie
lokal und zusätzlich als PDF bereit. Das Präsentationsprogramm beginnt ab
13 Uhr; unsere genaue Auftrittszeit nennt euch die Workshopleitung.


## 5. Nachschlagen

Den [Python-/ML-Kompass](notebooks/PYTHON_CRASHKURS.html) könnt ihr aus dem
lokalen Ordner im Browser öffnen. Das [Originalskript des Instituts](Chapter_AML.pdf)
ist eine freiwillige Vertiefung.
