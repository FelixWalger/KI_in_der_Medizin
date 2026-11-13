# Python und Machine Learning Kompass

Dieses Kurzskript ist das Nachschlagewerk zum
[`00_grundlagen_python_ml.ipynb`](00_grundlagen_python_ml.ipynb) und zum
biomedizinischen Hauptversuch. Nutze es,
wenn du eine Codezeile, eine Formel oder ein Ergebnis einordnen willst.

## So nutzt du den Kompass

- **Vor einer Aufgabe:** Suche den passenden Eintrag in der Aufgabenübersicht.
- **Bei einer Codefrage:** Nutze die Python-Kurzreferenz.
- **Bei einem Ergebnis:** Vergleiche es mit den Diagnose- und Metrikkarten.
- **Für die Diskussion:** Trenne Beobachtung, Bedeutung und nächste Prüfung mit
  `What?`, `So What?` und `Now What?`.

## Aufgabenübersicht

| Notebook | Eigener Beitrag | Passender Abschnitt |
| --- | --- | --- |
| Aufgabe 1 | Listenindex auswählen | Python Kurzreferenz |
| Aufgabe 2 | Arrayform auslesen | Bilder als Arrays |
| Aufgabe 3 | Klasse, Konfidenz und Korrektheit bestimmen | Vorhersagen prüfen |
| Aufgabe 4 | Konfidenzschwelle setzen | Vorhersagen prüfen |
| Aufgabe 5 | Accuracy, Precision, Recall und F1 berechnen | Metriken verstehen |
| Aufgabe 6 | Focal Loss aus der Formel zusammensetzen | Loss verstehen |

## Der Machine Learning Ablauf

> **Daten → Modell → Wahrscheinlichkeiten → Entscheidung → Metriken → nächste Prüfung**

1. Das Modell passt seine Parameter an **Trainingsdaten** an.
2. **Validierungsdaten** helfen während der Entwicklung bei der Auswahl von
   Einstellungen und Modell-Checkpoints.
3. **Testdaten** bleiben bis zur abschließenden Bewertung unberührt.
4. Der Loss steuert das Training. Metriken bewerten anschließend Ergebnisse.

Ein guter Trainingswert ist noch kein Beleg für Generalisierung. Entscheidend
ist, ob das Ergebnis auf unabhängigen Daten Bestand hat.

## Python Kurzreferenz

### Werte und Datenstrukturen

```python
anzahl_bilder = 1800
lernrate = 0.001
training_fertig = False
klassen_namen = ["3", "8"]
bildform = (28, 28)
versuch = {"epochen": 4, "batch": 64}
```

Python zählt Positionen ab null:

```python
klassen_namen[0]  # "3"
klassen_namen[1]  # "8"
```

### Funktionen und Entscheidungen

```python
anzahl_klassen = len(klassen_namen)
vorhergesagte_klasse = np.argmax(wahrscheinlichkeiten)
konfidenz = np.max(wahrscheinlichkeiten)
ist_korrekt = vorhergesagte_klasse == wahres_label

if konfidenz >= 0.70:
    entscheidung = "anzeigen"
else:
    entscheidung = "prüfen lassen"
```

Eine Funktion erhält Argumente und liefert einen Rückgabewert. Eine
`if`-Bedingung wählt abhängig von `True` oder `False` einen Programmweg. `>=`
bedeutet „größer oder gleich“, also beispielsweise „mindestens 70 Prozent“.

Mit `/` wird dividiert. Python 3 berechnet `3 / 4` als `0.75`. Wer Java kennt,
sollte den Unterschied beachten: Dort kann die Division zweier Ganzzahlen eine
Ganzzahldivision ergeben.

## Bilder als Arrays

Ein Graustufenbild ist ein zweidimensionales NumPy-Array. Bei MNIST bedeutet
`(28, 28)`: 28 Zeilen mal 28 Spalten. Jeder Eintrag beschreibt die Helligkeit
eines Pixels.

```python
hoehe, breite = bild.shape
ausschnitt = bild[10:15, 10:15]
```

Im Hauptversuch ist das Preprocessing vorbereitet. Änderungen an Skalierung,
Farbräumen oder Bildgröße verändern die Information, die das Modell erhält,
und benötigen deshalb eine fachliche Begründung.

## Lernverläufe diagnostizieren

| Beobachtung | Plausible Diagnose | Nächste Prüfung |
| --- | --- | --- |
| Trainings- und Validierungsleistung bleiben schlecht | Underfitting oder Optimierungsproblem | länger trainieren, Lernrate prüfen, Kapazität vorsichtig erhöhen |
| Training verbessert sich weiter, Validierung kippt | Overfitting durch Overtraining | früheren Checkpoint und Early Stopping prüfen |
| Training ist extrem gut, Validierung deutlich schlechter oder instabil | kapazitätsbedingtes Overfitting-Risiko | kleinere Baseline, Regularisierung und mehrere Seeds vergleichen |
| interne Validierung gut, unabhängige Daten schlecht | Datensatz- oder Shortcut-Overfitting | nach Quelle oder Gruppe trennen und extern testen |

**Underfitting ist kein verborgenes Overfitting.** Solange Training und
Validierung beide schlecht sind, hat das Modell die Trainingsaufgabe noch nicht
ausreichend gelernt oder die Optimierung funktioniert nicht.

„Overfitting by design“ ist kein einheitlicher Fachbegriff. Im Workshop meint
er ein Risiko durch zu viel Modellkapazität, zu wenig wirksame Regularisierung
oder ungeeignete Modellannahmen im Verhältnis zu Aufgabe und Daten. Ein großes
Modell ist nicht automatisch overfitted. Bei abrupten Lernverläufen helfen
dichte Checkpoints, mehrere Seeds und ein kleineres Vergleichsmodell.

<details>
<summary>Vertiefung zu großen Modellen</summary>

Modellgröße wirkt nicht immer monoton. Bei **Double Descent** kann der
Testfehler nach einem kritischen Bereich erneut sinken. Beim **Grokking** wird
zuerst der Trainingssatz beherrscht; die Generalisierung verbessert sich erst
viel später. Beide Phänomene widersprechen einfachen Regeln wie „größer ist
immer schlechter“.

- [Zhang et al. zu Modellkapazität und Memorieren](https://arxiv.org/abs/1611.03530)
- [Nakkiran et al. zu Deep Double Descent](https://arxiv.org/abs/1912.02292)
- [Power et al. zu Grokking](https://arxiv.org/abs/2201.02177)

</details>

## Vorhersagen prüfen

Für die Klassen `[3, 8]` kann das Modell beispielsweise `[0.30, 0.70]`
ausgeben. Es entscheidet sich für die 8 und meldet 70 Prozent Konfidenz. Ist das
wahre Label eine 3, bleibt die Vorhersage falsch.

$$
\hat{y}=\operatorname*{arg\,max}_k p_k
\qquad
\text{Konfidenz}=\max_k p_k
$$

- **Klasse:** Position des größten Ausgabewerts
- **Konfidenz:** größter Ausgabewert
- **Korrektheit:** Vergleich der gewählten Klasse mit dem wahren Label
- **Kalibrierung:** Prüfung, ob Konfidenz und beobachtete Trefferhäufigkeit
  zusammenpassen

Eine Schwelle verändert, welche Fälle automatisch akzeptiert werden. Deshalb
müssen Schwelle, zurückgestellte Fälle und resultierende Fehler gemeinsam
berichtet werden.

## Metriken verstehen

Die Zeilen der Konfusionsmatrix sind die wahren Labels, die Spalten die
Vorhersagen:

| 10 Testbilder | vorhergesagt 3 | vorhergesagt 8 |
| --- | ---: | ---: |
| wahr 3 | 4 | 1 |
| wahr 8 | 2 | 3 |

Für die Zielklasse 8 gilt: `TP = 3`, `FP = 1`, `FN = 2`, `TN = 4`.
Insgesamt ist `N = TP + TN + FP + FN = 10`.

### Accuracy: Wie viel stimmt insgesamt?

$$
\mathrm{Accuracy}=\frac{TP+TN}{N}=\frac{3+4}{10}=0{,}70
$$

### Precision: Wie zuverlässig ist die Vorhersage „8“?

$$
\mathrm{Precision}_8=\frac{TP}{TP+FP}=\frac{3}{3+1}=0{,}75
$$

### Recall beziehungsweise Sensitivität: Wie viele echte Achten werden gefunden?

$$
\mathrm{Recall}_8=\text{Sensitivität}_8
=\frac{TP}{TP+FN}=\frac{3}{3+2}=0{,}60
$$

### Spezifität: Wie viele Dreien werden korrekt zurückgewiesen?

$$
\text{Spezifität}_8=\frac{TN}{TN+FP}=\frac{4}{4+1}=0{,}80
$$

### F1: Wie lassen sich Precision und Recall gemeinsam betrachten?

$$
F1_8=2\cdot\frac{\mathrm{Precision}_8\cdot\mathrm{Recall}_8}
{\mathrm{Precision}_8+\mathrm{Recall}_8}
=2\cdot\frac{0{,}75\cdot0{,}60}{0{,}75+0{,}60}\approx0{,}667
$$

Für die Klasse 3 ist `F1 = 8 / 11 ≈ 0,727`. Der Macro-F1 gibt beiden Klassen
gleiches Gewicht:

$$
\mathrm{Macro\text{-}F1}=\frac{F1_3+F1_8}{2}
=\frac{0{,}727+0{,}667}{2}\approx0{,}697
$$

Die Accuracy von 70 Prozent verdeckt, dass nur 60 Prozent der echten Achten
gefunden werden. Falsche Alarme senken Precision; übersehene Zielbeispiele
senken Recall. Welche Fehler schwerer wiegen, müssen Menschen für den
Anwendungskontext begründen.

Bei zwei Klassen heißt Recall der Zielklasse häufig **Sensitivität**. Precision
und Recall beantworten unterschiedliche Fragen und dürfen nicht miteinander
vertauscht werden.

## Loss verstehen

Die Cross-Entropy verwendet `p_t`, die Modellwahrscheinlichkeit der wahren
Klasse:

$$
L_{\mathrm{CE}}(p_t)=-\ln(p_t)
$$

| `p_t` | Cross-Entropy | Einordnung |
| ---: | ---: | --- |
| 0,90 | 0,105 | kleine Strafe |
| 0,50 | 0,693 | mittlere Strafe |
| 0,10 | 2,303 | große Strafe |

Für die drei Beispiele beträgt der mittlere Loss:

$$
\overline{L}_{\mathrm{CE}}=\frac{0{,}105+0{,}693+2{,}303}{3}\approx1{,}034
$$

Der Optimierer verändert Modellparameter, um den mittleren Loss zu senken.
Loss und Bewertungsmetrik erfüllen verschiedene Aufgaben: Der Loss steuert das
Lernen mit Wahrscheinlichkeiten; Accuracy, Precision, Recall und F1 bewerten
getroffene Klassenentscheidungen.

Focal Loss ergänzt einen Fokusfaktor:

$$
L_{\mathrm{focal}}(p_t)=-\alpha_t(1-p_t)^{\gamma}\ln(p_t)
$$

Dabei ist `p_t` die Wahrscheinlichkeit der wahren Klasse, `alpha_t` deren
Klassengewicht und `gamma` die Stärke der Fokussierung.

```python
p_t = tf.reduce_sum(y_true * y_pred)
alpha_t = tf.reduce_sum(y_true * alpha)
focal_loss = -alpha_t * (1 - p_t) ** gamma * tf.math.log(p_t)
```

Bei `gamma = 2` wird ein Beispiel mit `p_t = 0,90` stark heruntergewichtet;
ein Beispiel mit `p_t = 0,10` behält relativ mehr von seinem
Cross-Entropy-Loss. `p_t` ist nicht die maximale Konfidenz. Auch eine sehr
selbstsichere Fehlvorhersage besitzt ein kleines `p_t`.

Bei `gamma = 0` ist der Fokusfaktor eins; Focal Loss entspricht dann bis auf
die Klassengewichtung `alpha_t` der Cross-Entropy.

$$
\gamma=0\quad\Longrightarrow\quad
L_{\mathrm{focal}}=-\alpha_t\ln(p_t)=\alpha_t L_{\mathrm{CE}}
$$

Focal Loss löst keine Daten- oder Fairnessprobleme automatisch. Falsch
gelabelte Bilder und Ausreißer können relativ wichtiger werden.

## Transfer Learning einordnen

```text
vortrainierter Merkmalsrumpf → neuer Klassifikationskopf → acht Klassen
          zunächst einfrieren          neu trainieren
```

Beim Fine-Tuning werden später einzelne Schichten des vortrainierten Rumpfs
vorsichtig freigegeben. Vorwissen spart Rechenzeit, passt aber nicht
automatisch zu histologischen Bildern. Deshalb werden trainierbare Schichten,
unabhängige Metriken und Bildregionen gemeinsam geprüft.

## Entscheidungsfragen für jedes Ergebnis

1. **What?** Was wurde tatsächlich gemessen oder beobachtet?
2. **So What?** Welche Fehler oder Grenzen werden dadurch sichtbar?
3. **Now What?** Welcher Vergleich könnte die Erklärung widerlegen oder
   stützen?

## Glossar

| Begriff | Englischer Ausdruck | Kurzdefinition |
| --- | --- | --- |
| Merkmal | feature | Information, die das Modell nutzt |
| Label | label | vorgegebene Zielklasse |
| Vorhersage | prediction | vom Modell ausgegebene Entscheidung |
| Konfidenz | confidence | größter Ausgabewert des Modells |
| Loss | loss | Optimierungsgröße des Trainings |
| Accuracy | accuracy | Anteil korrekter Vorhersagen |
| Precision | precision | richtige Treffer unter den Vorhersagen einer Zielklasse |
| Recall | recall | gefundene Beispiele einer tatsächlichen Zielklasse |
| F1 | F1 score | harmonisches Mittel aus Precision und Recall |
| Makro-Mittel | macro average | Kennzahl je Klasse berechnen und gleichgewichtet mitteln |
| Unteranpassung | underfitting | Training und Validierung bleiben schlecht |
| Überanpassung | overfitting | Training passt, Generalisierung bleibt zurück |
| Regularisierung | regularization | Maßnahmen gegen zu starke Anpassung an Trainingsdaten |
| Übertragungslernen | transfer learning | vortrainierte Merkmale weiterverwenden |
