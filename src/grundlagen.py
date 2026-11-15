"""Getestete Hilfsfunktionen fuer den deutschen ML-Grundlagenkurs.

Das Schuelernotebook soll fachliche Entscheidungen sichtbar machen, aber nicht
mit Datenlade-, Trainings- oder Plot-Interna ueberfrachten. Diese Datei ist
deshalb die einzige ausfuehrbare Quelle fuer die technische Infrastruktur des
Grundlagen-Notebooks.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)


SEED = 313
KLASSENNAMEN = ("3", "8")


def _zeige_abbildung(fig, alternativtext: str) -> None:
    """Zeige eine Matplotlib-Abbildung mit Alternativtext im Notebook."""

    from IPython.display import display

    display(fig, metadata={"image/png": {"alt": alternativtext}})
    plt.close(fig)


def lade_mnist_grundlagen(projekt_root: str | Path) -> dict[str, np.ndarray]:
    """Lade und validiere das lokal mitgelieferte MNIST-Subset."""

    root = Path(projekt_root)
    data_path = root / "data" / "mnist_3_8.npz"
    if not data_path.exists():
        raise FileNotFoundError(
            "Das lokale Datenpaket data/mnist_3_8.npz fehlt. "
            "Bitte die Workshop-Materialien vollstaendig entpacken."
        )

    with np.load(data_path, allow_pickle=False) as data:
        result = {key: data[key] for key in data.files}

    _validiere_datensatz(result)
    return result


def _validiere_datensatz(data: dict[str, np.ndarray]) -> None:
    expected_sizes = {"train": 1200, "val": 300, "test": 300}
    all_ids: list[str] = []
    for split, size in expected_sizes.items():
        x = data[f"x_{split}"]
        y = data[f"y_{split}"]
        ids = data[f"ids_{split}"]
        if x.shape != (size, 28, 28) or y.shape != (size,):
            raise ValueError(f"Unerwartete Form im {split}-Split: {x.shape}, {y.shape}")
        if x.dtype != np.uint8 or not set(np.unique(y)).issubset({0, 1}):
            raise ValueError(f"Unerwartete Werte im {split}-Split")
        if not np.array_equal(np.bincount(y, minlength=2), [size // 2, size // 2]):
            raise ValueError(f"Der {split}-Split ist nicht ausgeglichen")
        all_ids.extend(ids.astype(str).tolist())
    if len(all_ids) != len(set(all_ids)):
        raise ValueError("Trainings-, Validierungs- und Test-IDs ueberlappen sich")


def bereite_bilder_vor(x: np.ndarray) -> np.ndarray:
    """Skaliere Graustufenbilder nach [0, 1] und ergaenze den Kanal."""

    return (x.astype("float32") / 255.0)[..., np.newaxis]


def erstelle_grundlagenmodell(seed: int = SEED):
    """Erzeuge das kleine, CPU-taugliche Zwei-Klassen-CNN."""

    import tensorflow as tf

    tf.keras.utils.set_random_seed(seed)
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(28, 28, 1), name="bild"),
            tf.keras.layers.Conv2D(8, 3, activation="relu", name="merkmalsfilter"),
            tf.keras.layers.MaxPooling2D(name="verkleinerung"),
            tf.keras.layers.Flatten(name="merkmale_auflisten"),
            tf.keras.layers.Dense(16, activation="relu", name="entscheidung"),
            tf.keras.layers.Dense(2, activation="softmax", name="wahrscheinlichkeiten"),
        ],
        name="mini_ziffern_cnn",
    )
    model.compile(
        # In TensorFlow 2.15 ist der Legacy-Adam auf Apple-Silicon-CPUs deutlich
        # schneller und vermeidet eine irrefuehrende Workshop-Warnung.
        optimizer=tf.keras.optimizers.legacy.Adam(learning_rate=0.001),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def trainiere_grundlagenmodell(
    data: dict[str, np.ndarray], epochs: int = 4, batch_size: int = 64
):
    """Trainiere das Mini-CNN reproduzierbar und ohne Testdatenkontakt."""

    model = erstelle_grundlagenmodell()
    history = model.fit(
        bereite_bilder_vor(data["x_train"]),
        data["y_train"],
        validation_data=(bereite_bilder_vor(data["x_val"]), data["y_val"]),
        epochs=epochs,
        batch_size=batch_size,
        shuffle=True,
        verbose=0,
    )
    return model, history


def lade_grundlagen_ergebnisse(projekt_root: str | Path):
    """Lade den gekennzeichneten Offline-Ersatzlauf mit gleicher Schnittstelle."""

    import tensorflow as tf

    root = Path(projekt_root)
    reference_dir = root / "assets" / "reference_results" / "grundlagen"
    model_path = reference_dir / "mini_ziffern_cnn.keras"
    result_path = reference_dir / "ergebnisse.npz"
    history_path = reference_dir / "history.json"
    if not (model_path.exists() and result_path.exists() and history_path.exists()):
        raise FileNotFoundError(
            "Der lokale Ersatzlauf fehlt. Bitte die Workshop-Materialien "
            "vollstaendig entpacken."
        )
    model = tf.keras.models.load_model(model_path, compile=False)
    with history_path.open(encoding="utf-8") as handle:
        history = SimpleNamespace(history=json.load(handle))
    with np.load(result_path, allow_pickle=False) as data:
        predictions = data["predictions"]
    print("Hinweis: Es werden vorbereitete Referenzergebnisse verwendet.")
    return model, history, predictions


def berechne_grundlagenmetriken(
    y_true: np.ndarray, probabilities: np.ndarray
) -> dict[str, Any]:
    """Berechne mehrere Metriken fuer die gemeinsame Interpretation."""

    y_pred = np.argmax(probabilities, axis=1)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=[0, 1]),
    }


def pruefe_antwort(
    name: str,
    antwort: Any,
    erwartet: Any,
    erfolg: str = "Genau!",
    hinweis: str = "Schau dir den Hinweis unter der Aufgabe an.",
) -> bool:
    """Gib motivierendes Feedback, ohne den linearen Notebooklauf abzubrechen."""

    try:
        korrekt = bool(np.allclose(antwort, erwartet))
    except (TypeError, ValueError):
        korrekt = antwort == erwartet
    symbol = "✓" if korrekt else "↻"
    text = erfolg if korrekt else hinweis
    print(f"{symbol} {name}: {text}")
    return korrekt


def zeige_beispielbilder(x: np.ndarray, y: np.ndarray, anzahl: int = 6) -> None:
    fig, axes = plt.subplots(1, anzahl, figsize=(11, 2.2))
    for ax, image, label in zip(axes, x[:anzahl], y[:anzahl]):
        ax.imshow(image, cmap="gray")
        ax.set_title(f"Label: {KLASSENNAMEN[int(label)]}")
        ax.axis("off")
    fig.suptitle("Unsere Trainingsbeispiele", fontsize=15)
    plt.tight_layout()
    _zeige_abbildung(
        fig,
        "Sechs handgeschriebene Beispielziffern mit ihren Labels 3 oder 8.",
    )


def zeige_datenaufteilung(data: dict[str, np.ndarray]) -> None:
    labels = ["Training\nModell lernt", "Validierung\nWir vergleichen", "Test\nErst am Ende"]
    values = [len(data["y_train"]), len(data["y_val"]), len(data["y_test"])]
    colors = ["#0072B2", "#E69F00", "#009E73"]
    fig, ax = plt.subplots(figsize=(8, 3.4))
    bars = ax.bar(labels, values, color=colors, edgecolor="black")
    ax.bar_label(bars, labels=[str(v) for v in values], padding=3)
    ax.set_ylabel("Anzahl Bilder")
    ax.set_title("Drei getrennte Rollen für unsere Daten")
    ax.set_ylim(0, max(values) * 1.18)
    plt.tight_layout()
    _zeige_abbildung(
        fig,
        "Balkendiagramm mit 1200 Trainings-, 300 Validierungs- und 300 Testbildern.",
    )


def zeige_trainingsverlauf(history) -> None:
    epochs = np.arange(1, len(history.history["loss"]) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
    axes[0].plot(epochs, history.history["loss"], "o-", label="Training", color="#0072B2")
    axes[0].plot(epochs, history.history["val_loss"], "s--", label="Validierung", color="#D55E00")
    axes[0].set(title="Loss: kleiner ist besser", xlabel="Epoche", ylabel="Loss")
    axes[1].plot(epochs, history.history["accuracy"], "o-", label="Training", color="#0072B2")
    axes[1].plot(epochs, history.history["val_accuracy"], "s--", label="Validierung", color="#D55E00")
    axes[1].set(title="Genauigkeit: größer ist besser", xlabel="Epoche", ylabel="Anteil korrekt", ylim=(0, 1.02))
    for ax in axes:
        ax.grid(alpha=0.25)
        ax.legend()
    plt.tight_layout()
    _zeige_abbildung(
        fig,
        "Trainings- und Validierungskurven für Loss und Accuracy über vier Epochen.",
    )


def zeige_fit_muster() -> None:
    """Zeige schematische Loss-Verläufe als Diagnosehilfe, nicht als Messdaten."""

    epochs = np.arange(1, 9)
    muster = [
        (
            "Underfitting",
            [1.20, 1.10, 1.03, 0.98, 0.94, 0.91, 0.89, 0.87],
            [1.22, 1.14, 1.08, 1.05, 1.03, 1.02, 1.03, 1.04],
            "beide Loss-Werte bleiben hoch",
        ),
        (
            "Passende Lernphase",
            [1.20, 0.85, 0.62, 0.48, 0.40, 0.36, 0.34, 0.33],
            [1.25, 0.90, 0.68, 0.55, 0.49, 0.47, 0.48, 0.49],
            "niedrig und ähnlich",
        ),
        (
            "Overtraining",
            [1.20, 0.80, 0.55, 0.38, 0.25, 0.17, 0.12, 0.09],
            [1.25, 0.85, 0.62, 0.52, 0.55, 0.65, 0.80, 1.00],
            "Validierung kippt nach Minimum",
        ),
        (
            "Validierung bleibt schwach",
            [1.25, 1.22, 1.18, 1.10, 0.45, 0.15, 0.08, 0.05],
            [1.28, 1.24, 1.20, 1.15, 1.05, 1.04, 1.06, 1.08],
            "kein guter Modellstand im gezeigten Lauf",
        ),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(10, 7), sharex=True, sharey=True)
    for ax, (title, train, val, note) in zip(axes.flat, muster):
        ax.plot(epochs, train, "o-", color="#0072B2", label="Training")
        ax.plot(epochs, val, "s--", color="#D55E00", label="Validierung")
        ax.set_title(f"{title}\n{note}", fontsize=11)
        ax.set_xlabel("Epoche")
        ax.grid(alpha=0.25)
    axes[0, 0].set_ylabel("Loss")
    axes[1, 0].set_ylabel("Loss")
    axes[0, 0].legend(fontsize=9.5)
    fig.suptitle("Vier schematische Fit-Muster – Diagnose erst aus beiden Kurven", fontsize=15)
    plt.tight_layout(rect=(0, 0, 1, 0.96))
    _zeige_abbildung(
        fig,
        "Vier schematische Trainings- und Validierungs-Loss-Kurven für Underfitting, passende Lernphase, Overtraining und Overfitting bei durchgehend schwacher Validierung.",
    )


def zeige_vorhersage(image: np.ndarray, probabilities: np.ndarray, true_label: int) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(7.5, 3.2))
    axes[0].imshow(image, cmap="gray")
    axes[0].set_title(f"Wahres Label: {KLASSENNAMEN[int(true_label)]}")
    axes[0].axis("off")
    bars = axes[1].bar(KLASSENNAMEN, probabilities, color=["#56B4E9", "#E69F00"], edgecolor="black")
    axes[1].bar_label(bars, labels=[f"{p:.1%}" for p in probabilities], padding=3)
    axes[1].set(title="Modellwahrscheinlichkeiten", xlabel="Vorhergesagte Klasse", ylabel="Wahrscheinlichkeit", ylim=(0, 1.08))
    plt.tight_layout()
    _zeige_abbildung(
        fig,
        "Testbild neben einem Balkendiagramm der Modellwahrscheinlichkeiten für 3 und 8.",
    )


def zeige_konfusionsmatrix(matrix: np.ndarray) -> None:
    fig, ax = plt.subplots(figsize=(5.8, 4.4))
    image = ax.imshow(matrix, cmap="Blues", vmin=0)
    for row in range(2):
        for col in range(2):
            value = int(matrix[row, col])
            ax.text(col, row, str(value), ha="center", va="center", fontsize=14,
                    color="white" if value > matrix.max() / 2 else "black")
    ax.set_xticks([0, 1], KLASSENNAMEN)
    ax.set_yticks([0, 1], KLASSENNAMEN)
    ax.set_xlabel("Vorhergesagtes Label")
    ax.set_ylabel("Wahres Label")
    ax.set_title("Konfusionsmatrix: Wo verwechselt sich das Modell?", fontsize=12)
    fig.colorbar(image, ax=ax, label="Anzahl Bilder")
    plt.tight_layout()
    _zeige_abbildung(
        fig,
        "Zwei-mal-zwei-Konfusionsmatrix für wahre und vorhergesagte Ziffern 3 und 8.",
    )


def zeige_transfer_learning() -> None:
    fig, ax = plt.subplots(figsize=(10, 2.8))
    ax.axis("off")
    boxes = [
        (0.03, "Viele Bilder\nvorher"),
        (0.28, "Vortrainierter\nMerkmals-Detektor"),
        (0.57, "Neuer kleiner\nKlassifikationskopf"),
        (0.83, "Unsere\n8 Klassen"),
    ]
    colors = ["#CC79A7", "#0072B2", "#E69F00", "#009E73"]
    for (x, label), color in zip(boxes, colors):
        ax.text(x, 0.5, label, transform=ax.transAxes, ha="left", va="center",
                fontsize=11, bbox=dict(boxstyle="round,pad=0.5", facecolor=color, alpha=0.25, edgecolor=color, linewidth=2))
    for start, end in [(0.18, 0.27), (0.48, 0.56), (0.74, 0.82)]:
        ax.annotate("", xy=(end, 0.5), xytext=(start, 0.5), xycoords=ax.transAxes,
                    arrowprops=dict(arrowstyle="->", linewidth=2, color="#333333"))
    ax.text(0.37, 0.09, "zunächst einfrieren", transform=ax.transAxes, ha="center", fontsize=10)
    ax.text(0.65, 0.09, "neu trainieren", transform=ax.transAxes, ha="center", fontsize=10)
    ax.set_title("Übertragungslernen: vorhandenes Wissen an eine neue Aufgabe anpassen", fontsize=14)
    _zeige_abbildung(
        fig,
        "Ablauf des Transfer Learning vom vortrainierten Merkmalsrumpf zum neuen Acht-Klassen-Kopf.",
    )
