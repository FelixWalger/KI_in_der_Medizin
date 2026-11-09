"""Bereitgestellte, geprüfte Infrastruktur für den achtklassigen Lehrversuch.

Bilder bleiben bis zum Modelleingang RGB-Werte in [0, 255]. Das Modell
übernimmt Resize und MobileNetV2-Skalierung; niemals zusätzlich preprocess_input
aufrufen. Die Testdaten werden weder zur Modellwahl noch zum Training benutzt.
"""
from __future__ import annotations

import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score


def _labels(y):
    values = np.asarray(y)
    if values.ndim == 2:
        if values.shape[1] < 2 or not np.all(np.isin(values, [0, 1])) or not np.all(values.sum(axis=1) == 1):
            raise ValueError("One-Hot-Labels müssen pro Zeile genau eine Klasse markieren.")
        values = values.argmax(axis=1)
    if values.ndim != 1 or not np.all(np.isfinite(values)) or not np.all(values == values.astype(int)):
        raise ValueError("Labels müssen Klassenindizes oder One-Hot-Zeilen sein.")
    return values.astype(int)


def split_data(x, y, test_size=500, val_size=500, subset_frac=.5, seed=313):
    """Stratifizierte disjunkte Splits; Indizes dokumentieren die Trennung."""
    x, labels = np.asarray(x), _labels(y)
    if len(x) != len(labels) or not 0 < subset_frac <= 1:
        raise ValueError("Bild-/Labelanzahl und subset_frac prüfen.")
    ids = np.arange(len(labels))
    train, test = train_test_split(ids, test_size=test_size, stratify=labels, random_state=seed)
    train, val = train_test_split(train, test_size=val_size, stratify=labels[train], random_state=seed)
    if subset_frac < 1:
        train, _ = train_test_split(train, train_size=subset_frac, stratify=labels[train], random_state=seed)
    result = {}
    for name, selected in (("train", train), ("val", val), ("test", test)):
        result[f"x_{name}"] = x[selected]
        result[f"y_{name}"] = labels[selected]
        result[f"ids_{name}"] = selected
    return result


def make_imbalanced(x, y, minority_class=0, keep_fraction=.1, seed=313):
    """Entferne nur Beispiele der gewählten Klasse; andere bleiben gleich."""
    x, labels = np.asarray(x), _labels(y)
    if len(x) != len(labels) or not 0 < keep_fraction <= 1:
        raise ValueError("Bild-/Labelanzahl und keep_fraction prüfen.")
    minority = np.flatnonzero(labels == minority_class)
    if not len(minority):
        raise ValueError("Die gewählte Minderheitsklasse fehlt bereits im Datensatz.")
    rng = np.random.default_rng(seed)
    retained = rng.choice(minority, max(1, int(len(minority) * keep_fraction)), replace=False)
    selected = np.sort(np.concatenate([np.flatnonzero(labels != minority_class), retained]))
    return x[selected], labels[selected]


def class_weights(y, num_classes=8):
    labels = _labels(y)
    if num_classes < 2 or np.any(labels < 0) or np.any(labels >= num_classes):
        raise ValueError("Klassenindizes passen nicht zur Klassenanzahl.")
    counts = np.bincount(labels, minlength=num_classes)
    if np.any(counts == 0):
        raise ValueError("Für Klassen ohne Trainingsbeispiele kann kein Gewicht berechnet werden.")
    return {i: float(len(labels) / (num_classes * n)) for i, n in enumerate(counts)}


@tf.keras.utils.register_keras_serializable(package="ScienceCollege")
class CategoricalFocalLoss(tf.keras.losses.Loss):
    """-alpha_t (1-p_t)^gamma log(p_t), kategorisch und One-Hot-kodiert."""
    def __init__(self, gamma=2., alpha=None, reduction="sum_over_batch_size", name="categorical_focal_loss"):
        super().__init__(reduction=reduction, name=name)
        if not np.isfinite(gamma) or gamma < 0:
            raise ValueError("gamma muss endlich und nicht negativ sein.")
        if alpha is not None:
            a = np.asarray(alpha, dtype=float)
            if a.ndim != 1 or len(a) < 2 or not np.all(np.isfinite(a)) or np.any(a <= 0):
                raise ValueError("alpha muss positive endliche Klassengewichte enthalten.")
            alpha = a.tolist()
        self.gamma, self.alpha = float(gamma), alpha

    def call(self, y_true, y_pred):
        y_true = tf.cast(y_true, y_pred.dtype)
        tf.debugging.assert_equal(tf.shape(y_true), tf.shape(y_pred))
        if self.alpha is None:
            alpha = tf.ones_like(y_pred)
        else:
            tf.debugging.assert_equal(tf.shape(y_pred)[-1], len(self.alpha))
            alpha = tf.cast(tf.constant(self.alpha), y_pred.dtype)
        probability = tf.clip_by_value(y_pred, tf.keras.backend.epsilon(), 1 - tf.keras.backend.epsilon())
        return -tf.reduce_sum(y_true * alpha * tf.pow(1 - probability, self.gamma) * tf.math.log(probability), axis=-1)

    def get_config(self):
        return {**super().get_config(), "gamma": self.gamma, "alpha": self.alpha}


def categorical_focal_loss(gamma=2., alpha=None):
    return CategoricalFocalLoss(gamma=gamma, alpha=alpha)


@tf.keras.utils.register_keras_serializable(package="ScienceCollege")
class MacroF1(tf.keras.metrics.Metric):
    """F1 aus epochweit aufsummierten Zählwerten für eine feste Klassenliste."""
    def __init__(self, num_classes=8, name="macro_f1", **kwargs):
        super().__init__(name=name, **kwargs)
        self.num_classes = int(num_classes)
        if self.num_classes < 2:
            raise ValueError("Mindestens zwei Klassen erforderlich.")
        self.counts = self.add_weight("confusion_counts", shape=(num_classes, num_classes), initializer="zeros")

    def update_state(self, y_true, y_pred, sample_weight=None):
        labels = tf.argmax(y_true, axis=-1, output_type=tf.int32) if y_true.shape.rank == y_pred.shape.rank else tf.cast(y_true, tf.int32)
        predictions = tf.argmax(y_pred, axis=-1, output_type=tf.int32)
        weights = None if sample_weight is None else tf.broadcast_to(tf.cast(sample_weight, self.dtype), tf.shape(labels))
        self.counts.assign_add(tf.math.confusion_matrix(tf.reshape(labels, [-1]), tf.reshape(predictions, [-1]), num_classes=self.num_classes, weights=None if weights is None else tf.reshape(weights, [-1]), dtype=self.dtype))

    def result(self):
        tp = tf.linalg.diag_part(self.counts)
        denominators = tf.reduce_sum(self.counts, axis=0) + tf.reduce_sum(self.counts, axis=1)
        return tf.reduce_mean(tf.math.divide_no_nan(2 * tp, denominators))

    def reset_state(self):
        self.counts.assign(tf.zeros_like(self.counts))

    def get_config(self):
        return {**super().get_config(), "num_classes": self.num_classes}


def build_model(num_classes=8, num_unfrozen_layers=3, learning_rate=.001, weights="imagenet", input_shape=(150, 150, 3)):
    if num_classes < 2 or not isinstance(num_unfrozen_layers, int) or num_unfrozen_layers < 0:
        raise ValueError("Klassenanzahl und Zahl freizugebender Schichten prüfen.")
    if input_shape[-1] != 3 or not np.isfinite(learning_rate) or learning_rate <= 0:
        raise ValueError("RGB-Bilder und positive Lernrate erforderlich.")
    base = tf.keras.applications.MobileNetV2(include_top=False, weights=weights, input_shape=(224, 224, 3))
    if num_unfrozen_layers > len(base.layers):
        raise ValueError("Mehr freizugebende Schichten als im Basismodell vorhanden.")
    # Die Elternebene muss bei Fine-Tuning trainierbar bleiben, ihre einzelnen
    # Layer werden gezielt eingefroren. BatchNorm bleibt immer unverändert.
    base.trainable = num_unfrozen_layers > 0
    for i, layer in enumerate(base.layers):
        layer.trainable = bool(num_unfrozen_layers and i >= len(base.layers) - num_unfrozen_layers and not isinstance(layer, tf.keras.layers.BatchNormalization))
    backbone = tf.keras.Model(base.input, base.output, name="backbone")
    inputs = tf.keras.Input(shape=input_shape, name="rgb_bild")
    resized = tf.keras.layers.Resizing(224, 224, name="bildgroesse")(inputs)
    scaled = tf.keras.layers.Rescaling(1 / 127.5, offset=-1, name="mobilenet_skalierung")(resized)
    features = backbone(scaled, training=False)
    pooled = tf.keras.layers.GlobalAveragePooling2D(name="merkmale")(features)
    hidden = tf.keras.layers.Dense(512, activation="relu", name="modellkopf")(pooled)
    output = tf.keras.layers.Dense(num_classes, activation="softmax", name="klassen")(hidden)
    model = tf.keras.Model(inputs, output, name="gewebe_lehrmodell")
    model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=learning_rate, momentum=.9, nesterov=True), loss="categorical_crossentropy", metrics=["accuracy", MacroF1(num_classes)])
    return model


def _probabilities(values, num_classes=None):
    values = np.asarray(values, dtype=float)
    if values.ndim != 2 or not len(values) or (num_classes is not None and values.shape[1] != num_classes):
        raise ValueError("Wahrscheinlichkeiten als Tabelle Bilder × Klassen übergeben.")
    if not np.all(np.isfinite(values)) or np.any(values < 0) or not np.allclose(values.sum(axis=1), 1, atol=1e-5):
        raise ValueError("Ungültige Wahrscheinlichkeiten: Zeilensummen müssen eins ergeben.")
    return values


def evaluate_probabilities(y, probabilities, num_classes=8):
    labels, probabilities = _labels(y), _probabilities(probabilities, num_classes)
    if len(labels) != len(probabilities) or np.any(labels < 0) or np.any(labels >= num_classes):
        raise ValueError("Wahre Labels passen nicht zu den Vorhersagen.")
    predicted = probabilities.argmax(axis=1)
    return {"accuracy": float(accuracy_score(labels, predicted)), "macro_f1": float(f1_score(labels, predicted, labels=np.arange(num_classes), average="macro", zero_division=0)), "confusion_matrix": confusion_matrix(labels, predicted, labels=np.arange(num_classes))}


def choose_examples(listprobabilities, y_true):
    labels = _labels(y_true)
    probs = [_probabilities(p) for p in listprobabilities]
    if len(probs) != 4 or any(p.shape != probs[0].shape for p in probs) or len(labels) != len(probs[0]):
        raise ValueError("Genau vier Modelltabellen mit denselben Bildern erforderlich.")
    if np.any(labels < 0) or np.any(labels >= probs[0].shape[1]):
        raise ValueError("Wahre Klassen liegen außerhalb der Vorhersagetabellen.")
    correct = np.array([p.argmax(axis=1) == labels for p in probs])
    confidence = np.array([p.max(axis=1) for p in probs])
    case1 = np.flatnonzero(~correct[1] & correct[0] & correct[2] & correct[3])
    candidates = np.flatnonzero(np.all(correct, axis=0))
    case2 = candidates[np.argsort(-np.var(confidence[:, candidates], axis=0), kind="stable")]
    return {"case1": case1, "case2": case2, "fallback": 0, "case1_missing": len(case1) == 0, "case2_missing": len(case2) == 0}


def gradcam(model, image, target_class=None):
    """Grad-CAM für eine explizite Zielklasse; Nullaktivierungen bleiben null."""
    image = np.asarray(image, dtype=np.float32)
    if image.ndim != 3 or image.shape[-1] != 3 or not np.all(np.isfinite(image)):
        raise ValueError("Ein RGB-Bild mit endlichen Werten erforderlich.")
    backbone = model.get_layer("backbone")
    feature_model = tf.keras.Model(backbone.input, [backbone.get_layer("Conv_1").output, backbone.output])
    inputs = tf.convert_to_tensor(image[None])
    processed = model.get_layer("mobilenet_skalierung")(model.get_layer("bildgroesse")(inputs))
    if target_class is None:
        target_class = int(tf.argmax(model(inputs, training=False)[0]))
    if not 0 <= int(target_class) < model.output_shape[-1]:
        raise ValueError("Zielklasse liegt außerhalb des Modelloutputs.")
    with tf.GradientTape() as tape:
        activations, features = feature_model(processed, training=False)
        result = model.get_layer("klassen")(model.get_layer("modellkopf")(model.get_layer("merkmale")(features)))
        score = result[:, int(target_class)]
    gradients = tape.gradient(score, activations)
    if gradients is None:
        raise RuntimeError("Grad-CAM-Gradient ist nicht mit dem Zielscore verbunden.")
    weights = tf.reduce_mean(gradients, axis=(0, 1, 2))
    heatmap = tf.maximum(tf.reduce_sum(activations[0] * weights, axis=-1), 0)
    heatmap = tf.math.divide_no_nan(heatmap, tf.reduce_max(heatmap))
    return tf.image.resize(heatmap[..., None], image.shape[:2])[..., 0].numpy()


def plot_evaluation(probabilities, y, label_names, title=""):
    """Konfusionsmatrix und klassenweise Kalibrierung (nicht Top-Label-Accuracy)."""
    import matplotlib.pyplot as plt
    from sklearn.calibration import calibration_curve
    labels, probabilities = _labels(y), _probabilities(probabilities, len(label_names))
    metrics = evaluate_probabilities(labels, probabilities, len(label_names))
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    cm = metrics["confusion_matrix"]
    axes[0].imshow(cm, cmap="Blues")
    for row in range(len(label_names)):
        for column in range(len(label_names)):
            axes[0].text(column, row, str(cm[row, column]), ha="center", va="center", color="white" if cm[row, column] > cm.max() / 2 else "black")
    axes[0].set(xticks=range(len(label_names)), yticks=range(len(label_names)), xticklabels=label_names, yticklabels=label_names, xlabel="Vorhergesagte Klasse", ylabel="Wahre Klasse", title="Konfusionsmatrix")
    axes[0].tick_params(axis="x", labelrotation=55)
    for c, name in enumerate(label_names):
        observed, predicted = calibration_curve(labels == c, probabilities[:, c], n_bins=5, strategy="uniform")
        axes[1].plot(predicted, observed, marker=("o", "s", "^", "D", "v", "P", "X", "<")[c % 8], label=name)
    axes[1].plot([0, 1], [0, 1], "k--", label="ideal")
    axes[1].set(xlabel="Mittlere Wahrscheinlichkeit für die jeweilige Klasse", ylabel="Beobachteter Anteil dieser Klasse", title="Klassenweise Kalibrierung", xlim=(0, 1), ylim=(0, 1))
    axes[1].legend(fontsize=8)
    fig.suptitle(title + f" · Accuracy {metrics['accuracy']:.3f} · Macro-F1 {metrics['macro_f1']:.3f}")
    fig.tight_layout()
    plt.show()
    return metrics


def plot_example(model, image, true_label, label_names, target_class=None):
    import matplotlib.pyplot as plt
    probability = model(np.asarray(image)[None], training=False).numpy()[0]
    predicted = int(probability.argmax())
    target = predicted if target_class is None else int(target_class)
    heatmap = gradcam(model, image, target)
    fig, axes = plt.subplots(1, 2, figsize=(8, 3.8))
    axes[0].imshow(np.asarray(image).astype(np.uint8))
    axes[1].imshow(np.asarray(image).astype(np.uint8))
    axes[1].imshow(heatmap, cmap="magma", alpha=.45, vmin=0, vmax=1)
    for ax in axes:
        ax.axis("off")
    axes[0].set_title("Lehrbild")
    axes[1].set_title(f"Grad-CAM für {label_names[target]}")
    fig.suptitle(f"Label: {label_names[int(true_label)]} · Vorhersage: {label_names[predicted]} · Konfidenz {probability[predicted]:.2f}")
    fig.tight_layout()
    plt.show()
    return heatmap
