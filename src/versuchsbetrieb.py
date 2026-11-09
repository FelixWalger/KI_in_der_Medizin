"""Lokale Daten, nachvollziehbare Trainingsstände und kleine Aufgabenkontrollen."""
from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path

import numpy as np


def lade_histologie(root):
    file = Path(root) / 'data' / 'histologie.npz'
    if not file.exists():
        raise FileNotFoundError(
            'Lokales Histologiepaket fehlt. Die IT führt vor dem Workshop '
            'scripts/bereite_hauptversuch_vor.py aus und verteilt das Paket. '
            'Im Workshop werden keine Daten heruntergeladen.'
        )
    with np.load(file, allow_pickle=False) as data:
        x, y, names = data['bilder'], data['labels'], data['klassennamen'].tolist()
    if x.ndim != 4 or x.shape[1:] != (150, 150, 3) or x.dtype != np.uint8:
        raise ValueError('Histologiebilder müssen RGB-Arrays mit 150×150 Pixeln sein.')
    if len(x) != len(y) or len(names) != 8 or set(np.unique(y)) != set(range(8)):
        raise ValueError('Das Histologiepaket muss alle acht Klassen enthalten.')
    return x, y, names


def kontrolliere(name, antwort, erwartet):
    """Lernfeedback; offene Antworten blockieren keine vorbereiteten Modellläufe."""
    try:
        ok = antwort is not None and np.shape(antwort) == np.shape(erwartet) and bool(
            np.allclose(antwort, erwartet, atol=1e-6)
        )
    except (TypeError, ValueError):
        ok = False
    print(('✓ ' if ok else '↻ ') + name + (
        ': stimmt.' if ok else ': noch offen. Nutze Hinweis oder Lösung. '
        'Die folgenden Experimente verwenden den bereitgestellten Referenzcode.'
    ))
    return ok


def _fingerprint(config, x_train, y_train, x_val, y_val):
    digest = hashlib.sha256(json.dumps(config, sort_keys=True).encode())
    for array in (x_train, y_train, x_val, y_val):
        array = np.ascontiguousarray(array)
        digest.update(str((array.shape, str(array.dtype))).encode())
        digest.update(memoryview(array).cast('B'))
    return digest.hexdigest()


def _write_json(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding='utf-8')
    os.replace(temporary, path)


def modelllauf(root, name, x_train, y_train, x_val, y_val, *, epochs=30,
               num_unfrozen_layers=3, learning_rate=.001, class_weight=None,
               loss='cross_entropy', alpha=None, gamma=2., weights='imagenet',
               batch_size=64, seed=313, training_erlaubt=True):
    """Lade passenden Stand oder trainiere weiter; Testbilder werden nicht übergeben.

    Jeder abgeschlossene Epochenstand erhält eine eigene Modelldatei. Erst nach
    erfolgreichem Speichern verweist stand.json darauf. Abbruch verliert so
    höchstens die laufende Epoche. Identische Konfiguration kann weiterlaufen.
    """
    import tensorflow as tf
    import hauptversuch as hv
    from training_betrieb import trainiere_ohne_ruhezustand

    if not isinstance(name, str) or not re.fullmatch(r'[A-Za-z0-9_-]+', name):
        raise ValueError('Laufname muss ein einfacher Ordnername sein.')
    if not isinstance(epochs, int) or epochs < 1:
        raise ValueError('epochs muss eine positive ganze Zahl sein.')
    if loss not in ('cross_entropy', 'focal'):
        raise ValueError('Unbekannte Verlustfunktion.')
    config = dict(version=1, epochs=epochs, num_unfrozen_layers=num_unfrozen_layers,
                  learning_rate=learning_rate, class_weight=class_weight,
                  loss=loss, alpha=None if alpha is None else np.asarray(alpha).tolist(),
                  gamma=gamma, weights=weights, batch_size=batch_size, seed=seed)
    signature = _fingerprint(config, x_train, y_train, x_val, y_val)
    folder = Path(root) / 'results' / 'hauptversuch' / name
    folder.mkdir(parents=True, exist_ok=True)
    manifest = folder / 'stand.json'
    history = {}
    done = 0
    tf.keras.utils.set_random_seed(seed)
    if manifest.exists():
        state = json.loads(manifest.read_text(encoding='utf-8'))
        if state['fingerprint'] != signature:
            raise ValueError(f'Lauf {name}: Einstellungen oder Daten geändert. '
                             'Neuen Laufnamen verwenden; gespeicherte Daten bleiben erhalten.')
        model = tf.keras.models.load_model(folder / state['model'])
        done, history = state['completed_epochs'], state['history']
        print(f'Gespeicherter eigener Lauf {name}: {done}/{epochs} Epochen.')
        if done < epochs and not training_erlaubt:
            raise RuntimeError(f'Lauf {name} hat erst {done}/{epochs} Epochen. '
                               'Training nach Absprache fortsetzen.')
    else:
        if not training_erlaubt:
            raise FileNotFoundError(f'Für {name} fehlt ein abgeschlossener eigener Lauf. '
                                    'TRAINING_STARTEN nach Absprache auf True setzen '
                                    'oder die IT-Vorbereitung durchführen.')
        model_weights = weights
        if weights == 'imagenet':
            keras_home = Path(os.environ.get('KERAS_HOME', Path.home() / '.keras'))
            weightfile = keras_home / 'models' / 'mobilenet_v2_weights_tf_dim_ordering_tf_kernels_1.0_224_no_top.h5'
            if not weightfile.is_file():
                raise FileNotFoundError('ImageNet-Gewichte fehlen im lokalen Keras-Cache. '
                                        'Die IT lädt sie vor dem Workshop mit --weights.')
            # Übergabe als Datei vermeidet Keras-Download auch bei nur lesbarem Cache.
            model_weights = str(weightfile)
        model = hv.build_model(8, num_unfrozen_layers, learning_rate, weights=model_weights)
        if loss == 'focal':
            model.compile(optimizer=model.optimizer,
                          loss=hv.categorical_focal_loss(gamma=gamma, alpha=alpha),
                          metrics=['accuracy', hv.MacroF1(num_classes=8)])

    if done < epochs:
        class SaveEpoch(tf.keras.callbacks.Callback):
            def on_epoch_end(self, epoch, logs=None):
                for key, value in (logs or {}).items():
                    history.setdefault(key, []).append(float(value))
                filename = f'epoche_{epoch + 1:03d}.keras'
                temporary = folder / f'epoche_{epoch + 1:03d}.tmp.keras'
                self.model.save(temporary)
                os.replace(temporary, folder / filename)
                _write_json(manifest, dict(fingerprint=signature, config=config,
                                          completed_epochs=epoch + 1, model=filename,
                                          history=history))

        # Datenerweiterung auf Rohbildern; das Modell selbst skaliert nach [-1, 1].
        generator = tf.keras.preprocessing.image.ImageDataGenerator(
            rotation_range=10, width_shift_range=.1, height_shift_range=.1,
            horizontal_flip=True, zoom_range=.2, brightness_range=[.8, 1.2])
        trainiere_ohne_ruhezustand(
            model, generator.flow(x_train, tf.keras.utils.to_categorical(y_train, 8),
                                  batch_size=batch_size, seed=seed),
            validation_data=(x_val, tf.keras.utils.to_categorical(y_val, 8)),
            initial_epoch=done, epochs=epochs, class_weight=class_weight,
            callbacks=[SaveEpoch()], verbose=2)
    return model, history


def werte_aus(root, name, model, history, x_test, y_test, klassennamen):
    """Auswertung mit getrennten Testbildern; gemessene Wahrscheinlichkeiten sichern."""
    import hauptversuch as hv
    probabilities = model.predict(x_test, verbose=0)
    metrics = hv.evaluate_probabilities(y_test, probabilities, num_classes=8)
    print(f'{name}: Accuracy={metrics["accuracy"]:.3f}, Macro-F1={metrics["macro_f1"]:.3f}')
    hv.plot_evaluation(probabilities, y_test, klassennamen, title=name)
    folder = Path(root) / 'results' / 'hauptversuch' / name
    folder.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(folder / 'testvorhersagen.npz', probabilities=probabilities,
                        labels=y_test)
    return probabilities
