"""Temporärer Windows-Energieschutz ausschließlich während eines Trainings.

Verhindert automatischen Leerlauf-Ruhezustand, nicht Sperren des Bildschirms.
Manuelles Schlafen, Herunterfahren und zentrale Richtlinien bleiben wirksam.
Quelle: Microsoft Learn, SetThreadExecutionState (winbase.h).
"""

import ctypes
import sys
import warnings
from contextlib import contextmanager


@contextmanager
def training_wachhalten():
    """Energieanforderung auf dem aufrufenden Thread setzen und zurücknehmen."""
    if sys.platform != "win32":
        warnings.warn("Kein Windows: Ruhezustand für das Training selbst prüfen.", RuntimeWarning)
        yield
        return

    continuous = 0x80000000
    system_required = 0x00000001
    set_state = ctypes.WinDLL("kernel32", use_last_error=True).SetThreadExecutionState
    set_state.argtypes = [ctypes.c_uint32]
    set_state.restype = ctypes.c_uint32
    previous = set_state(continuous | system_required)
    if not previous:
        raise OSError("Windows konnte den Trainings-Energieschutz nicht aktivieren. IT informieren.")
    try:
        yield
    finally:
        # Auch bei Fehler oder Notebook-Unterbrechung den früheren Zustand herstellen.
        if not set_state(previous | continuous):
            warnings.warn("Windows-Energieanforderung konnte nicht zurückgesetzt werden.", RuntimeWarning)


def trainiere_ohne_ruhezustand(model, *args, **kwargs):
    """model.fit mit zeitlich begrenztem Energieschutz; Rückgabewert unverändert."""
    with training_wachhalten():
        return model.fit(*args, **kwargs)
