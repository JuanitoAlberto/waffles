"""Procesado de bajo nivel de las waveforms de las XA del IR02.

Operaciones vectorizadas sobre un array 2D de waveforms crudas (n_eventos,
n_muestras) tal como salen de CoMPASS. Aqui se centraliza la convencion de
baseline, polaridad y gates de carga; el resto de modulos (carga/ganancia, dark
rate, estabilidad, visores) parten de estas funciones para no duplicar criterios.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

import xa_config as cfg


@dataclass
class WaveformFeatures:
    """Magnitudes por evento extraidas de un conjunto de waveforms (un canal)."""

    baseline: np.ndarray    # nivel de baseline en cuentas ADC
    rms: np.ndarray         # RMS del baseline (ruido) en cuentas ADC
    amplitude: np.ndarray   # amplitud de pico (cuentas, ya con polaridad corregida)
    peak_pos: np.ndarray    # posicion del pico (muestra)
    qshort: np.ndarray      # integral gate corto
    qlong: np.ndarray       # integral gate largo


def subtract_baseline(
    waveforms: np.ndarray,
    baseline_window: slice = cfg.BASELINE_WINDOW,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Devuelve (signal, baseline, rms).

    'signal' tiene la polaridad corregida (pulsos positivos) y la baseline
    restada: signal = POLARITY * (adc - baseline).
    """
    waveforms = np.asarray(waveforms, dtype=np.float64)
    base = waveforms[:, baseline_window].mean(axis=1)
    rms = waveforms[:, baseline_window].std(axis=1)
    signal = cfg.POLARITY * (waveforms - base[:, None])
    return signal, base, rms


def compute_features(
    waveforms: np.ndarray,
    baseline_window: slice = cfg.BASELINE_WINDOW,
    peak_window: slice = cfg.PEAK_SEARCH_WINDOW,
    qshort_gate: int = cfg.QSHORT_GATE_SAMPLES,
    qlong_gate: int = cfg.QLONG_GATE_SAMPLES,
    pre_samples: int = 5,
) -> WaveformFeatures:
    """Extrae baseline, ruido, amplitud, posicion de pico y cargas por evento.

    Las cargas se integran desde 'pre_samples' muestras antes del pico, sobre la
    'signal' con polaridad corregida y baseline restada (unidades: cuentas*muestra).
    """
    signal, base, rms = subtract_baseline(waveforms, baseline_window)
    n_events, n_samples = signal.shape

    seg = signal[:, peak_window]
    amplitude = seg.max(axis=1)
    peak_pos = seg.argmax(axis=1) + (peak_window.start or 0)

    start = np.clip(peak_pos - pre_samples, 0, n_samples - 1)
    # Suma acumulada para integrar gates de longitud fija de forma vectorizada.
    csum = np.concatenate([np.zeros((n_events, 1)), np.cumsum(signal, axis=1)], axis=1)

    def gate_integral(gate: int) -> np.ndarray:
        end = np.clip(start + gate, 0, n_samples)
        return csum[np.arange(n_events), end] - csum[np.arange(n_events), start]

    return WaveformFeatures(
        baseline=base,
        rms=rms,
        amplitude=amplitude,
        peak_pos=peak_pos,
        qshort=gate_integral(qshort_gate),
        qlong=gate_integral(qlong_gate),
    )


def time_axis_ns(n_samples: int = cfg.NSAMPLES) -> np.ndarray:
    """Eje temporal de una waveform, en ns."""
    return np.arange(n_samples) * cfg.SAMPLE_NS


def adc_to_mV(adc: np.ndarray | float) -> np.ndarray | float:
    """Convierte cuentas ADC a mV usando el rango dinamico del DT2730."""
    return np.asarray(adc) * (cfg.ADC_DYNAMIC_VPP / (cfg.ADC_MAX + 1)) * 1e3


def get_observable(features: WaveformFeatures, name: str) -> np.ndarray:
    """Devuelve el array del observable pedido ('qshort'|'qlong'|'amplitude')."""
    try:
        return getattr(features, name)
    except AttributeError as exc:
        raise KeyError(
            f"Observable desconocido '{name}'. Usa qshort, qlong o amplitude."
        ) from exc
