"""Visores y plots de comparacion para las XA del IR02.

Cubre:
  - Objetivo 2: comparar la respuesta de los dos canales de la XA (y LED vs laser).
  - Objetivo 3: waveforms, persistencia e histogramas de ADC, y comparacion del
    histograma de energia de CoMPASS (integral on-board) con la carga calculada
    a partir de la waveform (analisis tradicional).

Sin dependencia de ipywidgets: funciones de ploteo directas (mismo criterio que
los visores de PMTcheck por defecto).
"""

from __future__ import annotations

import numpy as np

import xa_config as cfg
from xa_waveforms import compute_features, get_observable, subtract_baseline, time_axis_ns


def plot_waveforms(arrays, channel, n=20, baseline_subtracted=True, ax=None):
    """Dibuja las primeras n waveforms de un canal (crudas o con baseline restada)."""
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots(figsize=(9, 4.5))
    wfs = arrays[channel].waveforms[:n]
    t = time_axis_ns(wfs.shape[1])
    y = subtract_baseline(wfs)[0] if baseline_subtracted else wfs.astype(float)
    for w in y:
        ax.plot(t, w, lw=0.6, alpha=0.6)
    ax.set_xlabel("t (ns)")
    ax.set_ylabel("signal (cuentas)" if baseline_subtracted else "ADC")
    ax.set_title(f"{cfg.CHANNEL_LABELS.get(channel, channel)} — {len(y)} waveforms")
    return ax


def average_waveform(arrays, channel):
    """Waveform promedio (con baseline restada y polaridad corregida) de un canal."""
    signal = subtract_baseline(arrays[channel].waveforms)[0]
    return time_axis_ns(signal.shape[1]), signal.mean(axis=0)


def plot_channel_comparison_avg(arrays, channels=None, ax=None):
    """Objetivo 2: superpone la waveform promedio de los dos canales de la XA."""
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots(figsize=(9, 4.5))
    for ch in (channels or sorted(arrays)):
        t, avg = average_waveform(arrays, ch)
        ax.plot(t, avg, lw=1.5, label=cfg.CHANNEL_LABELS.get(ch, f"CH{ch}"))
    ax.set_xlabel("t (ns)")
    ax.set_ylabel("signal media (cuentas)")
    ax.set_title("Waveform promedio por canal")
    ax.legend()
    return ax


def persistence(arrays, channel, t_bins=300, y_bins=300, baseline_subtracted=True, ax=None):
    """Diagrama de persistencia (heatmap 2D) de las waveforms de un canal."""
    import matplotlib.pyplot as plt

    wfs = arrays[channel].waveforms
    y = subtract_baseline(wfs)[0] if baseline_subtracted else wfs.astype(float)
    t = time_axis_ns(wfs.shape[1])
    tt = np.repeat(t[None, :], y.shape[0], axis=0).ravel()
    if ax is None:
        _, ax = plt.subplots(figsize=(9, 4.5))
    ax.hist2d(tt, y.ravel(), bins=[t_bins, y_bins], cmap="viridis",
              norm="log")
    ax.set_xlabel("t (ns)")
    ax.set_ylabel("signal (cuentas)" if baseline_subtracted else "ADC")
    ax.set_title(f"Persistencia — {cfg.CHANNEL_LABELS.get(channel, channel)}")
    return ax


def adc_sample_histogram(arrays, channel, bins=200, ax=None):
    """Objetivo 3: histograma de TODAS las muestras ADC crudas de un canal."""
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots(figsize=(7, 4.5))
    ax.hist(arrays[channel].waveforms.ravel(), bins=bins, histtype="step")
    ax.set_xlabel("ADC (cuentas)")
    ax.set_ylabel("muestras")
    ax.set_yscale("log")
    ax.set_title(f"Histograma de ADC — {cfg.CHANNEL_LABELS.get(channel, channel)}")
    return ax


def compare_channels_observable(arrays, observable="qshort", bins=200, channels=None, ax=None):
    """Objetivo 2: superpone el histograma de un observable para los dos canales."""
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots(figsize=(7, 4.5))
    for ch in (channels or sorted(arrays)):
        feats = compute_features(arrays[ch].waveforms)
        values = get_observable(feats, observable)
        ax.hist(values, bins=bins, histtype="step",
                label=cfg.CHANNEL_LABELS.get(ch, f"CH{ch}"))
    ax.set_xlabel(f"{observable} (cuentas*muestra)")
    ax.set_ylabel("eventos")
    ax.set_title("Comparacion entre canales")
    ax.legend()
    return ax


def compare_energy_vs_charge(arrays, channel, bins=200, ax=None):
    """Objetivo 3: histograma de la energia on-board de CoMPASS vs la carga
    integrada de la waveform (analisis tradicional). Se normalizan en area para
    poder comparar la forma de ambos espectros."""
    import matplotlib.pyplot as plt

    data = arrays[channel]
    feats = compute_features(data.waveforms)
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 4.5))
    ax.hist(data.energy, bins=bins, histtype="step", density=True,
            label="energia CoMPASS (on-board)")
    ax.hist(feats.qlong, bins=bins, histtype="step", density=True,
            label="carga waveform (qlong)")
    ax.set_xlabel("observable (normalizado)")
    ax.set_ylabel("densidad")
    ax.set_title(f"CoMPASS vs waveform — {cfg.CHANNEL_LABELS.get(channel, channel)}")
    ax.legend()
    return ax
