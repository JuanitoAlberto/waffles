"""Corriente oscura (dark count rate) a nivel de SPE para las XA del IR02.

Objetivo 4: comprobar si es posible medir la tasa de cuentas oscuras a nivel de
fotoelectron unico y a varios over-voltages (OV).

Dos vias complementarias:
  - Staircase: a partir de las waveforms de un run de dark current (self-trigger),
    tasa(umbral) = N(amplitud > umbral) / tiempo_vivo. Con la ganancia SPE el
    umbral se expresa en PE; la tasa a 0.5 PE es la DCR.
  - MCS: CoMPASS guarda en OTHER/ un CSV de cuentas vs tiempo (rate-vs-time) que
    sirve para validar la estabilidad de la tasa durante el run.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

import xa_config as cfg
from xa_waveforms import WaveformFeatures, compute_features


@dataclass
class StaircaseResult:
    thresholds_pe: np.ndarray   # umbrales en unidades de PE
    rate_hz: np.ndarray         # tasa (Hz) por encima de cada umbral
    rate_05pe_hz: float         # DCR a 0.5 PE
    live_time_s: float
    gain: float
    pedestal_mean: float


def amplitude_to_pe(amplitude, gain, pedestal_mean):
    """Convierte amplitud/observable a PE: (valor - pedestal) / gain."""
    return (np.asarray(amplitude) - pedestal_mean) / gain


def staircase(
    features: WaveformFeatures,
    live_time_s: float,
    gain: float,
    pedestal_mean: float,
    observable: str = "amplitude",
    thresholds_pe: np.ndarray | None = None,
) -> StaircaseResult:
    """Calcula la tasa de cuentas oscuras en funcion del umbral (en PE)."""
    values = getattr(features, observable)
    pe = amplitude_to_pe(values, gain, pedestal_mean)
    if thresholds_pe is None:
        thresholds_pe = np.linspace(0.1, 4.0, 40)

    rate = np.array([(pe > thr).sum() / live_time_s for thr in thresholds_pe])
    rate_05 = float((pe > cfg.DARKRATE["threshold_pe"]).sum() / live_time_s)
    return StaircaseResult(
        thresholds_pe=thresholds_pe,
        rate_hz=rate,
        rate_05pe_hz=rate_05,
        live_time_s=live_time_s,
        gain=gain,
        pedestal_mean=pedestal_mean,
    )


def staircase_run(
    arrays: dict,
    channel: int,
    live_time_s: float,
    gain: float,
    pedestal_mean: float,
    observable: str = "amplitude",
) -> StaircaseResult:
    """Atajo: features + staircase para un canal de un run de dark current."""
    feats = compute_features(arrays[channel].waveforms)
    return staircase(feats, live_time_s, gain, pedestal_mean, observable)


def parse_mcs_csv(path) -> tuple[np.ndarray, np.ndarray]:
    """Lee un CSV MCSgraph de CoMPASS (cuentas vs tiempo). Devuelve (t, cuentas)."""
    import csv
    from pathlib import Path

    rows = []
    with open(Path(path), newline="", encoding="utf-8", errors="ignore") as fh:
        for row in csv.reader(fh, delimiter=";"):
            nums = []
            for tok in row:
                try:
                    nums.append(float(tok.replace(",", ".")))
                except ValueError:
                    nums = []
                    break
            if len(nums) >= 2:
                rows.append(nums[:2])
    arr = np.array(rows) if rows else np.empty((0, 2))
    return (arr[:, 0], arr[:, 1]) if arr.size else (arr, arr)


def plot_staircase(result: StaircaseResult, ax=None, label: str | None = None):
    """Dibuja la curva tasa-vs-umbral (escala log) y marca el punto a 0.5 PE."""
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots(figsize=(7, 4.5))
    ax.semilogy(result.thresholds_pe, np.clip(result.rate_hz, 1e-9, None), "-o", ms=3,
                label=label)
    ax.axvline(cfg.DARKRATE["threshold_pe"], color="gray", ls="--", lw=1)
    ax.set_xlabel("umbral (PE)")
    ax.set_ylabel("tasa (Hz)")
    ax.set_title(f"DCR(0.5 PE) = {result.rate_05pe_hz:.1f} Hz")
    if label:
        ax.legend()
    return ax
