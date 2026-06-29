"""Ganancia y relacion senal/ruido (S/N) de las XA a partir de espectros SPE.

Objetivo 1: con datos de trigger externo a nivel de fotoelectron unico (SPE), se
construye el histograma de carga (finger plot), se localizan los picos 0-PE,
1-PE, ... y se extrae:
    - ganancia  = separacion media entre picos PE consecutivos (cuentas*muestra/PE)
    - S/N       = (mu_1PE - mu_0PE) / sigma_0PE

Mismo criterio que el CalibrationHistogram de waffles (picos PE equiespaciados),
pero autocontenido para los datos de CoMPASS del IR02. Sirve igual para comparar
LED vs laser: se procesan los dos runs y se comparan ganancia y S/N.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

import xa_config as cfg
from xa_waveforms import WaveformFeatures, compute_features, get_observable


@dataclass
class CalibrationResult:
    observable: str
    centers: np.ndarray
    counts: np.ndarray
    peak_positions: np.ndarray      # posicion (valor del observable) de cada pico PE
    peak_indices: np.ndarray        # indice de bin de cada pico
    gain: float                     # separacion media entre picos consecutivos
    gain_err: float
    snr: float                      # (mu1 - mu0) / sigma0
    pedestal_mean: float
    pedestal_sigma: float
    onepe_mean: float
    onepe_sigma: float
    fits: dict = field(default_factory=dict)


def _gaussian(x, a, mu, sigma):
    return a * np.exp(-0.5 * ((x - mu) / sigma) ** 2)


def build_histogram(values: np.ndarray, bins: int, rng):
    if rng is None:
        lo, hi = np.percentile(values, [0.5, 99.5])
        rng = (lo, hi)
    counts, edges = np.histogram(values, bins=bins, range=rng)
    centers = 0.5 * (edges[:-1] + edges[1:])
    return counts.astype(float), centers


def find_pe_peaks(centers, counts, params=cfg.CALIB):
    """Localiza los picos PE en el histograma (find_peaks con suavizado)."""
    from scipy.ndimage import gaussian_filter1d
    from scipy.signal import find_peaks

    smooth = gaussian_filter1d(counts, params["smooth_sigma_bins"])
    prominence = params["prominence"] * smooth.max()
    distance = max(1, int(params["min_peak_distance_frac"] * len(centers)))
    idx, _ = find_peaks(smooth, prominence=prominence, distance=distance)
    if params["max_peaks"]:
        # Quedarse con los max_peaks mas prominentes, reordenados por posicion.
        order = np.argsort(smooth[idx])[::-1][: params["max_peaks"]]
        idx = np.sort(idx[order])
    return idx


def _fit_peak(centers, counts, mu0, half_width):
    """Ajusta una gaussiana en una ventana alrededor de mu0. Devuelve (a, mu, sigma)."""
    from scipy.optimize import curve_fit

    mask = np.abs(centers - mu0) < half_width
    if mask.sum() < 4:
        return None
    x, y = centers[mask], counts[mask]
    p0 = [y.max(), mu0, half_width / 2]
    try:
        popt, _ = curve_fit(_gaussian, x, y, p0=p0, maxfev=10000)
        return float(popt[0]), float(popt[1]), abs(float(popt[2]))
    except (RuntimeError, ValueError):
        return None


def calibrate(
    features: WaveformFeatures,
    params: dict = cfg.CALIB,
) -> CalibrationResult:
    """Construye el histograma de calibracion y extrae ganancia y S/N."""
    values = get_observable(features, params["observable"])
    counts, centers = build_histogram(values, params["bins"], params["range"])
    idx = find_pe_peaks(centers, counts, params)
    peak_pos = centers[idx]

    if len(peak_pos) >= 2:
        spacings = np.diff(peak_pos)
        gain = float(spacings.mean())
        gain_err = float(spacings.std(ddof=1) / np.sqrt(len(spacings))) if len(spacings) > 1 else 0.0
    else:
        gain = gain_err = float("nan")

    fits = {}
    half = 0.4 * gain if np.isfinite(gain) else (centers[-1] - centers[0]) / 20
    ped = _fit_peak(centers, counts, peak_pos[0], half) if len(peak_pos) >= 1 else None
    one = _fit_peak(centers, counts, peak_pos[1], half) if len(peak_pos) >= 2 else None
    if ped:
        fits["pedestal"] = ped
    if one:
        fits["onepe"] = one

    ped_mean, ped_sigma = (ped[1], ped[2]) if ped else (float("nan"), float("nan"))
    one_mean, one_sigma = (one[1], one[2]) if one else (float("nan"), float("nan"))
    snr = (one_mean - ped_mean) / ped_sigma if ped and one and ped_sigma > 0 else float("nan")

    return CalibrationResult(
        observable=params["observable"],
        centers=centers,
        counts=counts,
        peak_positions=peak_pos,
        peak_indices=idx,
        gain=gain,
        gain_err=gain_err,
        snr=snr,
        pedestal_mean=ped_mean,
        pedestal_sigma=ped_sigma,
        onepe_mean=one_mean,
        onepe_sigma=one_sigma,
        fits=fits,
    )


def calibrate_run(arrays: dict, channel: int, params: dict = cfg.CALIB) -> CalibrationResult:
    """Atajo: features + calibracion para un canal de un run (salida de load_run_arrays)."""
    feats = compute_features(arrays[channel].waveforms)
    return calibrate(feats, params)


def plot_calibration(result: CalibrationResult, ax=None, label: str | None = None):
    """Dibuja el finger plot con los picos PE y los fits de pedestal/1PE."""
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots(figsize=(8, 4.5))
    ax.step(result.centers, result.counts, where="mid", label=label)
    ax.plot(result.peak_positions, result.counts[result.peak_indices], "rv", ms=8,
            label="picos PE")
    for name, (a, mu, sigma) in result.fits.items():
        xx = np.linspace(mu - 4 * sigma, mu + 4 * sigma, 200)
        ax.plot(xx, _gaussian(xx, a, mu, sigma), "--", lw=1.5)
    ax.set_xlabel(f"{result.observable} (cuentas*muestra)")
    ax.set_ylabel("eventos")
    title = f"Ganancia = {result.gain:.1f} ± {result.gain_err:.1f}   S/N = {result.snr:.2f}"
    ax.set_title(title)
    ax.legend()
    return ax
