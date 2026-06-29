"""Estabilidad del LED durante la toma de datos de las XA del IR02.

Objetivo 5: seguir en el tiempo la respuesta a la luz del LED para detectar
derivas. Dos escalas:
  - inter-run: media y dispersion de un observable (p.ej. qshort) run a run,
    ordenados por hora de inicio (estabilidad entre dias).
  - intra-run: perfil del observable frente al tiempo del evento dentro de un run
    (usando los timestamps de CoMPASS), para ver derivas cortas.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

import xa_config as cfg
from xa_io import load_run_arrays
from xa_waveforms import compute_features, get_observable


@dataclass
class RunSummary:
    campaign: str
    run: int
    channel: int
    observable: str
    n: int
    mean: float
    std: float
    median: float
    sem: float       # error estandar de la media


def summarize_run(
    campaign: str,
    run: int,
    channel: int = cfg.STABILITY["reference_channel"],
    observable: str = cfg.STABILITY["observable"],
    max_events: int | None = None,
) -> RunSummary:
    """Resumen estadistico de un observable para un canal de un run."""
    arrays = load_run_arrays(campaign, run, max_events=max_events)
    feats = compute_features(arrays[channel].waveforms)
    values = get_observable(feats, observable)
    n = len(values)
    std = float(values.std(ddof=1)) if n > 1 else 0.0
    return RunSummary(
        campaign=campaign,
        run=run,
        channel=channel,
        observable=observable,
        n=n,
        mean=float(values.mean()),
        std=std,
        median=float(np.median(values)),
        sem=std / np.sqrt(n) if n > 0 else float("nan"),
    )


def summarize_runs(
    campaign: str,
    runs: list[int],
    channel: int = cfg.STABILITY["reference_channel"],
    observable: str = cfg.STABILITY["observable"],
    max_events: int | None = None,
):
    """DataFrame con el resumen por run (para la estabilidad inter-run)."""
    import pandas as pd

    rows = []
    for run in runs:
        try:
            s = summarize_run(campaign, run, channel, observable, max_events)
            rows.append(vars(s))
        except (FileNotFoundError, KeyError, RuntimeError) as exc:
            print(f"  run {run}: omitido ({exc})")
    return pd.DataFrame(rows)


def time_profile(
    campaign: str,
    run: int,
    channel: int = cfg.STABILITY["reference_channel"],
    observable: str = cfg.STABILITY["observable"],
    n_bins: int = 50,
    max_events: int | None = None,
):
    """Perfil intra-run: media del observable en bins de tiempo del evento.

    Devuelve (t_min, mean, sem) con el tiempo relativo al primer evento en minutos.
    """
    arrays = load_run_arrays(campaign, run, max_events=max_events)
    data = arrays[channel]
    feats = compute_features(data.waveforms)
    values = get_observable(feats, observable)

    t_s = (data.timestamps_ps.astype(np.float64) - data.timestamps_ps[0]) * 1e-12
    t_min = t_s / 60.0
    edges = np.linspace(t_min.min(), t_min.max(), n_bins + 1)
    idx = np.clip(np.digitize(t_min, edges) - 1, 0, n_bins - 1)

    centers, means, sems = [], [], []
    for b in range(n_bins):
        sel = idx == b
        if sel.sum() == 0:
            continue
        v = values[sel]
        centers.append(0.5 * (edges[b] + edges[b + 1]))
        means.append(v.mean())
        sems.append(v.std(ddof=1) / np.sqrt(len(v)) if len(v) > 1 else 0.0)
    return np.array(centers), np.array(means), np.array(sems)


def plot_inter_run(df, ax=None, time_col: str = "start_time"):
    """Media ± std del observable por run. Si el catalogo aporta 'start_time' se
    usa como eje; si no, el numero de run."""
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots(figsize=(9, 4.5))
    x = df[time_col] if time_col in df.columns else df["run"].astype(str)
    ax.errorbar(range(len(df)), df["mean"], yerr=df["std"], fmt="o-", capsize=3)
    ax.set_xticks(range(len(df)))
    ax.set_xticklabels(x, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel(f"{df['observable'].iloc[0]} (media ± std)" if len(df) else "media")
    ax.set_title("Estabilidad del LED entre runs")
    ax.grid(True, alpha=0.3)
    return ax
