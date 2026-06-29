"""Carga de runs de las XA del IR02.

Dos niveles de acceso:
  - load_run_arrays(): rapido, sin waffles. Devuelve los arrays numpy crudos de
    CoMPASS (timestamps, energias, waveforms). Lo usan los analisis de carga,
    dark rate, estabilidad e histogramas de ADC.
  - load_run(): construye un WaveformSet de waffles (con cache HDF5), para los
    analisis que quieran reutilizar la infraestructura de waffles.
"""

from __future__ import annotations

from pathlib import Path

import xa_config as cfg
from xa_compass_reader import (
    CompassData,
    compass_run_to_wfset,
    find_channel_files,
    parse_compass_bin,
)


def campaign_dir(campaign: str) -> Path:
    """Ruta en EOS de la campana ('spe' | 'darkcurrent' | 'highlight')."""
    if campaign not in cfg.CAMPAIGNS:
        raise KeyError(
            f"Campana desconocida '{campaign}'. Opciones: {list(cfg.CAMPAIGNS)}"
        )
    return cfg.EOS_BASE / cfg.CAMPAIGNS[campaign] / "DAQ"


def run_dir(campaign: str, run: int | str) -> Path:
    """Directorio del run. 'run' puede ser el numero (1) o el run_id completo."""
    base = campaign_dir(campaign)
    if isinstance(run, int) or str(run).isdigit():
        candidates = sorted(base.glob(f"*run_{int(run)}"))
        candidates = [c for c in candidates if c.name.endswith(f"run_{int(run)}")]
        if not candidates:
            raise FileNotFoundError(f"No hay run_{run} en {base}")
        return candidates[0]
    d = base / str(run)
    if not d.is_dir():
        raise FileNotFoundError(f"No existe {d}")
    return d


def list_runs(campaign: str) -> list[int]:
    """Numeros de run disponibles en una campana, ordenados."""
    runs = []
    for d in campaign_dir(campaign).glob("*run_*"):
        try:
            runs.append(int(d.name.rsplit("run_", 1)[1]))
        except (IndexError, ValueError):
            continue
    return sorted(set(runs))


def cache_path(campaign: str, run: int, outdir: Path = cfg.CACHE_DIR) -> Path:
    return Path(outdir) / f"wfset_{campaign}_run{int(run):04d}.hdf5"


def load_run_arrays(
    campaign: str,
    run: int | str,
    max_events: int | None = None,
    load_waveforms: bool = True,
) -> dict[int, CompassData]:
    """Devuelve {canal: CompassData} leyendo directamente los .BIN (sin waffles)."""
    rdir = run_dir(campaign, run)
    files = find_channel_files(rdir)
    return {
        ch: parse_compass_bin(path, max_events=max_events, load_waveforms=load_waveforms)
        for ch, path in files.items()
    }


def load_run(
    campaign: str,
    run: int,
    use_cache: bool = True,
    max_events: int | None = None,
    channels: list[int] | None = None,
):
    """Devuelve el WaveformSet de waffles del run (con cache HDF5)."""
    from waffles.input_output.pickle_hdf5_reader import WaveformSet_from_hdf5_pickle

    if use_cache:
        path = cache_path(campaign, run)
        if not path.exists():
            raise FileNotFoundError(
                f"No existe el cache {path}.\n"
                f"Generalo con: python xa_preprocess.py --campaign {campaign} --run {run}"
            )
        return WaveformSet_from_hdf5_pickle(str(path))

    rdir = run_dir(campaign, run)
    print(f"Leyendo {campaign} run {run} desde {rdir}")
    return compass_run_to_wfset(rdir, run_number=run, max_events=max_events, channels=channels)
