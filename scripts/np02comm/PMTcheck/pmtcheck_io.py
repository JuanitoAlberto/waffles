"""Carga de WaveformSets para el analisis PMTcheck.

Sustituye el antiguo bloque "Opcion A / Opcion B" del notebook:
- use_cache=True lee el HDF5 generado por pmt_preprocess.py (rapido).
- use_cache=False carga desde EOS y ejecuta runBasicWfAnaNP02 (lento).
"""

from pathlib import Path

from waffles.input_output.pickle_hdf5_reader import WaveformSet_from_hdf5_pickle
from waffles.np02_utils.load_utils import open_processed
from waffles.np02_utils.PlotUtils import runBasicWfAnaNP02

from pmtcheck_config import BASELINE_FINISH, CACHE_DIR, DEFAULT_DATADIR, THRESHOLD


def cache_path(run: int, dettype: str = "pmt", outdir: Path = CACHE_DIR) -> Path:
    return Path(outdir) / f"wfset_run{run:06d}_{dettype}_ana.hdf5"


def load_wfset(
    run: int,
    dettype: str = "pmt",
    use_cache: bool = True,
    datadir: str = DEFAULT_DATADIR,
    nwaveforms: int | None = None,
    mergefiles: bool = True,
):
    """Devuelve el WaveformSet del run con el analisis 'std' ya calculado."""
    if use_cache:
        path = cache_path(run, dettype)
        if not path.exists():
            raise FileNotFoundError(
                f"No existe el cache {path}.\n"
                f"Generalo con: python pmt_preprocess.py --run {run}"
            )
        return WaveformSet_from_hdf5_pickle(str(path))

    print(f"Loading run {run} ({dettype}) from {datadir}")
    wfset = open_processed(
        run, dettype=dettype, datadir=datadir, nwaveforms=nwaveforms, mergefiles=mergefiles
    )
    print("Running BasicWfAna...")
    runBasicWfAnaNP02(wfset, baselinefinish=BASELINE_FINISH, onlyoptimal=True, threshold=THRESHOLD)
    return wfset
