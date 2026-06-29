#!/usr/bin/env python
"""Pre-procesa runs de las XA del IR02: lee los .BIN de CoMPASS, construye el
WaveformSet de waffles y lo guarda en cache/ como HDF5 (carga rapida en los
notebooks). Equivalente a pmt_preprocess.py de PMTcheck.

Ejemplos:
    python xa_preprocess.py --campaign spe --run 1
    python xa_preprocess.py --campaign spe --all
    python xa_preprocess.py --campaign highlight --run 13 --max-events 5000
"""

import argparse
from pathlib import Path

from waffles.input_output.persistence_utils import WaveformSet_to_file

import xa_config as cfg
from xa_io import cache_path, list_runs, load_run


def process_one(campaign: str, run: int, outdir: Path, max_events, force: bool) -> None:
    out = cache_path(campaign, run, outdir)
    if out.exists() and not force:
        print(f"Cache ya existe: {out}  (usa --force para sobrescribir)")
        return

    wfset = load_run(campaign, run, use_cache=False, max_events=max_events)
    outdir.mkdir(parents=True, exist_ok=True)
    print(f"Guardando {out}")
    WaveformSet_to_file(wfset, str(out), overwrite=force, format="hdf5")


def main() -> None:
    argp = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Convierte runs de CoMPASS (XA IR02) a cache HDF5 de waffles.",
    )
    argp.add_argument("--campaign", required=True, choices=list(cfg.CAMPAIGNS),
                      help="Campana de medida.")
    group = argp.add_mutually_exclusive_group(required=True)
    group.add_argument("--run", type=int, help="Numero de run a procesar.")
    group.add_argument("--all", action="store_true", help="Procesa todos los runs de la campana.")
    argp.add_argument("--outdir", type=str, default=str(cfg.CACHE_DIR),
                      help="Directorio del cache HDF5.")
    argp.add_argument("--max-events", type=int, default=None,
                      help="Maximo de eventos por canal (None = todos).")
    argp.add_argument("--force", action="store_true", help="Sobrescribe el cache existente.")
    args = argp.parse_args()

    outdir = Path(args.outdir)
    runs = list_runs(args.campaign) if args.all else [args.run]
    for run in runs:
        process_one(args.campaign, run, outdir, args.max_events, args.force)
    print("Hecho.")


if __name__ == "__main__":
    main()
