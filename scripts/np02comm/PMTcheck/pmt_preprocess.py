#!/usr/bin/env python

import argparse
from pathlib import Path

from waffles.input_output.persistence_utils import WaveformSet_to_file

from pmtcheck_config import CACHE_DIR, DEFAULT_DATADIR
from pmtcheck_io import cache_path, load_wfset


def main(run: int, dettype: str, datadir: str, outdir: Path, nwaveforms: int | None, force: bool) -> None:
    out = cache_path(run, dettype, outdir)

    if out.exists() and not force:
        print(f"Cache already exists: {out}\nUse --force to overwrite.")
        return

    wfset = load_wfset(
        run, dettype=dettype, use_cache=False, datadir=datadir, nwaveforms=nwaveforms
    )

    outdir.mkdir(parents=True, exist_ok=True)
    print(f"Saving to {out}")
    WaveformSet_to_file(wfset, str(out), overwrite=force, format="hdf5")
    print("Done.")


if __name__ == "__main__":
    argp = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Pre-process a PMT run: load and run BasicWfAna, then save to cache.",
    )
    argp.add_argument("--run",        type=int,   required=True,               help="Run number to process.")
    argp.add_argument("--dettype",    type=str,   default="pmt",               help="Detector type (pmt / cathode / membrane).")
    argp.add_argument("--datadir",    type=str,   default=DEFAULT_DATADIR,     help="Directory with processed waveform files.")
    argp.add_argument("--outdir",     type=str,   default=str(CACHE_DIR),      help="Directory where the cache HDF5 will be written.")
    argp.add_argument("--nwaveforms", type=int,   default=None,                help="Max waveforms to load (None = all).")
    argp.add_argument("--force",      action="store_true",                     help="Overwrite existing cache file.")

    args = argp.parse_args()
    main(
        run=args.run,
        dettype=args.dettype,
        datadir=args.datadir,
        outdir=Path(args.outdir),
        nwaveforms=args.nwaveforms,
        force=args.force,
    )
