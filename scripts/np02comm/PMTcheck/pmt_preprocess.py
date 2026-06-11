#!/usr/bin/env python

import os
import argparse
from pathlib import Path

from waffles.data_classes.Waveform import Waveform
from waffles.data_classes.WaveformSet import WaveformSet
from waffles.np02_utils.PlotUtils import runBasicWfAnaNP02
from waffles.np02_utils.load_utils import open_processed
from waffles.input_output.persistence_utils import WaveformSet_to_file


DEFAULT_DATADIR = "/eos/experiment/neutplatform/protodune/experiments/ProtoDUNE-VD/commissioning/"
DEFAULT_OUTDIR  = Path(__file__).parent / "cache"

BASELINE_FINISH = 65
THRESHOLD       = 25


def cache_path(run: int, dettype: str, outdir: Path) -> Path:
    return outdir / f"wfset_run{run:06d}_{dettype}_ana.hdf5"


def adjust_offset(wf: Waveform) -> bool:
    # return -5 < wf.daq_window_timestamp - wf.timestamp < 25
    return True


def main(run: int, dettype: str, datadir: str, outdir: Path, nwaveforms: int | None, force: bool) -> None:
    out = cache_path(run, dettype, outdir)

    if out.exists() and not force:
        print(f"Cache already exists: {out}\nUse --force to overwrite.")
        return

    print(f"Loading run {run} ({dettype}) from {datadir}")
    wfset_full = open_processed(run, dettype=dettype, datadir=datadir, nwaveforms=nwaveforms, mergefiles=True)

    wfset_triggered_all = WaveformSet.from_filtered_WaveformSet(wfset_full, adjust_offset, show_progress=True)
    del wfset_full

    print("Running BasicWfAna...")
    runBasicWfAnaNP02(wfset_triggered_all, baselinefinish=BASELINE_FINISH, onlyoptimal=True, threshold=THRESHOLD)

    outdir.mkdir(parents=True, exist_ok=True)
    print(f"Saving to {out}")
    WaveformSet_to_file(wfset_triggered_all, str(out), overwrite=force, format="hdf5")
    print("Done.")


if __name__ == "__main__":
    argp = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Pre-process a PMT run: load, filter and run BasicWfAna, then save to cache.",
    )
    argp.add_argument("--run",        type=int,   required=True,                   help="Run number to process.")
    argp.add_argument("--dettype",    type=str,   default="pmt",                   help="Detector type (pmt / cathode / membrane).")
    argp.add_argument("--datadir",    type=str,   default=DEFAULT_DATADIR,         help="Directory with processed waveform files.")
    argp.add_argument("--outdir",     type=str,   default=str(DEFAULT_OUTDIR),     help="Directory where the cache HDF5 will be written.")
    argp.add_argument("--nwaveforms", type=int,   default=None,                    help="Max waveforms to load (None = all).")
    argp.add_argument("--force",      action="store_true",                         help="Overwrite existing cache file.")

    args = argp.parse_args()
    main(
        run=args.run,
        dettype=args.dettype,
        datadir=args.datadir,
        outdir=Path(args.outdir),
        nwaveforms=args.nwaveforms,
        force=args.force,
    )
