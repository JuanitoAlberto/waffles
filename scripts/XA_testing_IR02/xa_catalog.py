"""Catalogo de runs de las XA del IR02.

Extrae de cada run los metadatos del info.txt y del settings.xml de CoMPASS y los
vuelca a runs_catalog.csv. Las columnas que dependen del logbook (source LED/laser,
ov_V, light_level, notes) NO se sobrescriben al regenerar: se preservan las que ya
hayas rellenado a mano.

Uso:
    python xa_catalog.py                 # (re)genera runs_catalog.csv para todas las campanas
    python xa_catalog.py --campaign spe  # solo una campana
"""

from __future__ import annotations

import argparse
import csv
import re
import xml.etree.ElementTree as ET
from pathlib import Path

import xa_config as cfg
from xa_io import campaign_dir, list_runs, run_dir

# Columnas que rellena el usuario y que se preservan entre regeneraciones.
_USER_COLUMNS = ("source", "ov_V", "light_level", "notes")

_OUTPUT_RE = re.compile(r"Output counts\s*=\s*(\d+);\s*average rate \(cps\)\s*=\s*([\d.]+)")
_REALTIME_RE = re.compile(r"Real time\s*=\s*(\d+):(\d+):([\d.]+)")
_CH_HEADER_RE = re.compile(r"^CH(\d+)@")


def parse_info_txt(path: Path) -> dict:
    """Lee el *_info.txt y devuelve start/stop, duracion y, por canal, counts y rate."""
    info: dict = {"start_time": "", "stop_time": "", "duration_s": "", "channels": {}}
    if not path.exists():
        return info

    current_ch = None
    text = path.read_text(encoding="utf-8", errors="ignore")
    for line in text.splitlines():
        s = line.strip()
        if s.startswith("Start time"):
            info["start_time"] = s.split("=", 1)[1].strip()
        elif s.startswith("Stop time"):
            info["stop_time"] = s.split("=", 1)[1].strip()
        else:
            mh = _CH_HEADER_RE.match(s)
            if mh:
                current_ch = int(mh.group(1))
                info["channels"].setdefault(current_ch, {})
            elif current_ch is not None:
                mo = _OUTPUT_RE.search(s)
                if mo:
                    info["channels"][current_ch]["n_events"] = int(mo.group(1))
                    info["channels"][current_ch]["rate_cps"] = float(mo.group(2))
                mr = _REALTIME_RE.search(s)
                if mr and not info["duration_s"]:
                    h, m, sec = mr.groups()
                    info["duration_s"] = round(int(h) * 3600 + int(m) * 60 + float(sec), 3)
    return info


def parse_settings_dcoffset(path: Path) -> str:
    """Lee SRV_PARAM_CH_BLINE_DCOFFSET del settings.xml (porcentaje de baseline)."""
    if not path.exists():
        return ""
    try:
        root = ET.parse(path).getroot()
    except ET.ParseError:
        return ""
    for entry in root.iter("entry"):
        key = entry.find("key")
        if key is not None and key.text == "SRV_PARAM_CH_BLINE_DCOFFSET":
            val = entry.find("value")
            if val is not None and val.text:
                return val.text.strip()
    return ""


def _row_for_run(campaign: str, run: int) -> dict:
    rdir = run_dir(campaign, run)
    run_id = rdir.name
    info = parse_info_txt(rdir / f"{run_id}_info.txt")
    dc = parse_settings_dcoffset(rdir / "settings.xml")
    ch = info["channels"]
    return {
        "campaign": campaign,
        "run": run,
        "run_id": run_id,
        "source": "",
        "ov_V": "",
        "light_level": {"spe": "spe", "darkcurrent": "dark", "highlight": "high"}.get(campaign, ""),
        "start_time": info["start_time"],
        "stop_time": info["stop_time"],
        "duration_s": info["duration_s"],
        "n_events_ch0": ch.get(0, {}).get("n_events", ""),
        "n_events_ch1": ch.get(1, {}).get("n_events", ""),
        "rate_ch0_cps": ch.get(0, {}).get("rate_cps", ""),
        "rate_ch1_cps": ch.get(1, {}).get("rate_cps", ""),
        "dc_offset_pct": dc,
        "notes": "",
    }


def _load_existing(csv_path: Path) -> dict[tuple[str, str], dict]:
    """Devuelve {(campaign, run): fila} del CSV existente, para preservar columnas de usuario."""
    if not csv_path.exists():
        return {}
    with open(csv_path, newline="", encoding="utf-8") as fh:
        return {(r["campaign"], r["run"]): r for r in csv.DictReader(fh)}


def build_catalog(campaigns: list[str] | None = None, csv_path: Path = cfg.CATALOG_CSV) -> Path:
    """(Re)genera el catalogo, preservando las columnas de usuario ya rellenadas."""
    campaigns = campaigns if campaigns is not None else list(cfg.CAMPAIGNS)
    existing = _load_existing(csv_path)

    rows = []
    for campaign in campaigns:
        for run in list_runs(campaign):
            row = _row_for_run(campaign, run)
            prev = existing.get((campaign, str(run)))
            if prev:  # preservar lo que el usuario haya rellenado
                for col in _USER_COLUMNS:
                    if prev.get(col):
                        row[col] = prev[col]
            rows.append(row)

    # Conservar filas de campanas no regeneradas en esta llamada.
    regenerated = {(r["campaign"], str(r["run"])) for r in rows}
    for key, prev in existing.items():
        if key not in regenerated:
            rows.append({c: prev.get(c, "") for c in cfg.CATALOG_COLUMNS})

    rows.sort(key=lambda r: (r["campaign"], int(r["run"])))
    with open(csv_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=cfg.CATALOG_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Catalogo escrito: {csv_path}  ({len(rows)} runs)")
    return csv_path


def load_catalog(csv_path: Path = cfg.CATALOG_CSV):
    """Carga el catalogo como DataFrame de pandas (para los notebooks)."""
    import pandas as pd

    if not csv_path.exists():
        raise FileNotFoundError(f"No existe {csv_path}. Generalo con: python xa_catalog.py")
    return pd.read_csv(csv_path)


if __name__ == "__main__":
    argp = argparse.ArgumentParser(description=__doc__,
                                   formatter_class=argparse.RawDescriptionHelpFormatter)
    argp.add_argument("--campaign", choices=list(cfg.CAMPAIGNS), default=None,
                      help="Solo esta campana (por defecto, todas).")
    args = argp.parse_args()
    build_catalog([args.campaign] if args.campaign else None)
