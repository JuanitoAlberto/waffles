"""Lector de los ficheros binarios RAW de CoMPASS (CAEN) para las XA del IR02.

Formato validado contra los datos de Test_XA_11-06-2026 (digitizador DT2730,
firmware DPP-PSD, waveforms activadas). Cada fichero .BIN corresponde a UN canal
y empieza con una cabecera de 2 bytes (mascara de campos presentes), seguida de
un evento tras otro con la estructura:

    board        uint16   (2 B)
    channel      uint16   (2 B)
    timestamp    uint64   (8 B)   en picosegundos
    energy       uint16   (2 B)   gate largo (DPP-PSD)
    energy_short uint16   (2 B)   gate corto (DPP-PSD)
    flags        uint32   (4 B)
    code         uint8    (1 B)   tipo de probe de la waveform
    n_samples    uint32   (4 B)   numero de muestras (constante por fichero)
    samples      int16 x n_samples

El nombre del fichero es DataR_CH<ch>@<board>_<serial>_<run_id>.BIN.

Funciones principales:
    parse_compass_bin(path)        -> CompassData (arrays numpy, lazy en waveforms)
    find_channel_files(run_dir)    -> {canal: Path}
    compass_run_to_wfset(run_dir)  -> WaveformSet de waffles (ambos canales)
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np

import xa_config as cfg

# Cabecera global del fichero (2 bytes) + parte fija del evento (sin samples).
FILE_HEADER_BYTES = 2
_FIXED_FIELDS_BYTES = 2 + 2 + 8 + 2 + 2 + 4 + 1 + 4  # = 25 B

# Regex del nombre de fichero: DataR_CH0@DT2730_66353_11Junio_run_1.BIN
_FILENAME_RE = re.compile(
    r"DataR_CH(?P<ch>\d+)@(?P<board>[^_]+)_(?P<serial>\d+)_(?P<run_id>.+)\.BIN",
    re.IGNORECASE,
)


@dataclass
class CompassData:
    """Contenido de un .BIN de un canal."""

    channel: int
    board: str
    serial: int
    run_id: str
    timestamps_ps: np.ndarray  # uint64, picosegundos
    energy: np.ndarray         # uint16, gate largo
    energy_short: np.ndarray   # uint16, gate corto
    flags: np.ndarray          # uint32
    waveforms: np.ndarray      # int16, shape (n_events, n_samples)

    @property
    def n_events(self) -> int:
        return len(self.timestamps_ps)

    @property
    def n_samples(self) -> int:
        return self.waveforms.shape[1] if self.waveforms.size else 0


def _event_dtype(n_samples: int) -> np.dtype:
    """dtype estructurado de un evento con n_samples muestras (little-endian)."""
    return np.dtype([
        ("board", "<u2"),
        ("channel", "<u2"),
        ("timestamp", "<u8"),
        ("energy", "<u2"),
        ("energy_short", "<u2"),
        ("flags", "<u4"),
        ("code", "u1"),
        ("n_samples", "<u4"),
        ("samples", "<i2", (n_samples,)),
    ])


def _peek_n_samples(path: Path) -> int:
    """Lee el primer evento para averiguar el numero de muestras (record length)."""
    with open(path, "rb") as fh:
        fh.seek(FILE_HEADER_BYTES + _FIXED_FIELDS_BYTES - 4)
        n_samples = int(np.frombuffer(fh.read(4), dtype="<u4")[0])
    if n_samples <= 0 or n_samples > 1_000_000:
        raise ValueError(f"n_samples sospechoso ({n_samples}) en {path}")
    return n_samples


def parse_compass_bin(
    path: str | Path,
    max_events: int | None = None,
    load_waveforms: bool = True,
) -> CompassData:
    """Parsea un .BIN de CoMPASS (un canal) y devuelve un CompassData.

    El numero de muestras se asume constante dentro del fichero (asi lo escribe
    CoMPASS con record length fijo); se valida que el tamano del fichero sea
    consistente.
    """
    path = Path(path)
    m = _FILENAME_RE.search(path.name)
    if m is None:
        raise ValueError(f"Nombre de fichero CoMPASS no reconocido: {path.name}")

    n_samples = _peek_n_samples(path)
    ev_dtype = _event_dtype(n_samples)

    body_bytes = path.stat().st_size - FILE_HEADER_BYTES
    if body_bytes % ev_dtype.itemsize != 0:
        raise ValueError(
            f"{path.name}: el cuerpo ({body_bytes} B) no es multiplo del tamano "
            f"de evento ({ev_dtype.itemsize} B); n_samples={n_samples} incorrecto?"
        )
    n_total = body_bytes // ev_dtype.itemsize
    count = n_total if max_events is None else min(max_events, n_total)

    events = np.fromfile(path, dtype=ev_dtype, count=count, offset=FILE_HEADER_BYTES)

    waveforms = (
        np.ascontiguousarray(events["samples"])
        if load_waveforms
        else np.empty((0, n_samples), dtype="<i2")
    )

    return CompassData(
        channel=int(m.group("ch")),
        board=m.group("board"),
        serial=int(m.group("serial")),
        run_id=m.group("run_id"),
        timestamps_ps=np.ascontiguousarray(events["timestamp"]),
        energy=np.ascontiguousarray(events["energy"]),
        energy_short=np.ascontiguousarray(events["energy_short"]),
        flags=np.ascontiguousarray(events["flags"]),
        waveforms=waveforms,
    )


def find_channel_files(run_dir: str | Path) -> dict[int, Path]:
    """Devuelve {canal: ruta_al_BIN} para un directorio de run de CoMPASS."""
    raw = Path(run_dir) / cfg.RAW_SUBDIR
    if not raw.is_dir():
        raise FileNotFoundError(f"No existe {raw}")
    out: dict[int, Path] = {}
    for p in sorted(raw.glob("DataR_CH*@*.BIN")):
        m = _FILENAME_RE.search(p.name)
        if m is not None:
            out[int(m.group("ch"))] = p
    if not out:
        raise FileNotFoundError(f"No hay ficheros DataR_*.BIN en {raw}")
    return out


def compass_run_to_wfset(
    run_dir: str | Path,
    run_number: int,
    max_events: int | None = None,
    channels: list[int] | None = None,
):
    """Convierte un run de CoMPASS (ambos canales) en un WaveformSet de waffles.

    Cada evento de CoMPASS se mapea a un Waveform con:
      - timestamp: timestamp de CoMPASS (ps).
      - time_step_ns: cfg.SAMPLE_NS.
      - endpoint: cfg.ENDPOINT (ficticio, para el modelo de waffles).
      - channel: canal del digitizador (0/1).
      - record_number: indice del evento dentro del canal.
    """
    # Import diferido: waffles solo es necesario para construir el WaveformSet,
    # no para el parseo crudo (que usan los analisis basados en numpy).
    from waffles.data_classes.Waveform import Waveform
    from waffles.data_classes.WaveformSet import WaveformSet

    channels = channels if channels is not None else cfg.CHANNELS
    files = find_channel_files(run_dir)

    waveforms = []
    for ch in channels:
        if ch not in files:
            continue
        data = parse_compass_bin(files[ch], max_events=max_events)
        for i in range(data.n_events):
            waveforms.append(
                Waveform(
                    timestamp=int(data.timestamps_ps[i]),
                    time_step_ns=cfg.SAMPLE_NS,
                    daq_window_timestamp=int(data.timestamps_ps[i]),
                    adcs=data.waveforms[i].astype(np.float64),
                    run_number=run_number,
                    record_number=i,
                    endpoint=cfg.ENDPOINT,
                    channel=ch,
                )
            )

    if not waveforms:
        raise RuntimeError(f"No se han construido waveforms para {run_dir}")
    return WaveformSet(*waveforms)
