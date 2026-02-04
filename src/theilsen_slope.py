from __future__ import annotations
from dataclasses import dataclass, field
from collections import deque
import numpy as np

@dataclass
class MinuteMeanBuilder:
    """
    Constrói PV_1min (média por minuto) a partir de amostras QCS (t_sec, pv).

    Regra:
    - minuto = floor(t_sec / 60)
    - enquanto estiver no mesmo minuto: acumula soma/contagem
    - ao mudar o minuto: fecha o minuto anterior e devolve (minute_idx, PV_1min)
    - se um minuto ficar sem amostra (caso raro), o minuto só “aparece” quando voltar a ter amostra;
      a lógica de HOLD para minutos ausentes pode ser feita externamente se desejado.
    """
    current_minute: int | None = None
    acc_sum: float = 0.0
    acc_n: int = 0
    last_mean: float | None = None

    def update(self, t_sec: float, pv: float):
        minute = int(t_sec // 60.0)
        out = []

        if self.current_minute is None:
            self.current_minute = minute

        # Virou o minuto -> fecha o minuto anterior
        if minute != self.current_minute:
            if self.acc_n > 0:
                mean = self.acc_sum / self.acc_n
            else:
                mean = self.last_mean if self.last_mean is not None else pv

            self.last_mean = mean
            out.append((self.current_minute, mean))

            # reseta para o novo minuto
            self.current_minute = minute
            self.acc_sum = 0.0
            self.acc_n = 0

        # acumula amostra do minuto atual
        self.acc_sum += pv
        self.acc_n += 1
        return out

    def flush(self):
        """Fecha o minuto atual (se existir)."""
        if self.current_minute is None:
            return []
        if self.acc_n > 0:
            mean = self.acc_sum / self.acc_n
        else:
            mean = self.last_mean if self.last_mean is not None else 0.0
        self.last_mean = mean
        out = [(self.current_minute, mean)]
        self.acc_sum = 0.0
        self.acc_n = 0
        return out


@dataclass
class TheilSenSlopeEstimator:
    """
    Estima slope (%/min) via Theil–Sen usando os últimos N endpoints:
    - endpoints = N últimas médias de minuto (PV_1min)
    - tempo é o índice do minuto (regular), então dt=1 min entre pontos consecutivos
    """
    window_n: int = 11
    clip_abs: float = 0.20  # %/min
    mins: deque = field(default_factory=lambda: deque(maxlen=11))  # (minute_idx, pv_1min)

    def set_window(self, n: int):
        n = int(max(3, n))
        if n != self.window_n:
            self.window_n = n
            old = list(self.mins)
            self.mins = deque(old[-n:], maxlen=n)

    def set_clip(self, clip_abs: float):
        self.clip_abs = float(max(0.0, clip_abs))

    def add_minute_mean(self, minute_idx: int, pv_1min: float) -> float:
        self.mins.append((int(minute_idx), float(pv_1min)))
        return self.slope()

    def slope(self) -> float:
        pts = list(self.mins)
        if len(pts) < 2:
            return 0.0

        t = np.array([p[0] for p in pts], dtype=float)
        y = np.array([p[1] for p in pts], dtype=float)

        slopes = []
        for i in range(len(pts) - 1):
            dt = t[i+1:] - t[i]
            dy = y[i+1:] - y[i]
            valid = dt != 0
            if np.any(valid):
                slopes.extend(list((dy[valid] / dt[valid])))

        s = float(np.median(np.array(slopes, dtype=float))) if slopes else 0.0
        if self.clip_abs > 0:
            s = float(np.clip(s, -self.clip_abs, +self.clip_abs))
        return s
