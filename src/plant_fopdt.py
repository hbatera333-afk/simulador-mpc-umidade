from __future__ import annotations
from dataclasses import dataclass, field
from collections import deque
import math
import random

@dataclass
class FOPDTPlant:
    """
    Planta FOPDT simples para PV (umidade) em função de MV (vapor) e distúrbios.
    Convenção:
      - MV ↑  => PV ↓ (mais vapor seca)
      - Portanto, K_mv deve ser NEGATIVO (ex.: -0.08 %/bar)

    PV_true evolui para PV_ss com 1a ordem e atraso puro (dead time).
    """
    pv0: float = 10.0
    mv0: float = 3.8
    prod0: float = 170.0   # t/h (aprox 4080 ADT/d)
    broke0: float = 0.0
    ph0: float = 7.0

    K_mv: float = -0.08     # % PV por bar
    K_prod: float = +0.002  # % PV por t/h
    K_broke: float = +0.010 # % PV por t/h
    K_ph: float = +0.050    # % PV por pH

    tau_min: float = 10.0
    deadtime_min: float = 7.0

    noise_std: float = 0.02  # % (process noise)
    pv_true: float = field(default=10.0)

    # fila para dead time (armazenar entradas atrasadas)
    _u_queue: deque = field(default_factory=deque)

    def reset(self):
        self.pv_true = float(self.pv0)
        self._u_queue.clear()

    def _pv_ss(self, mv: float, prod: float, broke: float, ph: float) -> float:
        return (self.pv0
                + self.K_mv * (mv - self.mv0)
                + self.K_prod * (prod - self.prod0)
                + self.K_broke * (broke - self.broke0)
                + self.K_ph * (ph - self.ph0))

    def step(self, dt_min: float, mv: float, prod: float, broke: float, ph: float) -> float:
        dt_min = max(1e-6, float(dt_min))
        # empilha entrada
        self._u_queue.append((mv, prod, broke, ph))
        # calcula quantas amostras equivalem ao deadtime
        n_delay = int(round(self.deadtime_min / dt_min))
        if n_delay < 0:
            n_delay = 0

        # pega entrada atrasada
        if len(self._u_queue) > n_delay:
            mv_d, prod_d, broke_d, ph_d = self._u_queue[-(n_delay+1)]
        else:
            mv_d, prod_d, broke_d, ph_d = self._u_queue[0]

        pv_ss = self._pv_ss(mv_d, prod_d, broke_d, ph_d)

        # 1a ordem
        alpha = dt_min / max(1e-6, self.tau_min)
        if alpha > 1.0:
            alpha = 1.0
        self.pv_true = self.pv_true + alpha * (pv_ss - self.pv_true)

        # ruído de processo pequeno
        if self.noise_std > 0:
            self.pv_true += random.gauss(0.0, self.noise_std)

        return self.pv_true
