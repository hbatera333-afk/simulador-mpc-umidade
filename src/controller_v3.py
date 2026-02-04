from __future__ import annotations
from dataclasses import dataclass, field
from collections import deque
import math

from .utils import clamp, sign_nz, safe_std, EPS

@dataclass
class FastWindow:
    win_time_min: float = 0.0
    n: int = 0
    mean_pv: float = 0.0
    m2_pv: float = 0.0
    mean_mv: float = 0.0
    m2_mv: float = 0.0
    mean_err: float = 0.0
    m2_err: float = 0.0
    prev_err_sign: int = 0
    zero_cross: int = 0

    def reset(self):
        self.win_time_min = 0.0
        self.n = 0
        self.mean_pv = self.m2_pv = 0.0
        self.mean_mv = self.m2_mv = 0.0
        self.mean_err = self.m2_err = 0.0
        self.prev_err_sign = 0
        self.zero_cross = 0

    def update(self, pv: float, err: float, mv_fb: float):
        # Igual ao VBA: incrementar n 1x e atualizar mean/m2 de cada variável com o mesmo n.
        if self.n <= 0:
            self.n = 1
            self.mean_pv, self.m2_pv = pv, 0.0
            self.mean_err, self.m2_err = err, 0.0
            self.mean_mv, self.m2_mv = mv_fb, 0.0
        else:
            self.n += 1

            d = pv - self.mean_pv
            self.mean_pv += d / self.n
            self.m2_pv += d * (pv - self.mean_pv)

            d = err - self.mean_err
            self.mean_err += d / self.n
            self.m2_err += d * (err - self.mean_err)

            d = mv_fb - self.mean_mv
            self.mean_mv += d / self.n
            self.m2_mv += d * (mv_fb - self.mean_mv)

        s = sign_nz(err)
        if s != 0:
            if self.prev_err_sign != 0 and s != self.prev_err_sign:
                self.zero_cross += 1
            self.prev_err_sign = s

    def stds(self):
        return (safe_std(self.n, self.m2_pv),
                safe_std(self.n, self.m2_mv),
                safe_std(self.n, self.m2_err))

    def zc_ratio(self):
        if self.n > 1:
            return self.zero_cross / (self.n - 1)
        return 0.0


@dataclass
class SupervisorBuffer:
    # Rolling por TEMPO (min)
    cap: int = 5000
    w: deque = field(default_factory=deque)
    pv: deque = field(default_factory=deque)
    mv: deque = field(default_factory=deque)
    prod: deque = field(default_factory=deque)
    broke: deque = field(default_factory=deque)
    ph: deque = field(default_factory=deque)
    err: deque = field(default_factory=deque)

    sumW: float = 0.0
    sumPV: float = 0.0
    sumPV2: float = 0.0
    sumMV: float = 0.0
    sumMV2: float = 0.0

    sumProd: float = 0.0
    sumProd2: float = 0.0
    sumMVProd: float = 0.0

    sumBroke: float = 0.0
    sumBroke2: float = 0.0
    sumMVBroke: float = 0.0

    sumPH: float = 0.0
    sumPH2: float = 0.0
    sumMVPH: float = 0.0

    timer_min: float = 0.0

    def reset(self):
        self.w.clear(); self.pv.clear(); self.mv.clear()
        self.prod.clear(); self.broke.clear(); self.ph.clear(); self.err.clear()
        self.sumW = 0.0
        self.sumPV = self.sumPV2 = 0.0
        self.sumMV = self.sumMV2 = 0.0
        self.sumProd = self.sumProd2 = self.sumMVProd = 0.0
        self.sumBroke = self.sumBroke2 = self.sumMVBroke = 0.0
        self.sumPH = self.sumPH2 = self.sumMVPH = 0.0
        self.timer_min = 0.0

    def _remove_oldest(self):
        if not self.w:
            return
        w = self.w.popleft()
        pv = self.pv.popleft()
        mv = self.mv.popleft()
        xp = self.prod.popleft()
        xb = self.broke.popleft()
        xh = self.ph.popleft()
        e = self.err.popleft()

        self.sumW -= w
        self.sumPV -= w * pv
        self.sumPV2 -= w * pv * pv
        self.sumMV -= w * mv
        self.sumMV2 -= w * mv * mv

        self.sumProd -= w * xp
        self.sumProd2 -= w * xp * xp
        self.sumMVProd -= w * mv * xp

        self.sumBroke -= w * xb
        self.sumBroke2 -= w * xb * xb
        self.sumMVBroke -= w * mv * xb

        self.sumPH -= w * xh
        self.sumPH2 -= w * xh * xh
        self.sumMVPH -= w * mv * xh

    def add(self, pv: float, mv_fb: float, prod_dev: float, broke_dev: float, ph_dev: float, err: float, dt_min: float, win_min: float):
        if dt_min <= 0:
            return
        win_min = max(10.0, float(win_min))

        # cap
        while len(self.w) >= self.cap:
            self._remove_oldest()

        self.w.append(dt_min)
        self.pv.append(pv)
        self.mv.append(mv_fb)
        self.prod.append(prod_dev)
        self.broke.append(broke_dev)
        self.ph.append(ph_dev)
        self.err.append(err)

        self.sumW += dt_min

        self.sumPV += dt_min * pv
        self.sumPV2 += dt_min * pv * pv

        self.sumMV += dt_min * mv_fb
        self.sumMV2 += dt_min * mv_fb * mv_fb

        self.sumProd += dt_min * prod_dev
        self.sumProd2 += dt_min * prod_dev * prod_dev
        self.sumMVProd += dt_min * mv_fb * prod_dev

        self.sumBroke += dt_min * broke_dev
        self.sumBroke2 += dt_min * broke_dev * broke_dev
        self.sumMVBroke += dt_min * mv_fb * broke_dev

        self.sumPH += dt_min * ph_dev
        self.sumPH2 += dt_min * ph_dev * ph_dev
        self.sumMVPH += dt_min * mv_fb * ph_dev

        # corta por TEMPO
        while self.sumW > win_min and len(self.w) > 1:
            self._remove_oldest()

    def std_pv(self) -> float:
        if self.sumW <= 0:
            return 0.0
        mean = self.sumPV / self.sumW
        var = self.sumPV2 / self.sumW - mean * mean
        return math.sqrt(max(0.0, var))

    def std_mv(self) -> float:
        if self.sumW <= 0:
            return 0.0
        mean = self.sumMV / self.sumW
        var = self.sumMV2 / self.sumW - mean * mean
        return math.sqrt(max(0.0, var))

    def effort(self) -> float:
        return self.std_mv() / (self.std_pv() + 1e-9)

    def zc_ratio(self) -> float:
        if len(self.err) < 2:
            return 0.0
        prev = 0
        zc = 0
        for e in self.err:
            s = sign_nz(e)
            if s != 0:
                if prev != 0 and s != prev:
                    zc += 1
                prev = s
        return zc / (len(self.err) - 1)


@dataclass
class DryerHumidityCtrlV3:
    # estados principais
    init_done: bool = False
    int_term: float = 0.0
    prev_enable: bool = False

    kp_eff: float = 0.02
    ti_eff: float = 10.0

    # PV acceptance
    pv_last_acc: float = 0.0
    slope_last_acc: float = 0.0
    time_since_acc_min: float = 0.0
    dt_real_min: float = 0.0
    has_accepted_once: bool = False

    # fast window
    fast: FastWindow = field(default_factory=FastWindow)

    # safety/revert
    prev_std_pv: float = 0.0
    kp_stable: float = 0.02
    ti_stable: float = 10.0
    worsen_streak: int = 0
    hold_windows: int = 0

    # supervisor buffer
    sup: SupervisorBuffer = field(default_factory=SupervisorBuffer)

    # FF effective
    kff_prod_eff: float = 0.0
    kff_broke_eff: float = 0.0
    kff_ph_eff: float = 0.0

    kff_prod_base: float = 0.0
    kff_broke_base: float = 0.0
    kff_ph_base: float = 0.0

    # KPIs (telemetria)
    kpi_std_pv: float = 0.0
    kpi_zc_ratio: float = 0.0
    kpi_effort: float = 0.0

    # logs
    logs: deque = field(default_factory=lambda: deque(maxlen=500))

    def reset(self, mv_track: float, pv_raw: float, slope_raw: float, params: dict):
        self.int_term = mv_track
        self.prev_enable = bool(params.get("en_auto", False))

        self.kp_eff = float(params.get("Kp_base", params.get("kp_base", 0.02)))
        self.ti_eff = float(params.get("Ti_base_min", params.get("ti_base", 10.0)))

        self.pv_last_acc = pv_raw
        self.slope_last_acc = slope_raw
        self.time_since_acc_min = 0.0
        self.dt_real_min = float(params.get("Ts_exec_min", 0.1667))
        self.has_accepted_once = False

        self.fast.reset()

        self.prev_std_pv = 0.0
        self.worsen_streak = 0
        self.hold_windows = 0
        self.kp_stable = self.kp_eff
        self.ti_stable = self.ti_eff

        self.kff_prod_eff = float(params.get("Kff_Prod_base", 0.0))
        self.kff_broke_eff = float(params.get("Kff_Broke_base", 0.0))
        self.kff_ph_eff = float(params.get("Kff_pH_base", 0.0))

        self.kff_prod_base = self.kff_prod_eff
        self.kff_broke_base = self.kff_broke_eff
        self.kff_ph_base = self.kff_ph_eff

        self.sup.reset()

        self.kpi_std_pv = 0.0
        self.kpi_zc_ratio = 0.0
        self.kpi_effort = 0.0

        self.logs.clear()
        self.init_done = True
        self._log("RESET", f"Reset aplicado. Kp={self.kp_eff:.5f} Ti={self.ti_eff:.3f}")

    def _log(self, tag: str, msg: str):
        self.logs.appendleft((tag, msg))

    def step(self, inputs: dict, params: dict) -> dict:
        """Executa 1 ciclo do bloco V3. Retorna outputs e flags úteis para trend/log."""
        # Entradas
        sp = float(inputs.get("sp", 0.0))
        pv_raw = float(inputs.get("pv_raw", 0.0))
        slope_raw = float(inputs.get("slope_raw", 0.0))
        mv_track = float(inputs.get("mv_track", 0.0))

        prod = float(inputs.get("prod", 0.0))
        broke = float(inputs.get("broke", 0.0))
        ph = float(inputs.get("ph", 7.0))

        en_auto = bool(inputs.get("en_auto", False))
        en_ff = bool(inputs.get("en_ff", True))
        do_reset = bool(inputs.get("do_reset", False))

        # Parâmetros (IPs)
        Ts_exec_min = float(params.get("Ts_exec_min", 0.1667))
        Ts_exec_min = max(1e-6, Ts_exec_min)

        LAMBDA_min = max(0.0, float(params.get("LAMBDA_min", 3.0)))

        kp_base = float(params.get("Kp_base", 0.02))
        ti_base = float(params.get("Ti_base_min", 10.0))
        if ti_base <= 0: ti_base = 10.0

        mvMin = float(params.get("MV_Min_bar", 0.0))
        mvMax = float(params.get("MV_Max_bar", 10.0))
        if mvMax < mvMin: mvMax = mvMin

        kffProd_base = float(params.get("Kff_Prod_base", 0.0))
        kffBroke_base = float(params.get("Kff_Broke_base", 0.0))
        kffPH_base = float(params.get("Kff_pH_base", 0.0))

        gammaSlope = float(params.get("GAMMA_SLOPE_RAW", 0.5))
        gammaSlope = clamp(gammaSlope, 0.0, 1.0)

        errScale = float(params.get("ERR_STEP", 0.1))
        if errScale <= 0: errScale = 0.1

        kpMin = float(params.get("Kp_min", 0.002))
        kpMax = float(params.get("Kp_max", 0.2))
        if kpMax < kpMin: kpMax = kpMin

        # PV acceptance
        pvEps = float(params.get("PV_EPS_ACCEPT", 0.002))
        if pvEps < 0: pvEps = 0.0
        pvMaxHold_sec = float(params.get("PV_MAX_HOLD_sec", 45.0))
        if pvMaxHold_sec <= 0: pvMaxHold_sec = 45.0
        satMargin = float(params.get("AT_SAT_MARGIN", 0.05))
        if satMargin < 0: satMargin = 0.0

        # AutoTune FAST
        atFastEn = bool(params.get("AUTOTUNE_FAST_EN", False))
        targetStdPV = float(params.get("TARGET_STD_PV", 0.10))
        if targetStdPV <= 0: targetStdPV = 0.10

        dt_min = float(params.get("DEADTIME_min", 7.0))
        if dt_min <= 0: dt_min = 7.0

        winMult = float(params.get("WIN_MULT_DT", 3.0))
        if winMult < 1: winMult = 1.0

        maxStepFrac = float(params.get("MAX_STEP_FRAC", 0.10))
        if maxStepFrac < 0: maxStepFrac = 0.0

        # referências FF
        prodRef = float(params.get("PROD_REF", prod))
        brokeRef = float(params.get("BROKE_REF", broke))
        phRef = float(params.get("PH_REF", ph))

        # Supervisor + AutoFF
        kpiWin_min = float(params.get("KPI_WIN_min", 480.0))
        if kpiWin_min < 10: kpiWin_min = 10.0
        kpiUpdate_min = float(params.get("KPI_UPDATE_min", 60.0))
        if kpiUpdate_min < 1: kpiUpdate_min = 1.0

        kpSupEn = bool(params.get("KP_SUPERVISOR_EN", False))
        kpSupMaxStep = float(params.get("KP_SUP_MAXSTEP_FRAC", 0.03))
        if kpSupMaxStep < 0: kpSupMaxStep = 0.0

        ffAutoEn = bool(params.get("FF_AUTOTUNE_EN", False))
        ffStepProd = float(params.get("FF_MAXSTEP_PROD", 0.05))
        ffStepBroke = float(params.get("FF_MAXSTEP_BROKE", 0.05))
        ffStepPH = float(params.get("FF_MAXSTEP_PH", 0.05))
        ffRangeFrac = float(params.get("FF_RANGE_FRAC", 1.0))
        if ffStepProd < 0: ffStepProd = 0.0
        if ffStepBroke < 0: ffStepBroke = 0.0
        if ffStepPH < 0: ffStepPH = 0.0
        if ffRangeFrac < 0: ffRangeFrac = 0.0

        # Reset
        if (not self.init_done) or do_reset:
            self.reset(mv_track, pv_raw, slope_raw, {
                "en_auto": en_auto,
                "Ts_exec_min": Ts_exec_min,
                "kp_base": kp_base,
                "Kp_base": kp_base,
                "ti_base": ti_base,
                "Ti_base_min": ti_base,
                "Kff_Prod_base": kffProd_base,
                "Kff_Broke_base": kffBroke_base,
                "Kff_pH_base": kffPH_base,
            })

        if self.kp_eff <= 0: self.kp_eff = kp_base
        if self.ti_eff <= 0: self.ti_eff = ti_base

        # 4) PV Acceptance
        self.time_since_acc_min += Ts_exec_min

        pvAccepted = False
        if not self.has_accepted_once:
            pvAccepted = True
            self.has_accepted_once = True
        else:
            if abs(pv_raw - self.pv_last_acc) >= pvEps:
                pvAccepted = True
            elif self.time_since_acc_min >= (pvMaxHold_sec / 60.0):
                pvAccepted = True

        if pvAccepted:
            self.dt_real_min = self.time_since_acc_min
            if self.dt_real_min <= 0:
                self.dt_real_min = Ts_exec_min
            self.pv_last_acc = pv_raw
            self.slope_last_acc = slope_raw
            self.time_since_acc_min = 0.0
        else:
            self.dt_real_min = 0.0

        pv = self.pv_last_acc
        slope = self.slope_last_acc

        # 5) erros
        e_now = sp - pv
        pv_future = pv + slope * LAMBDA_min
        e_pred = sp - pv_future

        e_eff = (1.0 - gammaSlope) * e_now + gammaSlope * e_pred
        normErr = e_eff / errScale

        # 6) Feedforward (com ganhos efetivos quando AutoFF ligado)
        if not ffAutoEn:
            self.kff_prod_eff = kffProd_base
            self.kff_broke_eff = kffBroke_base
            self.kff_ph_eff = kffPH_base
            kffProd = kffProd_base
            kffBroke = kffBroke_base
            kffPH = kffPH_base
            self.kff_prod_base = kffProd_base
            self.kff_broke_base = kffBroke_base
            self.kff_ph_base = kffPH_base
        else:
            kffProd = self.kff_prod_eff
            kffBroke = self.kff_broke_eff
            kffPH = self.kff_ph_eff

        mvFF = 0.0
        if en_ff:
            mvFF += kffProd * (prod - prodRef)
            mvFF += kffBroke * (broke - brokeRef)
            mvFF += kffPH * (ph - phRef)

        # 7) PI preditivo
        pTerm = -self.kp_eff * normErr
        if (self.dt_real_min > 0) and (self.ti_eff > 0):
            Ki_step = self.kp_eff * self.dt_real_min / self.ti_eff
        else:
            Ki_step = 0.0

        mvFB = 0.0
        if not en_auto:
            mvCmd = mv_track
            self.int_term = mv_track - pTerm - mvFF
        else:
            if (not self.prev_enable) and en_auto:
                self.int_term = mv_track - pTerm - mvFF
            if Ki_step != 0.0:
                self.int_term = self.int_term - Ki_step * normErr
            mvFB = pTerm + self.int_term
            mvCmd = mvFF + mvFB

        self.prev_enable = en_auto

        # 8) saturação + antiwindup
        sat = 0
        if mvCmd < mvMin:
            mvCmd = mvMin
            sat = -1
            if en_auto:
                self.int_term = mvCmd - pTerm - mvFF
        elif mvCmd > mvMax:
            mvCmd = mvMax
            sat = +1
            if en_auto:
                self.int_term = mvCmd - pTerm - mvFF

        nearSat = (mvCmd <= mvMin + satMargin) or (mvCmd >= mvMax - satMargin)

        # 9) AutoTune FAST (anti-oscilacao)
        fast_event = None
        if atFastEn and en_auto:
            if pvAccepted and (not nearSat):
                self.fast.win_time_min += self.dt_real_min
                self.fast.update(pv, e_now, mvFB)

                T_win = dt_min * winMult
                if T_win < 5: T_win = 5.0

                if self.fast.win_time_min >= T_win:
                    fast_event = self._evaluate_fast_autotune(
                        targetStdPV=targetStdPV,
                        maxStepFrac=maxStepFrac,
                        kpMin=kpMin,
                        kpMax=kpMax,
                        ti_base=ti_base
                    )
                    self.fast.reset()

        # 10) Supervisor + AutoFF
        sup_event = None
        ff_event = None
        if en_auto and pvAccepted and (not nearSat):
            self.sup.add(
                pv=pv,
                mv_fb=mvFB,
                prod_dev=(prod - prodRef),
                broke_dev=(broke - brokeRef),
                ph_dev=(ph - phRef),
                err=e_now,
                dt_min=self.dt_real_min,
                win_min=kpiWin_min
            )
            self.sup.timer_min += self.dt_real_min

            if self.sup.timer_min >= kpiUpdate_min:
                self.kpi_std_pv = self.sup.std_pv()
                self.kpi_zc_ratio = self.sup.zc_ratio()
                self.kpi_effort = self.sup.effort()

                if ffAutoEn:
                    ff_event = self._apply_autoff(
                        targetStdPV=targetStdPV,
                        stepProd=ffStepProd,
                        stepBroke=ffStepBroke,
                        stepPH=ffStepPH,
                        rangeFrac=ffRangeFrac
                    )

                if kpSupEn:
                    sup_event = self._apply_kp_supervisor(
                        targetStdPV=targetStdPV,
                        kpMin=kpMin,
                        kpMax=kpMax,
                        kpSupMaxStep=kpSupMaxStep
                    )

                self.sup.timer_min = 0.0

        out = {
            "MV_CMD": mvCmd,
            "MV_FF": mvFF,
            "MV_FB": mvFB,
            "ERR_NOW": e_now,
            "ERR_PRED": e_pred,
            "PV_PRED": pv_future,
            "KP_EFF": self.kp_eff,
            "TI_EFF": self.ti_eff,
            "KFF_PROD_EFF": self.kff_prod_eff,
            "KFF_BROKE_EFF": self.kff_broke_eff,
            "KFF_PH_EFF": self.kff_ph_eff,
            "KPI_STD_PV": self.kpi_std_pv,
            "KPI_ZC_RATIO": self.kpi_zc_ratio,
            "KPI_EFFORT": self.kpi_effort,
            "PV_ACCEPTED": pvAccepted,
            "DT_REAL_MIN": self.dt_real_min,
            "NEAR_SAT": nearSat,
            "SAT": sat,
            "FAST_EVENT": fast_event,
            "SUP_EVENT": sup_event,
            "FF_EVENT": ff_event,
            "pTerm": pTerm,
            "iTerm": self.int_term,
            "e_eff": e_eff,
            "normErr": normErr,
            "gammaSlope": gammaSlope,
            "pv_used": pv,
            "slope_used": slope,
        }
        return out


    def _evaluate_fast_autotune(self, targetStdPV: float, maxStepFrac: float, kpMin: float, kpMax: float, ti_base: float):
        if self.fast.n < 10:
            return None

        stdPV, stdMV, stdErr = self.fast.stds()
        zcRatio = self.fast.zc_ratio()
        effort = stdMV / (stdPV + 1e-9)

        isOsc = (zcRatio > 0.22) or (effort > 2.7)
        isTooAgg = (zcRatio > 0.25) or (effort > 3.0)

        hi = targetStdPV * 1.05
        lo = targetStdPV * 0.95

        stepKp = 0.0
        stepTi = 0.0

        tiMinLim = max(1.0, ti_base * 0.25)
        tiMaxLim = min(120.0, ti_base * 4.0)

        if self.hold_windows > 0:
            self.hold_windows -= 1
            self.prev_std_pv = stdPV
            return {"type": "FAST_HOLD", "stdPV": stdPV, "zc": zcRatio, "effort": effort}

        if stdPV > hi:
            if isOsc:
                stepKp = -maxStepFrac
                stepTi = +0.5 * maxStepFrac
        elif stdPV < lo:
            if isTooAgg:
                stepKp = -0.5 * maxStepFrac
                stepTi = +0.5 * maxStepFrac

        kpBefore = self.kp_eff
        tiBefore = self.ti_eff

        if stepKp != 0.0:
            self.kp_eff = self.kp_eff * (1.0 + stepKp)
            self.kp_eff = clamp(self.kp_eff, kpMin, kpMax)

        if stepTi != 0.0:
            self.ti_eff = self.ti_eff * (1.0 + stepTi)
            self.ti_eff = clamp(self.ti_eff, tiMinLim, tiMaxLim)

        worsened = False
        if self.prev_std_pv > 0 and stdPV > (self.prev_std_pv * 1.02):
            worsened = True

        if (self.prev_std_pv <= 0) or (stdPV <= self.prev_std_pv) or (stdPV <= hi):
            self.kp_stable = self.kp_eff
            self.ti_stable = self.ti_eff

        if (stepKp != 0.0) or (stepTi != 0.0):
            if worsened:
                self.worsen_streak += 1
            else:
                self.worsen_streak = 0

            if self.worsen_streak >= 2:
                self.kp_eff = self.kp_stable
                self.ti_eff = self.ti_stable
                self.hold_windows = 1
                self.worsen_streak = 0
                self._log("FAST_REVERT", f"Reverteu p/ estável. Kp={self.kp_eff:.5f} Ti={self.ti_eff:.3f}")

        else:
            self.worsen_streak = 0

        self.prev_std_pv = stdPV

        if (stepKp != 0.0) or (stepTi != 0.0):
            self._log("FAST_TUNE",
                      f"stdPV={stdPV:.3f} zc={zcRatio:.3f} eff={effort:.2f} | "
                      f"Kp {kpBefore:.5f}->{self.kp_eff:.5f} (step {stepKp:+.3f}) | "
                      f"Ti {tiBefore:.3f}->{self.ti_eff:.3f} (step {stepTi:+.3f})")
            return {
                "type": "FAST_TUNE",
                "stdPV": stdPV, "zc": zcRatio, "effort": effort,
                "kpBefore": kpBefore, "kpAfter": self.kp_eff,
                "tiBefore": tiBefore, "tiAfter": self.ti_eff,
                "stepKp": stepKp, "stepTi": stepTi
            }
        return {"type": "FAST_NOACTION", "stdPV": stdPV, "zc": zcRatio, "effort": effort}


    def _apply_kp_supervisor(self, targetStdPV: float, kpMin: float, kpMax: float, kpSupMaxStep: float):
        stdPV = self.kpi_std_pv
        if stdPV <= targetStdPV:
            return None

        if (self.kpi_zc_ratio < 0.18) and (self.kpi_effort < 3.0):
            ratio = (stdPV / targetStdPV) - 1.0
            ratio = clamp(ratio, 0.0, 1.0)
            step = kpSupMaxStep * ratio
            if step > 0:
                kpBefore = self.kp_eff
                self.kp_eff = self.kp_eff * (1.0 + step)
                self.kp_eff = clamp(self.kp_eff, kpMin, kpMax)
                self._log("SUP_KP",
                          f"KPI stdPV={stdPV:.3f} alvo={targetStdPV:.3f} | "
                          f"Kp {kpBefore:.5f}->{self.kp_eff:.5f} (step {step:+.3f})")
                return {"type": "SUP_KP", "kpBefore": kpBefore, "kpAfter": self.kp_eff, "step": step}
        return {"type": "SUP_HOLD"}


    def _clamp_ff(self, k: float, kBase: float, rangeFrac: float) -> float:
        baseAbs = abs(kBase)
        if baseAbs < 0.001: baseAbs = 0.001
        kMin = kBase - rangeFrac * baseAbs
        kMax = kBase + rangeFrac * baseAbs
        return clamp(k, kMin, kMax)

    def _autoff_one(self, mu: float, meanMV: float, sumX: float, sumX2: float, sumMVX: float, maxStepFrac: float, kCurrent: float, kBase: float):
        if self.sup.sumW <= 0:
            return 0.0
        meanX = sumX / self.sup.sumW
        varX = sumX2 / self.sup.sumW - meanX * meanX
        varX = max(0.0, varX)
        if varX < 1e-9:
            return 0.0

        cov = (sumMVX / self.sup.sumW) - meanMV * meanX
        raw = mu * cov / (varX + 1e-9)

        refAbs = abs(kCurrent)
        if refAbs < abs(kBase) * 0.10:
            refAbs = abs(kBase) * 0.10
        if refAbs < 0.001:
            refAbs = 0.001

        maxAbs = maxStepFrac * refAbs
        raw = clamp(raw, -maxAbs, +maxAbs)
        return raw

    def _apply_autoff(self, targetStdPV: float, stepProd: float, stepBroke: float, stepPH: float, rangeFrac: float):
        if self.sup.sumW <= 0:
            return None

        # gate de segurança (igual ao VBA)
        if (self.kpi_zc_ratio > 0.22) or (self.kpi_effort > 3.0):
            return {"type": "FF_GATE"}

        mu = 0.25
        meanMV = self.sup.sumMV / self.sup.sumW

        changes = {}

        dK = self._autoff_one(mu, meanMV, self.sup.sumProd, self.sup.sumProd2, self.sup.sumMVProd, stepProd, self.kff_prod_eff, self.kff_prod_base)
        if dK != 0.0:
            before = self.kff_prod_eff
            self.kff_prod_eff = self._clamp_ff(self.kff_prod_eff + dK, self.kff_prod_base, rangeFrac)
            changes["prod"] = (before, self.kff_prod_eff, dK)

        dK = self._autoff_one(mu, meanMV, self.sup.sumBroke, self.sup.sumBroke2, self.sup.sumMVBroke, stepBroke, self.kff_broke_eff, self.kff_broke_base)
        if dK != 0.0:
            before = self.kff_broke_eff
            self.kff_broke_eff = self._clamp_ff(self.kff_broke_eff + dK, self.kff_broke_base, rangeFrac)
            changes["broke"] = (before, self.kff_broke_eff, dK)

        dK = self._autoff_one(mu, meanMV, self.sup.sumPH, self.sup.sumPH2, self.sup.sumMVPH, stepPH, self.kff_ph_eff, self.kff_ph_base)
        if dK != 0.0:
            before = self.kff_ph_eff
            self.kff_ph_eff = self._clamp_ff(self.kff_ph_eff + dK, self.kff_ph_base, rangeFrac)
            changes["ph"] = (before, self.kff_ph_eff, dK)

        if changes:
            parts = []
            for k, (b,a,dk) in changes.items():
                parts.append(f"{k}: {b:.4f}->{a:.4f} (dK {dk:+.4f})")
            self._log("AUTOFF", " | ".join(parts))
            return {"type": "AUTOFF", "changes": changes}
        return {"type": "AUTOFF_NOACTION"}
