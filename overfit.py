"""Was that a result, or was it the best of many tries?

Two measures from Bailey and Lopez de Prado, implemented with numpy
and the standard library only.

DEFLATED SHARPE RATIO
---------------------
Run N strategies with no skill at all and the best of them still shows
a positive Sharpe. How positive depends on N and on how much the trials
differ from each other. The deflated Sharpe works out that bar and asks
whether the observed Sharpe clears it.

The bar is sqrt(V) * ((1-g)*Z(1-1/N) + g*Z(1-1/(N*e))), where V is the
variance of the trial Sharpes and g is Euler's constant. So a grid where
everything lands in the same place gets a low bar, and one that sprays
gets a high one.

Skew and kurtosis go into the probability. Returns from real strategies
are not normal, and ignoring that reads high.

PROBABILITY OF BACKTEST OVERFITTING (CSCV)
------------------------------------------
Split the sample into S pieces, take every way of using half for
training and half for testing, pick the training winner in each, and
see where it lands out of sample. The share of splits where it falls
below the median is the PBO.

A PBO near 0.5 says "pick whatever won in training" is a coin flip.
Near 0 says the selection is doing something.

BOTH ARE DIAGNOSTICS. They do not improve a strategy. They tell you
how to read the number it produced.

IMPORTANT: pass every trial you ran, losers included. Handing it only
the winners shrinks N, lowers the bar, and hides the exact bias you
are trying to measure.
"""
from __future__ import annotations

import itertools
import math

import numpy as np


def _normal_ppf(p: float) -> float:
    """Inverse normal CDF.

    Acklam's approximation, so scipy stays out of the dependency
    list. Absolute error around 1e-9, which is far more than this
    needs.
    """
    if not 0.0 < p < 1.0:
        return float("nan")
    a = (-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
         1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00)
    b = (-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
         6.680131188771972e+01, -1.328068155288572e+01)
    c = (-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
         -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00)
    d = (7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
         3.754408661907416e+00)
    pl, ph = 0.02425, 1 - 0.02425
    if p < pl:
        q = math.sqrt(-2 * math.log(p))
        return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q
                + c[5]) / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)
    if p > ph:
        q = math.sqrt(-2 * math.log(1 - p))
        return -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q
                 + c[5]) / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)
    q, r = p - 0.5, (p - 0.5) ** 2
    return (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r
            + a[5]) * q / (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r
                            + b[4]) * r + 1)


def _normal_cdf(z: float) -> float:
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0))) if np.isfinite(z) else float("nan")


def sharpe(seri: np.ndarray, yil_bar: float = 365.0) -> float:
    """Annualised Sharpe. Risk-free rate is taken as zero."""
    v = np.asarray(seri, dtype=float)
    v = v[np.isfinite(v)]
    if len(v) < 3:
        return float("nan")
    sd = v.std(ddof=1)
    # GORELI esik. Sabit bir seride std tam sifir cikmiyor: ortalama
    # kayan noktada birikince ~1e-19 kalinti kaliyor ve `sd > 0`
    # kontrolu bunu geciriyor, Sharpe astronomik ama sonlu oluyor.
    if not sd > 1e-12 * max(1.0, float(np.abs(v).mean())):
        return float("nan")
    return float(v.mean() / sd * math.sqrt(yil_bar))


def olasilikli_sharpe(seri: np.ndarray, esik: float = 0.0,
                      yil_bar: float = 365.0) -> float:
    """Gözlenen Sharpe'ın `esik`i gerçekten aştığı olasılığı.

    Çarpıklık ve basıklık düzeltmeli. Normal olmayan bir seride bu
    düzeltme olmadan olasılık yukarı kaçar -- negatif çarpıklık ve
    kalın kuyruk Sharpe'ın standart hatasını büyütür.
    """
    v = np.asarray(seri, dtype=float)
    v = v[np.isfinite(v)]
    n = len(v)
    if n < 3:
        return float("nan")
    sd = v.std(ddof=1)
    if not sd > 1e-12 * max(1.0, float(np.abs(v).mean())):
        return float("nan")
    sr = v.mean() / sd                      # bar bazinda
    sr_esik = esik / math.sqrt(yil_bar)
    z = (v - v.mean()) / sd
    carpiklik = float((z ** 3).mean())
    basiklik = float((z ** 4).mean())
    payda = math.sqrt(max(1e-12,
                          1 - carpiklik * sr + (basiklik - 1) / 4 * sr ** 2))
    return _normal_cdf((sr - sr_esik) * math.sqrt(n - 1) / payda)


def sisirilmis_sharpe(seriler: "dict[str, np.ndarray] | list",
                      secilen: "str | int | None" = None,
                      yil_bar: float = 365.0) -> dict:
    """Denenen bütün kurgular verilince en iyisinin gerçekliği.

    `seriler` denenen HER kurgunun getiri serisi. Eksik bırakmak
    sonucu iyimser yapar: N küçüldükçe eşik düşer, yani yalnızca
    kazananları vermek tam olarak ölçmeye çalıştığımız yanlılığı
    gizler.
    """
    if isinstance(seriler, dict):
        adlar, diziler = list(seriler.keys()), list(seriler.values())
    else:
        adlar, diziler = list(range(len(seriler))), list(seriler)
    sr = np.array([sharpe(x, yil_bar) for x in diziler], dtype=float)
    gecerli = np.isfinite(sr)
    if gecerli.sum() < 2:
        return {"ok": False, "reason": "en az iki gecerli kurgu gerekli"}
    n_deneme = int(gecerli.sum())
    if secilen is None:
        i = int(np.nanargmax(np.where(gecerli, sr, -np.inf)))
    else:
        i = adlar.index(secilen) if isinstance(secilen, str) else int(secilen)

    # Denemelerin Sharpe yayilimi esigi belirliyor: butun kurgular ayni
    # sonucu veriyorsa arama az sey kazandirmistir ve esik alcaktir.
    v_sr = float(np.var(sr[gecerli], ddof=1))
    g = 0.5772156649015329
    z1 = _normal_ppf(max(1e-12, min(1 - 1e-12, 1.0 - 1.0 / n_deneme)))
    z2 = _normal_ppf(max(1e-12, min(1 - 1e-12,
                                    1.0 - 1.0 / (n_deneme * math.e))))
    beklenen_max = math.sqrt(max(0.0, v_sr)) * ((1 - g) * z1 + g * z2)
    return {
        "ok": True,
        "secilen": adlar[i],
        "sharpe": float(sr[i]),
        "deneme": n_deneme,
        "sharpe_yayilimi": math.sqrt(max(0.0, v_sr)),
        # Beceri yoksa N denemenin en iyisinden BEKLENEN Sharpe.
        "sans_esigi": float(beklenen_max),
        "sisirilmis": float(olasilikli_sharpe(diziler[i], beklenen_max,
                                              yil_bar)),
        "olasilikli_sifira_karsi": float(olasilikli_sharpe(diziler[i], 0.0,
                                                           yil_bar)),
    }


def pbo(seriler: "dict[str, np.ndarray] | list", s_parca: int = 10) -> dict:
    """Arka test aşırı uydurma olasılığı (CSCV).

    Pencere `s_parca` eşit parçaya bölünür; yarısı eğitim yarısı test
    olan bütün kombinasyonlarda eğitimin en iyisi seçilip testteki
    yüzdelik sırasına bakılır. Ortanca sıranın altına düşme oranı PBO.
    """
    if isinstance(seriler, dict):
        diziler = list(seriler.values())
    else:
        diziler = list(seriler)
    if len(diziler) < 2:
        return {"ok": False, "reason": "en az iki kurgu gerekli"}
    n = min(len(x) for x in diziler)
    if s_parca % 2 or n < s_parca * 4:
        return {"ok": False, "reason": f"{n} gozlem / {s_parca} parca yetersiz"}
    M = np.array([np.asarray(x, dtype=float)[:n] for x in diziler])
    kes = np.array_split(np.arange(n), s_parca)

    mantik = []
    for egitim in itertools.combinations(range(s_parca), s_parca // 2):
        test = [j for j in range(s_parca) if j not in egitim]
        ie = np.concatenate([kes[j] for j in egitim])
        it = np.concatenate([kes[j] for j in test])
        se = np.array([sharpe(M[k, ie]) for k in range(len(M))])
        st = np.array([sharpe(M[k, it]) for k in range(len(M))])
        if not np.isfinite(se).any() or not np.isfinite(st).any():
            continue
        en_iyi = int(np.nanargmax(np.where(np.isfinite(se), se, -np.inf)))
        gecerli = np.isfinite(st)
        if gecerli.sum() < 2 or not np.isfinite(st[en_iyi]):
            continue
        # Secilenin testteki yuzdelik sirasi; 0,5 ortanca.
        w = float((st[gecerli] < st[en_iyi]).sum()) / float(gecerli.sum() - 1 or 1)
        w = min(max(w, 1e-9), 1 - 1e-9)
        mantik.append(math.log(w / (1 - w)))
    if not mantik:
        return {"ok": False, "reason": "hicbir bolunme olculemedi"}
    m = np.array(mantik)
    return {
        "ok": True,
        "pbo": float((m <= 0).mean()),
        "bolunme": len(m),
        "ortanca_mantik": float(np.median(m)),
    }
