"""FCFP with adaptive or hybrid confidence weights, used only to show what the
fixed weights prevent.

omega_alpha(B) = alpha * omega^(C) + (1 - alpha) * omega^(B), where omega^(B) is the
tie-aware k_omega-NN purity computed on the attributes in B (object excluded).
alpha = 1 is FCFP.  The search is Algorithm 2 of the paper applied to this
measure: forward, add the attribute with the largest value while M(B) < theta
(also when that value is below the current one), theta = rho * M(C); then
backward, remove any a with M(B - {a}) >= theta until no change.
"""
from __future__ import annotations

import numpy as np

import engine as E
from reducers_r1 import knn_purity


class FCFPHybrid:
    def __init__(self, alpha=0.0, delta=0.3, k_omega=5, rho=0.95):
        self.alpha, self.delta, self.k_omega, self.rho = alpha, delta, k_omega, rho

    def _value(self, cols):
        om = self.omega_C if self.alpha >= 1 else \
            self.alpha * self.omega_C + (1 - self.alpha) * knn_purity(self.X[:, cols], self.y, self.k_omega)
        meas = E.FCFPMeasure(om, self.delta)
        return E.score(meas, self.X, self.y, E.full_base(meas, self.X, cols), None, len(cols))

    def fit_reduction(self, X, y):
        self.X, self.y = np.asarray(X, float), np.asarray(y)
        m = self.X.shape[1]
        self.omega_C = knn_purity(self.X, self.y, self.k_omega)
        self.theta_ = self.rho * self._value(list(range(m)))
        B, g, self.negative_steps_ = [], 0.0, 0
        while g < self.theta_ - 1e-12 and len(B) < m:
            vals = {a: self._value(sorted(B + [a])) for a in range(m) if a not in B}
            a = max(vals, key=vals.get)
            if vals[a] < g - 1e-12:
                self.negative_steps_ += 1
            B.append(a); g = vals[a]
        changed = True
        while changed and len(B) > 1:
            changed = False
            for a in list(B):
                rest = sorted(set(B) - {a})
                if self._value(rest) >= self.theta_ - 1e-12:
                    B.remove(a); changed = True
                    break
        self.reduct_ = sorted(int(a) for a in B)
        del self.X
        return self
