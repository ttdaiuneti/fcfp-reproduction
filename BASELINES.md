# Baseline implementations

Each baseline follows the measure, the search procedure and the parameters of its original paper. This file lists, for each one, the source, the equations followed, the choices we had to make where the paper is not explicit, and the checks we ran. All data are min–max scaled on the training part of each fold before any method runs.

## NRS — Hu, Yu, Liu and Wu (2008), Information Sciences 178(18):3577–3594
- File: `algorithms/NRS_Hu.py`.
- Model: Euclidean distance on [0, 1]-scaled attributes; neighbourhood δ_B(x) = {y : ‖x − y‖_B ≤ δ}; POS_B = objects whose neighbourhood lies in their own class; γ_B = |POS_B| / |U|. δ = 0.15 (the paper recommends 0.1–0.2).
- Search: Algorithm 1 (NFARNRS), add the attribute with the largest significance while it is larger than ε. The paper only says ε is a small positive number; γ moves in steps of 1/|U|, so every ε < 1/|U| gives the same result, and we use ε = 0.
- Check: Example 1 (Table 1, δ = 0.1). Five of the six neighbourhoods match. The paper lists δ(x4) = {x4, x5} although δ(x6) = {x4, x5, x6}; since |0.74 − 0.72| = 0.02 < 0.1, x6 must also be in δ(x4). We follow the definition.

## FNRS — Cornelis, Jensen, Hurtado and Ślęzak (2010), Information Sciences 180(2):209–224
- File: `algorithms/FNRS_Cornelis.py`.
- Model: per-attribute relation R_a(x, y) = max(0, 1 − |a(x) − a(y)| / σ_a) with σ_a the sample standard deviation (Eq. 15); Łukasiewicz t-norm over B (Eq. 16); positive region by Prop. 1 with the Łukasiewicz implicator; γ_B = |POS_B| / |POS_A| (Eq. 21).
- Search: forward selection until γ_B ≥ 0.95 (a fuzzy decision reduct of degree α = 0.95, Def. 1), then backward elimination.
- Check: Examples 1, 4 and 7 of the paper: σ_a4 = 16.385, σ_a5 = 131.176, POS_B = (1, .808, .473, .473, 1, 1, 1), γ_B = 0.822, POS_A ≡ 1; all match to three digits.

## SFRS — Hu, An and Yu (2010), Information Sciences 180(22):4384–4400
- File: `algorithms/SFRS_Hu.py`.
- Model: R(x, y) = exp(−‖x − y‖² / δ) with δ = 0.15 (Eq. 10, Sec. 6.2); soft distance with β = 0.1 (Def. 3, Sec. 6.1); soft lower approximation SD(x, U − X) (Eqs. 19–20); positive region as the supremum over classes (Def. 5); γ as the mean (Eq. 28).
- Search: Table 1 of the paper — rank all attributes by forward selection on the soft dependency, then keep the prefix with the best 3-NN accuracy (k = 3, Sec. 6.3). The paper does not give the number of cross-validation folds; we use ten stratified folds inside the training part (seed 42), and the shortest prefix on ties.
- `SFRS-eq`: the same ranking cut at the FCFP reduct size of the same fold.
- Check: the worked example on p. 4389 (SD = 0.49 with β = 0.06) is reproduced.

## GBNRS — Xia et al. (2022), IEEE TKDE 34(3)
- Files: `algorithms/vendor_gbnrs_xia.py` (the functions of `GBNRS_new/xin_de.py` from https://github.com/syxiaa/GBNRS, commit 0f2d4ee6 of 30 Jan 2024, copied verbatim; only the `mean_std` helper and the `__main__` block with hard-coded paths were removed) and `algorithms/GBNRS_Xia.py` (wrapper).
- As in the authors' code: purity 1; backward elimination by the number of objects in pure balls; the minimum ball size is chosen on the grid range(2, 2m, m//5) by the resubstitution 5-NN accuracy on the training part. The paper describes k-means splitting, whereas the code does not use k-means; we follow the code.

## FGB — Ji, Peng, Zhao and Yao (2023), Information Sciences 640:119071
- File: `algorithms/FGB_Ji.py`.
- Model: Algorithm 1 (k = number of classes, initial centres = class means, k-means; balls with purity below δ are split again); Eqs. 12–20 (centre, radius, purity, BC, SC, FPOS, FR); δ = 0.75 (Sec. 6).
- Search: Algorithm 2 (rank single attributes by FR, then stages 2 and 3).
- Where the paper is not explicit: splitting recurses until purity ≥ δ or a ball cannot be split; "generate balls" means k-means run to convergence.
- Check: no numerical example in the paper. Three hand-built cases: two separated classes give FR = 1; a foreign object at the centre gives FR = 0.8; a foreign object on the boundary gives FR = 1 (case 2 of Sec. 4.2).

## WFRS — Wang, Deng, An and Huang (2025), IEEE Trans. Fuzzy Systems 33(10):3504–3514
- File: `algorithms/WFRS_Wang.py`.
- Model: directed distance with class-wise sample variances (Eq. 9; λ = 1 when a variance is zero); R = exp(−d) (Eq. 10); object weight = share of same-class objects among the 5 nearest neighbours under d_A, object excluded (Defs. 3.3–3.4), computed once on all attributes; R^w = w·R (Eq. 15); fuzzy decision (Eq. 21); lower approximation (Eq. 17); dependency (Eqs. 23–24).
- Search: Algorithm 1, forward selection while the significance exceeds 0.005 (the paper recommends 0.005–0.008).
- Where the paper is not explicit: Eq. 9 is printed with a sum over all attributes; we sum over B.
- Check: Tables I–III of the paper could not be reproduced from the printed formulas (Table II gives a non-zero classical lower approximation of another class at an object itself, which the classical definition excludes), so no numerical check was possible.

## AFRSFCV — Wang, Zhang, An and Deng (2026), Pattern Recognition 170:112014
- File: `algorithms/AFRSFCV_Wang.py`.
- Model: relation of Eq. 7 with class-wise sample variance; lower approximation (Eqs. 8–9); decision matrix (Eqs. 12–13); loss (Eq. 14) with w = softmax(τ); update (Eq. 23); 200 iterations (Algorithm 1); attributes ranked by the learned weights (step 19).
- Choices: the printed derivatives (Eqs. 16, 19, 22) are not the derivatives of Eq. 14, so we use the exact derivative (checked by finite differences, difference 2.3e−11); λ1 = λ2 = 1 (within the recommended range; the paper selects them on test accuracy, which we do not do); step size 0.5, τ0 = 0.
- The selector keeps the FCFP reduct size of the same fold.

## mRMR, ReliefF, L1 — `algorithms/rankers.py`
- mRMR: MID criterion (Peng, Long and Ding, 2005); each attribute discretised into three levels at μ ± σ; mutual information from contingency counts.
- ReliefF: skrebate 0.8.3, 10 neighbours.
- L1: logistic regression with an ℓ1 penalty (scikit-learn, saga, C = 1); attributes ranked by the largest absolute coefficient.
- Each keeps the FCFP reduct size of the same fold. `Random` draws the same number of attributes uniformly (seed 1000 + fold).

## Earlier implementations
An earlier version of this repository contained simplified versions of NRS, FNRS, SFRS, GBNRS and FGB, and an implementation of FPGB. They were replaced by the implementations above; FPGB was removed because its original paper could not be obtained to verify the implementation.
