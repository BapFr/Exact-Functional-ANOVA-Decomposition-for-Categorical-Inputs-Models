"""
Functional ANOVA Decomposition Basis for MNIST.

Dependencies:
    - numpy
    - tqdm
"""

# ==========================================
# IMPORTS
# ==========================================

import numpy as np
from tqdm import tqdm
from scipy.linalg import cho_factor, cho_solve

# ==========================================
# Class Basis for Binary case
# ==========================================

class BinaryBasisExtractor:
    """
    Requested final configuration:
      - pair_pool_strategy="top_variance"
      - pair_probs_mode="on_the_fly" (no pre-computation)
      - reorthogonalize=False
      - SINGLE tqdm progress bar: 0..n_set (number of accepted sets)

    Pipeline:
      1) A = ∅ (Empty set)
      2) Singletons [1],...,[d]
      3) User-provided sets in A (pruned if singleton is rejected)
      4) Automatic pair pool from maxpool variables (top-variance),
         excluding pairs already present in A
      Stop: as soon as n_set sets have been ACCEPTED.
    """

    def __init__(self, X, A, n_set, rtol=1e-6, atol=1e-8, maxpool=100, verbose=True):
        # -------------------------
        # Data & unique patterns
        # -------------------------
        self.X = np.asarray(X, dtype=np.int8)
        self.n, self.d = self.X.shape

        self.L, self.counts = np.unique(self.X, axis=0, return_counts=True)
        self.L = self.L.astype(np.int8, copy=False)
        self.counts = self.counts.astype(np.int64, copy=False)

        self.k = self.L.shape[0]
        self.n_total = int(self.counts.sum())

        # -------------------------
        # Parameters
        # -------------------------
        self.n_set = int(n_set)
        if self.n_set < 0:
            raise ValueError("n_set must be >= 0.")

        self.rtol = float(rtol)
        self.atol = float(atol)

        self.maxpool = int(maxpool) if maxpool is not None else self.d
        self.maxpool = min(max(self.maxpool, 0), self.d)

        self.verbose = bool(verbose)

        # Enforced configuration
        self.pair_pool_strategy = "top_variance"
        self.pair_probs_mode = "on_the_fly"
        self.reorthogonalize = False

        # -------------------------
        # Pre-computations
        # -------------------------
        self.weights = self.counts.astype(np.float64) / float(self.n_total)
        self.S = 1 - 2 * self.L  # (-1)^L

        c1 = np.dot(self.counts, self.L)  # (d,)
        self.p1 = c1.astype(np.float64) / float(self.n_total)

        # -------------------------
        # Outputs
        # -------------------------
        self.power_set = []   # accepted sets (1-based)
        self.S_reject = []    # rejected sets (tested OR pruned) (1-based)
        self.M_final = None

        # -------------------------
        # Preparation of sets A
        # -------------------------
        self.A_input_0 = self._sanitize_A(A)           # 0-based tuples
        self._A_input_set = set(self.A_input_0)        # to exclude from pool (exact pairs)
        self._rejected_singleton = np.zeros(self.d, dtype=bool)

        # Execution
        self._run()

    # ==========================================
    # Public Methods
    # ==========================================

    def get_patterns(self):
        return self.L

    def get_P(self):
        return self.weights

    def get_sets(self):
        return self.power_set

    def get_rejected_sets(self):
        return self.S_reject

    def get_matrix(self):
        return self.M_final

    # ==========================================
    # Internals: sets / pool
    # ==========================================

    def _sanitize_A(self, A):
        out = []
        for s in (A or []):
            if s is None:
                continue
            if len(s) == 0:
                out.append(tuple())
                continue
            ss = sorted(set(int(x) for x in s if 1 <= int(x) <= self.d))
            out.append(tuple(x - 1 for x in ss))
        return out

    def _build_pair_pool_vars(self):
        m = self.maxpool
        if m <= 0:
            return np.array([], dtype=np.int32)
        if m >= self.d:
            return np.arange(self.d, dtype=np.int32)

        score = self.p1 * (1.0 - self.p1)
        idx = np.argpartition(-score, kth=m - 1)[:m]
        idx = idx[np.argsort(-score[idx])]
        return idx.astype(np.int32, copy=False)

    # ==========================================
    # Internals: v(A) computation (fully on-the-fly)
    # ==========================================

    def _v_empty(self):
        return np.ones(self.k, dtype=np.float64)

    def _v_singleton(self, j0):
        num = self.S[:, j0].astype(np.float64, copy=False)
        pj = float(self.p1[j0])
        den = np.where(self.L[:, j0] == 1, pj, 1.0 - pj).astype(np.float64, copy=False)
        if np.any(den <= 0.0):
            return None
        return num / den

    def _v_pair(self, j0, l0):
        num = (self.S[:, j0] * self.S[:, l0]).astype(np.float64, copy=False)

        pj = float(self.p1[j0])
        pl = float(self.p1[l0])

        # on-the-fly: p11 recalculated for each pair
        p11 = float(np.dot(self.counts, self.L[:, j0] * self.L[:, l0])) / float(self.n_total)

        p10 = pj - p11
        p01 = pl - p11
        p00 = 1.0 - pj - pl + p11

        den_table = np.array([p00, p01, p10, p11], dtype=np.float64)
        if np.any(den_table <= 0.0):
            return None

        code = (self.L[:, j0].astype(np.int8) << 1) + self.L[:, l0].astype(np.int8)
        den = den_table[code]
        return num / den

    def _v_general(self, A0):
        cols = list(A0)
        num = np.prod(self.S[:, cols], axis=1, dtype=np.float64)

        sub_L = self.L[:, cols]
        _, indices = np.unique(sub_L, axis=0, return_inverse=True)

        probs = np.bincount(indices, weights=self.counts, minlength=int(indices.max()) + 1).astype(np.float64)
        probs /= float(self.n_total)

        den = probs[indices]
        if np.any(den <= 0.0):
            return None
        return num / den

    def _compute_v(self, A0):
        if len(A0) == 0:
            return self._v_empty()
        if len(A0) == 1:
            return self._v_singleton(A0[0])
        if len(A0) == 2:
            return self._v_pair(A0[0], A0[1])
        return self._v_general(A0)

    # ==========================================
    # Internals: Gram–Schmidt (without re-orthogonalization)
    # ==========================================

    def _try_add_vector(self, Q, M, rank, v):
        v = v.astype(np.float64, copy=False)
        norm_v = np.linalg.norm(v)
        if (not np.isfinite(norm_v)) or norm_v < self.atol:
            return rank, False

        if rank == 0:
            v_res = v
        else:
            Qi = Q[:, :rank]
            coeffs = Qi.T @ v
            v_res = v - Qi @ coeffs

        norm_res = np.linalg.norm(v_res)
        if not np.isfinite(norm_res):
            return rank, False

        tol = self.atol + self.rtol * norm_v
        if norm_res < tol:
            return rank, False

        M[:, rank] = v
        Q[:, rank] = v_res / norm_res
        return rank + 1, True

    def _reject(self, A0):
        self.S_reject.append([a + 1 for a in A0])
        if len(A0) == 1:
            self._rejected_singleton[A0[0]] = True

    def _pruned_by_singleton(self, A0):
        return any(self._rejected_singleton[j] for j in A0)

    # ==========================================
    # Main Loop (single progress bar)
    # ==========================================

    def _run(self):
        if self.n_set == 0:
            self.M_final = np.empty((self.k, 0), dtype=np.float64)
            return
        if self.k == 0:
            self.M_final = np.empty((0, 0), dtype=np.float64)
            return

        M = np.empty((self.k, self.n_set), dtype=np.float64)
        Q = np.empty((self.k, self.n_set), dtype=np.float64)

        rank = 0
        accepted = set()  # 0-based tuples accepted

        pbar = tqdm(total=self.n_set, desc="Accepted sets", colour="green") if self.verbose else None

        def accept(A0, v):
            nonlocal rank
            rank_new, ok = self._try_add_vector(Q, M, rank, v)
            if ok:
                if pbar is not None:
                    pbar.update(rank_new - rank)
                rank = rank_new
                accepted.add(A0)
                self.power_set.append([a + 1 for a in A0] if len(A0) > 0 else [])
            else:
                self._reject(A0)

        try:
            # ---- Phase 0
            if rank < self.n_set:
                A0 = tuple()
                v = self._compute_v(A0)
                accept(A0, v)

            # ---- Phase 1: singletons
            for j in range(self.d):
                if rank >= self.n_set:
                    break
                A0 = (j,)
                v = self._compute_v(A0)
                if v is None:
                    self._reject(A0)
                    continue
                accept(A0, v)

            # ---- Phase 2: sets A (unique + pruning)
            if rank < self.n_set:
                A_unique, seen = [], set()
                for s in self.A_input_0:
                    if s in seen:
                        continue
                    seen.add(s)
                    A_unique.append(s)

                for A0 in A_unique:
                    if rank >= self.n_set:
                        break
                    if A0 in accepted:
                        continue
                    if self._pruned_by_singleton(A0):
                        self._reject(A0)
                        continue
                    v = self._compute_v(A0)
                    if v is None:
                        self._reject(A0)
                        continue
                    accept(A0, v)

            # ---- Phase 3: top-variance pool (on-the-fly)
            if rank < self.n_set:
                pool_vars = self._build_pair_pool_vars()
                excluded = self._A_input_set
                m = pool_vars.size

                for a in range(m):
                    if rank >= self.n_set:
                        break
                    j = int(pool_vars[a])
                    if self._rejected_singleton[j]:
                        continue
                    for b in range(a + 1, m):
                        if rank >= self.n_set:
                            break
                        l = int(pool_vars[b])
                        if self._rejected_singleton[l]:
                            continue

                        A0 = (j, l)
                        if A0 in excluded or A0 in accepted:
                            continue

                        v = self._compute_v(A0)
                        if v is None:
                            self._reject(A0)
                            continue
                        accept(A0, v)

        finally:
            if pbar is not None:
                pbar.close()
            self.M_final = M[:, :rank] if rank > 0 else np.empty((self.k, 0), dtype=np.float64)


# ==========================================
# Select neighbors to prioritize
# ==========================================

def get_neighbor_pairs(n):
    """
    Generates the list of neighbor pairs (8-connectivity) for an n x n grid.
    Indices from 1 to n*n.
    """
    pairs = []
    
    for r in range(n):
        for c in range(n):
            current_idx = r * n + c + 1
            
            moves = [(0, 1), (1, -1), (1, 0), (1, 1)]
            
            for dr, dc in moves:
                nr, nc = r + dr, c + dc
                
                if 0 <= nr < n and 0 <= nc < n:
                    neighbor_idx = nr * n + nc + 1
                    pairs.append([current_idx, neighbor_idx])
                    
    return pairs