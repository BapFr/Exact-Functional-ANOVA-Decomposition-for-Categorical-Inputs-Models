"""
Functiona ANOVA Decomposition Basis.

This module implements an incremental hierarchical-orthogonal basis construction for 
discrete multivariate data. It computes the ANOVA-like decomposition 
matrix and solves the regularized linear system for model interpretation.

Dependencies:
    - numpy
    - scipy
    - tqdm
"""

# ==========================================
# IMPORTS
# ==========================================

from typing import Tuple, List, Optional, Dict, Any
import numpy as np
from tqdm import tqdm
from itertools import combinations
from math import prod
from scipy.linalg import cho_factor, cho_solve

# ====================================================
# CATEGORICAL FONCTIONAL ANOVA (SPARSE DATASET BASED)
# ====================================================

# ==========================================
# 1. Pre-computation Helpers
# ==========================================

def _compute_patterns(X: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Computes distinct patterns and empirical probabilities from the dataset.

    Args:
        X (np.ndarray): Input dataset of shape (n_samples, n_features).

    Returns:
        combs (np.ndarray): Distinct rows (patterns) present in X, shape (r, d).
        P (np.ndarray): Empirical probabilities for each pattern, shape (r,).
        N (np.ndarray): Cardinality (number of modalities) for each variable, shape (d,).
    """
    X = np.asarray(X, dtype=int)
    combs, counts = np.unique(X, axis=0, return_counts=True)
    P = counts.astype(float) / counts.sum()
    N = combs.max(axis=0) + 1
    return combs, P, N


# ==========================================
# 2. Subset Iteration Logic
# ==========================================

def _next_set(d: int, current_A: List[int]) -> Optional[List[int]]:
    """
    Generates the next subset A (1-based indices) in lexicographical order 
    by increasing size.

    Args:
        d (int): Total number of dimensions/features.
        current_A (List[int]): The current subset of indices (1-based).

    Returns:
        List[int] or None: The next subset, or None if all subsets are exhausted.
    """
    size = len(current_A)

    # First set initialization
    if size == 0:
        return [1] if d >= 1 else None

    # Search for a pivot to increment
    pivot = -1
    for i in range(size - 1, -1, -1):
        if current_A[i] < d - (size - 1 - i):
            pivot = i
            break

    if pivot != -1:
        next_A = list(current_A)
        next_A[pivot] += 1
        for j in range(pivot + 1, size):
            next_A[j] = next_A[j - 1] + 1
        return next_A

    # No pivot found: increase subset size if possible
    if size < d:
        return list(range(1, size + 2))
    else:
        return None


# ==========================================
# 3. Y-Vector Generation for Subsets
# ==========================================

def _get_y_from_N(N: np.ndarray, S: List[int]) -> np.ndarray:
    """
    Generates the grid of coordinate vectors 'y' for a given subset S.

    For a subset S, generates matrix Y (m x d) where each row y satisfies:
      - for j in S: y_j in {0, ..., N_j - 2}
      - for j not in S: y_j = 0
    
    Where m = product_{i in S}(N_i - 1).

    Args:
        N (np.ndarray): Number of modalities per variable.
        S (List[int]): List of 1-based variable indices.

    Returns:
        np.ndarray: The grid of y vectors.
    """
    N = np.asarray(N, dtype=int)
    d = N.size

    S = sorted(S)
    S0 = [s - 1 for s in S]  # Convert to 0-based indexing
    k = len(S0)

    if k == 0:
        return np.zeros((1, d), dtype=int)

    sizes = N[S0] - 1
    if np.any(sizes <= 0):
        return np.zeros((0, d), dtype=int)

    # Create grid on coordinates in S
    grids = np.indices(sizes, dtype=int)  # Shape: (k, sizes[0], ..., sizes[k-1])
    vals_S = grids.reshape(k, -1).T       # Shape: (m, k)

    Y = np.zeros((vals_S.shape[0], d), dtype=int)
    Y[:, S0] = vals_S
    return Y


# ==========================================
# 4. Probability Computation
# ==========================================

def _compute_P_A_for_patterns(combs: np.ndarray, P: np.ndarray, A0: List[int]) -> np.ndarray:
    """
    Computes marginal probabilities P(X_A = x_A) for all patterns in a vectorized manner.

    Args:
        combs (np.ndarray): Distinct patterns (r, d).
        P (np.ndarray): Empirical probabilities (r,).
        A0 (List[int]): List of 0-based indices for the subset A.

    Returns:
        np.ndarray: A vector v of size r, where v[i] = P(X_A = combs[i, A]).
    """
    sub = combs[:, A0]                                     # (r, |A|)
    _, inv = np.unique(sub, axis=0, return_inverse=True)   # inv: (r,)
    sums = np.bincount(inv, weights=P)                     # (num_distinct_values,)
    return sums[inv]                                       # (r,)


# ==========================================
# 5. Basis Function Evaluation
# ==========================================

def _psi_from_precomputed(xA: np.ndarray, 
                          yA: np.ndarray, 
                          Ni_minus1: np.ndarray, 
                          sign: np.ndarray, 
                          P_A_vec: np.ndarray) -> np.ndarray:
    """
    Computes the basis vector e_{S,y}(x) using pre-calculated components.

    Args:
        xA (np.ndarray): Values of X restricted to S, shape (r, |S|).
        yA (np.ndarray): Current y vector restricted to S, shape (|S|,).
        Ni_minus1 (np.ndarray): N[A0] - 1 values, shape (|S|,).
        sign (np.ndarray): Precomputed sign term (-1)^{1{x_i = N_i-1}}, shape (r,).
        P_A_vec (np.ndarray): Marginal probabilities P(X_S = x_S), shape (r,).

    Returns:
        np.ndarray: The evaluated basis vector of size r.
    """
    # Check condition: x_i must be equal to y_i or (N_i - 1)
    mask_in = (xA == yA) | (xA == Ni_minus1)  # (r, |S|)
    valid = np.all(mask_in, axis=1)           # (r,)

    vec = np.zeros(xA.shape[0], dtype=float)
    if np.any(valid):
        vec[valid] = sign[valid] / P_A_vec[valid]
    return vec


# ==========================================
# 6. Orthogonal Basis Updates
# ==========================================

def _update_basis(Q: Optional[np.ndarray], 
                  v: np.ndarray, 
                  rtol: float = 1e-3, 
                  atol: float = 1e-3) -> Tuple[Optional[np.ndarray], bool]:
    """
    Incrementally updates the orthonormal basis Q with vector v using Gram-Schmidt.

    Args:
        Q (np.ndarray or None): Current orthonormal basis matrix.
        v (np.ndarray): New candidate vector.
        rtol (float): Relative tolerance for linear independence check.
        atol (float): Absolute tolerance for linear independence check.

    Returns:
        Tuple[np.ndarray, bool]: The updated basis Q (or original if dependent), 
                                 and a boolean indicating if v was linearly independent.
    """
    v = np.asarray(v, dtype=float)
    norm_v = np.linalg.norm(v)

    if Q is None or Q.size == 0:
        if norm_v < atol:
            return Q, False
        return (v / norm_v)[:, None], True

    # Orthogonalize v against Q
    coeffs = Q.T @ v
    v_res = v - Q @ coeffs
    norm_res = np.linalg.norm(v_res)

    # Check for linear independence
    tol = atol + rtol * norm_v
    if norm_res < tol:
        return Q, False

    # Normalize and append
    v_res /= norm_res
    Q_new = np.column_stack((Q, v_res))
    return Q_new, True


# ==========================================
# 7. Main Matrix Construction Routine
# ==========================================

def get_matrix_optimized(X: np.ndarray, 
                         tol_percent: float = 100, 
                         eps: float = 1e-3) -> Tuple[np.ndarray, List[List[int]]]:
    """
    Constructs a matrix where columns are linearly independent basis functions 
    e_{S,y} evaluated on the distinct patterns of X.

    Optimized version using incremental QR updates and pre-computations.

    Args:
        X (np.ndarray): Input dataset.
        tol_percent (float): Percentage of the total rank (r) to target.
        eps (float): Tolerance for linear independence checks.

    Returns:
        matrix (np.ndarray): The constructed basis matrix (r, r_partial).
        complete_powerset (List[List[int]]): List of subsets S corresponding to columns.
    """
    # Pre-computations
    combs, P, N = _compute_patterns(X)
    d = N.size
    r = combs.shape[0]

    if r == 0:
        return np.zeros((0, 0)), []

    # Determine target rank
    target_rank = int(r * tol_percent / 100)
    target_rank = max(1, min(target_rank, r))

    # First column: Constant function 1
    first_col = np.ones(r, dtype=float)
    matrix = first_col[:, None]
    Q, _ = _update_basis(None, first_col, atol=eps)
    actual_rank = Q.shape[1]

    S = []
    complete_powerset = [[]]
    cache_P_A = {}  # Cache for P(X_A = x_A) vectors

    # Progress bar tracking rank increase
    pbar = tqdm(total=target_rank, desc="Constructing Basis Matrix", colour="green")
    pbar.update(actual_rank)

    try:
        while actual_rank < target_rank:
            S = _next_set(d, S)
            if S is None:
                break

            Y = _get_y_from_N(N, S)
            if Y.shape[0] == 0:
                continue

            # Convert S to 0-based indices for internal logic
            A0 = tuple(a - 1 for a in S)

            # Retrieve or compute P(X_S = x_S)
            if A0 not in cache_P_A:
                cache_P_A[A0] = _compute_P_A_for_patterns(combs, P, list(A0))
            P_A_vec = cache_P_A[A0]

            # Pre-computations independent of y
            A0_list = list(A0)
            xA = combs[:, A0_list]
            NA = N[A0_list]
            Ni_minus1 = NA - 1

            # Compute sign term (-1)^{1{x_i = N_i-1}} (product over i in S)
            sign_i = np.where(xA == Ni_minus1, -1.0, 1.0)
            sign = np.prod(sign_i, axis=1)

            # Iterate over valid y vectors for this S
            for y in Y:
                yA = y[A0_list]
                vec = _psi_from_precomputed(xA, yA, Ni_minus1, sign, P_A_vec)

                # Check linear independence via incremental basis Q
                old_rank = actual_rank
                Q_new, independent = _update_basis(Q, vec, atol=eps)
                
                if not independent:
                    continue

                # Update basis and matrix
                Q = Q_new
                matrix = np.column_stack((matrix, vec))
                complete_powerset.append(list(S))
                actual_rank = Q.shape[1]

                # Update progress bar
                delta_rank = actual_rank - old_rank
                if delta_rank > 0:
                    pbar.update(min(delta_rank, target_rank - pbar.n))

                if actual_rank >= target_rank or matrix.shape[1] >= r:
                    break

            if actual_rank >= target_rank or matrix.shape[1] >= r:
                break
    finally:
        pbar.close()

    return matrix, complete_powerset

# ==========================================
# 8. Main Analysis
# ==========================================

# Auxiliary function

def aggregate_duplicate_coalitions(
    S: List[List[int]], 
    M: np.ndarray
) -> Tuple[List[List[int]], np.ndarray]:
    """
    Aggregates columns of matrix M corresponding to identical sets in S.
    
    This function identifies duplicate sets in the list S. For every unique set found,
    it sums the corresponding columns in M. The order of unique sets in the output
    preserves their first appearance in the input S.

    Parameters
    ----------
    S : List[List[int]]
        A list of sets (represented as lists), potentially containing duplicates.
        Length = k.
    M : np.ndarray
        Input matrix of shape (n, k). Each column j corresponds to S[j].

    Returns
    -------
    reduced_S : List[List[int]]
        List of unique sets, length r.
    reduced_M : np.ndarray
        Aggregated matrix of shape (n, r).
    """
    # Dictionary to map a unique set (as a tuple) to a list of its column indices
    # Structure: { (1, 2): [7, 8, 9, 10], ... }
    set_to_indices_map = {}
    
    for col_idx, subset in enumerate(S):
        # Convert list to tuple to make it hashable (dict key).
        # We sort to ensure {1, 2} is treated the same as {2, 1}.
        key = tuple(sorted(subset))
        
        if key not in set_to_indices_map:
            set_to_indices_map[key] = []
        set_to_indices_map[key].append(col_idx)
    
    # Lists to store the result
    unique_sets = []
    aggregated_columns = []
    
    # Iterate over the unique keys. 
    # Python 3.7+ dicts guarantee iteration in insertion order.
    for key_tuple, indices in set_to_indices_map.items():
        # 1. Reconstruct the set list
        unique_sets.append(list(key_tuple))
        
        # 2. Sum the corresponding columns in M
        if len(indices) == 1:
            # Optimization: no sum needed if only one occurrence
            col_sum = M[:, indices[0]]
        else:
            # Slice M to get all relevant columns, then sum across axis 1 (horizontally)
            col_sum = np.sum(M[:, indices], axis=1)
            
        aggregated_columns.append(col_sum)
    
    # Stack the result columns to form the (n, r) matrix
    reduced_M = np.column_stack(aggregated_columns)
    
    return unique_sets, reduced_M

# Main Class

class ModelAnalysis:
    """
    Main framework for analyzing model responses using orthogonal decomposition.
    """

    def __init__(self, 
                 X_encoded: np.ndarray, 
                 f_model: callable, 
                 percentage_set: float, 
                 a_tol: float, 
                 eps_reg_gamma: float = 1e-10):
        """
        Initializes the analysis and performs all computations.

        Args:
            X_encoded (np.ndarray): Integer-encoded input dataset.
            f_model (callable): The model function to analyze (takes X_encoded as input).
            percentage_set (float): Percentage of rank to preserve in basis construction.
            a_tol (float): Tolerance for linear independence checks.
            eps_reg_gamma (float, optional): Regularization term for Gamma matrix inversion.
        """
        
        # 1. Matrix and Pattern Computation
        self._G_matrix = get_matrix_optimized(X_encoded, percentage_set, a_tol)
        
        patterns = _compute_patterns(X_encoded)
        self._X_uniq = patterns[0]
        self._P = patterns[1]
        
        self._M = self._G_matrix[0]  # Basis Matrix
        self._S = self._G_matrix[1]  # Corresponding Subsets
        
        # 2. Model Application and Variance Calculation
        self._Y = f_model(self._X_uniq)
        self._var = self._Y**2 @ self._P - (self._Y @ self._P)**2

        # 3. Gamma Matrix Construction and Regularization
        self._Gamma = self._M.T @ np.diag(self._P) @ self._M
        self._Gamma[np.diag_indices_from(self._Gamma)] += eps_reg_gamma 

        # 4. Linear System Resolution (Computing Lambda)
        self._mu = (self._M.T * self._P) @ self._Y 
            
        # Using Cholesky decomposition for stability
        c, low = cho_factor(self._Gamma, lower=True) 
        self._lamb = cho_solve((c, low), self._mu) 

        # 5. Functional ANOVA
        self._Obliq_matrix = self._M * self._lamb
        self._functional_decomposition = aggregate_duplicate_coalitions( self._S , self._Obliq_matrix )

        # 5. Final Metrics (Oblique Projection, Error, R2)

        self._Err_L2 = (np.sum(self._Obliq_matrix, axis=1) - self._Y)**2 @ self._P
        
        # Relative L2 Error handling division by zero if norm is 0
        norm_Y = self._Y**2 @ self._P
        self._Err_L2_rel = (self._Err_L2 / norm_Y) if norm_Y > 1e-12 else 0.0
        
        self._R_2 = 1 - (self._Err_L2) / (self._var) if self._var > 1e-12 else 0.0
        
        print("Computations complete. Results ready.")

    # ==========================================
    #           Getters
    # ==========================================

    # Main Result
    def functional_anova(self):
        """Returns Sets and f_A(X_A)"""
        return self._functional_decomposition
    
    # Other Useful Getters

    def get_M(self) -> np.ndarray:
        """Returns the basis matrix M."""
        return self._M
    
    def get_S(self) -> List[List[int]]:
        """Returns the list of subsets corresponding to M's columns."""
        return self._S

    def get_P(self) -> np.ndarray:
        """Returns the empirical probability vector."""
        return self._P

    def get_X_uniq(self) -> np.ndarray:
        """Returns the unique patterns found in X."""
        return self._X_uniq

    def get_Y(self) -> np.ndarray:
        """Returns the model response vector f_model(X_uniq)."""
        return self._Y

    def get_Gamma(self) -> np.ndarray:
        """Returns the regularized Gamma matrix."""
        return self._Gamma

    def get_mu(self) -> np.ndarray:
        """Returns the mean vector (mu)."""
        return self._mu

    def get_lambda(self) -> np.ndarray:
        """Returns the coefficient vector lambda (system solution)."""
        return self._lamb

    def get_var(self) -> float:
        """Returns the calculated variance of Y."""
        return self._var

    def get_Obliq_matrix(self) -> np.ndarray:
        """Returns the final oblique matrix (M * lambda)."""
        return self._Obliq_matrix

    def get_L2_Error(self) -> float:
        """Returns the weighted Mean Squared Error (L2)."""
        return self._Err_L2
    
    def get_L2_Error_rel(self) -> float:
        """Returns the relative L2 error."""
        return self._Err_L2_rel

    def get_R2(self) -> float:
        """Returns the coefficient of determination R²."""
        return self._R_2
    
# ==========================================
# 9. Shapley Values from Harsanyi Dividends
# ==========================================

def batch_shapley_values(
    n_players: int, 
    coalitions: List[List[int]], 
    dividends_matrix: np.ndarray
) -> np.ndarray:
    """
    Vectorized computation of Shapley values from a batch of Harsanyi dividends.

    This function linearly transforms Harsanyi dividends into Shapley values 
    using a pre-computed weight matrix. This allows for processing multiple 
    game instances (samples) simultaneously.

    The relationship relies on the axiom that the Shapley value of a player i 
    is the sum of dividends of all coalitions T containing i, divided by the 
    size of T:
        phi_i = sum_{T : i in T} ( dividend(T) / |T| )

    Parameters
    ----------
    n_players : int
        The total dimension of the game (number of players), denoted as d.
    coalitions : List[List[int]]
        A list of length k representing the coalitions (subsets) associated 
        with the columns of the dividends matrix. 
        Note: Indices are expected to be 1-based (e.g., {1, 2, ...}).
    dividends_matrix : np.ndarray
        A matrix of shape (n_samples, n_coalitions) containing the Harsanyi 
        dividends. Each row p corresponds to a specific game instance v_p, 
        and columns correspond to the coalitions in S.

    Returns
    -------
    np.ndarray
        The matrix of Shapley values Phi of shape (n_samples, n_players).
        Phi[p, :] corresponds to the Shapley vector for the game instance p.

    Raises
    ------
    ValueError
        If the number of coalitions in S does not match the number of columns 
        in dividends_matrix.
    """
    # Ensure input is a float array
    V = np.asarray(dividends_matrix, dtype=float)
    n_samples, n_coalitions = V.shape

    # Validation
    if n_coalitions != len(coalitions):
        raise ValueError(
            f"Dimension mismatch: length of coalitions list ({len(coalitions)}) "
            f"must match number of columns in dividends_matrix ({n_coalitions})."
        )

    # 1. Construct the assignment/weight matrix A of shape (k, d)
    #    A[l, j] = 1/|S_l| if player j is in coalition S_l, else 0.
    #    This matrix distributes the dividend of a coalition equally among its members.
    A = np.zeros((n_coalitions, n_players), dtype=float)

    for idx, indices_set in enumerate(coalitions):
        cardinality = len(indices_set)
        
        if cardinality == 0:
            continue  # The empty set contributes nothing
            
        share = 1.0 / cardinality
        
        # Convert 1-based indices (input) to 0-based indices (internal)
        # We subtract 1 from every player index j
        idx_zero_based = [j - 1 for j in indices_set]
        
        # Vectorized assignment for the row corresponding to this coalition
        A[idx, idx_zero_based] = share

    # 2. Matrix multiplication to compute Shapley values for the entire batch
    #    Phi = V @ A
    #    Shape: (n_samples, n_coalitions) @ (n_coalitions, n_players) -> (n_samples, n_players)
    shapley_values = V @ A
    
    return shapley_values


# ==============================================================================================================================
# ==============================================================================================================================
# ==============================================================================================================================

# ====================================================
# FULL SUPPORT ANOVA (ULTRA FAST CLOSED FORMULA)
# ====================================================

class FullSupportAnova:
    def __init__(self, N, P, f):
        """
        Initializes the FullSupportAnova class.
        
        Args:
            N (array-like): List or array of dimensions (N1, ..., Nd).
            P (array-like): 1D vector of probabilities (size prod(N)) 
                            or nD array of shape N.
            f (callable): Function to apply on the generated tuples.
        """
        self.N = np.asarray(N, dtype=int)
        self.f = f
        self.d = self.N.size
        
        # P verification and preparation
        P = np.asarray(P, dtype=float)
        expected_size = int(np.prod(self.N))
        if P.size != expected_size:
            raise ValueError(f"Size of P ({P.size}) != prod(N) ({expected_size})")
        
        # Store flattened P and shaped P_nd
        self.P = P
        self.P_nd = P.reshape(tuple(self.N), order="C")

    # ==========================================
    # 1. Utility Functions (Indices and Grid Generation)
    # ==========================================

    def _generate_subsets(self):
        """
        Generates subsets of {1,...,d} (1-based indices).
        """
        subsets = [[]]
        for k in range(1, self.d + 1):
            for comb in combinations(range(1, self.d + 1), k):
                subsets.append(list(comb))
        return subsets

    def _repeated_subsets(self):
        """
        Returns a list of subsets S ⊆ {1,...,d} (in 1-based indices),
        in itertools order, where each S is repeated
            ∏_{i∈S} (N_i - 1)
        times, consecutively.
        """
        subsets = self._generate_subsets()
        N_list = list(self.N)
        
        result = []
        for S in subsets:
            # product_{i∈S} (N_i - 1), 1-based indices -> N[i-1]
            if len(S) == 0:
                reps = 1  # empty product = 1
            else:
                reps = prod(N_list[i - 1] - 1 for i in S)
            
            if reps > 0:
                result.extend([S] * reps)
        return result

    def _generate_tuples(self):
        """
        Generates all d-tuples (x1,...,xd) in canonical order (C-order).
        Returns a matrix of shape (prod(N), d).
        """
        grids = np.indices(self.N)
        return grids.reshape(self.d, -1).T

    def _generate_y(self, S):
        """
        Generates y vectors for a given subset S.
        Returns Y of shape (∏_{i∈S} (Ni-1), d).
        """
        S = sorted(S)
        S0 = [s - 1 for s in S] # 0-based indices
        k = len(S0)

        if k == 0:
            return np.zeros((1, self.d), dtype=int)

        sizes = self.N[S0] - 1
        if np.any(sizes <= 0):
            return np.zeros((0, self.d), dtype=int)

        grids = np.indices(sizes)
        vals_S = grids.reshape(k, -1).T  # (m, k)

        Y = np.zeros((vals_S.shape[0], self.d), dtype=int)
        Y[:, S0] = vals_S
        return Y

    # ==========================================
    # 2. Calculation Functions (Numerators and Denominators)
    # ==========================================

    def _compute_denominators(self, combs, S):
        """
        Calculates the denominator vector 'den' of size n = prod(N) such that
        den[j] = P(X_S = x_S) where x = combs[j].
        """
        S0 = [s - 1 for s in S] # 0-based indices

        if len(S) == 0:
            # convention : P(X_∅ = ·) = 1
            n = combs.shape[0]
            return np.ones(n, dtype=float)

        # Summation over axes NOT in S
        axes_sum = tuple(ax for ax in range(self.d) if ax not in S0)
        
        # P_marg has the shape of dimensions present in S
        P_marg = self.P_nd.sum(axis=axes_sum)

        # Retrieve corresponding values for each row in combs
        idx = tuple(combs[:, S0].T)
        den = P_marg[idx]
        
        return den.astype(float)

    def _compute_numerators(self, combs, S, Y_S):
        """
        Calculates the numerator matrix 'num' of shape (n, m).
        num[j, k] corresponds to the orthogonal interaction term.
        """
        n = combs.shape[0]
        m = Y_S.shape[0]

        if len(S) == 0:
            return np.ones((n, m), dtype=int)

        S0 = np.array(S, dtype=int) - 1  # 0-based
        num = np.ones((n, m), dtype=int)

        # Loop over active dimensions of S
        for ax in S0:
            Xi = combs[:, ax][:, None]     # (n, 1)
            Yi = Y_S[:, ax][None, :]       # (1, m)
            Ni_minus1 = self.N[ax] - 1

            # Indicator logic
            mask_in = (Xi == Yi) | (Xi == Ni_minus1)
            sign_i = np.where(Xi == Ni_minus1, -1, 1)

            factor = sign_i * mask_in      # bool -> 0/1 * sign
            num *= factor

        return num

    # ==========================================
    # 3. Main Logic (Matrix Construction)
    # ==========================================

    def _build_base_matrix(self):
        """
        Main function constructing the base matrix.
        
        Returns:
            The base matrix (n, columns).
        """
        # Pre-calculation of grids and subsets
        combs = self._generate_tuples()  # (n, d)
        subsets = self._generate_subsets()
        n = combs.shape[0]

        # First column: 1 (intercept)
        cols = [np.ones((n, 1), dtype=float)]

        # Loop over each non-empty subset S
        for S in subsets[1:]:
            Y_S = self._generate_y(S)        # (m, d)
            m = Y_S.shape[0]
            
            if m == 0:
                continue

            den = self._compute_denominators(combs, S) # (n,)
            num = self._compute_numerators(combs, S, Y_S)    # (n, m)

            # psi_S(x,y) = num(x,y) / P(X_S = x_S)
            # Note: handling division by zero if necessary, here assuming P > 0 on support
            with np.errstate(divide='ignore', invalid='ignore'):
                psi_S = num / den[:, None]
                psi_S = np.nan_to_num(psi_S) # Safety check if null probability

            cols.append(psi_S)

        return np.concatenate(cols, axis=1)

    def get_obliq_matrix(self):
        """
        Computes the oblique matrix by solving the linear system against f(X).
        """
        X_numpy = self._generate_tuples()
        base_matrix = self._build_base_matrix()
        
        # Apply function f to the grid
        y = self.f(X_numpy)
        
        # Solve for coefficients
        coeff = np.linalg.solve(base_matrix, y)
        
        obliq_matrix = base_matrix * coeff
        return obliq_matrix

    def get_anova_full(self):
        """
        Computes the full support ANOVA decomposition.
        Requires 'aggregate_duplicate_coalitions' to be defined externally.
        """
        matrix = self.get_obliq_matrix()
        sets = self._repeated_subsets()
        
        # Calling the external function as requested
        anova_indep = aggregate_duplicate_coalitions(sets, matrix)
        return anova_indep