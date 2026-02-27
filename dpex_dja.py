"""
DPEX_DJA – Dual-Population EXchange with Discrete Jaya Algorithm
=================================================================
Population A  – global exploration via DJA update rule (discrete Jaya)
Population B  – local refinement via tabu-enhanced neighbourhood search
Exchange      – periodically injects best-of-A into B, evicts worst-of-B

Reference pseudocode: DPEX_DJA_algorithm.md

Algorithm structure
-------------------
FOR each iteration t:
    Part A  : apply DJA update to every member of pop_A
                  ai' = ai + r1*(best_A - |ai|) - r2*(worst_A - |ai|)
              (discrete: probabilistic component-slot attraction/repulsion)
    Part B  : tabu-enhanced local search on pop_B elites
              generate k neighbours per elite via synthon similarity,
              block tabu moves unless aspiration holds
    Part C  : every T_ex iters, exchange m best-of-A into B
              (pop_B is trimmed to N_B after merge)
    Global  : accumulate all scored candidates into top_pool
"""
from __future__ import annotations

import random
import bittensor as bt
import pandas as pd
from collections import deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

from molecules import MoleculeManager
from tools import SynthonLibrary

# ── tunables ──────────────────────────────────────────────────────────────────
N_A_DEFAULT  = 200   # population A capacity  (moving-window of scored mols)
N_B_DEFAULT  = 100   # population B capacity  (elite pool for tabu search)
T_EX_DEFAULT = 3     # exchange every T_ex iterations
M_EX_DEFAULT = 20    # molecules exchanged per cycle
TABU_MAXLEN  = 50    # maximum tabu entries per component role
# ──────────────────────────────────────────────────────────────────────────────


@dataclass
class DPEXDJAState:
    """Persistent DPEX_DJA state carried across iterations."""
    pop_A:    List[Dict]        = field(default_factory=list)
    pop_B:    List[Dict]        = field(default_factory=list)
    tabu:     Dict[str, deque]  = field(default_factory=lambda: {
        'A': deque(maxlen=TABU_MAXLEN),
        'B': deque(maxlen=TABU_MAXLEN),
        'C': deque(maxlen=TABU_MAXLEN),
    })
    N_A:      int = N_A_DEFAULT
    N_B:      int = N_B_DEFAULT
    T_ex:     int = T_EX_DEFAULT
    m_ex:     int = M_EX_DEFAULT
    iteration: int = 0


# ── helpers ───────────────────────────────────────────────────────────────────

def _parse(name: str) -> Tuple[Optional[int], Optional[int], Optional[int], Optional[int]]:
    """Return (rxn_id, A_id, B_id, C_id).  All-None on parse failure."""
    parts = name.split(":")
    if len(parts) < 4:
        return None, None, None, None
    try:
        return (
            int(parts[1]),
            int(parts[2]),
            int(parts[3]),
            int(parts[4]) if len(parts) > 4 else None,
        )
    except (ValueError, IndexError):
        return None, None, None, None


def _build(rxn: int, A: int, B: int, C: Optional[int]) -> str:
    return f"rxn:{rxn}:{A}:{B}" if C is None else f"rxn:{rxn}:{A}:{B}:{C}"


# ── Part A – DJA global update ────────────────────────────────────────────────

def _dja_move(
    name:      str,
    best_name: str,
    worst_name: str,
    manager:   MoleculeManager,
    avoid:     Set[str],
) -> Optional[str]:
    """
    Discrete DJA update rule (per component slot):

        val = current_id
        if rand() > 0.5  →  val = best's component_id        (attraction)
        if rand() > 0.5 AND val == worst's component_id
                         →  val = random from pool             (repulsion)

    Returns None when the move yields no change or a seen molecule.
    """
    rxn, A,  B,  C  = _parse(name)
    _,   bA, bB, bC = _parse(best_name)
    _,   wA, wB, wC = _parse(worst_name)

    if rxn is None or bA is None or wA is None:
        return None

    def _step(cur: int, best: int, worst: int, pool: List[int]) -> int:
        val = cur
        if random.random() > 0.5:
            val = best
        if random.random() > 0.5 and val == worst:
            val = random.choice(pool)
        return val

    nA = _step(A, bA, wA, manager.moles_A_id)
    nB = _step(B, bB, wB, manager.moles_B_id)

    nC: Optional[int] = None
    if manager.is_three_component and C is not None:
        nC = _step(
            C,
            bC if bC is not None else C,
            wC if wC is not None else C,
            manager.moles_C_id,
        )

    new_name = _build(rxn, nA, nB, nC)
    return None if (new_name == name or new_name in avoid) else new_name


def dja_generate(
    state:    DPEXDJAState,
    manager:  MoleculeManager,
    n_samples: int,
    avoid:    Set[str],
) -> pd.DataFrame:
    """
    Part A – DJA global search.

    Applies the DJA update rule to every member of pop_A once.
    Shortfalls are filled by re-perturbing randomly selected members.
    Returns an *unvalidated* DataFrame with column 'name'.
    """
    if not state.pop_A:
        return pd.DataFrame(columns=["name"])

    by_score  = sorted(state.pop_A, key=lambda x: x.get('score', float('-inf')), reverse=True)
    best_mol  = by_score[0]
    worst_mol = by_score[-1]

    new_names: Set[str] = set()

    # One DJA step per population member
    for mol in state.pop_A:
        if len(new_names) >= n_samples:
            break
        n = _dja_move(mol['name'], best_mol['name'], worst_mol['name'], manager, avoid)
        if n:
            new_names.add(n)

    # Fill shortfall with re-perturbed randoms
    attempts = 0
    while len(new_names) < n_samples and attempts < n_samples * 4:
        attempts += 1
        mol = random.choice(state.pop_A)
        n = _dja_move(mol['name'], best_mol['name'], worst_mol['name'], manager, avoid)
        if n and n not in new_names:
            new_names.add(n)

    return pd.DataFrame({"name": list(new_names)}) if new_names else pd.DataFrame(columns=["name"])


# ── Part B – tabu-enhanced local search ──────────────────────────────────────

def _tabu_hit(tabu_set: Set[Tuple[int, int]], old_id: int, new_id: int) -> bool:
    return (old_id, new_id) in tabu_set


def tabu_generate(
    state:             DPEXDJAState,
    synthon_lib:       SynthonLibrary,
    manager:           MoleculeManager,
    avoid:             Set[str],
    k_per_elite:       int   = 15,
    global_best_score: float = float('-inf'),
) -> Tuple[pd.DataFrame, List[Tuple[str, int, int]]]:
    """
    Part B – tabu-enhanced local search on pop_B elites.

    For each elite molecule, synthon similarity generates k candidate neighbours.
    A tabu move (recently tried component swap) is blocked unless aspiration holds:
        - candidate is unseen (not in avoid)
        - elite sits at ≥ 97 % of the current global best score

    Returns:
        (DataFrame of candidate molecule names, list of applied moves)
    Applied moves are (role, old_id, new_id) tuples for tabu list update.
    """
    if not state.pop_B or synthon_lib is None:
        return pd.DataFrame(columns=["name"]), []

    # Build fast-lookup tabu sets from the deques
    tabu_sets: Dict[str, Set] = {r: set(state.tabu[r]) for r in ('A', 'B', 'C')}

    new_names:    List[str]                   = []
    applied_moves: List[Tuple[str, int, int]] = []

    n_elites = min(10, len(state.pop_B))

    for mol in state.pop_B[:n_elites]:
        rxn, A, B, C = _parse(mol['name'])
        mol_score    = mol.get('score', float('-inf'))
        if rxn is None:
            continue

        # Aspiration: allow tabu moves when this elite is near the global best
        aspiration = (
            global_best_score > float('-inf')
            and mol_score >= global_best_score * 0.97
        )

        similar = synthon_lib.find_similar_to_molecule_name(
            mol['name'],
            vary_component='both' if not manager.is_three_component else 'all',
            top_k_per_component=k_per_elite,
            min_similarity=0.50,
        )

        for new_A in similar.get('A', [])[:k_per_elite]:
            nn = _build(rxn, new_A, B, C)
            if _tabu_hit(tabu_sets['A'], A, new_A) and not aspiration:
                continue
            if nn not in avoid and nn not in new_names:
                new_names.append(nn)
                applied_moves.append(('A', A, new_A))

        for new_B in similar.get('B', [])[:k_per_elite]:
            nn = _build(rxn, A, new_B, C)
            if _tabu_hit(tabu_sets['B'], B, new_B) and not aspiration:
                continue
            if nn not in avoid and nn not in new_names:
                new_names.append(nn)
                applied_moves.append(('B', B, new_B))

        if manager.is_three_component and C is not None:
            for new_C in similar.get('C', [])[:k_per_elite]:
                nn = _build(rxn, A, B, new_C)
                if _tabu_hit(tabu_sets['C'], C, new_C) and not aspiration:
                    continue
                if nn not in avoid and nn not in new_names:
                    new_names.append(nn)
                    applied_moves.append(('C', C, new_C))

    return (
        pd.DataFrame({"name": new_names}) if new_names else pd.DataFrame(columns=["name"]),
        applied_moves,
    )


def update_tabu(state: DPEXDJAState, moves: List[Tuple[str, int, int]]) -> None:
    """Record newly applied moves in the tabu list (FIFO, bounded by TABU_MAXLEN)."""
    for role, old_id, new_id in moves:
        if role in state.tabu:
            state.tabu[role].append((old_id, new_id))


# ── Part C – dual-population exchange ────────────────────────────────────────

def dpex_exchange(state: DPEXDJAState) -> None:
    """
    Inject the m_ex best molecules from pop_A into pop_B.
    Deduplicates and trims pop_B back to N_B after re-ranking by score.
    """
    if not state.pop_A or state.m_ex <= 0:
        return

    best_of_A = sorted(
        state.pop_A,
        key=lambda x: x.get('score', float('-inf')),
        reverse=True,
    )[:state.m_ex]

    seen:   Set[str]   = set()
    merged: List[Dict] = []
    for mol in list(state.pop_B) + best_of_A:
        if mol['name'] not in seen:
            seen.add(mol['name'])
            merged.append(mol)

    merged.sort(key=lambda x: x.get('score', float('-inf')), reverse=True)
    state.pop_B = merged[:state.N_B]

    bt.logging.info(
        f"[DPEX] Exchange: injected {state.m_ex} best from A → B  |  pop_B size = {len(state.pop_B)}"
    )


# ── population maintenance ────────────────────────────────────────────────────

def update_populations(
    state:    DPEXDJAState,
    scored_A: pd.DataFrame,
    scored_B: pd.DataFrame,
) -> None:
    """
    Absorb freshly scored candidates into pop_A and pop_B.

    pop_A: replaced entirely by the latest DJA batch (keeps positions "moving").
    pop_B: merged with new tabu results and trimmed to N_B (cumulative elite pool).
    """
    _required = ('name', 'smiles', 'score')

    def _to_records(df: pd.DataFrame) -> List[Dict]:
        if df.empty or not all(c in df.columns for c in _required):
            return []
        cols = [c for c in ('name', 'smiles', 'score', 'target', 'anti') if c in df.columns]
        return df[cols].dropna(subset=['score']).to_dict('records')

    # Pop A: replace with the latest DJA-scored batch
    if not scored_A.empty:
        state.pop_A = sorted(
            _to_records(scored_A),
            key=lambda x: x.get('score', float('-inf')),
            reverse=True,
        )[:state.N_A]

    # Pop B: merge tabu results into existing elite pool
    new_B = _to_records(scored_B)
    if new_B:
        by_name = {mol['name']: mol for mol in state.pop_B}
        for mol in new_B:
            by_name[mol['name']] = mol          # overwrite with fresher score if revisited
        state.pop_B = sorted(
            by_name.values(),
            key=lambda x: x.get('score', float('-inf')),
            reverse=True,
        )[:state.N_B]
