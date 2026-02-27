import os
import time
import json
import pandas as pd
import bittensor as bt
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import nova_ph2

BASE_DIR   = os.path.abspath(os.path.join(os.path.dirname(__file__)))
OUTPUT_DIR = os.environ.get("OUTPUT_DIR", "/output")
DB_PATH    = str(Path(nova_ph2.__file__).resolve().parent / "combinatorial_db" / "molecules.sqlite")
TIME_LIMIT        = 1800
LIMIT_PER_REACTANT = 1000

from molecules import (
    MoleculeManager,
    MoleculeUtils,
)
from tools import (
    IterationParams,
    SynthonLibrary,
    generate_valid_random_molecules,
    cpu_random_candidates_with_similarity,
    build_component_weights,
)
from models import ModelManager
from exploit import (
    get_top_n_unexploited,
    run_exploit,
)
from dpex_dja import (
    DPEXDJAState,
    dja_generate,
    tabu_generate,
    update_tabu,
    dpex_exchange,
    update_populations,
)


def get_config(input_file: str = os.path.join(BASE_DIR, "input.json")):
    with open(input_file, "r") as f:
        d = json.load(f)
    return {**d.get("config", {}), **d.get("challenge", {})}


model_manager    = None
molecule_manager = None


def initialize_solution(config: dict):
    global molecule_manager, model_manager
    molecule_manager = MoleculeManager(config=config, db_path=DB_PATH)
    model_manager    = ModelManager(config)


def find_solution(config: dict, time_start: float):
    """
    Main search loop implementing the DPEX_DJA metaheuristic.

    Generation priority each iteration
    -----------------------------------
    1. Exploit mode   – structure-guided deep search (activates on stagnation)
    2. DPEX-DJA       – runs once pop_A is seeded (iteration ≥ 2)
         Part A: DJA global update on pop_A
         Part B: tabu-enhanced local search on pop_B  (needs synthon lib)
    3. Fallback       – standard random / GA  (iteration 1 or pop_A empty)

    After scoring every iteration:
        - pop_A replaced with latest DJA batch
        - pop_B merged with tabu results, refreshed from top_pool
        - Part C exchange fires every T_ex iterations
    """
    global molecule_manager, model_manager

    iteration = 0
    n_workers = os.cpu_count() or 1
    bt.logging.info(f"[Solution] CPU Workers: {n_workers}")

    params   = IterationParams(config=config)
    dpex     = DPEXDJAState()
    seed_df  = pd.DataFrame(columns=["name", "smiles"])
    top_pool = pd.DataFrame(columns=["name", "smiles", "inchi", "score", "target", "anti"])

    with ProcessPoolExecutor(max_workers=n_workers) as cpu_executor:
        while time.time() - time_start < TIME_LIMIT:
            iteration          += 1
            dpex.iteration      = iteration
            iteration_start     = time.time()
            remaining_time      = TIME_LIMIT - (iteration_start - time_start)

            bt.logging.info(f"[Solution] --- Iteration {iteration} [DPEX-DJA] ---")

            n_samples         = params.get_nsamples_from_time(remaining_time)
            component_weights = (
                build_component_weights(top_pool.head(config["num_molecules"]), molecule_manager.rxn_id)
                if not top_pool.empty else None
            )
            elite_df    = (
                MoleculeUtils.select_diverse_elites(top_pool, min(150, len(top_pool)))
                if not top_pool.empty else pd.DataFrame()
            )
            elite_names = elite_df["name"].tolist() if not elite_df.empty else None

            # ── Exploit mode gate ─────────────────────────────────────────────
            if params.no_improvement_counter >= 2:
                params.use_exploit_mode = True
                bt.logging.info(f"[Solution] === EXPLOIT MODE  (no_improvement={params.no_improvement_counter}) ===")

            # ── Synthon library (built once on iteration 2) ───────────────────
            if iteration >= 2 and params.synthon_lib is None:
                try:
                    bt.logging.info("[Solution] Building synthon library ...")
                    t0 = time.time()
                    params.synthon_lib     = SynthonLibrary(molecule_manager=molecule_manager)
                    params.use_synthon_search = True
                    bt.logging.info(f"[Solution] Synthon library ready in {time.time()-t0:.2f}s")
                except Exception as e:
                    bt.logging.warning(f"[Solution] Synthon library failed: {e}")
                    params.synthon_lib        = None
                    params.use_synthon_search = False

            # ── Refresh pop_B from current top_pool elites ────────────────────
            # This ensures tabu always operates on the best known molecules.
            if not top_pool.empty:
                cols       = [c for c in ('name', 'smiles', 'score', 'target', 'anti') if c in top_pool.columns]
                top_records = top_pool[cols].head(dpex.N_B).to_dict('records')
                # Merge incoming top_pool elites with tabu-discovered molecules
                existing   = {m['name']: m for m in dpex.pop_B}
                for mol in top_records:
                    if mol['name'] not in existing:
                        existing[mol['name']] = mol
                dpex.pop_B = sorted(
                    existing.values(),
                    key=lambda x: x.get('score', float('-inf')),
                    reverse=True,
                )[:dpex.N_B]

            # ─────────────────────────────────────────────────────────────────
            # CANDIDATE GENERATION
            # ─────────────────────────────────────────────────────────────────
            data             = pd.DataFrame(columns=["name", "smiles"])
            data_dja         = pd.DataFrame(columns=["name"])
            data_tabu        = pd.DataFrame(columns=["name"])
            data_tabu_moves: list = []
            exploited_status = False
            exploit_summary  = None

            # ── Priority 1: Exploit ───────────────────────────────────────────
            if params.use_exploit_mode:
                bt.logging.info("[Solution] Exploit: structure-guided deep search ...")
                all_top_mols = top_pool.to_dict("records")
                try:
                    unexploited = get_top_n_unexploited(all_top_mols, params.exploited_reactants)
                    if unexploited:
                        t0 = time.time()
                        exploit_results, exploit_summary = run_exploit(
                            manager=molecule_manager,
                            config=config,
                            top_molecules=unexploited,
                            limit_per_reactant=LIMIT_PER_REACTANT,
                            avoid_names=params.seen_molecules,
                            exploited_reactants=params.exploited_reactants,
                        )
                        bt.logging.info(
                            f"[Solution] Exploit: {len(exploit_results)} candidates "
                            f"in {time.time()-t0:.1f}s"
                        )
                        if exploit_results:
                            data             = pd.DataFrame(exploit_results)
                            exploited_status = True
                        else:
                            raise Exception("Exploit returned no molecules.")
                    else:
                        raise Exception("No unexploited top molecules available.")
                except Exception as e:
                    bt.logging.warning(f"[Solution] Exploit skipped: {e}")

            # ── Priority 2 & 3: DPEX-DJA or fallback ─────────────────────────
            if not exploited_status:

                if iteration == 1 or not dpex.pop_A:
                    # ── Initialisation (iteration 1 / cold start) ─────────────
                    bt.logging.info(
                        f"[Solution] Init: generating {params.n_samples_start} random molecules"
                    )
                    data = generate_valid_random_molecules(
                        config=config,
                        manager=molecule_manager,
                        n_samples=params.n_samples_start,
                        mutation_prob=0,
                        elite_prob=0,
                        executor=cpu_executor,
                        n_workers=n_workers,
                        avoid_names=params.seen_molecules,
                        elite_names=None,
                        component_weights=component_weights,
                    )

                else:
                    # ── Part A: DJA global update ─────────────────────────────
                    n_dja  = int(n_samples * 0.60)
                    n_tabu = n_samples - n_dja

                    bt.logging.info(
                        f"[Solution] DJA: generating {n_dja} candidates "
                        f"(pop_A size = {len(dpex.pop_A)})"
                    )
                    raw_dja = dja_generate(
                        state=dpex,
                        manager=molecule_manager,
                        n_samples=n_dja,
                        avoid=params.seen_molecules,
                    )
                    if not raw_dja.empty:
                        data_dja = molecule_manager.validate_molecules(
                            config, raw_dja,
                            time_elapsed=iteration_start - time_start,
                        )
                        bt.logging.info(f"[Solution] DJA: {len(data_dja)} validated")

                    # ── Part B: tabu local search ─────────────────────────────
                    if params.synthon_lib is not None and dpex.pop_B:
                        global_best = (
                            top_pool['score'].max()
                            if not top_pool.empty else float('-inf')
                        )
                        bt.logging.info(
                            f"[Solution] Tabu: generating candidates from "
                            f"pop_B ({len(dpex.pop_B)} elites), n_tabu≈{n_tabu}"
                        )
                        raw_tabu, data_tabu_moves = tabu_generate(
                            state=dpex,
                            synthon_lib=params.synthon_lib,
                            manager=molecule_manager,
                            avoid=params.seen_molecules,
                            k_per_elite=15,
                            global_best_score=global_best,
                        )
                        if not raw_tabu.empty:
                            data_tabu = molecule_manager.validate_molecules(
                                config, raw_tabu,
                                time_elapsed=iteration_start - time_start,
                            )
                            # Remove any overlap with DJA candidates
                            if not data_dja.empty:
                                data_tabu = data_tabu[
                                    ~data_tabu["name"].isin(data_dja["name"].tolist())
                                ]
                            bt.logging.info(f"[Solution] Tabu: {len(data_tabu)} validated")

                    # Combine DJA + Tabu candidates
                    parts = [df for df in [data_dja, data_tabu] if not df.empty]
                    if parts:
                        data = pd.concat(parts, ignore_index=True).drop_duplicates(subset=["name"])
                    else:
                        # Fallback to GA when both DJA and Tabu produce nothing
                        bt.logging.warning(
                            "[Solution] DJA+Tabu yielded no candidates; falling back to GA"
                        )
                        data = generate_valid_random_molecules(
                            config=config,
                            manager=molecule_manager,
                            n_samples=n_samples,
                            mutation_prob=params.mutation_prob,
                            elite_prob=params.elite_prob,
                            executor=cpu_executor,
                            n_workers=n_workers,
                            avoid_names=params.seen_molecules,
                            elite_names=elite_names,
                            component_weights=component_weights,
                        )

            # ── Generation summary ────────────────────────────────────────────
            gen_time = time.time() - iteration_start
            bt.logging.info(
                f"[Solution] Iteration {iteration}: {len(data)} candidates "
                f"generated in {gen_time:.2f}s (pre-score)"
            )
            if data.empty:
                bt.logging.warning(f"[Solution] No valid molecules this iteration; skipping")
                continue

            # ── Merge seed_df from previous CPU background futures ────────────
            if not seed_df.empty:
                data    = pd.concat([data, seed_df], ignore_index=True).drop_duplicates(subset=["name"])
                seed_df = pd.DataFrame(columns=["name", "smiles"])

            # ── Deduplication vs seen_molecules ──────────────────────────────
            try:
                filtered  = data[~data["name"].isin(params.seen_molecules)]
                dup_ratio = (len(data) - len(filtered)) / max(1, len(data))

                if dup_ratio > 0.7:
                    params.mutation_prob = min(0.9, params.mutation_prob * 1.5)
                    params.elite_prob    = max(0.15, params.elite_prob * 0.7)
                    bt.logging.warning(
                        f"[Solution] SEVERE duplication ({dup_ratio:.2%}) | "
                        f"mut={params.mutation_prob:.2f} elite={params.elite_prob:.2f}"
                    )
                elif dup_ratio > 0.5:
                    params.mutation_prob = min(0.7, params.mutation_prob * 1.3)
                    params.elite_prob    = max(0.2, params.elite_prob * 0.8)
                    bt.logging.warning(
                        f"[Solution] High duplication ({dup_ratio:.2%}) | "
                        f"mut={params.mutation_prob:.2f} elite={params.elite_prob:.2f}"
                    )
                elif dup_ratio < 0.15 and not top_pool.empty and iteration > 10:
                    params.mutation_prob = max(0.05, params.mutation_prob * 0.95)
                    params.elite_prob    = min(0.85, params.elite_prob * 1.05)

                data = filtered
            except Exception as e:
                bt.logging.warning(f"[Solution] Deduplication error: {e}")

            if data.empty:
                bt.logging.error(
                    f"[Solution] All molecules were duplicates; boosting diversity"
                )
                params.mutation_prob = min(0.95, params.mutation_prob * 2.0)
                params.elite_prob    = max(0.10, params.elite_prob * 0.5)
                continue

            data = data.reset_index(drop=True)

            # ── Background CPU similarity futures ─────────────────────────────
            cpu_futures = []
            if not top_pool.empty and iteration > 3 and params.score_improvement_rate < 0.01:
                cpu_futures.append((
                    cpu_executor.submit(
                        cpu_random_candidates_with_similarity,
                        molecule_manager, 40, config,
                        top_pool.head(5)[["name", "smiles"]],
                        params.seen_molecules, 0.80,
                    ), "tight-top5"
                ))
                cpu_futures.append((
                    cpu_executor.submit(
                        cpu_random_candidates_with_similarity,
                        molecule_manager, 30, config,
                        top_pool.head(20)[["name", "smiles"]],
                        params.seen_molecules, 0.65,
                    ), "medium-top20"
                ))

            # ── GPU scoring ───────────────────────────────────────────────────
            bt.logging.info(f"[Solution] Scoring {len(data)} molecules on GPU ...")
            gpu_start       = time.time()
            data["target"]  = model_manager.get_target_score_from_data(data["smiles"])
            data["anti"]    = model_manager.get_antitarget_score()
            data["score"]   = data["target"] - (config["antitarget_weight"] * data["anti"])
            bt.logging.info(f"[Solution] GPU scoring time: {time.time()-gpu_start:.2f}s")

            # ── Collect CPU futures → seed_df ─────────────────────────────────
            if cpu_futures:
                for fut, strategy_name in cpu_futures:
                    try:
                        cpu_df = fut.result(timeout=0)
                        if not cpu_df.empty:
                            seed_df = (
                                pd.concat([seed_df, cpu_df], ignore_index=True)
                                if not seed_df.empty else cpu_df.copy()
                            )
                            bt.logging.info(
                                f"[Solution] CPU ({strategy_name}): {len(cpu_df)} candidates"
                            )
                    except TimeoutError:
                        pass
                    except Exception as e:
                        bt.logging.warning(f"[Solution] CPU ({strategy_name}) failed: {e}")
                if not seed_df.empty:
                    seed_df = seed_df.drop_duplicates(subset=["name"])

            # ── DPEX-DJA population update ────────────────────────────────────
            # Identify rows that came from each sub-strategy for selective update.
            dja_names  = set(data_dja["name"].tolist())  if not data_dja.empty  else set()
            tabu_names = set(data_tabu["name"].tolist()) if not data_tabu.empty else set()

            # On iteration 1 (cold start), the entire scored batch seeds pop_A.
            scored_for_A = data[data["name"].isin(dja_names)]  if dja_names  else data
            scored_for_B = data[data["name"].isin(tabu_names)] if tabu_names else pd.DataFrame(columns=data.columns)

            update_populations(dpex, scored_for_A, scored_for_B)

            # Update tabu list with the moves that were applied this iteration
            if data_tabu_moves:
                update_tabu(dpex, data_tabu_moves)

            # Part C: exchange every T_ex iterations
            if iteration % dpex.T_ex == 0:
                dpex_exchange(dpex)

            bt.logging.debug(
                f"[DPEX] pop_A={len(dpex.pop_A)}  pop_B={len(dpex.pop_B)}"
            )

            # ── top_pool accumulation ─────────────────────────────────────────
            data["inchi"]      = data["smiles"].map(MoleculeUtils.generate_inchikey)
            params.seen_molecules = params.seen_molecules | set(data["name"].tolist())

            prev_avg  = top_pool.head(config["num_molecules"])['score'].mean() if not top_pool.empty else None
            total_data = data[["name", "smiles", "inchi", "score", "target", "anti"]]

            if not total_data.empty:
                top_pool = pd.concat(
                    [top_pool, total_data] if not top_pool.empty else [total_data],
                    ignore_index=True,
                )
                top_pool = top_pool.sort_values(by="score", ascending=False)
                top_pool = top_pool.drop_duplicates(subset=["inchi"], keep="first")
                top_pool = top_pool.head(config["num_molecules"] + 50)
            else:
                bt.logging.warning(f"[Solution] Iteration {iteration}: No valid scored data")

            # ── Score improvement tracking ────────────────────────────────────
            current_avg = (
                top_pool.head(config["num_molecules"])['score'].mean()
                if not top_pool.empty else None
            )
            if current_avg is not None and prev_avg is not None:
                params.score_improvement_rate = (
                    (current_avg - prev_avg) / max(abs(prev_avg), 1e-6)
                )
            elif current_avg is not None:
                params.score_improvement_rate = 1.0

            if params.score_improvement_rate == 0.0:
                params.no_improvement_counter += 1
            else:
                params.no_improvement_counter = 0
                params.use_exploit_mode       = False   # reset exploit on improvement

            if (
                exploit_summary
                and 'exploited_reactant_ids' in exploit_summary
                and params.score_improvement_rate == 0.0
            ):
                params.exploited_reactants.update(exploit_summary['exploited_reactant_ids'])

            # ── Logging ───────────────────────────────────────────────────────
            iter_time  = time.time() - iteration_start
            total_time = time.time() - time_start
            pool_avg   = top_pool.head(config["num_molecules"])['score'].mean()
            pool_max   = top_pool['score'].max()
            try:
                pool_entropy = MoleculeUtils.compute_maccs_entropy(
                    top_pool.head(config["num_molecules"])['smiles'].tolist()
                )
            except Exception:
                pool_entropy = 0.0

            if exploited_status:
                mode_str = "EXPLOIT"
            elif iteration == 1 or not dpex.pop_A:
                mode_str = "INIT"
            elif params.synthon_lib is not None:
                mode_str = "DJA+TABU"
            else:
                mode_str = "DJA"

            bt.logging.info(
                f"Iteration {iteration} | {iter_time:.1f}s | Total: {total_time:.0f}s | "
                f"Mode: {mode_str} | "
                f"popA={len(dpex.pop_A)} popB={len(dpex.pop_B)} | "
                f"Pool: avg={pool_avg:.4f} max={pool_max:.4f} ent={pool_entropy:.3f}"
            )

            # ── Save result ───────────────────────────────────────────────────
            top_entries = {"molecules": top_pool.head(config["num_molecules"])["name"].tolist()}
            if pool_entropy > config['entropy_min_threshold']:
                with open(os.path.join(OUTPUT_DIR, "result.json"), "w") as f:
                    json.dump(top_entries, f, ensure_ascii=False, indent=2)
                bt.logging.info("[Solution] Top entries saved.")


if __name__ == "__main__":
    config     = get_config()
    time_start = time.time()

    initialize_solution(config)
    bt.logging.info(f"[Solution] Init time: {time.time()-time_start:.2f}s")

    find_solution(config, time_start)
