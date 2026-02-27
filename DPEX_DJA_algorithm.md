Initialize population A = {a1, a2, ..., aN} randomly
Initialize population B = {b1, b2, ..., bN} randomly

Evaluate fitness of all solutions in A and B
Set global_best = best solution in (A ∪ B)

Initialize Tabu lists for population B

FOR t = 1 TO T DO

    # ----- Part A: EA-DJA Global Search -----
    Identify best_A and worst_A in population A

    FOR each solution ai in A DO
        Generate candidate ai' using DJA update rule:
            ai' = ai + rand() * (best_A - |ai|)
                      - rand() * (worst_A - |ai|)
        Repair ai' to feasible discrete solution
        IF f(ai') ≤ f(ai) THEN
            ai = ai'
        END IF
    END FOR

    # ----- Part B: Tabu-Enhanced Local Search -----
    FOR each selected elite solution bj in B DO
        Generate k neighbors using problem-specific move
        Exclude moves in Tabu list unless aspiration holds
        Select best neighbor bj'
        Update Tabu list with applied move
        IF f(bj') ≤ f(bj) THEN
            bj = bj'
        END IF
    END FOR

    # ----- Part C: Dual-Population Exchange -----
    IF t mod T_ex == 0 THEN
        Select m best solutions from A
        Select m worst solutions from B
        Exchange selected solutions (preserve population size)
    END IF

    # ----- Global Update -----
    Update global_best if improved

END FOR

RETURN global_best
