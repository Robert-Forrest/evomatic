"""Module providing competition operations."""

from typing import Optional

import numpy as np
import pandas as pd

import evomatic as evo


def tournaments(
    alloys: pd.DataFrame, selection_percentage: float, tournament_size: int
) -> pd.DataFrame:
    """Holds tournaments between members of the population, winners progress to
    the recombination stage, losers are discarded.

    :group: genetic.operators.competition

    Parameters
    ----------

    alloys
        The current population of alloy candidates.
    selection_percentage
        The percentage of the population to be selected as winners, to enter
        the next generation.
    tournament_size
        The number of alloys to compete in each tournament.

    """

    tmp_alloys = alloys.copy()
    tmp_alloys = tmp_alloys.sort_values("rank")

    winners = []
    num_winners = int(np.floor(len(tmp_alloys) * selection_percentage))
    while len(winners) < num_winners:
        if len(tmp_alloys) >= tournament_size:

            contestants = tmp_alloys.sample(n=tournament_size, replace=False)

            winner = evo.fitness.compare_candidates(
                contestants.iloc[0], contestants.iloc[1]
            )
            if winner is not None:
                winners.append(winner)
                tmp_alloys.drop([winner.name])

        else:
            break

    return pd.DataFrame(winners)
