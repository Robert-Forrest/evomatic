import numpy as np
import pandas as pd

import evomatic as evo


def tournaments(alloys: pd.DataFrame) -> pd.DataFrame:
    """Holds tournaments between members of the population, winners progress to
    the recombination stage, losers are discarded.

    :group: genetic.operators.competition

    Parameters
    ----------

    alloys
        The current population of alloy candidates.

    """

    tmp_alloys = alloys.copy()
    tmp_alloys = tmp_alloys.sort_values("rank")

    winners = []
    num_winners = int(
        np.floor(len(tmp_alloys) * evo.parameters["selection_percentage"])
    )
    while len(winners) < num_winners:
        if len(tmp_alloys) >= evo.parameters["tournament_size"]:

            contestants = tmp_alloys.sample(
                n=evo.parameters["tournament_size"], replace=False
            )

            winner = evo.fitness.compare_candidates(
                contestants.iloc[0], contestants.iloc[1]
            )
            winners.append(winner)
            tmp_alloys.drop([winner.name])

        else:
            break

    return pd.DataFrame(winners)


def compete(alloys: pd.DataFrame) -> pd.DataFrame:
    """Applies the competition operator to the alloy candidate population.

    :group: genetic.operators
    """

    if evo.parameters["competition_type"] == "tournament":
        return tournaments(alloys)
