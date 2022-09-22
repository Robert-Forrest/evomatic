import numpy as np
import pandas as pd

import evomatic as evo
from . import fitness


def hold_tournaments(alloys):

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

            winner = fitness.compare_candidates(
                contestants.iloc[0], contestants.iloc[1]
            )
            winners.append(winner)
            tmp_alloys.drop([winner.name])

        else:
            break

    return pd.DataFrame(winners)
