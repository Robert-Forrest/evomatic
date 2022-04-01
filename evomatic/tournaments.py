import numpy as np
import pandas as pd

import evomatic as evo
from . import fitness


def hold_tournaments(compositions):

    tmp_compositions = compositions.copy()
    tmp_compositions = tmp_compositions.sort_values('rank')

    winners = []
    num_winners = int(np.floor(len(tmp_compositions) *
                               evo.parameters['selection_percentage']))
    while len(winners) < num_winners:
        if(len(tmp_compositions) >= evo.parameters['tournament_size']):

            contestants = tmp_compositions.sample(
                n=evo.parameters['tournament_size'], replace=False)

            winner = fitness.compare_candidates(
                contestants.iloc[0], contestants.iloc[1])
            winners.append(winner)
            tmp_compositions.drop([winner.name])

        else:
            break

    return pd.DataFrame(winners)
