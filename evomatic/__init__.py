import pandas as pd
import numpy as np
import elementy

from .genetic import evolve

parameters = None


def setup(in_parameters):
    global parameters

    parameters = in_parameters

    if 'targets' not in parameters:
        print("No targets set.")
        exit()
    elif 'maximise' not in parameters['targets'] and 'minimise' not in parameters['targets']:
        print("No targets set.")
        exit()

    if 'minimise' not in parameters['targets']:
        parameters['targets']['minimise'] = []
    if 'maximise' not in parameters['targets']:
        parameters['targets']['maximise'] = []
    for direction in ['minimise', 'maximise']:
        if direction not in parameters['targets']:
            targets[direction] = []

    parameters['target_normalisation'] = {}
    for target in parameters['targets']['maximise']+parameters['targets']['minimise']:
        parameters['target_normalisation'][target] = {
            'max': -np.Inf, 'min': np.Inf}

    if 'constraints' in parameters:

        required_elements = parameters['constraints']['elements']
        max_elements = parameters['constraints']['max_elements']
        if 'min_elements' in parameters['constraints']:
            min_elements = parameters['constraints']['min_elements']
        else:
            min_elements = 1

        disallowed_elements = parameters['constraints']['disallowed_elements']
        disallowed_properties = parameters['constraints']['disallowed_properties']

        allow_other_elements = False
        if len(required_elements) < 2:
            allow_other_elements = True
        elif len(required_elements) < max_elements:
            allow_other_elements = True

        allowed_elements = list(required_elements.keys())
        if allow_other_elements:
            allowed_elements = [e.symbol for e in elementy.elements.ELEMENTS]
            if len(disallowed_properties) > 0:
                elementData = features.getElementData()
                for element in allowed_elements:
                    if element not in disallowed_elements:
                        for item in disallowed_properties:
                            if item['property'] in elementData[element]:
                                if elementData[element][item['property']] == item['value']:
                                    disallowed_elements.append(element)

            if len(disallowed_elements) > 0:
                for element in disallowed_elements:
                    if element in allowed_elements:
                        allowed_elements.remove(element)

        parameters['allowed_elements'] = allowed_elements

        parameters['requirements'] = parse_required_elements(required_elements)

        if not allow_other_elements:
            if max_elements > len(required_elements):
                max_elements = len(required_elements)

    else:
        parameters['constraints'] = {}
        parameters['constraints']['min_elements'] = 1
        parameters['constraints']['max_elements'] = 8
        parameters['constraints']['allowed_elements'] = [
            e.symbol for e in elementy.elements.ELEMENTS]
        parameters['constraints']['elements'] = {}

    if 'max_iterations' not in parameters:
        parameters['max_iterations'] = 100
    if 'min_iterations' not in parameters:
        parameters['min_iterations'] = 10
    if 'convergence_window' not in parameters:
        parameters['convergence_window'] = 10
    if 'convergence_tolerance' not in parameters:
        parameters['convergence_tolerance'] = 0.01

    if 'selection_percentage' not in parameters:
        parameters['selection_percentage'] = 0.5
    if 'tournament_size' not in parameters:
        parameters['tournament_size'] = 2

    if 'recombination_rate' not in parameters:
        parameters['recombination_rate'] = 0.9

    if 'mutation_rate' not in parameters:
        parameters['mutation_rate'] = 0.05

    if 'percentage_step' not in parameters:
        parameters['percentage_step'] = percentage_step = 0.01

    if 'plot' not in parameters:
        parameters['plot'] = False
    else:
        parameters['output_directory'] = './'

    parameters['sigfigs'] = -int(f'{percentage_step:e}'.split('e')[-1])


def setup_history():
    global parameters

    history = {
        'averageComposition': [],
        'alloys': pd.DataFrame({})
    }
    for target in parameters['targets']['minimise'] +\
            parameters['targets']['maximise']:

        history[target] = []

    return history
