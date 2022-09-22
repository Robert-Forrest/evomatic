import pandas as pd
import numpy as np
import metallurgy as mg

from .genetic import evolve

parameters = None


def setup(in_parameters):
    global parameters

    parameters = in_parameters

    if "targets" not in parameters:
        print("No targets set.")
        exit()
    elif (
        "maximise" not in parameters["targets"]
        and "minimise" not in parameters["targets"]
    ):
        print("No targets set.")
        exit()

    if "minimise" not in parameters["targets"]:
        parameters["targets"]["minimise"] = []
    if "maximise" not in parameters["targets"]:
        parameters["targets"]["maximise"] = []
    for direction in ["minimise", "maximise"]:
        if direction not in parameters["targets"]:
            parameters["targets"][direction] = []

    parameters["timeSinceImprovement"] = 0

    parameters["target_normalisation"] = {}
    for target in (
        parameters["targets"]["maximise"] + parameters["targets"]["minimise"]
    ):
        parameters["target_normalisation"][target] = {
            "max": -np.Inf,
            "min": np.Inf,
        }

    if "constraints" in parameters:

        if "max_elements" not in parameters["constraints"]:
            parameters["constraints"]["max_elements"] = 8

        if "min_elements" not in parameters["constraints"]:
            parameters["constraints"]["min_elements"] = 1

        if "allowed_elements" not in parameters["constraints"]:
            parameters["constraints"]["allowed_elements"] = [
                e for e in mg.periodic_table.elements
            ]

        if "percentages" in parameters["constraints"]:
            allow_other_elements = False
            if (
                len(parameters["constraints"]["percentages"]) < 2
                or len(parameters["constraints"]["percentages"])
                < parameters["constraints"]["max_elements"]
            ):
                allow_other_elements = True

        else:
            allow_other_elements = True
            parameters["constraints"]["percentages"] = {}

        if allow_other_elements:
            if "disallowed_properties" in parameters["constraints"]:
                if len(parameters["constraints"]["disallowed_properties"]) > 0:
                    for element in parameters["allowed_elements"]:
                        if (
                            element
                            not in parameters["constraints"][
                                "disallowed_elements"
                            ]
                        ):
                            for item in parameters["constraints"][
                                "disallowed_properties"
                            ]:
                                if (
                                    item["property"]
                                    in mg.periodic_table.data[element]
                                ):
                                    if (
                                        mg.periodic_table.data[element][
                                            item["property"]
                                        ]
                                        == item["value"]
                                    ):
                                        parameters["constraints"][
                                            "disallowed_elements"
                                        ].append(element)

            if "disallowed_elements" in parameters["constraints"]:
                if len(parameters["constraints"]["disallowed_elements"]) > 0:
                    for element in parameters["constraints"][
                        "disallowed_elements"
                    ]:
                        if (
                            element
                            in parameters["constraints"]["allowed_elements"]
                        ):
                            parameters["constraints"][
                                "allowed_elements"
                            ].remove(element)
        else:
            if parameters["constraints"]["max_elements"] > len(
                parameters["constraints"]["percentages"]
            ):
                parameters["constraints"]["max_elements"] = len(
                    parameters["constraints"]["percentages"]
                )

            parameters["constraints"]["allowed_elements"] = list(
                parameters["constraints"]["percentages"].keys()
            )

    else:
        parameters["constraints"] = {}
        parameters["constraints"]["min_elements"] = 1
        parameters["constraints"]["max_elements"] = 8
        parameters["constraints"]["allowed_elements"] = [
            e for e in mg.periodic_table.elements
        ]
        parameters["constraints"]["percentages"] = {}

    if "max_iterations" not in parameters:
        parameters["max_iterations"] = 100
    if "min_iterations" not in parameters:
        parameters["min_iterations"] = 10
    if "convergence_window" not in parameters:
        parameters["convergence_window"] = 10
    if "convergence_tolerance" not in parameters:
        parameters["convergence_tolerance"] = 0.01

    if "selection_percentage" not in parameters:
        parameters["selection_percentage"] = 0.5
    if "tournament_size" not in parameters:
        parameters["tournament_size"] = 2

    if "recombination_rate" not in parameters:
        parameters["recombination_rate"] = 0.9

    if "mutation_rate" not in parameters:
        parameters["mutation_rate"] = 0.05

    if "percentage_step" not in parameters:
        parameters["percentage_step"] = percentage_step = 0.01

    if "plot" not in parameters:
        parameters["plot"] = False

    if "output_directory" not in parameters:
        parameters["output_directory"] = "./"
    if parameters["output_directory"][-1] != "/":
        parameters["output_directory"] += "/"

    parameters["sigfigs"] = -int(f"{percentage_step:e}".split("e")[-1])


def setup_history():
    global parameters

    history = {"average_alloy": [], "alloys": pd.DataFrame({})}
    for target in (
        parameters["targets"]["minimise"] + parameters["targets"]["maximise"]
    ):

        history[target] = []

    return history
