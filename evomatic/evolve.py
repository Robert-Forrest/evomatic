"""Module providing the basic evolutionary algorithm."""

import itertools
from typing import Optional

import pandas as pd
import numpy as np
import metallurgy as mg

import evomatic as evo


class Evolver:
    """Evolver class

    :group: evolver
    """

    def __init__(
        self,
        targets,
        population_size: int = 100,
        max_iterations: int = 200,
        min_iterations: int = 10,
        convergence_window: int = 10,
        convergence_tolerance: float = 0.01,
        constraints: Optional[dict] = None,
        competition_type: str = "tournament",
        selection_percentage: float = 0.5,
        tournament_size: Optional[int] = None,
        recombination_rate: float = 0.9,
        mutation_rate: float = 0.05,
        temperature: float = 100,
        cooling_rate: float = 0.9,
        model: Optional = None,
        model_uncertainty: float = False,
        verbosity: int = 1,
    ):
        """Evolver class

        :group: evolver

        Attributes
        ----------

        history
            The history dict, containing data from each iteration of the
            algorithm.
        targets
            Values to be optimised, either by maximisation or minimisation.
        population_size
            Number of candidates to form the population.
        max_iterations
            Maximum number of iterations the evolution can run for.
        min_iterations
            Minimum number of iterations the evolution must run for.
        convergence_window
            Number of iterations over which to measure for convergence.
        convergence_tolerance
            Tolerance of percentage increase in target performance used to
            detect convergence.
        constraints
            The constraints on alloy compositions that can be generated.
        competition_type
            The type of competition operator to apply.
        selection_percentage
            The percentage of the population to be selected by the competition
            operator.
        tournament_size
            The number of alloys to compete in each tournament.
        recombination_rate
            Percentage chance of recombination ocurring rather than simple
            copying of the parents into the next generation.
        mutation_rate
            The percentage chance that mutation occurs per candidate.
        temperature
            The effective temperature applied during simulated annealing.
        cooling_rate
            The rate at which the cooling schedule reduces the annealing temperature.
        model
            Cerebral model to use for on-the-fly predictions.
        model_uncertainty
            If True, Cerebral model predictions will be returned with
            uncertainty estimates.
        verbosity
            Determines the amount of output from evomatic (0=none, 1=all)

        """

        self.setup_targets(targets)
        self.setup_constraints(constraints)

        self.population_size = population_size
        self.max_iterations = max_iterations
        self.min_iterations = min([min_iterations, self.max_iterations])
        self.convergence_tolerance = convergence_tolerance
        self.convergence_window = convergence_window
        self.verbosity = verbosity

        self.competition_type = competition_type
        self.selection_percentage = selection_percentage
        if self.competition_type == "tournament":
            if tournament_size is None:
                self.tournament_size = 2
            else:
                self.tournament_size = tournament_size
        elif tournament_size is not None:
            raise ValueError(
                "Can only set tournament_size when competition_type is tournament"
            )

        self.recombination_rate = recombination_rate
        self.mutation_rate = mutation_rate

        self.temperature = temperature
        self.initial_temperature = temperature
        self.cooling_rate = cooling_rate

        self.model = model
        self.model_uncertainty = model_uncertainty
        if self.model is not None:
            import cerebral as cb

            mg.set_model(cb.models.load(self.model))

        self.setup_history()

    def setup_constraints(self, constraints: dict):
        """Sets up the constraints for alloy compositions during evolution.

        :group: utils

        Parameters
        ----------

        constraints
            The constraints on alloy compositions that can be generated.

        """
        if constraints is not None:

            if "max_elements" not in constraints:
                constraints["max_elements"] = 8

            if "min_elements" not in constraints:
                constraints["min_elements"] = 1

            if "allowed_elements" not in constraints:
                constraints["allowed_elements"] = [
                    e for e in mg.periodic_table.elements
                ]

            if "percentages" in constraints:
                allow_other_elements = False
                if (
                    len(constraints["percentages"]) < 2
                    or len(constraints["percentages"])
                    < constraints["max_elements"]
                ):
                    allow_other_elements = True

            else:
                allow_other_elements = True
                constraints["percentages"] = {}

            if allow_other_elements:
                if "disallowed_properties" in constraints:
                    if len(constraints["disallowed_properties"]) > 0:
                        for element in constraints["allowed_elements"]:
                            if (
                                element
                                not in constraints["disallowed_elements"]
                            ):
                                for item in constraints[
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
                                            constraints[
                                                "disallowed_elements"
                                            ].append(element)

                if "disallowed_elements" in constraints:
                    if len(constraints["disallowed_elements"]) > 0:
                        for element in constraints["disallowed_elements"]:
                            if element in constraints["allowed_elements"]:
                                constraints["allowed_elements"].remove(element)
            else:
                if constraints["max_elements"] > len(
                    constraints["percentages"]
                ):
                    constraints["max_elements"] = len(
                        constraints["percentages"]
                    )

                constraints["allowed_elements"] = list(
                    constraints["percentages"].keys()
                )

        else:
            constraints = {}
            constraints["min_elements"] = 1
            constraints["max_elements"] = 8
            constraints["allowed_elements"] = [
                e for e in mg.periodic_table.elements
            ]
            constraints["percentages"] = {}
        self.constraints = constraints

    def setup_targets(self, targets: dict):
        """Sets up the targets and target_normalisation attributes.

        :group: utils

        Parameters
        ----------

        targets
            Dict of targets for maximisation and minimisation.

        """

        self.targets = targets
        if self.targets is None or (
            "maximise" not in self.targets and "minimise" not in self.targets
        ):
            raise ValueError("No targets set.")

        if "minimise" not in self.targets:
            self.targets["minimise"] = []
        if "maximise" not in self.targets:
            self.targets["maximise"] = []
        for direction in ["minimise", "maximise"]:
            if direction not in self.targets:
                self.targets[direction] = []

        self.target_normalisation = {}
        for target in self.targets["maximise"] + self.targets["minimise"]:
            self.target_normalisation[target] = {
                "max": -np.Inf,
                "min": np.Inf,
            }

    def setup_history(self) -> dict:
        """Sets up the history object with empty data structures, to be filled in
        during an evolution.

        :group: utils
        """

        self.history = {"average_alloy": [], "alloys": pd.DataFrame({})}

        for target in self.targets["minimise"] + self.targets["maximise"]:
            self.history[target] = []

    def immigrate(self, num_immigrants: int) -> pd.DataFrame:
        """Creates a number of new random alloy compositions to join the population.

        :group: genetic

        Parameters
        ----------

        num_immigrants
            The number of new alloy compostions to be generated.
        """

        new_alloys = []

        for _ in range(num_immigrants):
            immigrant = mg.generate.random_alloy(
                min_elements=self.constraints["min_elements"],
                max_elements=self.constraints["max_elements"],
                percentage_constraints=self.constraints["percentages"],
                allowed_elements=self.constraints["allowed_elements"],
                constrain_alloy=True,
            )

            new_alloys.append({"alloy": immigrant})

        return pd.DataFrame(new_alloys)

    def check_converged(self) -> bool:
        """Determines if the evolutionary algorithm has converged, based on the
        improvement of performance on targets over recent history.

        :group: utils
        """

        converged = [False] * len(self.target_normalisation)
        j = 0
        for target in self.target_normalisation:
            converged_target = [False] * (self.convergence_window - 1)
            if len(self.history[target]) > self.convergence_window:

                # if target in self.targets["minimise"]:
                #     direction = "min"
                # else:
                #     direction = "max"
                direction = "average"

                tolerance = np.abs(
                    self.convergence_tolerance
                    * self.history[target][-1][direction]
                )
                for i in range(1, self.convergence_window):
                    if (
                        np.abs(
                            self.history[target][-1][direction]
                            - self.history[target][-1 - i][direction]
                        )
                        < tolerance
                    ):
                        converged_target[i - 1] = True

            if np.all(converged_target):
                converged[j] = True
            j += 1

        return np.all(converged)

    def compete(self) -> pd.DataFrame:
        """Applies the competition operator to the alloy candidate population.

        :group: genetic.operators.competition
        """

        if self.competition_type == "tournament":
            return evo.competition.tournaments(
                self.alloys, self.selection_percentage, self.tournament_size
            )
        else:
            raise NotImplementedError

    def accumulate_history(self) -> dict:
        """Appends data from the most recent iteration of the evolutionary algorithm
        to the history dictionary.

        :group: utils
        """

        for target in self.targets["minimise"] + self.targets["maximise"]:
            self.history[target].append(
                {
                    "average": np.average(self.alloys[target]),
                    "max": np.max(self.alloys[target]),
                    "min": np.min(self.alloys[target]),
                }
            )

        self.history["alloys"] = pd.concat(
            [self.history["alloys"], self.alloys], ignore_index=True
        )

        self.history["alloys"] = self.history["alloys"].drop_duplicates(
            subset="alloy"
        )

        total_composition = {}
        for _, row in self.alloys.iterrows():
            alloy = mg.Alloy(row["alloy"])
            for element in alloy.composition:
                if element not in total_composition:
                    total_composition[element] = 0
                total_composition[element] += alloy.composition[element]
        for element in total_composition:
            total_composition[element] /= len(self.alloys)
        self.history["average_alloy"].append(total_composition)

    def make_new_generation(self) -> pd.DataFrame:
        """Applies the genetic operators to the current population, creating the
        next generation.

        :group: genetic

        Parameters
        ----------

        alloys
            The current population of alloy candidates.

        """

        parents = self.compete()

        children = evo.recombination.recombine(
            parents, self.recombination_rate
        )

        children = evo.mutation.mutate(
            children, self.mutation_rate, self.constraints
        ).drop_duplicates(subset="alloy")

        while len(children) < self.population_size:
            immigrants = self.immigrate(self.population_size - len(children))
            children = pd.concat(
                [children, immigrants], ignore_index=True
            ).drop_duplicates(subset="alloy")

        return children

    def evolve(self) -> dict:
        """Runs the evolutionary algorithm, generating new candidates until
        performance on target objectives has converged.

        Returns the history dictionary, containing data from each iteration of
        the algorithm.

        :group: genetic

        """

        self.alloys = self.immigrate(self.population_size)
        self.alloys["generation"] = 0
        self.alloys = evo.fitness.calculate_features(
            self.alloys, self.targets, uncertainty=self.model_uncertainty
        )
        self.alloys = evo.fitness.calculate_fitnesses(
            self.alloys, self.targets, self.target_normalisation
        )

        sort_columns = ["rank"]
        sort_directions = [True]
        if len(self.targets["maximise"] + self.targets["minimise"]) > 1:
            sort_columns.append("crowding")
            sort_directions.append(False)

        iteration = 0
        converged = False
        while (
            iteration < self.max_iterations and not converged
        ) or iteration < self.min_iterations:

            if (
                iteration > self.convergence_window
                and iteration > self.min_iterations
            ):
                converged = self.check_converged()
                if converged:
                    break

            if iteration < self.max_iterations - 1:
                children = self.make_new_generation()

                children = evo.fitness.calculate_features(
                    children, self.targets, uncertainty=self.model_uncertainty
                )
                children = evo.fitness.calculate_fitnesses(
                    children, self.targets, self.target_normalisation
                )

                if self.temperature > 0:
                    children = evo.annealing.anneal(
                        children,
                        self.temperature,
                        self.constraints,
                        self.targets,
                        self.target_normalisation,
                    )

                    self.temperature = (
                        self.initial_temperature
                        * self.cooling_rate**iteration
                    )

                children["generation"] = iteration

                self.alloys = self.alloys.sort_values(
                    by=sort_columns, ascending=sort_directions
                ).head(self.population_size)

                self.alloys = pd.concat(
                    [self.alloys, children], ignore_index=True
                ).drop_duplicates(subset="alloy")
                self.alloys = evo.fitness.calculate_fitnesses(
                    self.alloys, self.targets, self.target_normalisation
                )

                # while len(self.alloys) < 2 * self.population_size:
                #     immigrants = self.immigrate(
                #         2 * self.population_size - len(self.alloys)
                #     )
                #     self.alloys = pd.concat(
                #         [self.alloys, immigrants], ignore_index=True
                #     ).drop_duplicates(subset="alloy")

            self.accumulate_history()

            if self.verbosity > 0:
                self.output_progress()

            iteration += 1

        self.history["alloys"] = evo.fitness.calculate_comparible_fitnesses(
            self.history["alloys"], self.targets, self.target_normalisation
        ).sort_values("fitness", ascending=False)

        return self.history

    def output_progress(self):
        """Prints a string summarising the current alloy population's performance on
        each of the targets.

        :group: utils

        """

        stats_string = "T=" + str(round(self.temperature, 4)) + ", "
        for target in self.targets["minimise"] + self.targets["maximise"]:
            stats_string += (
                target
                + ": "
                + str(round(self.history[target][-1]["min"], 4))
                + ":"
                + str(round(self.history[target][-1]["average"], 4))
                + ":"
                + str(round(self.history[target][-1]["max"], 4))
                + ", "
            )
        stats_string = stats_string[:-2]

        print(
            "Generation "
            + str(len(self.history[target]))
            + ", population:"
            + str(len(self.alloys))
            + ", "
            + stats_string
        )

    def output_results(self, output_directory="./"):
        """Triggers writing of output files, including raw data and plot images.

        :group: utils

        Parameters
        ----------

        output_directory
            The path to write output files into.

        """

        output_directory = evo.plots.ensure_output_directory(output_directory)

        self.write_output_file(output_directory)

        evo.plots.plot_targets(self.history, self.targets, output_directory)
        evo.plots.plot_alloy_percentages(self.history, output_directory)
        evo.plots.plot_per_element_targets(
            self.history, self.targets, output_directory
        )

        for pair in itertools.combinations(
            self.targets["minimise"] + self.targets["maximise"],
            2,
        ):
            # plots.pareto_front_plot(self.history, pair)
            for i in range(10):
                evo.plots.pareto_plot(
                    self.history,
                    pair,
                    self.targets,
                    self.target_normalisation,
                    top_percentage=round((i + 1) / 10, 1),
                    output_directory=output_directory,
                )

    def write_output_file(self, output_directory="./"):
        """Writes output files.

        :group: utils

        Parameters
        ----------

        output_directory
            Location to write output files.

        """

        with open(output_directory + "genetic.dat", "w") as genetic_file:
            header_string = "# rank generation alloy fitness"
            for target in self.targets["minimise"] + self.targets["maximise"]:
                header_string += " " + target
                if target + "_uncertainty" in self.history["alloys"]:
                    header_string += " " + target + "_uncertainty"

            genetic_file.write(header_string + "\n")

            i = 0
            for _, row in self.history["alloys"].iterrows():
                stats_string = str(round(row["fitness"], 4)) + " "
                for target in (
                    self.targets["minimise"] + self.targets["maximise"]
                ):
                    stats_string += str(round(row[target], 4)) + " "
                    if target + "_uncertainty" in row:
                        stats_string += (
                            str(round(row[target + "_uncertainty"], 4)) + " "
                        )

                genetic_file.write(
                    str(i)
                    + " "
                    + str(row["generation"])
                    + " "
                    + mg.Alloy(row["alloy"]).to_string()
                    + " "
                    + stats_string
                    + "\n"
                )
                i += 1

        elements = []
        for i in range(len(self.history["average_alloy"])):
            for element in self.history["average_alloy"][i]:
                if element not in elements:
                    elements.append(element)

        with open(
            output_directory + "genetic_extended.dat", "w"
        ) as genetic_file:
            genetic_file.write(
                "# rank generation "
                + " ".join(elements)
                + " fitness "
                + " ".join(self.targets["minimise"] + self.targets["maximise"])
                + "\n"
            )

            i = 0
            for _, row in self.history["alloys"].iterrows():
                stats_string = str(round(row["fitness"], 4)) + " "
                for target in (
                    self.targets["minimise"] + self.targets["maximise"]
                ):
                    stats_string += str(round(row[target], 4)) + " "

                percentage_string = ""
                alloy = mg.Alloy(row["alloy"])
                for element in elements:
                    if element in alloy.composition:
                        percentage_string += str(alloy.composition[element])
                    else:
                        percentage_string += "0"
                    percentage_string += " "
                percentage_string = percentage_string[:-1]

                genetic_file.write(
                    str(i)
                    + " "
                    + str(row["generation"])
                    + " "
                    + percentage_string
                    + " "
                    + stats_string
                    + "\n"
                )
                i += 1
