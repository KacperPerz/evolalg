import copy
import os
import random
from typing import List, Callable, Union
from random import randrange

import numpy as np

from evolalg.base.step import Step
import pickle
import time

from evolalg.base.union_step import UnionStep
from evolalg.selection.selection import Selection
from evolalg.utils.stable_generation import StableGeneration
import logging


class MultiExperiment:
    def __init__(self, init_population: List[Callable],
                 selection: Selection,
                 new_generation_steps: List[Union[Callable, Step]],
                 generation_modification: List[Union[Callable, Step]],
                 end_steps: List[Union[Callable, Step]],
                 population_size,
                 tournament_size,
                 when_merge,
                 subpop_num,
                 split_method,
                 checkpoint_path=None, checkpoint_interval=None):

        self.init_population = init_population
        self.running_time = 0
        self.step = StableGeneration(
            selection=selection,
            steps=new_generation_steps,
            population_size=population_size//subpop_num)
        self.generation_modification = UnionStep(generation_modification)

        self.end_steps = UnionStep(end_steps)

        self.checkpoint_path = checkpoint_path
        self.checkpoint_interval = checkpoint_interval
        self.generation = 0
        self.population = None
        self.tournament_size = tournament_size
        self.subpopulations = None
        self.when_merge = when_merge
        self.split_method = split_method
        self.subpop_num = subpop_num

    def init(self):
        self.generation = 0
        for s in self.init_population:
            if isinstance(s, Step):
                s.init()

        self.step.init()
        self.generation_modification.init()
        self.end_steps.init()
        self.population = []
        for s in self.init_population:
            self.population = s(self.population)

    def sub_the_population(self, splitting_method, num_groups):
        # equal number allocation
        if splitting_method == "ena":
            population = sorted(self.population, key=lambda x: getattr(x, "fitness"))
            return np.array_split(population, num_groups)

        # equal range allocation
        if splitting_method == "era":
            temp_population = copy.deepcopy(self.population)
            res = []
            number_of_elements = len(temp_population) // num_groups

            for i in range(num_groups):
                elems = [temp_population.pop(randrange(len(temp_population))) for _ in range(number_of_elements)]
                res.append(elems)
            return res

        # equal width allocation
        if splitting_method == "ewa":
            best = max(self.population, key=lambda x: getattr(x, "fitness"))
            worst = min(self.population, key=lambda x: getattr(x, "fitness"))

            intervals = np.linspace(getattr(worst, "fitness"), getattr(best, "fitness"), num_groups + 1)
            temp_population = copy.deepcopy(self.population)
            res = []
            elems = []

            # It's worth to notice that the same result can be reached through popping individuals from
            # temp_population after they are added to elems list. With current approach the complexity is O(n^2).

            # [0 ; x] (x ; 2x] ... ((n-1)x ; nx] - the pattern of dividing fitness into intervals.
            # The first interval is closed from right and left to ensure that the first subpopulation
            # will never be empty. Each subsequent interval is closed from left to ensure that
            # the last interval never will lose the best solution.
            for individual in temp_population:
                if intervals[0] <= getattr(individual, "fitness") <= intervals[1]:
                    elems.append(individual)
            res.append(elems)

            for i in range(1, len(intervals[:-1])):
                elems = []
                for individual in temp_population:
                    if intervals[i] < getattr(individual, "fitness") <= intervals[i + 1]:
                        elems.append(individual)
                if len(elems) == 0:
                    elems = res[i - 1]
                res.append(elems)

            return res

    def run(self, num_generations):
        flag = 1

        for i in range(self.generation + 1, num_generations + 1):
            print("GENERATION N.", i)
            start_time = time.time()
            self.generation = i

            # periodic splitting
            if flag != 1 and (i % self.when_merge) - 1 == 0:
                self.subpopulations = self.sub_the_population(self.split_method, self.subpop_num)

            # initial splitting (executed once)
            if flag == 1:
                self.subpopulations = self.sub_the_population(self.split_method, self.subpop_num)
                flag = 0

            # statistics for each subpopulation
            print("BEFORE")
            for subp in self.subpopulations:
                self.generation_modification(subp)

            # operations on each subpopulation
            self.subpopulations = [self.step(subp) for subp in self.subpopulations]

            print("AFTER")
            # statistics for each subpopulation
            for subp in self.subpopulations:
                self.generation_modification(subp)

            # merging subpopulations
            # statistics for merged population
            if i % self.when_merge == 0 or i == num_generations:
                self.population = [ind for subp in self.subpopulations for ind in subp]
                print("STATISTICS FOR MERGED POPULATION")
                self.population = self.generation_modification(self.population)

            self.running_time += time.time() - start_time
            if (self.checkpoint_path is not None
                    and self.checkpoint_interval is not None
                    and i % self.checkpoint_interval == 0):
                self.save_checkpoint()

        self.population = self.end_steps(self.population)

        # self.check_checkpoint()

    def save_checkpoint(self):
        tmp_filepath = self.checkpoint_path + "+pop_size=" + str(len(self.population)) + "+subpop_num=" + str(self.subpop_num) + "+when_merge=" + str(self.when_merge)
        try:
            with open(tmp_filepath, "wb") as file:
                pickle.dump(self, file)
            os.replace(tmp_filepath,
                       tmp_filepath)  # ensures the new file was first saved OK (e.g. enough free space on device), then replace
        except Exception as ex:
            raise RuntimeError(
                "Failed to save checkpoint '%s' (because: %s). This does not prevent the experiment from continuing, but let's stop here to fix the problem with saving checkpoints." % (
                tmp_filepath, ex))

    # def check_checkpoint(self):
    #    with open('C:\\framsticks\\library\\framspy\\checkpoints', 'rb') as f:
    #        data = pickle.load(f)
    #        with open('foo.txt', 'w') as r:
    #
    #        print(data)
    @staticmethod
    def restore(path):
        with open(path) as file:
            res = pickle.load(file)
        return res
