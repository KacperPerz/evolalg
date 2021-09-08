import os
from typing import List, Callable, Union

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
                 when_merge,
                 checkpoint_path=None, checkpoint_interval=None):

        self.init_population = init_population
        self.running_time = 0
        self.step = StableGeneration(
            selection=selection,
            steps=new_generation_steps,
            population_size=population_size)
        self.generation_modification = UnionStep(generation_modification)

        self.end_steps = UnionStep(end_steps)

        self.checkpoint_path = checkpoint_path
        self.checkpoint_interval = checkpoint_interval
        self.generation = 0
        self.population = None
        self.subpopulations = None
        self.when_merge = when_merge

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

    def sub_the_population(self):  # TODO add an option to choose which criterion of splitting is used
        population = sorted(self.population, key=lambda x: getattr(x, "fitness"))
        return np.array_split(population, 5)  # TODO add non-static parameter for the number of divisions

    def run(self, num_generations):
        flag = 1
        for i in range(self.generation + 1, num_generations + 1):
            print("GENERATION N.", i)
            start_time = time.time()
            self.generation = i

            # periodic splitting
            if flag != 1 and i % (self.when_merge + 1) == 0:
                self.subpopulations = self.sub_the_population()

            # initial splitting (executed once)
            if flag == 1:
                self.subpopulations = self.sub_the_population()
                flag = 0

            # operations on each subpopulation
            self.subpopulations = [self.step(subp) for subp in self.subpopulations]  # TODO decrease number of generated individuals in class StableGeneration

            # statistics for each subpopulation
            for subp in self.subpopulations:
                self.generation_modification(subp)

            # merging subpopulations
            # statistics for merged population
            if i % self.when_merge == 0 or i == num_generations:
                self.population = np.array(self.subpopulations).ravel() # TODO create counter indicating when subpopulations should be merged
                print("STATISTICS FOR MERGED POPULATION")
                self.population = self.generation_modification(self.population) # TODO statistics for each subpopulation and for merged population

            self.running_time += time.time() - start_time
            if (self.checkpoint_path is not None
                    and self.checkpoint_interval is not None
                    and i % self.checkpoint_interval == 0):
                self.save_checkpoint()

        self.population = self.end_steps(self.population)

    def save_checkpoint(self):
        tmp_filepath = self.checkpoint_path+"_tmp"
        try:
            with open(tmp_filepath, "wb") as file:
                pickle.dump(self, file)
            os.replace(tmp_filepath, self.checkpoint_path)  # ensures the new file was first saved OK (e.g. enough free space on device), then replace
        except Exception as ex:
            raise RuntimeError("Failed to save checkpoint '%s' (because: %s). This does not prevent the experiment from continuing, but let's stop here to fix the problem with saving checkpoints." % (tmp_filepath, ex))


    @staticmethod
    def restore(path):
        with open(path) as file:
            res = pickle.load(file)
        return res
