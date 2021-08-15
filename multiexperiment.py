import os
from typing import List, Callable, Union

from evolalg.base.step import Step
import pickle
import time

from evolalg.base.union_step import UnionStep
from evolalg.selection.selection import Selection
from evolalg.utils.stable_generation import StableGeneration
import logging

class Experiment:
    def __init__(self, init_population: List[List[Callable]], #lista list
                 selection: Selection,
                 new_generation_steps: List[Union[Callable, Step]],
                 generation_modification: List[Union[Callable, Step]],
                 end_steps: List[Union[Callable, Step]],
                 population_size,
                 how_many_subp, #ile podpopulacji
                 checkpoint_path=None, checkpoint_interval=None):

        self.init_population = init_population
        self.running_time = 0
        self.step = StableGeneration(
            selection=selection,
            steps=new_generation_steps,
            population_size=population_size)
        self.generation_modification = UnionStep(generation_modification)

        self.end_steps = UnionStep(end_steps) #czy UnionStep sie nadaje do podpopulacji?

        self.checkpoint_path = checkpoint_path
        self.checkpoint_interval = checkpoint_interval
        self.generation = 0
        self.population = None

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
            self.population = s(self.population) #co to robi?

    def run(self, num_generations):
        
        for i in range(self.generation + 1, num_generations + 1):
            self.generation = i
            start_time = time.time()
            
            #czy da się tu wprowadzić wielowątkowość? czy zamienić na map?
            for j in range(how_many_subp): 
                self.population[j] = self.step(self.population[j]) #wykonuje kroki dla każdej podpopulacji
                self.population[j] = self.generation_modification(self.population[j])
                
            #KIEDY MERGOWAĆ?    
                
            self.running_time += time.time() - start_time    
            if (self.checkpoint_path is not None
                    and self.checkpoint_interval is not None
                    and i % self.checkpoint_interval == 0):
                self.save_checkpoint()

        for i in range(how_many_subp):
            self.population[i] = self.end_steps(self.population[i])

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
