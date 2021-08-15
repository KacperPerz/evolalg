from evolalg.base.individual import Individual
from typing import List
import random
import math
import numpy as np
from random import randrange

from evolalg.base.step import Step
from evolalg.selection.selection import Selection

class ConvectionSelection(Selection):
    def __init__(self, tournament_size: int, number_of_divisions : int, fit_attr="fitness", copy=False, *args, **kwargs):
        super(ConvectionSelection, self).__init__(copy, *args, **kwargs)
        self.tournament_size = tournament_size
        self.number_of_divisions = number_of_divisions
        self.fit_attr = fit_attr
    
    def select_next(self, population):
        def sub_the_population(population):
            population = sorted(population, key=lambda x: getattr(x, self.fit_attr)) #sortuję według fitnessu
            res = np.array_split(population,self.number_of_divisions)
            return res
            
        subpopulations = sub_the_population(population)
        res = []
        for subp in subpopulations:
            selected = [random.choice(subp) for i in range(self.tournament_size)]
            res.append(max(selected, key=lambda x: getattr(x, self.fit_attr)))
        
        return max(res, key=lambda x: getattr(x, self.fit_attr))

    def randomAllocation(self, population, num_groups):
        ret = []
        temp_population = copy.deepcopy(population)
        for i in range(len(num_groups)):
            number_of_elements = len(temp_population) // (num_groups - i)
            elems = [temp_population.pop(randrange(len(temp_population))) for _ in range(number_of_elements)] #czy zabezpieczone przed indeksem -1 ? tak
            ret.append(elems)
        return ret
        
    def equalWidthAllocation(self, population, num_groups):
        best = max(population, key=lambda x: getattr(x, self.fit_attr)) 
        worst = min(population, key=lambda x: getattr(x, self.fit_attr))
        ret = []
        for i in range(num_groups):
            elems = []
            mini = getattr(worst, self.fit_attr) + (getattr(best, self.fit_attr) - getattr(worst, self.fit_attr)) * g / num_groups
            maxi = getattr(worst, self.fit_attr) + (getattr(best, self.fit_attr) - getattr(worst, self.fit_attr)) * (g+1) / num_groups
            
            for j in range(len(population)):
                if getattr(population[j], self.fit_attr) >= mini and getattr(population[j], self.fit_attr) <= maxi:
                    elems.append(population[j])
            if len(elems) == 0:
                if i == 0:
                    elems.append(population.index(worst))
                else:
                    elems = ret[len(ret) - 1]
            ret.append(elems)
            
                
            
            
        

