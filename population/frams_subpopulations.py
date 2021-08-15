from evolalg.base.frams_step import FramsStep
from evolalg.base.individual import Individual
import numpy as np

class FramsSubpopulations(FramsStep):
    def __init__(self, frams_lib, genetic_format, pop_size, commands=None, how_many_subp, *args, **kwargs):
        
        #jak działa commands? czy są to parametry wpisywane z konsoli? zapytać.
        if commands is None:
            commands = []
        super().__init__(frams_lib, commands,*args, **kwargs)
        
        self.pop_size = pop_size #czy może self.pop_size = pop_size // how_many_subp ? ustalić czym tak naprawdę ma tu być pop_size: liczbą osobników w ogóle czy w podpopulacji
        self.genetic_format = genetic_format
        
        #czy mogę tak przyjmować how_many_subp w __init__ ? czy muszę dodać ten parametr w innej klasie?
        self.how_many_subp = how_many_subp

    def call(self, population, *args, **kwargs):
        super(FramsPopulation, self).call(population) #jak to działa?
        
        population = [Individual(self.frams.getSimplest(self.genetic_format)) for _ in range(self.pop_size * self.how_many_subp)]
        population = sorted(population, key=lambda x: getattr(x, self.fit_attr)) #sortuję według fitnessu
        res = np.array_split(population, self.how_many_subp)
        return res
        
        #jeżeli populacje należy splitować w innej klasie użyć kodu poniżej. PRZEDYSKUTOWAĆ!!!
        #return [[Individual(self.frams.getSimplest(self.genetic_format)) for _ in range(self.pop_size)] for _ in range(self.how_many_subp)]

#alternatywnie: czy można zwracać listę obiektów FramsPopulation zamiast listy list osobników?
#raczej nie, ale przedyskutować

