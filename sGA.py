import random
import math

# Global variables
maxpop = 100
maxstring = 30

class Allele:
    def __init__(self):
        self.chrom = [0] * maxstring
        self.x = 0.0
        self.fitness = 0.0
        self.parent1 = 0
        self.parent2 = 0
        self.end = [0] * maxpop
        self.position = [0] * maxstring

class Genotype:
    def __init__(self):
        self.allele = [Allele() for _ in range(maxpop)]
        self.xsite = 0
        self.individual = 0

class Phenotype:
    def __init__(self):
        self.x = 0.0
        self.fitness = 0.0
        self.parent1 = 0
        self.parent2 = 0
        self.xsite = 0

class Population:
    def __init__(self, size):
        self.population = [Genotype() for _ in range(size)]

def initialize():
    pass  # Add initialization code as needed

def objfunc(x):
    coef = 1073741823.0
    n = 10
    return math.pow(x / coef, n)

def select(popsize, sumfitness, pop):
    rand = random.random() * sumfitness
    partsum = 0.0
    j = 0

    while partsum < rand and j < popsize:
        j += 1
        partsum += pop[j].allele.fitness

    return j

def decode(chrom, lbits):
    accum = 0.0
    powerof2 = 1

    for j in range(1, lbits + 1):
        if chrom[j - 1]:
            accum += powerof2
        powerof2 *= 2

    return accum

def mutation(alleleval, pmutation, nmutation):
    mutate = flip(pmutation)
    if mutate:
        nmutation += 1
        return not alleleval
    else:
        return alleleval

def crossover(parent1, parent2, child1, child2, lchrom, ncross, nmutation, jcross, peross, pmutation):
    if flip(peross):
        jcross = random.randint(1, lchrom - 1)
        ncross += 1
    else:
        jcross = lchrom

    for j in range(1, jcross + 1):
        child1[j - 1] = mutation(parent1[j - 1], pmutation, nmutation)
        child2[j - 1] = mutation(parent2[j - 1], pmutation, nmutation)

    if jcross != lchrom:
        for j in range(jcross, lchrom):
            child1[j] = mutation(parent2[j], pmutation, nmutation)
            child2[j] = mutation(parent1[j], pmutation, nmutation)

def statistics(popsize, max, avg, min, sumfitness, pop):
    sumfitness = pop[0].allele.fitness
    min = pop[0].allele.fitness
    max = pop[0].allele.fitness

    for j in range(1, popsize):
        sumfitness += pop[j].allele.fitness
        if pop[j].allele.fitness > max:
            max = pop[j].allele.fitness
        if pop[j].allele.fitness < min:
            min = pop[j].allele.fitness

    avg = sumfitness / popsize

def flip(prob):
    return random.random() < prob

def generation():
    global popsize, lchrom, gen, maxgen

    j = 1
    while j <= popsize:
        matel = select(popsize, sumfitness, oldpop)
        mate2 = select(popsize, sumfitness, oldpop)
        jcross = 0
        crossover(oldpop[matel - 1].allele.chrom, oldpop[mate2 - 1].allele.chrom,
                  newpop[j - 1].allele.chrom, newpop[j].allele.chrom, lchrom, ncross, nmutation, jcross, peross, pmutation)

        newpop[j - 1].phenotype.x = decode(newpop[j - 1].allele.chrom, lchrom)
        newpop[j - 1].phenotype.fitness = objfunc(newpop[j - 1].phenotype.x)
        newpop[j - 1].phenotype.parent1 = matel
        newpop[j - 1].phenotype.parent2 = mate2
        newpop[j - 1].phenotype.xsite = jcross

        newpop[j].phenotype.x = decode(newpop[j].allele.chrom, lchrom)
        newpop[j].phenotype.fitness = objfunc(newpop[j].phenotype.x)
        newpop[j].phenotype.parent1 = matel
        newpop[j].phenotype.parent2 = mate2
        newpop[j].phenotype.xsite = jcross

        j += 2
