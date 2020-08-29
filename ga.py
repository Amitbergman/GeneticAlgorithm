import numpy

def cal_pop_fitness(estimator, numberOfIterations, pop):
    # Calculating the fitness value of each solution in the current population.
    fitness = []
    for tax_policy in pop:
        fitness.append(calculate_estimated_fitness(tax_policy, estimator, numberOfIterations))
    return fitness

def calculate_estimated_fitness(tax_policy, estimator, numberOfIterations):
    results = []
    for i in range(numberOfIterations):
        results.append(estimator(tax_policy))
    return numpy.average(results) 

def select_mating_pool(pop, fitness, num_parents):
    # Selecting the best individuals in the current generation as parents for producing the offspring of the next generation.
    parents = numpy.empty((num_parents, pop.shape[1]))
    for parent_num in range(num_parents):
        max_fitness_idx = numpy.where(fitness == numpy.max(fitness))
        max_fitness_idx = max_fitness_idx[0][0]
        parents[parent_num, :] = pop[max_fitness_idx, :]
        fitness[max_fitness_idx] = -99999999999
    return parents

def crossover(parents, numberOfOffsprings, offspring_size):
    # Combine pairs of random tax policies to create new generation
    assert (len(parents) > 1)
    children = numpy.empty((numberOfOffsprings,offspring_size))
    for i in range(numberOfOffsprings):
        
        firstParentIndex = numpy.random.randint(len(parents))
        secondParentIndex = numpy.random.randint(len(parents))
        while (secondParentIndex == firstParentIndex):
            secondParentIndex = numpy.random.randint(len(parents))
        children[i]=createChild(parents[firstParentIndex], parents[secondParentIndex])

    return children

def createChild(parent_a, parent_b):
    return (parent_a+parent_b)/2

def mutation(offspring_crossover):
    mutated_children = numpy.empty(offspring_crossover.shape)

    # Mutation changes a single gene in each offspring randomly.
    for idx in range(offspring_crossover.shape[0]):
        # The random value to be added to the gene.
        random_index = numpy.random.randint(offspring_crossover.shape[1])
        random_value = numpy.random.uniform(-10, 10, 1)
        offspring_crossover[idx, random_index] = (offspring_crossover[idx, random_index]  + random_value)%100
    return offspring_crossover