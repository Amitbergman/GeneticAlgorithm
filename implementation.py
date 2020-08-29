import numpy
import ga

def calculateNumberOfIterationFromGeneration(iteration):
    return iteration

number_of_genes = 7

estimator = lambda tax_policy: numpy.dot(tax_policy, [1,-2,3,-4,3,4,5]) + numpy.random.uniform(-20, 20, 1)
real_function = lambda tax_policy: numpy.dot(tax_policy, [1,-2,3,-4,3,4,5])

def runIteration(num_generations, size_of_initial_population, goal_to_stop_at, function_estimator):
    assert (size_of_initial_population > 3)
    populationOfTaxPolicies = numpy.random.uniform(low=0, high=100.0, size=(size_of_initial_population,number_of_genes))
    maxer = 0
    for generation in range(1,num_generations+1):
        # Measuring the fitness of each chromosome in the population.
        fitness = ga.cal_pop_fitness(function_estimator, calculateNumberOfIterationFromGeneration(generation), populationOfTaxPolicies)

        # Selecting the best parents in the population for mating.
        parents = ga.select_mating_pool(populationOfTaxPolicies, fitness, 
                                        int(size_of_initial_population/2))

        # Generating next generation using crossover.
        offspring_crossover = ga.crossover(parents, int(size_of_initial_population/2), number_of_genes)
        # Adding some variations to the offsrping using mutation.
        offspring_mutation = ga.mutation(offspring_crossover)

        # Creating the new population based on the parents and offspring.
        populationOfTaxPolicies[0:parents.shape[0], :] = parents
        populationOfTaxPolicies[parents.shape[0]:, :] = offspring_mutation

        # The best result in the current iteration.
        fitness = ga.cal_pop_fitness(function_estimator, calculateNumberOfIterationFromGeneration(generation), populationOfTaxPolicies)
        if (numpy.max(fitness) > goal_to_stop_at):
            print("Goal achieved in generation:", generation)
            break
        if (numpy.max(fitness) > maxer):
            maxer = numpy.max(fitness)
            max_solution = populationOfTaxPolicies[numpy.argmax(fitness)]
            print("Current maximum is", maxer)

runIteration(500, 100, 1590, estimator)
def justEstimateWithRandomSamples(numberOfTimesToSample=1000000):
    result = 0
    numberOfSamples = 0
    while (result < 1590 and numberOfSamples < numberOfTimesToSample):
        random_tax = numpy.random.uniform(low=0, high=100.0, size=(1,number_of_genes))
        numberOfSamples +=1
        random_result = estimator(random_tax)
        if (random_result > result):
            result = random_result
            print(result, random_tax, numberOfSamples)