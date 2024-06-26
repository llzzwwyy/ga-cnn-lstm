
from __future__ import print_function

from evolver import Evolver

from tqdm import tqdm

import logging

import sys

# Setup logging.
logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%m/%d/%Y %I:%M:%S %p',
    level=logging.INFO#,
    #filename='log.txt'
)

def train_genomes(genomes):
    """Train each genome

    """
    logging.info("***train_networks(networks, dataset)***")

    pbar = tqdm(total=len(genomes))

    for genome in genomes:
        genome.train()
        pbar.update(1)
    
    pbar.close()

def get_average_accuracy(genomes):
    """Get the average accuracy for a group of networks/genomes.

    """
    total_accuracy = 0
    max_acc = 0
    for genome in genomes:
        if genome.accuracy > max_acc:
            max_acc = genome.accuracy
        total_accuracy += genome.accuracy

    return total_accuracy / len(genomes), max_acc

def generate(generations, population, all_possible_genes):
    """Generate a network with the genetic algorithm.

    Args:
        generations (int): Number of times to evolve the population
        population (int): Number of networks in each generation
        all_possible_genes (dict): Parameter choices for networks
        dataset (str): Dataset to use for training/evaluating

    """
    logging.info("***generate(generations, population, all_possible_genes, dataset)***")
    
    evolver = Evolver(all_possible_genes)
    
    genomes = evolver.create_population(population)

    avg_accs = []
    max_accs = []

    # Evolve the generation.
    for i in range(generations):

        logging.info("***Now in generation %d of %d***" % (i + 1, generations))

        print_genomes(genomes)
        
        # Train and get accuracy for networks/genomes.
        train_genomes(genomes)

        # Get the average accuracy for this generation.
        average_accuracy, max_acc = get_average_accuracy(genomes)
        avg_accs.append(average_accuracy)
        max_accs.append(max_acc)

        # Print out the average accuracy each generation.
        logging.info("Generation average: %.2f%%" % (average_accuracy * 100))
        logging.info("Generation max acc: %.2f%%" % (max_acc * 100))
        logging.info('-'*80) #-----------

        # Evolve, except on the last iteration.
        if i != generations - 1:
            # Evolve!
            genomes = evolver.evolve(genomes)

    # Sort our final population according to performance.
    genomes = sorted(genomes, key=lambda x: x.accuracy, reverse=True)

    # Print out the top 5 networks/genomes.
    print_genomes(genomes[:10])
    with open('top_genomes.txt', 'w') as file:
        for genome in genomes[:10]:
            file.write(str(genome.geneparam) + '\n')
            file.write(str(genome.accuracy) + '\n')

    print('avg acc of each generation: ', avg_accs)
    print('max acc of each generation: ', max_accs)

    with open('result.txt', 'w', encoding='utf8') as f:
        f.write('avg acc of each generation: ' + str(avg_accs) + '\n')
        f.write('max acc of each generation: ' + str(max_accs))
    #save_path = saver.save(sess, '/output/model.ckpt')
    #print("Model saved in file: %s" % save_path)

def print_genomes(genomes):
    """Print a list of genomes.

    Args:
        genomes (list): The population of networks/genomes

    """
    logging.info('-'*80)

    for genome in genomes:
        genome.print_genome()

def main():
    """Evolve a genome."""
    population = 30 # Number of networks/genomes in each generation

    generations = 20 # Number of times to evolve the population.


    all_possible_genes = {
        'filter': [32, 64, 128, 256],
        'filter1': [32, 64, 128, 256],
        'filter2': [32, 64, 128, 256],
        'filter3': [32, 64, 128, 256],
        'kernel_size': [3, 4, 5, 6],
        'kernel_size1': [3, 4, 5, 6],
        'kernel_size2': [3, 4, 5, 6],
        'kernel_size3': [3, 4, 5, 6],
        'activation': ['relu', 'tanh'],
        # 'optimizer':  ['rmsprop', 'adam'],
        'lr': [0.0005, 0.001, 0.002, 0.003],
        'batch_size': [32, 64, 128, 256],
        'epoch': [25, 30, 35, 40],
        # 'layer_size': [1, 2, 3, 4],
        'layer_size': [1, 2, 3],
        'dropout': [0.1, 0.2, 0.3],
        'lstm_units': [64, 128, 256, 512],
        'lstm_units1': [64, 128, 256, 512],
        'layer_size1': [1, 2]
    }

    print("***Evolving for %d generations with population size = %d***" % (generations, population))

    generate(generations, population, all_possible_genes)

if __name__ == '__main__':
    main()
