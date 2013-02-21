#ifndef _NEAT_H
#define _NEAT_H


typedef struct Individual {
	CPPN genotype;
	double score;
	unsigned int species_id;
	unsigned int id;
} Individual;

typedef struct Species {
	unsigned int id;
	int size;
	int stagnating_for;
	double mean_score;
} Species;

typedef struct Population {
	Individual *members;
	int num_members;

	Species *species;
	int num_species;
} Population;


// ** Constructors
// Spawn an population from scratch, using prototype as a seed. No mutation or randomisation is performed.
int create_Population( Population *pop, struct NEAT_Params *parameters, const CPPN *prototype );

// Allocate memory for an entire population, guided by the num_* values
int allocate_Population( Population *pop );

// ** Destructor
void delete_Population( Population *pop );

// ** Public methods

// Set up a new generation as per the NEAT algorithm and the gathered evaluation data
int epoch( Population *pop, struct NEAT_Params *params );

// Randomise the connection weights of each member of the population
void randomise_population( Population *pop, struct NEAT_Params *params );

// ** Private methods

// Initialise the next generation based on data from the current generation's evaluation
// representatives is populated with copies of individuals from the current generation to allow for speciation
int reproduce_population( Population *pop, struct NEAT_Params *params, Individual *representatives, Individual **champions );

// Apply mutation (without crossover) to each member of the population
int mutate_population( Population *pop, struct NEAT_Params *params, Individual **champions );

// Divide the population into species
// Individuals are preferentially kept within the species indicated by their species_id, if still compatible.
int speciate_population( Population *pop, struct NEAT_Params *params, const Individual *representatives );

// Determine the number of offspring each species is allowed
int get_population_fertility( Population *pop, struct NEAT_Params *params, int *num_offspring );

// Returns a 2D array of pointers sorting a population's members by species index (asc), then individual score (desc).
// Individuals are accessed at x[species_index][individual_index].
// At the end of life, simply free() the returned pointer.
Individual ***get_population_ranking( Population *pop, int *err );

#endif

