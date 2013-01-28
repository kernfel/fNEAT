#ifndef _NEAT_H
#define _NEAT_H


typedef struct Individual {
	CPPN genotype;
	int score;
	unsigned int species_id;
} Individual;

typedef struct Population {
	Individual *members;
	int num_members;

	unsigned int *species_ids;
	int *species_size;
	int num_species;
} Population;


// ** Constructors
// Spawn an population from scratch, using prototype as a seed
int create_Population( Population *pop, struct NEAT_Params *parameters, const Individual *prototype );

// Allocate memory for an entire population, guided by the num_* values
int allocate_Population( Population *pop );

// ** Destructor
void delete_Population( Population *pop );

// ** Public methods

// Set up a new generation as per the NEAT algorithm and the gathered evaluation data
int epoch( Population *pop, struct NEAT_Params *params );

// ** Private methods

// Apply mutation (without crossover) to each member of the population
int mutate_population( Population *pop, struct NEAT_Params *params );

// Divide the population into species, disregarding any individual's species_id
// representatives must not point to members of the population lest they be overwritten.
int speciate_population( Population *pop, struct NEAT_Params *params, const Individual *representatives );


void get_population_fertility( Population *pop, struct NEAT_Params *params, int *num_offspring );
int reproduce_population( Population *pop, struct NEAT_Params *params, int *num_offspring, Individual *reps );
Individual ***get_population_ranking( Population *pop, int *err );

#endif

