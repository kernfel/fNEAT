#ifndef _NEAT_H
#define _NEAT_H


typedef struct Individual {
	CPPN *genotype;
	int score;
	int species_id;
} Individual;

typedef struct Population {
	Individual *members;
	int num_species;
	int *species_ids;
} Population;


// ** Constructors
// Spawn an population from scratch, using prototype as a seed
/* NYI */ int create_Population( Population *pop, struct NEAT_Params *parameters, CPPN *prototype );

// Allocate memory for an entire population, without initialising its members
/* NYI */ int allocate_Population( Population *pop, struct NEAT_Params *parameters );

// ** Destructor
/* NYI */ void delete_Population( Population *pop );

// ** Public methods

// Set up a new generation as per the NEAT algorithm and the gathered evaluation data
int epoch( Population *pop, struct NEAT_Params *params );


#endif

