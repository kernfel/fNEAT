#include <stdio.h>
#include <time.h>

#include "params.h"

#include "cppn.h"
#include "neat.h"

int main() {
	struct NEAT_Params params;
	Population pop;
	Individual seed;
	
	params.flags = CFL_USE_BIAS;
	params.num_dimensions = 1;

	params.population_size = 100;
	params.extinction_threshold = 5;
	params.survival_quota = 0.2;

	params.speciation_threshold = 6.0;
	params.disjoint_factor = 2.0;
	params.excess_factor = 2.0;
	params.weight_factor = 1.0;

	params.add_link_prob = 0.1;
	params.add_node_prob = 0.1;
	params.change_weight_prob = 0.9;
	params.change_weight_rate = 1.0;
	params.enable_link_prob = 0.1;
	params.crossover_prob = 0.2;

	params.allowed_funcs = (enum CPPNFunc[1]){ CF_SIGMOID };
	params.num_allowed_funcs = 1;

	params.innov_counter = 0;
	params.species_counter = 0;


	create_CPPN( &seed.genotype, 1, (enum CPPNFunc[1]){CF_SIGMOID}, (int[1]){1}, 0, &params );
	
	create_Population( &pop, &params, &seed );
	
	delete_CPPN( &seed.genotype );

	delete_Population( &pop );
	
	return 0;
}

