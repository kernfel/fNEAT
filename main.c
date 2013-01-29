#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include "params.h"

#include "cppn.h"
#include "neat.h"

void dump_CPPN( CPPN *net );

int main() {
	struct NEAT_Params params;
	Population pop;
	CPPN seed;
	
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
	params.num_activations = 10;

	params.innov_counter = 0;
	params.species_counter = 0;


	srand(time(0));
	create_CPPN( &seed, 1, (enum CPPNFunc[1]){CF_SIGMOID}, (int[1]){1}, 0, &params );
	create_Population( &pop, &params, &seed );
	delete_CPPN( &seed );

	int i,j,k;
	double test[4][3] = {
		{0.0, 0.0, 0.0},
		{0.0, 1.0, 1.0},
		{1.0, 0.0, 0.0},
		{1.0, 1.0, 1.0}
	};
	double d,r=0.0;
	int winner;
	for ( i=0; i<20; i++ ) {
		double best_score=0.0;
		for ( j=0; j<pop.num_members; j++ ) {
			putchar( '.' );
			pop.members[j].score = 4.0;
			for ( k=0; k<4; k++ ) {
				read_CPPN( &pop.members[j].genotype, &params, test[k], &r );
				d = r-test[k][2];
				pop.members[j].score -= d*d;
			}
			if ( pop.members[j].score < 0 )
				pop.members[j].score = 0.0;
			if ( pop.members[j].score > best_score ) {
				best_score = pop.members[j].score;
				winner = j;
			}
		}
		printf( "\n---------\nGeneration %d - Winner is #%d with a score of %.2f:\n----\n", i, winner, best_score );
		dump_CPPN( &pop.members[winner].genotype );
		epoch( &pop, &params );
	}

	delete_Population( &pop );
	
	return 0;
}

void dump_CPPN( CPPN *net ) {
	int i;
	for ( i=0; i<net->num_links; i++ ) {
		printf(
			"[%04d] %2d --> %2d  (%.2f)\n",
			net->links[i].innov_id,
			net->links[i].from,
			net->links[i].to,
			net->links[i].weight
		);	
	}
}

