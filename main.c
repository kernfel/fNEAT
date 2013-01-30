#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>

#include "params.h"

#include "cppn.h"
#include "neat.h"

void dump_CPPN( CPPN *net );

struct NEAT_Params get_params() {
	struct NEAT_Params params;
	
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
	
	return params;
}

void xor() {
	struct NEAT_Params params = get_params();
	Population pop;
	CPPN seed;

	srand(time(0));
	create_CPPN( &seed, 1, (enum CPPNFunc[1]){CF_SIGMOID}, (int[1]){1}, 0, &params );
	create_Population( &pop, &params, &seed );
	delete_CPPN( &seed );

	int i,j,k;
	double test[4][3] = {
		{0.0, 0.0, 0.0},
		{0.0, 1.0, 1.0},
		{1.0, 0.0, 1.0},
		{1.0, 1.0, 0.0}
	};
	double d,result[4]={0.0}, best_result[4];
	int winner;
	for ( i=0; i<20; i++ ) {
		double best_score=0.0;
		for ( j=0; j<pop.num_members; j++ ) {
			putchar( '.' );
			pop.members[j].score = 4.0;
			for ( k=0; k<4; k++ ) {
				read_CPPN( &pop.members[j].genotype, &params, test[k], &result[k] );
				d = result[k]-test[k][2];
				pop.members[j].score -= d*d;
			}
			if ( pop.members[j].score < 0 )
				pop.members[j].score = 0.0;
			if ( pop.members[j].score > best_score ) {
				best_score = pop.members[j].score;
				memcpy( best_result, result, 4*sizeof *result );
				winner = j;
			}
		}
		printf( "\n---------\nGeneration %d - Winner is #%d with a score of %.2f:\n", i, winner, best_score );
		for ( k=0; k<4; k++ )
			printf( "%1.0f^%1.0f=%.2f  ", test[k][0], test[k][1], best_result[k] );
		putchar( '\n' );
		dump_CPPN( &pop.members[winner].genotype );
		epoch( &pop, &params );
	}

	delete_Population( &pop );
}

void eval_xor( Individual *dude, struct NEAT_Params *params, int record, int verbose ) {
	double test[4][3] = {
		{0.0, 0.0, 0.0},
		{0.0, 1.0, 1.0},
		{1.0, 0.0, 1.0},
		{1.0, 1.0, 0.0}
	};
	double d, result[4]={0.0};
	int j;
	dude->score = 4.0;
	for ( j=0; j<4; j++ ) {
		read_CPPN( &dude->genotype, params, test[j], &result[j] );
		d = result[j]-test[j][2];
		if ( record )
			dude->score -= d*d;
		if ( verbose )
			printf( "%1.0f^%1.0f->%.2f, (%+.2f off)\n", test[j][0], test[j][1], result[j], d );
	}
}

void functest() {
	struct NEAT_Params params = get_params();
	Population pop;
	CPPN seed;
	int err=0;
	
	params.population_size = 100;
	params.extinction_threshold = 2;
	params.allowed_funcs = (enum CPPNFunc[]){ CF_LINEAR };
	params.crossover_prob = 0;
	
	if (( err = create_CPPN( &seed, 1, params.allowed_funcs, (int[]){1}, 1, &params ) )
	 || ( err = create_Population( &pop, &params, &seed ) ))
		exit(err);
	delete_CPPN(&seed);
	
	
	int generation=0, i, j, winner;
	while (++generation) {
		printf( "\n-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*\nGeneration %d -- %d species present: ", generation, pop.num_species );
		for ( i=0; i<pop.num_species; i++ ) {
			printf( "%d of #%d", pop.species_size[i], pop.species_ids[i] );
			if ( i<pop.num_species-1 )
				printf( ", " );
		}
		printf( "\n" );
		double best_score = 0.0;
		for ( i=0; i<pop.num_members; i++ ) {
			for ( j=0; j<pop.num_species; j++ )
				if ( pop.members[i].species_id == pop.species_ids[j] )
					break;
/*			printf( "\nIndividual #%d (Species #%d) topology:\n", i+1, pop.species_ids[j] );*/
/*			dump_CPPN( &pop.members[i].genotype );*/
/*			printf( "Test results as follows:\n" );*/
			eval_xor( &pop.members[i], &params, 1, 0 );
			if ( pop.members[i].score < 0 )
				pop.members[i].score = 0;
			if ( pop.members[i].score > best_score )
				winner = i;
/*			printf( "Total score: %.2f/4\n", pop.members[i].score );*/
/*			while(!getchar());*/
		}
		
		for ( j=0; j<pop.num_species; j++ )
			if ( pop.members[winner].species_id == pop.species_ids[j] )
				break;
		printf( "The winner with score %.2f belongs to species #%d, here he is:\n", pop.members[winner].score, pop.species_ids[j] );
		dump_CPPN( &pop.members[winner].genotype );
		eval_xor( &pop.members[winner], &params, 0, 1 );
		
		while(!getchar());
		if (( err = epoch( &pop, &params ) ))
			exit(err);
	}
}

void dump_CPPN( CPPN *net ) {
	int i;
	for ( i=0; i<net->num_links; i++ ) {
		printf(
			"[%04d] %2d --%c %2d  (%+.2f)\n",
			net->links[i].innov_id,
			net->links[i].from,
			net->links[i].is_disabled ? 'x' : '>',
			net->links[i].to,
			net->links[i].weight
		);	
	}
}

int main() {
	srand(time(0));
	functest();
	return 0;
}

