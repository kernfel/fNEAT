#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>
#include <math.h>

#include "params.h"

#include "cppn.h"
#include "neat.h"

void dump_CPPN( CPPN *net );

void get_params( struct NEAT_Params *params ) {
	params->flags = CFL_USE_BIAS | CFL_NO_DISTCALC_NORM;

	params->num_dimensions = 1;
	params->num_outputs = 0;
	params->output_funcs = NULL;
	params->initially_linked_outputs = NULL;

	params->allowed_funcs = NULL;
	params->num_allowed_funcs = 0;
	params->num_activations = 10;

	params->population_size = 100;
	params->extinction_threshold = 5;
	params->survival_quota = 0.2;

	params->speciation_threshold = 3.0;
	params->disjoint_factor = 1.0;
	params->excess_factor = 1.0;
	params->weight_factor = 0.4;

	params->add_link_prob = 0.05;
	params->add_node_prob = 0.03;
	params->change_weight_prob = 0.8;
	params->change_weight_rate = 1.0;
	params->enable_link_prob = 0.1;
	params->crossover_prob = 0.2;

	params->innov_counter = 0;
	params->species_counter = 0;
}

void xor() {
	struct NEAT_Params params;
	Population pop;
	CPPN seed;
	
	get_params( &params );
	params.allowed_funcs = (enum CPPNFunc[]){CF_SIGMOID};
	params.num_allowed_funcs = 1;
	params.output_funcs = (enum CPPNFunc[]){CF_SIGMOID};
	params.num_outputs = 1;
	params.initially_linked_outputs = (int[]){1};

	srand(time(0));
	create_CPPN( &seed, &params );
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
	if ( record )
		dude->score = 4.0;
	for ( j=0; j<4; j++ ) {
		read_CPPN( &dude->genotype, params, test[j], &result[j] );
		d = result[j]-test[j][2];
		if ( record )
			dude->score -= fabs(d);
		if ( verbose )
			printf( "%1.0f^%1.0f->%.2f, (%+.2f off)\n", test[j][0], test[j][1], result[j], d );
	}
	if ( record )
		dude->score *= dude->score;
}

void functest() {
	struct NEAT_Params params;
	Population pop;
	CPPN seed;
	int err=0;
	
	get_params( &params );
	params.allowed_funcs = (enum CPPNFunc[]){ CF_SIGMOID };
	params.num_allowed_funcs = 1;
	params.output_funcs = (enum CPPNFunc[]){ CF_SIGMOID };
	params.num_outputs = 1;
	params.initially_linked_outputs = (int[]){1};
	
	if (( err = create_CPPN( &seed, &params ) )
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
		printf( "The winner with score %.2f belongs to species #%d, here it is:\n", pop.members[winner].score, pop.species_ids[j] );
		dump_CPPN( &pop.members[winner].genotype );
		eval_xor( &pop.members[winner], &params, 0, 1 );
		
		while(!getchar());
		if (( err = epoch( &pop, &params ) ))
			exit(err);
	}
}

void print_link_info( CPPN_Link *l ) {
	printf(
		"[%04d] %2d --%c %2d  (%+.2f)",
		l->innov_id,
		l->from,
		l->is_disabled ? 'x' : '>',
		l->to,
		l->weight
	);
}

void dump_CPPN( CPPN *net ) {
	int i;
	for ( i=0; i<net->num_links; i++ ) {
		print_link_info( &net->links[i] );
		putchar('\n');
	}
}

void juxtapose_CPPN( CPPN *net1, CPPN *net2 ) {
	int i=0,j=0;
	while ( i<net1->num_links && j<net2->num_links ) {
		if ( net1->links[i].innov_id == net2->links[j].innov_id ) {
			print_link_info( &net1->links[i] );
			putchar('\t');
			print_link_info( &net2->links[j] );
			putchar('\n');
			i++;
			j++;
		} else if ( net1->links[i].innov_id > net2->links[j].innov_id ) {
			printf( "\t\t\t\t" );
			print_link_info( &net2->links[j] );
			putchar('\n');
			j++;
		} else {
			print_link_info( &net1->links[i] );
			putchar('\n');
			i++;
		}
	}
	for ( ; i<net1->num_links; i++ ) {
		print_link_info( &net1->links[i] );
		putchar('\n');
	}
	for ( ; j<net2->num_links; j++ ) {
		printf( "\t\t\t\t" );
		print_link_info( &net2->links[j] );
		putchar('\n');
	}
}

void hillclimbing() {
	struct NEAT_Params params;
	CPPN net1, net2;
	char c=' ';
	
	get_params( &params );
	params.allowed_funcs = (enum CPPNFunc[]){CF_SIGMOID};
	params.num_allowed_funcs = 1;
	params.output_funcs = (enum CPPNFunc[]){CF_SIGMOID};
	params.num_outputs = 1;
	params.initially_linked_outputs = (int[]){1};
	
	create_CPPN( &net1, &params );
	clone_CPPN( &net2, &net1 );
	do {
		if ( c == '1' ) {
			crossover_CPPN( &net1, &net2, &params );
		} else if ( c == '2' ) {
			crossover_CPPN( &net2, &net1, &params );
		} else {
			mutate_CPPN( &net1, &params, 0, 0 );
			mutate_CPPN( &net2, &params, 0, 0 );
		}
		juxtapose_CPPN( &net1, &net2 );
		printf( "-- 1 to xover left with right, 2 to xover right with left, any key to mutate -- compatibility is %.2f\n", \
			get_genetic_distance( &net1, &net2, &params ) );
	} while (( c=getchar() ));
}

int main() {
	srand(time(0));
	functest();
	return 0;
}

