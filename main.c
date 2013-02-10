#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>
#include <math.h>

#include "params.h"
#include "util.h"

#include "cppn.h"
#include "neat.h"
#include "network.h"

void get_params( struct NEAT_Params *params ) {
	params->flags = CFL_USE_BIAS | CFL_USE_DIST | CFL_ALLOW_RECURRENCE;

	params->min_resolution = 2;
	params->max_resolution = 4;
	params->max_network_depth = 2;

	params->variance_threshold = 0.2;
	params->output_variance_weight[0] = 1;
	params->band_threshold = 0.2;
	params->output_bandpruning_weight[0] = 1;
	params->expression_thresholds[0] = 0.2;

	params->output_funcs[0] = CF_SIGMOID;
	params->initially_linked_outputs[0] = 1;

	params->allowed_funcs = NULL;
	params->num_allowed_funcs = 0;
	params->num_activations = 25;

	params->population_size = 100;
	params->extinction_threshold = 2;
	params->champion_threshold = 5;
	params->target_num_species = 10;

	params->survival_quota = 0.2;
	params->speciation_threshold = 3.0;
	params->d_speciation_threshold = 0.1;
	params->max_speciation_threshold = 5.0;
	params->disjoint_factor = 1.0;
	params->excess_factor = 1.0;
	params->weight_factor = 0.4;
	
	params->stagnation_age_threshold = 15;
	params->stagnation_score_threshold = 0.2;
	params->stagnation_penalty = 1;

	params->add_link_prob = 0.4;
	params->add_node_prob = 0.02;
	params->enable_link_prob = 0.25;
	params->disable_link_prob = 0.01;
	params->crossover_prob = 0.75;
	params->interspecies_mating_prob = 0.0;
	
	params->mutate_weights_prob = 0.8;
	params->perturb_weights_proportion = 0.75;
	params->perturb_weights_range = 4.5;
	params->reassign_weights_proportion = 0.25;
	params->random_weights_range = 2.5;

	params->innov_counter = 0;
	params->species_counter = 0;
}

void m_get_spread( pNode *in, int n, double x ) {
	int i;
	for ( i=1; i<=n; i++ ) {
		in[i-1].x[0] = x;
		in[i-1].x[1] = (-0.5) + i/(1.0+n);
	}
}

void list_nodes_in_sector( pNetwork *net, double xmin, double xmax, double ymin, double ymax ) {
	unsigned int i;
	int block=-1, index;
	for ( i=0; i<net->num_nodes; i++ ) {
		index = i % BLOCKSIZE_NODES;
		if ( ! index )
			block++;
		pNode *n = &net->p_nodes[block][index];
		if ( n->x[0] > xmin && n->x[0] < xmax && n->x[1] > ymin && n->x[1] < ymax )
			printf( "%4d @ %2d, %2d\n", i, (int)(n->x[0]*32), (int)(n->x[1]*32) );
	}
}

void print_node_matrix( pNetwork *net ) {
	unsigned char mat[32][32];
	unsigned int i, j;
	int block=0, index;
	for ( i=0; i<32; i++ )
		for ( j=0; j<32; j++ )
			mat[i][j] = 0;
	for ( i=net->num_inputs+net->num_outputs; i<net->num_nodes; i++ ) {
		index = i % BLOCKSIZE_NODES;
		if ( ! index )
			block++;
		pNode *n = &net->p_nodes[block][index];
		mat[(int)(n->x[0]*32)+16][(int)(n->x[1]*32)+16] += 1;
	}
	for ( i=0; i<32; i++ ) {
		for ( j=0; j<32; j++ )
			if ( mat[i][j] )
				putchar('*');
			else
				putchar('o');
		putchar('-');
		putchar('\n');
	}
}

void maxTheNodeCount() {
	int err=0;

	struct NEAT_Params params;
	get_params( &params );
	params.allowed_funcs = (enum CPPNFunc[]){CF_GAUSS, CF_SIGMOID, CF_SINE, CF_LINEAR};
	params.num_allowed_funcs = 4;

	CPPN cppn;
	if (( err = create_CPPN( &cppn, &params ) ))
		exit(err);

	Population pop;
	if (( err = create_Population( &pop, &params, &cppn ) ))
		exit(err);
	randomise_population( &pop, &params );

	pNetwork *net = malloc( sizeof(pNetwork) );
	if (( err = create_pNetwork( net ) ))
		exit(err);

	int num_inputs=5, num_outputs=3;
	pNode inputs[num_inputs], outputs[num_outputs];
	m_get_spread( inputs, num_inputs, -0.5 );
	m_get_spread( outputs, num_outputs, 0.5 );

	int i, gen;
	for ( gen=0; gen<100; gen++ ) {
		for ( i=0; i<100; i++ ) {
			reset_pNetwork( net );
			if (( err = build_pNetwork( net, &pop.members[i].genotype, &params, num_inputs, inputs, num_outputs, outputs ) ))
				exit(err);
			pop.members[i].score = net->num_links;
			if ( !(i%10) )
				putchar('\n');
			printf( "%9d ", net->num_links );
			fflush(stdout);
/*			if ( net->num_nodes == 348 ) {*/
/*				printf( "\n.....................................\n" );*/
/*				//list_nodes_in_sector( net, 0, 0.5, 0, 0.5 );*/
/*				print_node_matrix( net );*/
/*				printf( ".....................................\n" );*/
/*				while(!getchar());*/
/*			}*/
		}
		printf( "\n------------------------- %d ", gen );
		epoch( &pop, &params );
		printf( "-------------------------" );
	}
}

int main( int argc, char **argv ) {
	srand(time(0));
	maxTheNodeCount();
	return 0;
}

