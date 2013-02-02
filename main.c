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
	params->extinction_threshold = 2;
	params->champion_threshold = 5;
	params->survival_quota = 0.2;

	params->speciation_threshold = 3.0;
	params->disjoint_factor = 1.0;
	params->excess_factor = 1.0;
	params->weight_factor = 0.4;

	params->add_link_prob = 0.4;
	params->add_node_prob = 0.02;
	params->enable_link_prob = 0.25;
	params->disable_link_prob = 0.01;
	params->crossover_prob = 0.75;
	
	params->mutate_weights_prob = 0.8;
	params->perturb_weights_proportion = 0.75;
	params->perturb_weights_range = 4.5;
	params->reassign_weights_proportion = 0.25;
	params->random_weights_range = 2.5;

	params->innov_counter = 0;
	params->species_counter = 0;
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

void xor() {
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
	params.population_size = 150;

	if (( err = create_CPPN( &seed, &params ) ))
		exit(err);

	int generation, i, j, winner, run, n_solved;
	int solved_in[100], nodes[100];
	for ( run=0; run<100; run++ ) {
	
		if (( err = create_Population( &pop, &params, &seed ) ))
			exit(err);
		randomise_population( &pop, &params );
		
		for ( generation=0; generation < 200; generation++ ) {
			double best_score = 0.0;
			for ( i=0; i<pop.num_members; i++ ) {
				for ( j=0; j<pop.num_species; j++ )
					if ( pop.members[i].species_id == pop.species_ids[j] )
						break;
				eval_xor( &pop.members[i], &params, 1, 0 );
				if ( pop.members[i].score > best_score ) {
					winner = i;
					best_score = pop.members[i].score;
				}
			}

			if ( best_score > 15.5 ) {
				solved_in[n_solved] = generation;
				nodes[n_solved] = pop.members[winner].genotype.num_hidden;
				n_solved++;
				//printf( "Run %2d, %3d generations to a solution with %d hidden nodes\n", run+1, generation+1, pop.members[winner].genotype.num_hidden );
				winner = -1;
				break;
			}

			if (( err = epoch( &pop, &params ) ))
				exit(err);
		}

		delete_Population( &pop );
		putchar( winner==-1 ? '#':'-' );
		fflush(stdout);
	}
	
	int total_length=0, total_nodes=0;
	for ( i=0; i<n_solved; i++ ) {
		total_length += solved_in[i];
		total_nodes += nodes[i];
	}
	double mean_length=(double)total_length/n_solved, mean_nodes=(double)total_nodes/n_solved;
	double ss_length=0, ss_nodes=0;
	for ( i=0; i<n_solved; i++ ) {
		double a = solved_in[i] - mean_length, b = nodes[i] - mean_nodes;
		ss_length += a*a;
		ss_nodes += b*b;
	}
	printf( "\nAverage runtime: %.2f (sd=%.2f) generations | Average node count: %.2f (sd=%.2f) hidden | %d fails\n",
		mean_length,
		sqrt( ss_length / n_solved ),
		mean_nodes,
		sqrt( ss_nodes / n_solved ),
		100-n_solved
	);

	delete_CPPN(&seed);
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
	params.population_size = 150;
	
	if (( err = create_CPPN( &seed, &params ) )
	 || ( err = create_Population( &pop, &params, &seed ) ))
		exit(err);
	randomise_population( &pop, &params );
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
			if ( pop.members[i].score > best_score ) {
				winner = i;
				best_score = pop.members[i].score;
			}
/*			printf( "Total score: %.2f/4\n", pop.members[i].score );*/
/*			while(!getchar());*/
		}
		
		for ( j=0; j<pop.num_species; j++ )
			if ( pop.members[winner].species_id == pop.species_ids[j] )
				break;
		printf( "The winner belongs to species #%d (idx %d), here it is:\n", pop.species_ids[j], j );
		dump_CPPN( &pop.members[winner].genotype );
		eval_xor( &pop.members[winner], &params, 0, 1 );
		printf( "Score: %.2f\n", pop.members[winner].score );
		
		char foo[10];
		while(1) {
			if ( ! fgets( foo, 9, stdin ) )
				continue;
			if ( foo[0] == '\n' )
				break;
			char action;
			int index;
			sscanf( foo, "%c %d", &action, &index );
			if ( action == 's' ) {
				if ( index >= pop.num_species )
					continue;
				int found=0;
				for ( i=0; i<pop.num_species; i++ )
					if ( pop.species_ids[i] == index ) {
						found = 1;
						index = i;
						break;
					}
				if ( ! found )
					continue;
				printf( "Species #%2d (idx %2d), %3d members:\n", pop.species_ids[index], index, pop.species_size[index] );
				for ( i=0; i<pop.num_members; i++ ) {
					if ( pop.species_ids[index] == pop.members[i].species_id ) {
						printf( "\nMember #%3d, score: %.2f\n", i, pop.members[i].score );
						dump_CPPN( &pop.members[i].genotype );
					}
				}
			} else if ( action == 'i' ) {
				if ( index >= pop.num_members )
					continue;
				printf( "Member #%3d, score: %.2f\n", index, pop.members[index].score );
				dump_CPPN( &pop.members[index].genotype );
			}
		}

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
	randomise_CPPN_weights( &net1, &params );
	randomise_CPPN_weights( &net2, &params );
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
	xor();
	return 0;
}

