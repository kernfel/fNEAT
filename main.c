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
#include "robot-simplistic.h"

#define NUM_BEHAVIOUR_VARS 2

void get_neat_params( struct NEAT_Params *params ) {
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

	params->population_size = 500;
	params->extinction_threshold = 2;
	params->champion_threshold = 1;
	params->target_num_species = 40;

	params->survival_quota = 0.2;
	params->speciation_threshold = 3.0;
	params->d_speciation_threshold = 0.1;
	params->max_speciation_threshold = 5.0;
	params->disjoint_factor = 1.0;
	params->excess_factor = 1.0;
	params->weight_factor = 0.4;
	
	params->stagnation_age_threshold = 15;
	params->stagnation_score_threshold = 0.2;
	params->stagnation_penalty = 0.8;

	params->add_link_prob = 0.4;
	params->add_node_prob = 0.02;
	params->enable_link_prob = 0.025;
	params->disable_link_prob = 0.01;
	params->crossover_prob = 0.5;
	params->interspecies_mating_prob = 0.001;
	
	params->mutate_weights_prob = 0.8;
	params->perturb_weights_proportion = 0.75;
	params->perturb_weights_range = 4.5;
	params->reassign_weights_proportion = 0.25;
	params->random_weights_range = 2.5;

	params->innov_counter = 0;
	params->species_counter = 0;
	params->individual_counter = 0;
}

void get_robot_params( struct Robot_Params *params ) {
	params->radius = 1;
	
	params->motor_sensitivity = 0.01;
	params->motor_threshold = 0.1;
	params->turn_sensitivity = 0.01;
	params->turn_threshold = 0.1;
	
	params->num_dist_sensors = 0;
	params->dist_sensor_length = 1;
	params->dist_sensor_pos = NULL;
}

int run_trial( pNetwork *controller_structure, struct NEAT_Params *n_params, struct Robot_Params *r_params, TileMaze *maze, int *behaviour ) {
	int err=0;
	
	eNetwork controller;
	if (( err = create_eNetwork( &controller ) ))
		return err;
	if (( err = build_eNetwork( &controller, controller_structure, n_params ) )) {
		delete_eNetwork( &controller );
		return err;
	}

	Robot bot = { 1, 1, 0 };
	double sensors[NUM_SENSORS+1], motors[NUM_MOTORS];
	sensors[NUM_SENSORS] = 1;
	int i, tile_counter=0, tile=0, prev_tile;
	for ( i=0; i<20000 && tile_counter < 400; i++ ) {
		get_sensor_readings_in_tilemaze( &bot, maze, r_params, sensors );
		activate( &controller, sensors, motors );
		move_robot_in_tilemaze( &bot, maze, r_params, motors );
		
		prev_tile = tile;
		tile = (int)(bot.x / maze->tile_width) + maze->x*(int)(bot.y / maze->tile_width);
		tile_counter++;
		if ( tile != prev_tile )
			tile_counter = 0;

		++behaviour[tile];
	}

	delete_eNetwork( &controller );

	return err;
}

void m_get_spread( pNode *in, int n, double x ) {
	int i;
	for ( i=1; i<=n; i++ ) {
		in[i-1].x[0] = x;
		in[i-1].x[1] = (-0.5) + i/(1.0+n);
	}
}

int analyse_maze( TileMaze *m, int *behaviour, int print ) {
	int x, y, maxval=0, tiles_visited=0;
	for ( x=0; x<m->x*m->y; x++ ) {
		maxval = max(maxval, behaviour[x]);
		if ( behaviour[x] )
			tiles_visited++;
	}
	if ( print ) {
		for ( x=0; x<m->x; x++ ) {
			putchar('\n');
			for ( y=0; y<m->y; y++ ) {
				if ( behaviour[x + m->x*y] ) {
					int track = (int)(34 * behaviour[x + m->x*y] / maxval);
					if ( track < 9 )
						putchar( '1'+track );
					else
						putchar( 'A'+(track-9) );
				} else if ( m->tiles[x + m->x*y] )
					putchar('0');
				else
					putchar('#');
			}
		}
		putchar('\n');
		fflush(stdout);
	}
	return tiles_visited;
}

void bots_in_a_maze() {
	int err=0;

	struct NEAT_Params params;
	get_neat_params( &params );
	params.allowed_funcs = (enum CPPNFunc[]){CF_GAUSS, CF_SIGMOID, CF_SINE, CF_LINEAR};
	params.num_allowed_funcs = 4;

	struct Robot_Params bot_params;
	get_robot_params( &bot_params );
	bot_params.num_dist_sensors = 5;
	bot_params.dist_sensor_pos = (double[]){-M_PI/2, -M_PI/4, 0, M_PI/4, M_PI/2};

	TileMaze maze;
	maze.tile_width = 2;
	maze.x = 8;
	maze.y = 8;
	maze.tiles = (unsigned char []){
	1, 1, 1, 0, 1, 1, 1, 1,
	1, 0, 1, 0, 1, 0, 1, 0,
	1, 0, 1, 0, 1, 0, 1, 1,
	1, 0, 1, 0, 1, 1, 0, 1,
	1, 1, 1, 0, 0, 1, 1, 1,
	1, 0, 1, 1, 1, 1, 0, 1,
	1, 0, 0, 0, 1, 0, 0, 1,
	1, 0, 1, 1, 1, 1, 1, 1,
	};

	CPPN cppn;
	if (( err = create_CPPN( &cppn, &params ) ))
		exit(err);

	Population pop;
	if (( err = create_Population( &pop, &params, &cppn ) ))
		exit(err);
	randomise_population( &pop, &params );

	pNetwork net;
	if (( err = create_pNetwork( &net ) ))
		exit(err);

	eNetwork controller;
	if (( err = create_eNetwork( &controller ) ))
		exit(err);

	pNode sensors[NUM_SENSORS+1], motors[NUM_MOTORS];
	m_get_spread( sensors, NUM_SENSORS, -0.5 );
	m_get_spread( motors, NUM_MOTORS, 0.5 );
	sensors[NUM_SENSORS].x[0] = 0;
	sensors[NUM_SENSORS].x[1] = -1;

	FILE *fp = fopen( "logs/champions.txt", "w" );

	int i, gen;
	int behaviour[maze.x*maze.y], best_behaviour[maze.x*maze.y], best_score, winner, peak, peak_score=0;
	for ( gen=0; 1; gen++ ) {
		best_score = 0;
		for ( i=0; i<params.population_size; i++ ) {
			reset_pNetwork( &net );
			if (( err = build_pNetwork( &net, &pop.members[i].genotype, &params, NUM_SENSORS+1, sensors, NUM_MOTORS, motors ) ))
				exit(err);

			int s=0;
			if ( ! net.num_used_links || ! net.num_used_nodes ) {
				pop.members[i].score = 0;
			} else {
				memset( behaviour, 0, maze.x*maze.y*sizeof *behaviour );
				if (( err = run_trial( &net, &params, &bot_params, &maze, behaviour ) ))
					exit(err);
				s = analyse_maze( &maze, behaviour, 0 );
				pop.members[i].score = s*s;

				if ( s > best_score ) {
					best_score = s;
					memcpy( best_behaviour, behaviour, maze.x*maze.y*sizeof *behaviour );
					winner = i;
				}
			}

//			printf( "%5x: ", pop.members[i].id );
			printf( "%2d  ", s);
			fflush(stdout);
		}

		printf( "\n\nGeneration %d's champion explorer, #%5x of species #%3x, had a good look at %d tiles:",
			gen,
			pop.members[winner].id,
			pop.members[winner].species_id,
			best_score
		);
		analyse_maze( &maze, best_behaviour, 1 );
		
/*		if ( peak_score > best_score ) {*/
/*			fprintf( fp, "\n>>> Generation %d | #%5x | species #%3x | %2d tiles\n",*/
/*				gen,*/
/*				pop.members[peak].id,*/
/*				pop.members[peak].species_id,*/
/*				(int) sqrt(pop.members[peak].score)*/
/*			);*/
/*			dump_CPPN( &pop.members[peak].genotype, fp );*/

/*			reset_pNetwork( &net );*/
/*			if (( err = build_pNetwork( &net, &pop.members[peak].genotype, &params, NUM_SENSORS+1, sensors, NUM_MOTORS, motors ) ))*/
/*				exit(err);*/
/*			//dump_pNetwork( &net, fp );*/
/*			if (( err = build_eNetwork( &controller, &net, &params ) ))*/
/*				exit(err);*/
/*			dump_eNetwork( &controller, fp );*/
/*		} else {*/
/*			peak_score = best_score;*/
/*			peak = winner;*/
/*		}*/

/*		fprintf( fp, "\nGeneration %d | #%5x | species #%3x | %2d tiles\n",*/
/*			gen,*/
/*			pop.members[winner].id,*/
/*			pop.members[winner].species_id,*/
/*			best_score*/
/*		);*/
/*		dump_CPPN( &pop.members[winner].genotype, fp );*/
/*		reset_pNetwork( &net );*/
/*		if (( err = build_pNetwork( &net, &pop.members[winner].genotype, &params, NUM_SENSORS+1, sensors, NUM_MOTORS, motors ) ))*/
/*			exit(err);*/
/*		//dump_pNetwork( &net, fp );*/
/*		if (( err = build_eNetwork( &controller, &net, &params ) ))*/
/*			exit(err);*/
/*		dump_eNetwork( &controller, fp );*/

/*		if ( ! (gen%5) ) {*/
/*			fflush(fp);*/
/*			printf( "\nPaused... e=exit, anything else=continue: " );*/
/*			unsigned char c=getchar();*/
/*			if ( c == 'e' ) {*/
/*				fclose( fp );*/
/*				delete_eNetwork( &controller );*/
/*				delete_pNetwork( &net );*/
/*				delete_Population( &pop );*/
/*				delete_CPPN( &cppn );*/
/*				exit(0);*/
/*			}*/
/*		}*/

		epoch( &pop, &params );
	}
}

int main( int argc, char **argv ) {
	srand(time(0));
	bots_in_a_maze();
	return 0;
}

