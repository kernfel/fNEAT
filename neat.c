#include <string.h>
#include <stdlib.h>
#include <math.h>

#include "util.h"
#include "params.h"

#include "cppn.h"
#include "neat.h"

#ifdef VERBOSE
	#include <stdio.h>
#endif


int create_Population( Population *pop, struct NEAT_Params *params, const CPPN *prototype ) {
	int err=0;
	
	pop->num_members = params->population_size;
	pop->num_species = 1;
	
	if (( err = allocate_Population( pop ) ))
		return err;
	
	Individual adam;
	adam.genotype = *prototype;
	adam.species_id = ++params->species_counter;
	adam.id = ++params->individual_counter;
	
	pop->species[0].id = adam.species_id;
	pop->species[0].size = pop->num_members;
	
	int i;
	for ( i=0; i<pop->num_members; i++ ) {
		if (( err = clone_CPPN( &pop->members[i].genotype, prototype ) )) {
			pop->num_members = i;
			delete_Population( pop );
			return err;
		}
		pop->members[i].species_id = adam.species_id;
		pop->members[i].id = ++params->individual_counter;
	}
	
	return err;
}

int allocate_Population( Population *pop ) {
	int err=0;
	
	pop->members = 0;
	pop->species = 0;
	
	if (( err = Calloc( pop->members, pop->num_members, sizeof *pop->members ) )
	 || ( err = Calloc( pop->species, pop->num_species, sizeof *pop->species ) ))
		delete_Population( pop );

	return err;
}

void delete_Population( Population *pop ) {
	int i;
	for ( i=0; i<pop->num_members; i++ )
		delete_CPPN( &pop->members[i].genotype );
	free( pop->members );
	free( pop->species );
	pop->members = 0;
	pop->species = 0;
}

void randomise_population( Population *pop, struct NEAT_Params *params ) {
	int i;
	for ( i=0; i<pop->num_members; i++ )
		randomise_CPPN_weights( &pop->members[i].genotype, params );
}


int epoch( Population *pop, struct NEAT_Params *params ) {
	int i, err=0;

	// Initialise next generation whilst saving a set of species representatives
	Individual reps[pop->num_species];
	Individual *champions[pop->num_species];
	int num_reps = pop->num_species;
	if (( err = reproduce_population( pop, params, reps, champions ) ))
		goto cleanup;

	// Mutate the entire new generation
	if (( err = mutate_population( pop, params, champions ) ))
		goto cleanup;

	// Divide the offspring into species
	if (( err = speciate_population( pop, params, reps ) ))
		goto cleanup;
	
	// Adjust the speciation threshold for next generation
	if ( params->target_num_species ) {
		if ( pop->num_species < params->target_num_species && params->speciation_threshold > params->d_speciation_threshold )
			params->speciation_threshold -= params->d_speciation_threshold;
		else if ( pop->num_species > params->target_num_species \
		 && params->speciation_threshold + params->d_speciation_threshold <= params->max_speciation_threshold )
			params->speciation_threshold += params->d_speciation_threshold;
	}

cleanup:
	if ( err != E_NO_OFFSPRING )
		for ( i=0; i<num_reps; i++ )
			if ( reps[i].species_id )
				delete_CPPN( &reps[i].genotype );

	return err;
}

// Reproduction
int reproduce_population( Population *pop, struct NEAT_Params *params, Individual *reps, Individual **champions ) {
	int i, j, k=0, err=0;

	// Determine the number of offspring each species is allowed
	int num_offspring[pop->num_species];
	for ( i=0; i<pop->num_species; i++ )
		num_offspring[i]=0; // Appease valgrind at low cost
	if (( err = get_population_fertility( pop, params, num_offspring ) ))
		return err;

	// Get ranking
	Individual ***ranked = get_population_ranking( pop, &err );
	if ( err )
		return err;

	// Determine survivors, remove the rest
	int num_parents[pop->num_species];
	Individual *free_slots[pop->num_members];
	int species_processed=0;
	for ( i=0; i<pop->num_species; i++ ) {
		// Let subthreshold species go extinct
		if ( !num_offspring[i] ) {
			num_parents[i] = 0;
			reps[i].species_id = 0;
			champions[i] = 0;
		} else {
			// Find out how many old individuals get to reproduce
			num_parents[i] = 1 + (int)(params->survival_quota * pop->species[i].size);
			if ( num_parents[i] > num_offspring[i] )
				num_parents[i] = num_offspring[i];
			
			// Save a random parent as representative for speciation
			if (( err = clone_CPPN( &reps[i].genotype, &ranked[i][rand()%num_parents[i]]->genotype ) ))
				goto failure;
			reps[i].species_id = pop->species[i].id;
			
			// Make sure the champion survives unchanged
			if ( pop->species[i].size >= params->champion_threshold )
				champions[i] = ranked[i][0];
			else
				champions[i] = 0;
		}

		// Eliminate the infertile individuals and post job openings in free_slots
		for ( j=num_parents[i]; j<pop->species[i].size; j++ ) {
			free_slots[k] = ranked[i][j];
			delete_CPPN( &free_slots[k]->genotype );
			k++;
		}
		species_processed++;
	}

	// Error protection
	free_slots[k] = 0;

	// Repopulate from the survivors
	CPPN *g;
	k=0;
	for ( i=0; i<pop->num_species; i++ ) {
		for ( j=0; j<num_offspring[i]; j++ ) {
			// Retrieve a fertile individual
			g = &ranked[i][j%num_parents[i]]->genotype;
			
			// If necessary, spread its genes...
			if ( j >= num_parents[i] ) {
				free_slots[k]->species_id = pop->species[i].id;
				free_slots[k]->id = ++params->individual_counter;
				if (( err = clone_CPPN( &free_slots[k]->genotype, g ) ))
					goto failure;
				// ... and work with the result.
				g = &free_slots[k++]->genotype;

			// No further processing for champions
			} else if ( g == &champions[i]->genotype ) {
				continue;
			}

			// Perform crossover
			if ( num_parents[i] > 1 && params->crossover_prob * RAND_MAX > rand() ) {
				// Find a non-self partner in the same species
				int l=rand()%(num_parents[i]-1);
				if ( l >= j )
					l++;
				// Do the deed
				if (( err = crossover_CPPN( g, &ranked[i][l]->genotype, params ) ))
					goto failure;

			// Interspecies crossover
			}
			 else if ( pop->num_species > 1 \
			 && params->interspecies_mating_prob > 0 \
			 && params->interspecies_mating_prob * RAND_MAX > rand() ) {
				do {
					// Select a species to mate with
					int l=rand()%(pop->num_species-1);
					if ( l >= i )
						l++;
					// If that species doesn't do offspring, cancel the mating attempt
					if ( ! num_parents[l] )
						break;
					// Select a mate within the chosen species
					int m=rand()%num_parents[l];
					// Have at it
					if (( err = crossover_CPPN( g, &ranked[l][m]->genotype, params ) ))
						goto failure;
				} while (0);
			}
		}
	}

cleanup:
	free( ranked );

	return err;

failure:
	for ( i=0; i<species_processed; i++ ) {
		if ( reps[i].species_id ) {
			delete_CPPN( &reps[i].genotype );
			reps[i].species_id = 0;
		}
	}
	goto cleanup;
}

// Determine number of offspring per species
int get_population_fertility( Population *pop, struct NEAT_Params *params, int *num_offspring ) {
	int i, j;
	double species_scores[pop->num_species];
	for ( i=0; i<pop->num_species; i++ ) {
		species_scores[i] = 0.0;
	}

	// Add up all scores within a species
	for ( i=0; i<pop->num_members; i++ ) {
		for ( j=0; j<pop->num_species; j++ ) {
			if ( pop->species[j].id == pop->members[i].species_id ) {
				species_scores[j] += pop->members[i].score;
				#if defined VERBOSE && VERBOSE > 2
					printf( "Member #%5x in species #%3x, score: %2.2f\n", pop->members[i].id, pop->species[j].id, pop->members[i].score );
				#endif
				break;
			}
		}
	}

	double sum_average_scores = 0.0;
	int species_scoresort[pop->num_species];
	for ( i=0; i<pop->num_species; i++ ) {

		// Determine average score
		double mean_score = species_scores[i] / pop->species[i].size;

		// Increase stagnation counter, where appropriate
		if ( fabs( mean_score - pop->species[i].mean_score ) > params->stagnation_score_threshold )
			pop->species[i].stagnating_for = 0;
		else
			pop->species[i].stagnating_for++;

		pop->species[i].mean_score = mean_score;
		sum_average_scores += mean_score;

		// Sort species by average score, descending
		for ( j=0; j<i; j++ ) {
			if ( pop->species[i].mean_score > pop->species[species_scoresort[j]].mean_score ) {
				memmove( species_scoresort+j+1, species_scoresort+j, (i-j)*sizeof *species_scoresort );
				break;
			}
		}
		species_scoresort[j] = i;
	}

	// Assign offspring in proportion to a species' average score
	int unassigned_offspring = params->population_size, living_species=0;
	for ( i=0; i<pop->num_species; i++ ) {
		// Penalise stagnation
		if ( pop->species[i].stagnating_for < params->stagnation_age_threshold )
			num_offspring[i] = (int)(params->population_size * pop->species[i].mean_score/sum_average_scores);
		else
			num_offspring[i] = (int)(params->population_size * (1-params->stagnation_penalty) * pop->species[i].mean_score/sum_average_scores);
		// Kill insular species
		if ( num_offspring[i] < params->extinction_threshold )
			num_offspring[i] = 0;
		
		if ( num_offspring[i] ) {
			living_species++;
			unassigned_offspring -= num_offspring[i];
		}
	}

	if ( ! living_species )
		return E_NO_OFFSPRING;

	// Distribute leftovers
	int shared=unassigned_offspring/living_species, stolen=unassigned_offspring%living_species;
	if ( shared || stolen ) {
		for ( i=0; i<pop->num_species; i++ ) {
			if ( num_offspring[i] ) {
				num_offspring[i] += shared;
				if ( stolen ) {
					num_offspring[i]++;
					stolen--;
				}
			}
		}
	}
	
	#ifdef VERBOSE
		double sd[pop->num_species], dev;
		for ( j=0; j<pop->num_species; j++ ) {
			sd[j] = 0;
			for ( i=0; i<pop->num_members; i++ ) {
				if ( pop->species[j].id == pop->members[i].species_id ) {
					dev = pop->species[j].mean_score - pop->members[i].score;
					sd[j] += dev*dev;
				}
			}
			sd[j] = sqrt(sd[j]/pop->species[j].size);
		}
		
		for ( i=0; i<pop->num_species; i++ ) {
			printf( "Species #%3x (idx %2d) - %3d members, %3d offspring - total score: %8.2f | avg score: %8.2f | sd: %4.2f\n",
				pop->species[i].id,
				i,
				pop->species[i].size,
				num_offspring[i],
				species_scores[i],
				pop->species[i].mean_score,
				sd[i]
			);
		}
	#endif
	
	return 0;
}

Individual ***get_population_ranking( Population *pop, int *err ) {
	Individual ***ranked=0;

	if (( *err = Malloc( ranked, pop->num_species*sizeof *ranked + pop->num_members*sizeof **ranked ) ))
		return 0;

	int i, j, all_indiv_seen=0, sp;
	for ( sp=0; sp<pop->num_species; sp++ ) {
		// Assign the primary array index to point to the right chunk
		ranked[sp] = (Individual **) &ranked[pop->num_species + all_indiv_seen];
		
		// Find all members...
		int sp_indiv_seen=0;
		for ( i=0; i<pop->num_members; i++ ) {
			// ...of that species...
			if ( pop->members[i].species_id == pop->species[sp].id ) {
				// Go through the conspecifics already in place...
				for ( j=0; j<sp_indiv_seen; j++ ) {
					// Shove the losers back...
					if ( pop->members[i].score > ranked[sp][j]->score ) {
						memmove( ranked[sp]+j+1, ranked[sp]+j, (sp_indiv_seen-j)*sizeof **ranked );
						break;
					}
				}
				// ... so as to maintain order.
				ranked[sp][j] = pop->members + i;
				sp_indiv_seen++;
			}
		}
		all_indiv_seen += sp_indiv_seen;
	}

	return ranked;
}

int mutate_population( Population *pop, struct NEAT_Params *params, Individual **champions ) {
	int i, k, l, err=0;
	Node_Innovation ni[pop->num_members];
	Link_Innovation li[pop->num_members];
	int num_ni=0, num_li=0;
	for ( i=0; i<pop->num_members; i++ ) {
		// Don't mutate champions
		int is_champ=0;
		for ( k=0; k<pop->num_species; k++ ) {
			if ( &pop->members[i] == champions[k] ) {
				is_champ = 1;
				break;
			}
		}
		if ( is_champ )
			continue;
		
		// Shortcut
		CPPN *genotype =& pop->members[i].genotype;

		// Prepare for new innovations
		ni[num_ni].replaced_link = 0;
		li[num_li].innov_id = 0;

		// Mutate
		if (( err = mutate_CPPN( genotype, params, ni+num_ni, li+num_li ) ))
			return err;

		// Identify and collapse parallel node innovations
		if ( ni[num_ni].replaced_link ) {
			int collapsed=0;
			// Check through the previous innovations
			for ( k=0; k<num_ni; k++ ) {
				if ( ni[k].replaced_link == ni[num_ni].replaced_link ) {
					// An equivalent innovation is found. Find the relevant links and collapse their innov_id's
					// Note that updating moves the links within their containing array, hence the l++ as a security blanket
					for ( l=genotype->num_links-1; l>=0 && collapsed<2; l-- ) {
						if ( genotype->links[l].innov_id == ni[num_ni].link_out ) {
							CPPN_update_innov_id( genotype, genotype->links[l].innov_id, ni[k].link_out );
							collapsed++;
							l++;
						} else if ( genotype->links[l].innov_id == ni[num_ni].link_in ) {
							CPPN_update_innov_id( genotype, genotype->links[l].innov_id, ni[k].link_in );
							collapsed++;
							l++;
						}
					}
					ASSERT( collapsed==2 );
					break;
				}
			}
			
			// A truly new innovation
			if ( ! collapsed )
				num_ni++;
		}

		// Identify and collapse parallel link innovations
		if ( li[num_li].innov_id ) {
			int collapsed=0;
			// Check through the previous innovations
			for ( k=0; k<num_li; k++ ) {
				if ( li[k].type == li[num_li].type && li[k].from == li[num_li].from && li[k].to == li[num_li].to ) {
					// An equivalent innovation is found. Find the relevant link and collapse its innov_id
					for ( l=genotype->num_links-1; l>=0; l-- ) {
						if ( genotype->links[l].innov_id == li[num_li].innov_id ) {
							CPPN_update_innov_id( genotype, genotype->links[l].innov_id, li[k].innov_id );
							collapsed = 1;
							break;
						}
					}
					ASSERT( collapsed );
					break;
				}
			}
			
			// A truly new innovation
			if ( ! collapsed )
				num_li++;
		}
	}

	return err;
}

int speciate_population( Population *pop, struct NEAT_Params *params, const Individual *reps ) {
	int i, j, num_new_species=0, num_extinct=0;
	
	for ( i=0; i<pop->num_species; i++ ) {
		pop->species[i].size = 0;
	}
	
	int new_species_size[pop->num_members];
	Individual *new_reps[pop->num_members];
	for ( i=0; i<pop->num_members; i++ ) {
		int assigned=0;
		
		// Check for same species first...
		if ( pop->members[i].species_id ) {
			for ( j=0; j<pop->num_species; j++ ) {
				if ( reps[j].species_id == pop->members[i].species_id ) {
					double distance = get_genetic_distance( &pop->members[i].genotype, &reps[j].genotype, params );
					if ( distance < params->speciation_threshold ) {
						pop->species[j].size++;
						assigned = 1;
						break;
					}
				}
			}
		}
		
		// Check for other ancestor species...
		if ( ! assigned ) {
			for ( j=0; j<pop->num_species; j++ ) {
				if ( reps[j].species_id && reps[j].species_id != pop->members[i].species_id ) {
					double distance = get_genetic_distance( &pop->members[i].genotype, &reps[j].genotype, params );
					if ( distance < params->speciation_threshold ) {
						pop->members[i].species_id = reps[j].species_id;
						pop->species[j].size++;
						assigned = 1;
						break;
					}
				}
			}
		}
		
		// ... or to a newly created species...
		if ( ! assigned ) {
			for ( j=num_new_species-1; j>=0; j-- ) {
				double distance = get_genetic_distance( &pop->members[i].genotype, &new_reps[j]->genotype, params );
				if ( distance < params->speciation_threshold ) {
					pop->members[i].species_id = new_reps[j]->species_id;
					new_species_size[j]++;
					assigned = 1;
					break;
				}
			}
		}
		
		// ... or create a new species, proudly representing!
		if ( ! assigned ) {
			pop->members[i].species_id = ++params->species_counter;
			new_reps[num_new_species] = pop->members + i;
			new_species_size[num_new_species++] = 1;
		}
	}
	
	for ( i=0; i<pop->num_species; i++ ) {
		if ( ! pop->species[i].size )
			num_extinct++;
	}
	
	// No structural changes to species list, bail
	if ( ! num_new_species && ! num_extinct )
		return 0;
	
	// Compose a new list with all the (living) species and replace the old one with that
	int err=0;
	int num_all = pop->num_species + num_new_species - num_extinct;
	Species *all_species;
	if (( err = Calloc( all_species, num_all, sizeof *all_species ) ))
		return err;
	
	j=0;
	for ( i=0; i<pop->num_species; i++ ) {
		if ( pop->species[i].size ) {
			all_species[j] = pop->species[i];
			j++;
		}
	}
	for ( i=0; i<num_new_species; i++ ) {
		all_species[j].id = new_reps[i]->species_id;
		all_species[j].size = new_species_size[i];
		j++;
	}
	
	free( pop->species );
	pop->species = all_species;
	pop->num_species = num_all;
	
	return err;
}

