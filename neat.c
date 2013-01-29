#include <string.h>
#include <stdlib.h>

#include "util.h"
#include "params.h"

#include "cppn.h"
#include "neat.h"


int create_Population( Population *pop, struct NEAT_Params *params, const Individual *prototype ) {
	int err=0;
	
	pop->num_members = params->population_size;
	pop->num_species = 1;
	
	if (( err = allocate_Population( pop ) ))
		return err;
	
	pop->species_ids[0] = ++params->species_counter;
	pop->species_size[0] = pop->num_members;
	
	int i;
	for ( i=0; i<pop->num_members; i++ ) {
		if (( err = clone_CPPN( &pop->members[i].genotype, &prototype->genotype ) )) {
			pop->num_members = i;
			delete_Population( pop );
			return err;
		}
		pop->members[i].species_id = pop->species_ids[0];
	}
	if (( err = mutate_population( pop, params ) )
	 || ( err = speciate_population( pop, params, prototype ) ))
		delete_Population( pop );
	
	return err;
}

int allocate_Population( Population *pop ) {
	int err=0;
	
	pop->members = 0;
	pop->species_ids = 0;
	pop->species_size = 0;
	
	if (( err = Calloc( pop->members, pop->num_members, sizeof *pop->members ) )
	 || ( err = Malloc( pop->species_ids, pop->num_species*sizeof *pop->species_ids ) )
	 || ( err = Malloc( pop->species_size, pop->num_species*sizeof *pop->species_size ) ))
		delete_Population( pop );

	return err;
}

void delete_Population( Population *pop ) {
	int i;
	for ( i=0; i<pop->num_members; i++ )
		delete_CPPN( &pop->members[i].genotype );
	free( pop->members );
	free( pop->species_ids );
	free( pop->species_size );
	pop->members = 0;
	pop->species_ids = 0;
	pop->species_size = 0;
}


int epoch( Population *pop, struct NEAT_Params *params ) {
	int err=0;

	// Initialise next generation whilst saving a set of species representatives
	Individual reps[pop->num_species];
	if (( err = reproduce_population( pop, params, reps ) ))
		return err;

	// Mutate the entire new generation
	if (( err = mutate_population( pop, params ) ))
		return err;

	// Divide the offspring into species
	if (( err = speciate_population( pop, params, reps ) ))
		return err;

	return err;
}

// Reproduction
int reproduce_population( Population *pop, struct NEAT_Params *params, Individual *reps ) {
	int i, j, k=0, err=0;

	// Determine the number of offspring each species is allowed
	int num_offspring[pop->num_species];
	get_population_fertility( pop, params, num_offspring );

	// Get ranking
	Individual ***mem_by_spec = get_population_ranking( pop, &err );
	if ( err )
		return err;

	// Determine survivors, remove the rest
	int num_parents[pop->num_species];
	Individual *free_slots[pop->num_members];
	int species_processed=0;
	for ( i=0; i<pop->num_species; i++ ) {
		// Let subthreshold species go extinct
		if ( !num_offspring[i] || num_offspring[i] < params->extinction_threshold ) {
			num_parents[i] = num_offspring[i] = 0;
			reps[i].species_id = 0;
		} else {
			// Find out how many old individuals get to reproduce
			num_parents[i] = (int)(params->survival_quota * pop->species_size[i]);
			if ( num_parents[i] > num_offspring[i] )
				num_parents[i] = num_offspring[i];
			
			// Save a random parent as representative for speciation
			if (( err = clone_CPPN( &reps[i].genotype, &mem_by_spec[i][rand()%num_parents[i]]->genotype ) ))
				goto failure;
			reps[i].species_id = pop->species_ids[i];
		}

		// Eliminate the infertile individuals and post job openings in free_slots
		for ( j=num_parents[i]; j<pop->species_size[i]; j++ ) {
			free_slots[k] = mem_by_spec[i][j];
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
			g = &mem_by_spec[i][j%num_parents[i]]->genotype;
			
			// If necessary, spread its genes...
			if ( j >= num_parents[i] ) {
				free_slots[k]->species_id = pop->species_ids[i];
				if (( err = clone_CPPN( &free_slots[k]->genotype, g ) ))
					goto failure;
				// ... and work with the result.
				g = &free_slots[k++]->genotype;
			}

			// Perform crossover
			if ( params->crossover_prob * RAND_MAX > rand() ) {
				// Find a non-self partner in the same species
				int l=rand()%(num_parents[i]-1);
				if ( l >= j )
					l++;
				// Do the deed
				if (( err = crossover_CPPN( g, &mem_by_spec[i][l]->genotype, params ) ))
					goto failure;
			}
		}
	}

cleanup:
	free( mem_by_spec );

	return err;

failure:
	for ( i=0; i<species_processed; i++ )
		if ( reps[i].species_id )
			delete_CPPN( &reps[i].genotype );
	goto cleanup;
}

// Determine number of offspring per species
void get_population_fertility( Population *pop, struct NEAT_Params *params, int *num_offspring ) {
	int i, j;
	double species_scores[pop->num_species];
	for ( i=0; i<pop->num_species; i++ ) {
		species_scores[i] = 0.0;
	}

	// Add up all scores within a species
	for ( i=0; i<pop->num_members; i++ ) {
		for ( j=0; j<pop->num_species; j++ ) {
			if ( pop->species_ids[j] == pop->members[i].species_id ) {
				species_scores[j] += pop->members[i].score;
				break;
			}
		}
	}

	double average_scores[pop->num_species];
	double sum_average_scores;
	int species_scoresort[pop->num_species];
	for ( i=0; i<pop->num_species; i++ ) {

		// Determine average score per species
		average_scores[i] = species_scores[i] / pop->species_size[i];
		sum_average_scores += average_scores[i];

		// Sort species by average score, descending
		for ( j=0; j<i; j++ ) {
			if ( average_scores[i] > average_scores[species_scoresort[j]] ) {
				memmove( species_scoresort+j+1, species_scoresort+j, (i-j)*sizeof *species_scoresort );
				break;
			}
		}
		species_scoresort[j] = i;
	}

	// Assign offspring in proportion to a species' average score
	int unassigned_offspring;
	for ( i=0; i<pop->num_species; i++ ) {
		num_offspring[i] = (int)(average_scores[i]/sum_average_scores);
		unassigned_offspring -= num_offspring[i];
	}

	// Throw the winners some scraps from rounding error
	for ( i=0; i<unassigned_offspring; i++ ) {
		num_offspring[species_scoresort[i]]++;
	}
}

Individual ***get_population_ranking( Population *pop, int *err ) {
	Individual ***mem_by_spec=0;

	if (( *err = Malloc( mem_by_spec, pop->num_species*sizeof *mem_by_spec + pop->num_members*sizeof **mem_by_spec ) ))
		return 0;

	int i, j, all_indiv_seen=0, sp;
	for ( sp=0; sp<pop->num_species; sp++ ) {
		// Assign the primary array index to point to the right chunk
		mem_by_spec[sp] = mem_by_spec[pop->num_species] + all_indiv_seen;
		
		// Find all members...
		int sp_indiv_seen=0;
		for ( i=0; i<pop->num_members; i++ ) {
			// ...of that species...
			if ( pop->members[i].species_id == pop->species_ids[sp] ) {
				// Go through the conspecifics already in place...
				for ( j=0; j<sp_indiv_seen; j++ ) {
					// Shove the losers back...
					if ( pop->members[i].score > mem_by_spec[sp][j]->score ) {
						memmove( mem_by_spec[sp]+j+1, mem_by_spec[sp]+j, (sp_indiv_seen-j)*sizeof **mem_by_spec );
						break;
					}
				}
				// ... so as to maintain order.
				mem_by_spec[sp][j] = pop->members + i;
				sp_indiv_seen++;
			}
		}
		all_indiv_seen += sp_indiv_seen;
	}

	return mem_by_spec;
}

int mutate_population( Population *pop, struct NEAT_Params *params ) {
	int i, k, l, err=0;
	Node_Innovation ni[pop->num_members];
	Link_Innovation li[pop->num_members];
	int num_ni=0, num_li=0;
	for ( i=0; i<pop->num_members; i++ ) {
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
					for ( l=genotype->num_links-1; l>=0 && collapsed<2; l-- ) {
						if ( genotype->links[l].innov_id == ni[num_ni].link_in ) {
							CPPN_update_innov_id( genotype, genotype->links[l].innov_id, ni[k].link_in );
							collapsed++;
						} else if ( genotype->links[l].innov_id == ni[num_ni].link_out ) {
							CPPN_update_innov_id( genotype, genotype->links[l].innov_id, ni[k].link_out );
							collapsed++;
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
	int i, j, num_new_species=0;
	
	for ( i=0; i<pop->num_species; i++ ) {
		pop->species_size[i] = 0;
	}
	
	int new_species_size[pop->num_members];
	unsigned int new_species_ids[pop->num_members];
	Individual *new_reps[pop->num_members];
	for ( i=0; i<pop->num_members; i++ ) {
		int assigned=0;
		
		// Check for same species first...
		if ( pop->members[i].species_id ) {
			for ( j=0; j<pop->num_species; j++ ) {
				if ( reps[j].species_id == pop->members[i].species_id ) {
					double distance = get_genetic_distance( &pop->members[i].genotype, &reps[j].genotype, params );
					if ( distance < params->speciation_threshold ) {
						pop->species_size[j]++;
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
						pop->species_size[j]++;
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
			pop->members[i].species_id = \
			new_species_ids[num_new_species] = ++params->species_counter;
			new_reps[num_new_species] = pop->members + i;
			new_species_size[num_new_species++] = 1;
		}
	}
	
	int err=0;
	if (( err = Realloc( pop->species_ids, pop->num_species+num_new_species*sizeof *pop->species_ids ) ))
		return err;
	if (( err = Realloc( pop->species_size, pop->num_species+num_new_species*sizeof *pop->species_size ) )) {
		Realloc( pop->species_ids, pop->num_species*sizeof *pop->species_ids );
		return err;
	}
	memcpy( pop->species_ids+pop->num_species, new_species_ids, num_new_species*sizeof *new_species_ids );
	memcpy( pop->species_size+pop->num_species, new_species_size, num_new_species*sizeof *new_species_size );
	pop->num_species += num_new_species;
	
	return err;
}














