#include <string.h>
#include <stdlib.h>

#include "util.h"
#include "params.h"

#include "cppn.h"
#include "neat.h"






int epoch( Population *pop, struct NEAT_Params *params ) {
	int err=0;

// Init
	// Allocate offspring population
	Population *offspring=0;
	if (( err = allocate_Population( offspring, params ) ))
		return err;

	int i, j;
	double total_score=0.0;
	double species_scores[pop->num_species];
	int species_size[pop->num_species];
	Individual *sorted[params->population_size];

	for ( i=0; i<pop->num_species; i++ ) {
		species_scores[i] = 0.0;
		species_size[i] = 0;
	}


// Set up scores and sorting
	for ( i=0; i<params->population_size; i++ ) {
		int species_idx;
		
		// Determine species index
		for ( species_idx=0; pop->species_ids[species_idx] != pop->members[i].species_id; species_idx++ );
		
		// Gather statistics
		total_score =+ pop->members[i].score;
		species_scores[species_idx] += pop->members[i].score;
		species_size[species_idx]++;
		
		// Sort by score desc
		for ( j=0; j<i; j++ ) {
			if ( pop->members[i].score > sorted[i]->score ) {
				memmove( sorted+j+1, sorted+j, (i-j)*sizeof *sorted );
				break;
			}
		}
		sorted[j] = pop->members + i;
	}


// Determine reproduction credit ~= number of offspring per species
	double reproduction_credit[pop->num_species];
	int num_offspring[pop->num_species], species_creditsorted[pop->num_species];
	int unassigned_offspring = params->population_size;
	
	for ( i=0; i<pop->num_species; i++ ) {
		reproduction_credit[i] = species_scores[i] / species_size[i];
		num_offspring[i] = (int)(reproduction_credit[i]*params->population_size / total_score);
		unassigned_offspring -= num_offspring[i];
		
		// Sort species by reproduction credit, descending
		for ( j=0; j<i; j++ ) {
			if ( reproduction_credit[i] > reproduction_credit[species_creditsorted[j]] ) {
				memmove( species_creditsorted+j+1, species_creditsorted+j, (i-j)*sizeof *species_creditsorted );
				break;
			}
		}
		species_creditsorted[j] = i;
	}
	
	// Assign unassigned offspring (rounding errors) to the best species
	for ( i=0; i<unassigned_offspring; i++ ) {
		num_offspring[species_creditsorted[i]]++;
	}
	
	// Let too small species go extinct. Also, keep track of which species are being passed on for later reference.
	int offspring_species_ids[params->population_size];
	offspring->num_species = 0;
	for ( i=0; i<pop->num_species; i++ ) {
		if ( num_offspring[i] < params->extinction_threshold ) {
			num_offspring[i] = 0;
		} else {
			offspring_species_ids[offspring->num_species++] = pop->species_ids[i];
		}
	}


// Reproduction
	Individual *reps[pop->num_species];
	Individual *child = offspring->members;
	for ( i=0; i<pop->num_species; i++ ) {
		
		// Skip extinct species
		if ( ! num_offspring[i] )
			continue;
		
		// Collect all individuals eligible for reproduction
		int num_parents = (int)(species_size[i] * (1-params->elimination_quota));
		Individual *members[num_parents];
		int k=0;
		for ( j=0; j<params->population_size; j++ ) {
			if ( sorted[j]->species_id == pop->species_ids[i] ) {
				members[k] = sorted[j];
				if ( ++k == num_parents )
					break;
			}
		}
		
		// Select a random representative for species assignment
		reps[i] = members[(int)((double)num_parents*rand()/RAND_MAX)];
		
		// Reproduce!
		Node_Innovation node_innovations[num_offspring[i]];
		Link_Innovation link_innovations[num_offspring[i]];
		int num_node_innovations=0, num_link_innovations=0;
		for ( j=0; j<num_offspring[i]; j++ ) {
			
			if (( err = clone_CPPN( child->genotype, members[j%num_parents]->genotype ) ))
				goto fail;
			
			// Crossover with random eligible conspecific
			if ( num_parents > 1 && params->crossover_prob * RAND_MAX > rand() ) {
				k = rand() % (num_parents-1);
				if ( k >= j%num_parents )
					k++;
				if (( err = crossover_CPPN( child->genotype, members[k]->genotype ) ))
					goto fail;
			}
			
			// Mutate
			node_innovations[num_node_innovations].replaced_link = 0;
			link_innovations[num_link_innovations].innov_id = 0;
			if (( err = mutate_CPPN(
				child->genotype,
				params,
				node_innovations+num_node_innovations,
				link_innovations+num_link_innovations )
			))
				goto fail;
			
			// Identify and unify parallel innovations
			// Node insertion:
			if ( node_innovations[num_node_innovations].replaced_link ) {
				int unified=0;
				for ( k=0; k<num_node_innovations; k++ ) {
					if ( node_innovations[k].replaced_link == node_innovations[num_node_innovations].replaced_link ) {
						int l;
						for ( l=child->genotype->num_links-1; l>=0 && unified < 2; l-- ) {
							if ( child->genotype->links[l].innov_id == node_innovations[num_node_innovations].link_in ) {
								child->genotype->links[l].innov_id = node_innovations[k].link_in;
								unified++;
							} else if ( child->genotype->links[l].innov_id == node_innovations[num_node_innovations].link_out ) {
								child->genotype->links[l].innov_id = node_innovations[k].link_out;
								unified++;
							}
						}
						break;
					}
				}
				if ( ! unified )
					num_node_innovations++;
			}
			
			// Link insertion:
			if ( link_innovations[num_link_innovations].innov_id ) {
				int unified=0;
				for ( k=0; k<num_link_innovations; k++ ) {
					if ( link_innovations[k].type == link_innovations[num_link_innovations].type \
					  && link_innovations[k].from == link_innovations[num_link_innovations].from \
					  && link_innovations[k].to == link_innovations[num_link_innovations].to ) {
						int l;
						for ( l=child->genotype->num_links-1; l>=0; l-- ) {
							if ( child->genotype->links[l].innov_id == link_innovations[num_link_innovations].innov_id ) {
								child->genotype->links[l].innov_id = link_innovations[k].innov_id;
								unified = 1;
								break;
							}
						}
						break;
					}
				}
				if ( ! unified )
					num_link_innovations++;
			}
			
			child++;

		} // Inner reproduction loop
	} // Species loop


// Assign each offspring to species
	{
	int num_new_species=0;
	Individual *new_reps[params->population_size];
	for ( i=0; i<params->population_size; i++ ) {
		int assigned=0;
		
		// Find its allegiance to an ancestor species...
		for ( j=0; j<pop->num_species; j++ ) {
			if ( ! num_offspring[j] )
				continue;
			double distance = get_genetic_distance( offspring->members[i].genotype, reps[j]->genotype, params );
			if ( distance < params->speciation_threshold ) {
				offspring->members[i].species_id = reps[j]->species_id;
				assigned = 1;
				break;
			}
		}
		
		// ... or to a newly created species...
		if ( ! assigned ) {
			for ( j=0; j<num_new_species; j++ ) {
				double distance = get_genetic_distance( offspring->members[i].genotype, new_reps[j]->genotype, params );
				if ( distance < params->speciation_threshold ) {
					offspring->members[i].species_id = new_reps[j]->species_id;
					assigned = 1;
					break;
				}
			}
		}
		
		// ... or create a new species, proudly representing!
		if ( ! assigned ) {
			offspring_species_ids[offspring->num_species++] = params->species_counter;
			offspring->members[i].species_id = params->species_counter++;
			new_reps[num_new_species++] = offspring->members + i;
		}
	}}
	
	// Update the offspring species list
	if (( err = Malloc(offspring->species_ids, offspring->num_species*sizeof *offspring->species_ids) ))
		goto fail;
	memcpy( offspring->species_ids, offspring_species_ids, offspring->num_species*sizeof *offspring->species_ids );

	return err;

fail:
	// Cleanup after catastrophic failure
	delete_Population( offspring );
	return err;
}



















