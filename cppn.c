#include <math.h>
#include <string.h>
#include <stdlib.h>

#include "util.h"
#include "params.h"

#include "cppn.h"

// Constructor
int create_CPPN( CPPN *net, struct NEAT_Params *params )
{
	int i, j, err=0;
	
	// Determine number of nodes and links
	net->num_outputs = params->num_outputs;
	net->num_inputs = 2*params->num_dimensions + ((params->flags & CFL_USE_DIST)? 1:0) + ((params->flags & CFL_USE_BIAS)? 1:0);
	net->num_hidden = 0;
	if ( params->flags & CFL_MASK_INIT_UNLINKED ) {
		net->num_links = net->num_inputs * net->num_outputs;
	} else {
		net->num_links = 0;
		for ( i=0; i<net->num_outputs; i++ )
			if ( params->initially_linked_outputs[i] )
				net->num_links += net->num_inputs;
	}
	
	// Allocate the storage necessary
	if (( err = allocate_CPPN( net ) ))
		return err;

	// Create input and output nodes
	for ( i=0; i<net->num_inputs; i++ )
		net->nodes[i].func = CF_LINEAR;
	for ( i=0; i<net->num_outputs; i++ )
		net->nodes[i+net->num_inputs].func = params->output_funcs[i];

	// Create links from input to output nodes, where applicable
	for ( i=0; i<net->num_outputs; i++ ) {
		if ( ! (params->flags & CFL_MASK_INIT_UNLINKED) && ! params->initially_linked_outputs[i] )
			continue;
		for ( j=0; j<net->num_inputs; j++ ) {
			net->links[j].innov_id = ++params->innov_counter;
			net->links[j].from = j;
			net->links[j].to = i+net->num_inputs;
			net->links[j].weight = 0;
			net->links[j].is_disabled = ! params->initially_linked_outputs[i];
		}
	}

	return err;
}

int clone_CPPN( CPPN *net, const CPPN *original ) {
	int err=0;
	
	// Copy values over
	memcpy( net, original, sizeof *original );

	// Reassign pointers
	if (( err = allocate_CPPN( net ) ))
		return err;
	
	// Copy nodes and links
	memcpy( net->nodes, original->nodes, (net->num_inputs+net->num_outputs+net->num_hidden)*sizeof *net->nodes );
	memcpy( net->links, original->links, net->num_links*sizeof *net->links );

	return err;
}

int allocate_CPPN( CPPN *net ) {
	int err=0;

	net->nodes = 0;
	net->links = 0;

	if (( err = Malloc( net->links, net->num_links*sizeof *net->links ) ) \
	 || ( err = Malloc( net->nodes, (net->num_inputs+net->num_outputs+net->num_hidden)*sizeof *net->nodes ) )) {
		delete_CPPN( net );
	}

	return err;
}

void delete_CPPN( CPPN *net ) {
	free( net->nodes );
	free( net->links );
	net->links = 0;
	net->nodes = 0;
}

int mutate_CPPN( CPPN *net, struct NEAT_Params *params, Node_Innovation *ni, Link_Innovation *li ) {
	int i, j, err=0;
	double p, s;
	
	// Mutate link weights within +/- change_weight_rate, regardless of whether links are disabled.
	if ( params->change_weight_prob ) {
		p = params->change_weight_prob * RAND_MAX;
		s = params->change_weight_rate * 2.0 / RAND_MAX;
		for ( i=0; i<net->num_links; i++ ) {
			if ( p > rand() ) {
				net->links[i].weight += s*rand() - params->change_weight_rate;
			}
		}
	}
	
	// Enable disabled links
	if ( params->enable_link_prob ) {
		p = params->enable_link_prob * RAND_MAX;
		for ( i=0; i<net->num_links; i++ ) {
			if ( net->links[i].is_disabled && p > rand() ) {
				net->links[i].is_disabled = 0;
			}
		}
	}
	
	// Add a node
	if ( params->add_node_prob * RAND_MAX > rand() ) {
		// Determine the link to split
		int link_id = rand() % net->num_links;
		
		// id of the new node
		int node_id = net->num_inputs + net->num_outputs + net->num_hidden;
		
		// Disable the existing link
		net->links[link_id].is_disabled = 1;

		// Create a new, random-function node
		if (( err = Realloc( net->nodes, (node_id+1)*sizeof *net->nodes ) ))
			return err;
		net->nodes[node_id].func = params->allowed_funcs[ rand() % params->num_allowed_funcs ];
		net->num_hidden++;
		
		// Link it along the old link's route
		// Into the new node, with weight=1, and out of the new node, with inherited weight
		// Note, net->num_links will be updated within CPPN_insert_link.
		// Todo: Tidy clean-up
		if (( err = CPPN_insert_link( net, params, net->links[link_id].from, node_id, 1.0, 0, 0 ) ))
			return err;
		if (( err = CPPN_insert_link( net, params, node_id, net->links[link_id].to, net->links[link_id].weight, 0, 0 ) ))
			return err;
		
		// Provide details about the mutation
		if ( ni ) {
			ni->replaced_link = net->links[link_id].innov_id;
			ni->link_in = params->innov_counter-1;
			ni->link_out = params->innov_counter;
		}
	}
	
	// Add a link
	if ( params->add_link_prob * RAND_MAX > rand() ) {
		int	target_id = rand() % (net->num_outputs+net->num_hidden) + net->num_inputs,
			num_possible_sources = net->num_inputs+net->num_hidden+net->num_outputs,
			possible_sources[num_possible_sources],
			source_id;
		
		for ( i=0; i<num_possible_sources; i++ )
			possible_sources[i] = 1;
		
		// Exclude existing links
		for ( i=0; i<net->num_links; i++ ) {
			if ( net->links[i].to == target_id ) {
				possible_sources[net->links[i].from] = 0;
				num_possible_sources--;
			}
		}
		
		// Exclude recurrent links
		if ( !(params->flags & CFL_ALLOW_RECURRENCE) )
			num_possible_sources -= CPPN_exclude_recurrent_links( net, params, target_id, possible_sources );
		
		// Exclude output-to-output links
		if ( !(params->flags & CFL_ALLOW_O_TO_O) && target_id < net->num_inputs+net->num_outputs ) {
			for ( i=0; i<net->num_outputs; i++ ) {
				num_possible_sources -= possible_sources[i+net->num_inputs];
				possible_sources[i+net->num_inputs] = 0;
			}
		}
		
		// Determine source node and add link
		if ( num_possible_sources ) {
			j = rand() % num_possible_sources + 1;
			// Find the j:th blip in a sparse array
			for ( source_id=0; j; source_id++ ) {
				j -= possible_sources[source_id];
			}
			source_id--; // To correct for that last ++ before loop exit
			
			if (( err = CPPN_insert_link(
				net,
				params,
				source_id,
				target_id,
				2*params->change_weight_rate*rand()/RAND_MAX - params->change_weight_rate,
				0,
				0
			) ))
				return err;
		
			// Provide details about the mutation
			if ( li ) {
				li->innov_id = params->innov_counter;
				li->type = 0x00;
				li->from = source_id;
				li->to = target_id;
				if ( source_id >= net->num_inputs ) {
					li->type |= 0x01;
					int oldest_link = li->innov_id;
					for ( i=0; i<net->num_links; i++) {
						if (( net->links[i].to == source_id || net->links[i].from == source_id ) \
						  && net->links[i].innov_id < oldest_link )
							oldest_link = net->links[i].innov_id;
					}
					li->from = oldest_link;
				}
				if ( target_id >= net->num_inputs+net->num_outputs ) {
					li->type |= 0x10;
					int oldest_link = li->innov_id;
					for ( i=0; i<net->num_links; i++) {
						if (( net->links[i].to == target_id || net->links[i].from == target_id ) \
						  && net->links[i].innov_id < oldest_link )
							oldest_link = net->links[i].innov_id;
					}
					li->to = oldest_link;
				}
			}
		}
	}
	
	return err;
}

int CPPN_exclude_recurrent_links( const CPPN *net, const struct NEAT_Params *params, int target_id, int *possible_sources ) {
	int i, n=0;

	n += possible_sources[target_id];
	possible_sources[target_id] = 0;

	// Don't bother going any further if we're at a non-projecting output node
	if ( !(params->flags & CFL_ALLOW_O_TO_O) && target_id < net->num_inputs+net->num_outputs )
		return n;

	for ( i=0; i<net->num_links; i++ ) {
		if ( net->links[i].from == target_id ) {
			n += possible_sources[net->links[i].to];
			possible_sources[net->links[i].to] = 0;
			n += CPPN_exclude_recurrent_links( net, params, net->links[i].to, possible_sources );
		}
	}
	
	return n;
}

int CPPN_insert_link( CPPN *net, struct NEAT_Params *params, int from, int to, double weight, int is_disabled, unsigned int innov_id ) {
	int err=0;

	if (( err = Realloc( net->links, (net->num_links+1)*sizeof *net->links ) ))
		return err;
	
	if ( ! innov_id )
		innov_id = ++params->innov_counter;

	int i;
	// Find the right place in the innov_id-sorted array
	for ( i=net->num_links; i>0; i-- ) {
		if ( net->links[i-1].innov_id < innov_id )
			break;
	}

	CPPN_Link *l = &net->links[i];
	if ( net->num_links > i ) {
		// Move over, newbies!
		memmove( l+1, l, i*sizeof *l );
	}
	l->innov_id = innov_id;
	l->from = from;
	l->to = to;
	l->weight = weight;
	l->is_disabled = is_disabled;
	
	net->num_links++;
	
	return err;
}

int CPPN_update_innov_id( CPPN *net, unsigned int old_id, unsigned int new_id ) {
	int i,j;
	for ( i=net->num_links-1; i>=0; i-- ) {
		if ( net->links[i].innov_id == old_id ) {
			net->links[i].innov_id = new_id;

			// Check first: Do we need to move at all?
			if (
				( i==0 || net->links[i-1].innov_id < new_id ) // Lowest or younger than lower neighbour
				&& ( i==net->num_links-1 || net->links[i+1].innov_id > new_id ) // Highest or older than higher neighbour
			)
				break;
			
			CPPN_Link tmp;
			// Excise
			memcpy( &tmp, &net->links[i], sizeof tmp );
			if ( old_id > new_id ) {
				// Search downwards...
				for ( j=i; j>0; j-- )
					if ( net->links[j-1].innov_id < new_id )
						break;
				// Move the block [j..i] up by one
				memmove( &net->links[j+1], &net->links[j], (i-j)*sizeof tmp );
			} else {
				// Search upwards...
				for ( j=i; j<net->num_links-1; j++ )
					if ( net->links[j+1].innov_id > new_id )
						break;
				// Move the block [i..j] down by one
				memmove( &net->links[i-1], &net->links[i], (j-i)*sizeof tmp );
			}
			
			// Reinsert
			memcpy( &net->links[j], &tmp, sizeof tmp );
		}
	}

	return 0;
}

double CPPN_func( enum CPPNFunc fn, double x ) {
	switch ( fn ) {
		case CF_ABS:
			return fabs(x);
		case CF_GAUSS:
			return exp(-0.5*x*x)*M_2_SQRTPI*M_SQRT1_2*0.5; // sigma=1, mu=0
		case CF_SIGMOID:
			return tanh(x);
		case CF_SINE:
			return sin(x);
		case CF_STEP:
			return x<0.0 ? 0.0 : 1.0;
		case CF_LINEAR:
		default:
			return x;
	}
}

double read_CPPN( CPPN *net, const struct NEAT_Params *params, double *coords, double *output ) {
	int i,j;
	int num_nodes = net->num_inputs+net->num_hidden+net->num_outputs;
	
	// Set input values
	for ( i=0; i<2*params->num_dimensions; i++ ) {
		net->nodes[i].activation = CPPN_func(net->nodes[i].func, coords[i]);
	}
	if ( params->flags & CFL_USE_DIST ) {
		double d=0.0;
		for ( j=0; j<params->num_dimensions; j++ )
			d += pow(coords[j]-coords[j+params->num_dimensions], 2);
		net->nodes[i].activation = CPPN_func(net->nodes[i].func, (params->flags & CFL_SQUARE_DIST) ? d : sqrt(d));
		i++;
	}
	if ( params->flags & CFL_USE_BIAS ) {
		net->nodes[i].activation = CPPN_func(net->nodes[i].func, 1.0);
	}
	
	// Flush
	for ( i=net->num_inputs; i<num_nodes; i++ ) {
		net->nodes[i].activation = 0.0;
	}
	
	// Cycle through the net until nothing changes or num_activations is reached
	int k, k_incr, k_max;
	if ( params->flags & CFL_ALLOW_RECURRENCE ) {
		k_incr = 1;
		k_max = params->num_activations;
	} else {
		k_incr = 1;
		k_max = 0;
	}
	
	double diff;
	for ( k=0; !k_max || k<k_max; k+=k_incr ) {
		diff = 0.0;
		
		// Go through each node
		for ( i=net->num_inputs; i<num_nodes; i++ ) {
			double a=0.0;
			int n=0;
			
			// Find each of its input links
			for ( j=0; j<net->num_links; j++ ) {
				if ( net->links[j].to == i ) {
					a += net->links[j].weight * net->nodes[net->links[j].from].activation;
					n++;
				}
			}
			a = CPPN_func( net->nodes[i].func, a ) / n;
			diff += fabs( a - net->nodes[i].activation );
			net->nodes[i].activation = a;
		}

		if ( diff == 0.0 )
			break;
	}
	
	for ( i=0; i<net->num_outputs; i++ ) {
		output[i] = net->nodes[i+net->num_inputs].activation;
	}

	return diff;
}

double get_genetic_distance( const CPPN *net1, const CPPN *net2, const struct NEAT_Params *params ) {
	int disjoint=0, excess=0, matches=0, n=max(net1->num_links, net2->num_links);
	double w1, w2, wdiff=0.0;
	int i=0, j=0;
	
	while ( i<net1->num_links && j<net2->num_links ) {
		if ( net1->links[i].innov_id == net2->links[j].innov_id ) {
			matches++;
			w1 = net1->links[i].is_disabled ? 0.0 : net1->links[i].weight;
			w2 = net2->links[j].is_disabled ? 0.0 : net2->links[j].weight;
			wdiff += fabs(w1-w2);
			i++;
			j++;
		} else if ( net1->links[i].innov_id > net2->links[j].innov_id ) {
			disjoint++;
			j++;
		} else {
			disjoint++;
			i++;
		}
	}

	if ( i == net1->num_links )
		excess = net2->num_links - j;
	else
		excess = net1->num_links - i;
	
	return params->disjoint_factor*disjoint/n + params->excess_factor*excess/n + params->weight_factor*wdiff/n;
}

int crossover_CPPN( CPPN *net1, const CPPN *net2, struct NEAT_Params *params ) {
	int err=0, i=0, j=0;
	
	int nodemap[net2->num_inputs+net2->num_outputs+net2->num_hidden];
	for ( i=0; i<net2->num_inputs+net2->num_outputs+net2->num_hidden; i++ )
		nodemap[i] = -1;
	
	CPPN_Link *extra_links[net2->num_links];
	int num_extra=0;
	while ( i<net1->num_links && j<net2->num_links ) {
		if ( net1->links[i].innov_id == net2->links[j].innov_id ) {
			// Add matching nodes to the node mapping, then move on
			nodemap[net2->links[j].from] = net1->links[i].from;
			nodemap[net2->links[j].to] = net1->links[i].to;
			i++;
			j++;
		} else if ( net1->links[i].innov_id > net2->links[j].innov_id ) {
			// Disjoint in net2, track
			extra_links[num_extra++] = &net2->links[j];
			j++;
		} else {
			// Disjoint in net1, ignore
			i++;
		}
	}
	for ( ; j<net2->num_links; j++ ) {
		// Excess in net2
		extra_links[num_extra++] = &net2->links[j];
	}
	
	// Nothing to add, bail
	if ( ! num_extra )
		return 0;
	
	// Determine extra nodes - assume hidden for now
	CPPN_Node *extra_nodes[num_extra*2];
	int num_extra_nodes=0;
	int num_nodes_net1 = net1->num_inputs+net1->num_outputs+net1->num_hidden;
	for ( i=0; i<num_extra; i++ ) {
		if ( nodemap[extra_links[i]->from] == -1 ) {
			extra_nodes[num_extra_nodes] = &net2->nodes[extra_links[i]->from];
			nodemap[extra_links[i]->from] = num_nodes_net1 + num_extra_nodes;
			num_extra_nodes++;
		}
		if ( nodemap[extra_links[i]->to] == -1 ) {
			extra_nodes[num_extra_nodes] = &net2->nodes[extra_links[i]->to];
			nodemap[extra_links[i]->to] = num_nodes_net1 + num_extra_nodes;
			num_extra_nodes++;
		}
	}
	
	// Add extra nodes
	if ( num_extra_nodes ) {
		if (( err = Realloc( net1->nodes, num_nodes_net1+num_extra_nodes ) ))
			return err;
		for ( i=0; i<num_extra_nodes; i++ ) {
			memcpy( &net1->nodes[num_nodes_net1+i], extra_nodes[i], sizeof(CPPN_Node) );
		}
		net1->num_hidden += num_extra_nodes;
	}
	
	// Add extra links
	for ( i=0; i<num_extra; i++ ) {
		if (( err = CPPN_insert_link(
			net1,
			params,
			nodemap[extra_links[i]->from],
			nodemap[extra_links[i]->to],
			extra_links[i]->weight,
			extra_links[i]->is_disabled,
			extra_links[i]->innov_id
		) ))
			return err;
	}
	
	return err;
}

