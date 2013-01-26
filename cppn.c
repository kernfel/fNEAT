#include <math.h>
#include <string.h>
#include <stdlib.h>

#include "util.h"

#include "cppn.h"

// Constructor
int create_CPPN(	CPPN *net,
			int num_outputs,
			enum CPPNFunc *output_funcs,
			int *outputs_linked,
			int create_disabled_links,
			CPPN_Params *params )
{
	int i, j, num_linked_outputs = 0, err=0;

	if (( err = Malloc( net, sizeof *net ) ))
		return err;
	
	net->num_outputs = num_outputs;
	net->num_inputs = 2*params->num_dimensions + ((params->flags & CFL_USE_DIST)? 1:0) + ((params->flags & CFL_USE_BIAS)? 1:0);
	
	// Create input and output nodes
	if (( err = Malloc( net->nodes, (net->num_inputs+num_outputs)*sizeof *net->nodes ) )) {
		free( net );
		return err;
	}
	for ( i=0; i<net->num_inputs; i++ ) {
		net->nodes[i].func = CF_LINEAR;
	}
	for ( i=net->num_inputs; i<net->num_inputs+num_outputs; i++ ) {
		net->nodes[i].func = output_funcs[i];
		if ( outputs_linked[i-net->num_inputs] )
			num_linked_outputs++;
	}

	// Create links from input to output nodes, where applicable
	if ( create_disabled_links )
		net->num_links = num_outputs * net->num_inputs;
	else
		net->num_links = num_linked_outputs * net->num_inputs;
	if (( err = Malloc( net->links, net->num_links*sizeof *net->links ) )) {
		free( net->nodes );
		free( net );
		return err;
	}
	for ( i=0; i<num_outputs; i++ ) {
		if ( create_disabled_links && ! outputs_linked[i] )
			continue;
		for ( j=0; j<net->num_inputs; j++ ) {
			net->links[j].innov_id = params->innov_counter++;
			net->links[j].from = j;
			net->links[j].to = i+net->num_inputs;
			net->links[j].weight = 0;
			net->links[j].is_disabled = ! outputs_linked[i];
		}
	}

	return err;
}

int clone_CPPN( CPPN *net, const CPPN *original ) {
	int err=0;
	if (( err = Malloc( net, sizeof *net ) ))
		return err;
	memcpy( net, original, sizeof *original );

	if (( err = Malloc( net->nodes, (net->num_inputs+net->num_outputs+net->num_hidden)*sizeof *net->nodes ) )) {
		free( net );
		return err;
	}
	memcpy( net->nodes, original->nodes, (net->num_inputs+net->num_outputs+net->num_hidden)*sizeof *net->nodes );

	if (( err = Malloc( net->links, net->num_links*sizeof *net->links ) )) {
		free( net->nodes );
		free( net );
		return err;
	}
	memcpy( net->links, original->links, net->num_links*sizeof *net->links );

	return err;
}

void delete_CPPN( CPPN *net ) {
	free( net->nodes );
	free( net->links );
	free( net );
}

int mutate_CPPN( CPPN *net, CPPN_Params *params ) {
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
		if (( err = Realloc( net->links, (net->num_links+2)*sizeof *net->links ) ))
			return err;
		if (( err = Realloc( net->links_nodesort, (net->num_links+2)*sizeof *net->links_nodesort ) ))
			return err;
		if (( err = CPPN_insert_link( net, params, net->links[link_id].from, node_id, 1.0, 0, 1 ) ))
			return err;
		if (( err = CPPN_insert_link( net, params, node_id, net->links[link_id].to, net->links[link_id].weight, 0, 1 ) ))
			return err;
	}
	
	// Add a link
	if ( params->add_link_prob * RAND_MAX > rand() ) {
		int	target_id = rand() % (net->num_outputs+net->num_hidden) + net->num_inputs,
			num_possible_sources = net->num_inputs+net->num_hidden+net->num_outputs,
			possible_sources[num_possible_sources];
		
		for ( i=0; i<num_possible_sources; i++ )
			possible_sources[num_possible_sources] = 1;
		
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
			j = rand() % num_possible_sources;
			for ( i=0; j; i++ ) {
				j -= possible_sources[i];
			}
			// i-1 now is the index of the chosen, valid source node.
			
			if (( err = CPPN_insert_link(
				net,
				params,
				i-1,
				target_id,
				2*params->change_weight_rate*rand()/RAND_MAX - params->change_weight_rate,
				0,
				0
			) ))
				return err;
		}
	}
	
	return err;
}

int CPPN_exclude_recurrent_links( const CPPN *net, const CPPN_Params *params, int target_id, int *possible_sources ) {
	int i, n=0;

	// Don't bother going any further if we're at a non-projecting output node
	if ( !(params->flags & CFL_ALLOW_O_TO_O) && target_id < net->num_inputs+net->num_outputs )
		return n;

	for ( i=0; i<net->num_links; i++ ) {
		if ( net->links[i].from == target_id ) {
			n += possible_sources[net->links[i].to];
			possible_sources[net->links[i].to] = 0;
			n += CPPN_exclude_recurrent_links( net, params, target_id, possible_sources );
		}
	}
	
	return n;
}

int CPPN_insert_link( CPPN *net, CPPN_Params *params, int from, int to, double weight, int is_disabled, int no_realloc ) {
	int err=0;
	if ( ! no_realloc ) {
		if (( err = Realloc( net->links, (net->num_links+1)*sizeof *net->links ) ))
			return err;
		if (( err = Realloc( net->links_nodesort, (net->num_links+1)*sizeof *net->links_nodesort ) ))
			return err;
	}

	// Insert the new link and set its parameters
	CPPN_Link *l = net->links + net->num_links;
	l->innov_id = params->innov_counter++;
	l->from = from;
	l->to = to;
	l->weight = weight;
	l->is_disabled = is_disabled;
	
	net->num_links++;
	
	// Find the right place for the nodesorted pointer
	int i=0;
	CPPN_Link **lpp;
	for ( lpp=net->links_nodesort+net->num_links; lpp>=net->links_nodesort; lpp-- ) {
		if ( (*(lpp-1))->to <= to )
			break;
		i++;
	}
	
	// Shove everything back to make space for the newbie
	if ( i ) {
		memmove( lpp+1, lpp, i*sizeof *lpp );
	}
	
	// Link up
	*lpp = l;
	
	return err;
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

double read_CPPN( CPPN *net, const CPPN_Params *params, double *coords, double *output ) {
	int i,j;
	
	// Flush first
	for ( i=net->num_inputs; i<net->num_inputs+net->num_outputs+net->num_hidden; i++ ) {
		net->nodes[i].activation = 0.0;
	}
	
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
	
	// Cycle through the net until nothing changes or num_activations is reached
	CPPN_Link **lpp;
	int links_processed, k, k_incr, k_max, n;
	double diff, a;
	if ( params->flags & CFL_ALLOW_RECURRENCE ) {
		k_incr = 1;
		k_max = params->num_activations;
	} else {
		k_incr = 0;
		k_max = 1;
	}
	
	for ( k=0; k<k_max; k+=k_incr ) {
		lpp = net->links_nodesort;
		links_processed = 0;
		diff = 0.0;
		
		// foreach node i that has inputs
		while ( links_processed < net->num_links ) {
			a = 0.0;
			n = 0;
			
			// foreach node j that feeds into i
			for ( i=(*lpp)->to; (*lpp)->to == i; lpp++ ) {
				a += (*lpp)->weight * net->nodes[(*lpp)->from].activation;
				links_processed++;
				n++;
			}
			a = CPPN_func(net->nodes[i].func, a) / n;
			diff += fabs(a-net->nodes[i].activation);
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

double get_genetic_distance( CPPN *net1, CPPN *net2, const CPPN_Params *params ) {
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
		} else {
			disjoint++;
			if ( net1->links[i].innov_id > net2->links[j].innov_id )
				j++;
			else
				i++;
		}
	}
	
	// Correct for pacing error that counts the first excess gene as disjoint...
	if ( net1->links[net1->num_links-1].innov_id != net2->links[net2->num_links-1].innov_id )
		disjoint--;
	
	// ... but capitalise on the same by using i|j instead of (i-1)|(j-1) here
	if ( i == net1->num_links )
		excess = net2->num_links - j;
	else
		excess = net1->num_links - i;
	
	return params->disjoint_factor*disjoint/n + params->excess_factor*excess/n + params->weight_factor*wdiff/n;
}

