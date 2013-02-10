#include <string.h>

#include "params.h"
#include "util.h"

#include "extract.h"
#include "network.h"
#include "cppn.h"


int create_pNetwork( pNetwork *n ) {
	int err=0;
	n->p_nodes = 0;
	n->p_links = 0;
	if (( err = Malloc( n->p_nodes, sizeof *n->p_nodes ) )
	 || ( err = Malloc( n->p_links, sizeof *n->p_links ) ) ) {
		free( n->p_nodes );
		return err;
	}

	n->p_nodes[0] = 0;
	n->num_nodes = 0;
	n->num_node_blocks = 1;
	n->num_inputs = 0;
	n->num_outputs = 0;

	n->p_links[0] = 0;
	n->num_links = 0;
	n->num_link_blocks = 1;

	if (( err = Malloc( n->p_nodes[0], BLOCKSIZE_NODES*sizeof **n->p_nodes ) )
	 || ( err = Malloc( n->p_links[0], BLOCKSIZE_LINKS*sizeof **n->p_links ) )) {
		delete_pNetwork( n );
		return err;
	}
	
	return err;
}

void delete_pNetwork( pNetwork *n ) {
	int i;
	for ( i=0; i<n->num_node_blocks; i++ ) {
		free( n->p_nodes[i] );
	}
	for ( i=0; i<n->num_link_blocks; i++ ) {
		free( n->p_links[i] );
	}

	free( n->p_nodes );
	free( n->p_links );
}

void reset_pNetwork( pNetwork *net ) {
	// No further processing is needed, as the block allocations remain intact
	net->num_nodes = 0;
	net->num_links = 0;
}

int add_pNode( pNetwork *n, const double x[DIMENSIONS], pNode **result ) {
	int err=0;

	// If need be, allocate a new block
	if ( n->num_nodes == n->num_node_blocks * BLOCKSIZE_NODES ) {
		if (( err = Realloc( n->p_nodes, (n->num_node_blocks+1)*sizeof *n->p_nodes ) )
		 || ( err = Malloc( n->p_nodes[n->num_node_blocks], BLOCKSIZE_NODES*sizeof **n->p_nodes ) )) {
			return err;
		}
		n->num_node_blocks++;
	}

	int block = n->num_nodes / BLOCKSIZE_NODES;
	int index = n->num_nodes % BLOCKSIZE_NODES;
	pNode *new = &n->p_nodes[block][index];
	new->used = 0;
	memcpy( new->x, x, DIMENSIONS*sizeof *x );

	n->num_nodes++;
	
	if ( result )
		*result = new;

	return err;
}

int add_pLink( pNetwork *n, pNode *from, pNode *to, pLink **result ) {
	int err=0;

	// If need be, allocate a new block
	if ( n->num_links == n->num_link_blocks * BLOCKSIZE_LINKS ) {
		if (( err = Realloc( n->p_links, (n->num_link_blocks+1)*sizeof *n->p_links ) )
		 || ( err = Malloc( n->p_links[n->num_link_blocks], BLOCKSIZE_LINKS*sizeof **n->p_links ) )) {
			return err;
		}
		n->num_link_blocks++;
	}

	int block = n->num_links / BLOCKSIZE_LINKS;
	int index = n->num_links % BLOCKSIZE_LINKS;
	pLink *new = &n->p_links[block][index];
	new->from = from;
	new->to = to;
	new->used = 0;

	n->num_links++;

	if ( result )
		*result = new;

	return err;
}

int connect_pNet( BinLeaf *leaf, struct Extraction_Params *eparams ) {
	int err=0;

	// Determine source and target coordinates
	double source[DIMENSIONS], target[DIMENSIONS];
	if ( eparams->outgoing ) {
		memcpy( source, eparams->ref, DIMENSIONS*sizeof *source );
		memcpy( target, leaf->x, DIMENSIONS*sizeof *target );
	} else {
		memcpy( source, leaf->x, DIMENSIONS*sizeof *source );
		memcpy( target, eparams->ref, DIMENSIONS*sizeof *target );
	}

	// Find the source and target nodes
	unsigned int i;
	int block=-1, index;
	pNode *source_pnode=0, *target_pnode=0, *n;
	for ( i=0; i<eparams->net->num_nodes; i++ ) {
		index = i % BLOCKSIZE_NODES;
		if ( ! index )
			block++;
		n = &eparams->net->p_nodes[block][index];
		if ( ! memcmp( source, n->x, DIMENSIONS*sizeof *source ) ) {
			source_pnode = n;
			if ( target_pnode )
				break;
		}
		if ( ! memcmp( target, n->x, DIMENSIONS*sizeof *source ) ) {
			target_pnode = n;
			if ( source_pnode )
				break;
		}
	}

	// Create what does not exist
	if ( ! source_pnode ) {
		if ( ! eparams->create_nodes )
			return 0;
		if (( err = add_pNode( eparams->net, source, &source_pnode ) ))
			return err;
	}
	if ( ! target_pnode ) {
		if ( ! eparams->create_nodes )
			return 0;
		if (( err = add_pNode( eparams->net, target, &target_pnode ) ))
			return err;
	}

	pLink *plink;
	if ( eparams->outgoing ) {
		// Unless the extraction algorithm is out of whack, no link comparison should be necessary, so skip it
		if (( err = add_pLink( eparams->net, source_pnode, target_pnode, &plink ) ))
			return err;
	} else {
		// Check whether the targeted node lies within the substrate space
		unsigned char out_of_bounds=0;
		for ( i=0; i<DIMENSIONS; i++ ) {
			if ( target[i] <= -0.5 || target[i] >= 0.5 ) {
				out_of_bounds = 1;
				break;
			}
		}
		if ( out_of_bounds ) {
			// There's no way extraction could have produced links here, skip the check
			if (( err = add_pLink( eparams->net, source_pnode, target_pnode, &plink ) ))
				return err;
		} else {
			// Target node lies within substrate space, there may already be links to it
			block=-1;
			pLink *l;
			for ( i=0; i<eparams->net->num_links; i++ ) {
				index = i % BLOCKSIZE_LINKS;
				if ( ! index )
					block++;
				l = &eparams->net->p_links[block][index];
				if ( l->from == source_pnode && l->to == target_pnode ) {
					return 0;
				}
			}

			// No such luck, add a new one
			if (( err = add_pLink( eparams->net, source_pnode, target_pnode, &plink ) ))
				return err;
		}
	}

	memcpy( plink->r, leaf->r, N_OUTPUTS*sizeof *plink->r );

	return err;
}

int build_pNetwork( pNetwork *net, CPPN *cppn, struct NEAT_Params *params, int num_inputs, const pNode *inputs, int num_outputs, const pNode *outputs ) {
	int block, index, err=0;
	unsigned int i;

	// Set up static input and output nodes
	pNode *tmp;
	for ( i=0; i<num_inputs; i++ ) {
		if (( err = add_pNode( net, inputs[i].x, &tmp ) ))
			return err;
	}
	for ( i=0; i<num_outputs; i++ ) {
		if (( err = add_pNode( net, outputs[i].x, &tmp ) ))
			return err;
	}
	net->num_inputs = num_inputs;
	net->num_outputs = num_outputs;

	struct Extraction_Params eparams = {0};
	eparams.params = params;
	eparams.cppn = cppn;
	eparams.net = net;

	if ( params->max_network_depth ) {
		// Extract links from input to first hidden layer
		eparams.outgoing = 1;
		eparams.create_nodes = 1;
		block = -1;
		for ( i=0; i<num_inputs; i++ ) {
			index = i % BLOCKSIZE_NODES;
			if ( ! index )
				block++;
			memcpy( eparams.ref, net->p_nodes[block][index].x, DIMENSIONS*sizeof *eparams.ref );
			if (( err = extract_links( &eparams ) ))
				return err;
		}

		// Extract deeper layers
		int depth;
		unsigned int i_start=num_inputs+num_outputs, i_end;
		block = (num_inputs+num_outputs-1) / BLOCKSIZE_NODES;
		for ( depth=1; depth < params->max_network_depth; depth++ ) {
			i_end = net->num_nodes;
			for ( i=i_start; i<i_end; i++ ) {
				index = i % BLOCKSIZE_NODES;
				if ( ! index )
					block++;
				memcpy( eparams.ref, net->p_nodes[block][index].x, DIMENSIONS*sizeof *eparams.ref );
				if (( err = extract_links( &eparams ) ))
					return err;
			}
			i_start = i_end;
		}
	}

	// Extract links to the output layer
	eparams.outgoing = 0;
	eparams.create_nodes = 0;
	block = (num_inputs-1) / BLOCKSIZE_NODES;
	for ( i=num_inputs; i<num_inputs+num_outputs; i++ ) {
		index = i % BLOCKSIZE_NODES;
		if ( ! index )
			block++;
		memcpy( eparams.ref, net->p_nodes[block][index].x, DIMENSIONS*sizeof *eparams.ref );
		if (( err = extract_links( &eparams ) ))
			return err;

		// Mark everything that connects to this output node
		if ( ! (params->flags & EFL_NO_BACKTRACKING) ) {
			backtrack( net, &net->p_nodes[block][index] );
		}
	}

	return err;
}

void backtrack( pNetwork *net, pNode *n ) {
	n->used = 1;
	unsigned int i;
	int index, block=-1;
	pLink *l;
	for ( i = 0; i < net->num_links; i++ ) {
		index = i % BLOCKSIZE_LINKS;
		if ( ! index )
			block++;
		l = &net->p_links[block][index];
		if ( l->to == n ) {
			l->used = 1;
			if ( ! l->from->used ) {
				backtrack( net, l->from );
			}
		}
	}
}

















