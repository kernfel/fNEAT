#include <string.h>

#include "params.h"
#include "util.h"

#include "extract.h"
#include "network.h"
#include "cppn.h"

#ifndef BLOCKSIZE_NODES
#define BLOCKSIZE_NODES 256
#endif

#ifndef BLOCKSIZE_LINKS
#define BLOCKSIZE_LINKS 1024
#endif


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
	n->e_nodes[0] = 0;
	n->num_nodes = 0;
	n->num_node_blocks = 1;
	n->num_inputs = 0;
	n->num_outputs = 0;

	n->p_links[0] = 0;
	n->e_links[0] = 0;
	n->num_links = 0;
	n->num_link_blocks = 1;

	if (( err = Malloc( n->p_nodes[0], BLOCKSIZE_NODES*sizeof **n->p_nodes ) )
	 || ( err = Malloc( n->e_nodes[0], BLOCKSIZE_NODES*sizeof **n->e_nodes ) )
	 || ( err = Malloc( n->p_links[0], BLOCKSIZE_LINKS*sizeof **n->p_links ) )
	 || ( err = Malloc( n->e_links[0], BLOCKSIZE_LINKS*sizeof **n->e_links ) )) {
		delete_pNetwork( n, 0 );
		return err;
	}
	
	return err;
}

void delete_pNetwork( pNetwork *n, int retain_eNet ) {
	int i;
	if ( retain_eNet ) {
		for ( i=0; i<n->num_node_blocks; i++ ) {
			free( n->p_nodes[i] );
			free( n->e_nodes[i] );
		}
		for ( i=0; i<n->num_link_blocks; i++ ) {
			free( n->p_links[i] );
			free( n->e_links[i] );
		}
	} else {
		for ( i=0; i<n->num_node_blocks; i++ ) {
			free( n->p_nodes[i] );
		}
		for ( i=0; i<n->num_link_blocks; i++ ) {
			free( n->p_links[i] );
		}
	}

	// e_nodes and e_links can be freed even if the eNet is being retained, because they are only pointers to the blocks.
	// It is presumed that you have these elsewhere if you're deleting the pNetwork with retain_eNet==1!
	free( n->p_nodes );
	free( n->e_nodes );
	free( n->p_links );
	free( n->e_links );
}

int add_pNode( pNetwork *n, double x[DIMENSIONS], pNode **result ) {
	int err=0;

	// If need be, allocate a new block
	if ( n->num_nodes == n->num_node_blocks * BLOCKSIZE_NODES ) {
		if (( err = Realloc( n->p_nodes, (n->num_node_blocks+1)*sizeof *n->p_nodes ) )
		 || ( err = Malloc( n->p_nodes[n->num_node_blocks], BLOCKSIZE_NODES*sizeof **n->p_nodes ) )) {
			return err;
		}
		if (( err = Realloc( n->e_nodes, (n->num_node_blocks+1)*sizeof *n->e_nodes ) )
		 || ( err = Malloc( n->e_nodes[n->num_node_blocks], BLOCKSIZE_NODES*sizeof **n->e_nodes ) )) {
			free( n->p_nodes[n->num_node_blocks] );
			return err;
		}
		n->num_node_blocks++;
	}

	int block = n->num_nodes / BLOCKSIZE_NODES;
	int index = n->num_nodes % BLOCKSIZE_NODES;
	pNode *new = &n->p_nodes[block][index];
	new->n = &n->e_nodes[block][index];
	new->track = 0;
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
		if (( err = Realloc( n->e_links, (n->num_link_blocks+1)*sizeof *n->e_links ) )
		 || ( err = Malloc( n->e_links[n->num_link_blocks], BLOCKSIZE_LINKS*sizeof **n->e_links ) )) {
			free( n->p_links[n->num_link_blocks] );
			return err;
		}
		n->num_link_blocks++;
	}

	int block = n->num_links / BLOCKSIZE_LINKS;
	int index = n->num_links % BLOCKSIZE_LINKS;
	pLink *new = &n->p_links[block][index];
	new->l = &n->e_links[block][index];
	new->from = from;
	new->to = to;
	new->track = 0;

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
	unsigned int i, source_index, target_index;
	int block=-1, index;
	pNode *source_pnode=0, *target_pnode=0, *n;
	for ( i=0; i<eparams->net->num_nodes; i++ ) {
		index = i % BLOCKSIZE_NODES;
		if ( ! index )
			block++;
		n = &eparams->net->p_nodes[block][index];
		if ( ! memcmp( source, n->x, DIMENSIONS*sizeof *source ) ) {
			source_pnode = n;
			source_index = i;
			if ( target_pnode )
				break;
		}
		if ( ! memcmp( target, n->x, DIMENSIONS*sizeof *source ) ) {
			target_pnode = n;
			target_index = i;
			if ( source_pnode )
				break;
		}
	}

	// Create what does not exist
	if ( ! source_pnode ) {
		if ( ! eparams->create_nodes )
			return 0;
		source_index = eparams->net->num_nodes;
		if (( err = add_pNode( eparams->net, source, &source_pnode ) ))
			return err;
	}
	if ( ! target_pnode ) {
		if ( ! eparams->create_nodes )
			return 0;
		target_index = eparams->net->num_nodes;
		if (( err = add_pNode( eparams->net, target, &target_pnode ) ))
			return err;
	}

	// Unless the extraction algorithm is out of whack, no link comparison should be necessary, so skip it
	pLink *plink;
	if (( err = add_pLink( eparams->net, source_pnode, target_pnode, &plink ) ))
		return err;

	plink->l->from = source_index;
	plink->l->to = target_index;

	err = connect_eNet( source_pnode, target_pnode, plink, leaf, eparams );

	return err;
}

int build_network( pNetwork *net, CPPN *cppn, struct NEAT_Params *params, int num_inputs, pNode *inputs, int num_outputs, pNode *outputs ) {
	int block, index, err=0;
	unsigned int i;

	// Set up static input and output nodes
	pNode *tmp;
	for ( i=0; i<num_inputs; i++ ) {
		if (( err = add_pNode( net, inputs[i].x, tmp ) ))
			return err;
		if ( inputs[i].n )
			memcpy( tmp->n, inputs[i].n, sizeof *tmp->n );
	}
	net->num_inputs = num_inputs;
	for ( i=0; i<num_outputs; i++ ) {
		if (( err = add_pNode( net, outputs[i].x, tmp ) ))
			return err;
		if ( outputs[i].n )
			memcpy( tmp->n, outputs[i].n, sizeof *tmp->n );
	}
	net->num_outputs = num_outputs;

	struct Extraction_Params eparams = {0};
	eparams->params = params;
	eparams->cppn = cppn;
	eparams->net = net;

	// Extract links from input to first hidden layer
	eparams->outgoing = 1;
	eparams->create_nodes = 1;
	block = -1;
	for ( i=0; i<num_inputs; i++ ) {
		index = i % BLOCKSIZE_NODES;
		if ( ! index )
			block++;
		memcpy( eparams->ref, net->p_nodes[block][index].x, DIMENSIONS*sizeof *eparams->ref );
		if (( err = extract_links( eparams ) ))
			return err;
	}

	// Extract deeper layers
	int depth;
	unsigned int start_level=num_inputs+num_outputs, end_level;
	block = (num_inputs+num_outputs-1) / BLOCKSIZE_NODES;
	for ( depth=1; depth < params->max_network_depth; depth++ ) {
		end_level = net->num_nodes;
		for ( i=start_level; i<end_level; i++ ) {
			index = i % BLOCKSIZE_NODES;
			if ( ! index )
				block++;
			memcpy( eparams->ref, net->p_nodes[block][index].x, DIMENSIONS*sizeof *eparams->ref );
			if (( err = extract_links( eparams ) ))
				return err;
		}
		start_level = end_level;
	}

	// Extract links to the output layer
	eparams->outgoing = 0;
	eparams->create_nodes = 0;
	block = (num_inputs-1) / BLOCKSIZE_NODES;
	for ( i=num_inputs; i<num_inputs+num_outputs; i++ ) {
		index = i % BLOCKSIZE_NODES;
		if ( ! index )
			block++;
		memcpy( eparams->ref, net->p_nodes[block][index].x, DIMENSIONS*sizeof *eparams->ref );
		if (( err = extract_links( eparams ) ))
			return err;

		// Mark everything that connects to this output node
		if ( ! (params->flags & EFL_RETAIN_DEAD_ENDS) ) {
			backtrack( net, &net->p_nodes[block][index] );
		}
	}

	// Prune all the frills not connected to an output. (Yes, this may include inputs!)
	block = -1;
	// Todo, require slight rewrite: Don't refer to eNet/eLinks/eNodes before pruning; store CPPN activation in pLink
	// After pruning, compress pNet memory structure, and only then call eNet formation code.
}

void backtrack( pNetwork *net, pNode *n ) {
	unsigned int l_i;
	int l_index, l_block=-1;
	pLink *l;
	for ( l_i = 0; l_i < net->num_links; l_i++ ) {
		l_index = l_i % BLOCKSIZE_LINKS;
		if ( ! l_index )
			l_block++;
		l = &net->p_links[l_block][l_index];
		if ( l->to == n ) {
			l->track = 1;
			if ( ! l->from->track ) {
				l->from->track = 1;
				backtrack( net, l->from );
			}
		}
	}
}

















