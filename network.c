#include <string.h>

#include "params.h"
#include "util.h"

#include "extract.h"
#include "network.h"

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
		source_index = eparams->net->num_nodes;
		if (( err = add_pNode( eparams->net, source, &source_pnode ) ))
			return err;
	}
	if ( ! target_pnode ) {
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





















