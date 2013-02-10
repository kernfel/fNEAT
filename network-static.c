#include "params.h"
#include "util.h"

#include "network.h"
#include "extract.h"

#include "network-static.h"

int create_eNetwork( eNetwork *e ) {
	e->nodes = 0;
	e->inputs = 0;
	e->outputs = 0;
	e->links = 0;

	e->num_nodes = 0;
	e->num_inputs = 0;
	e->num_outputs = 0;
	e->num_links = 0;

	return 0;
}

void delete_eNetwork( eNetwork *e ) {
	free( e->nodes );
	free( e->inputs );
	free( e->outputs );
	free( e->links );

	e->nodes = 0;
	e->inputs = 0;
	e->outputs = 0;
	e->links = 0;

	e->num_nodes = 0;
	e->num_inputs = 0;
	e->num_outputs = 0;
	e->num_links = 0;
}

int build_eNetwork( pNetwork *p, eNetwork *e, struct NEAT_Params *params ) {
	int err=0;

	if (( err = Realloc( e->nodes, p->num_used_nodes*sizeof *e->nodes ) )
	 || ( err = Realloc( e->inputs, p->num_inputs*sizeof *e->inputs ) )
	 || ( err = Realloc( e->outputs, p->num_outputs*sizeof *e->outputs ) )
	 || ( err = Realloc( e->links, p->num_used_links*sizeof *e->links ) )) {
		free( e->nodes );
		free( e->inputs );
		free( e->outputs );
		return err;
	}
	e->num_nodes = p->num_used_nodes;
	e->num_inputs = p->num_inputs;
	e->num_outputs = p->num_outputs;
	e->num_links = p->num_used_links;

	// Assign eNode indices to the pNodes
	int block=-1, index;
	unsigned int i, j, node_iter=0;
	pNode *n;
	for ( i=0; i<p->num_nodes; i++ ) {
		index = i % BLOCKSIZE_NODES;
		if ( ! index )
			block++;
		n = &p->p_nodes[block][index];

		if ( i < e->num_inputs ) {
			if ( n->used )
				e->inputs[i] = e->nodes + node_iter;
			else
				e->inputs[i] = 0;
		} else if ( i < e->num_inputs+e->num_outputs ) {
			if ( n->used )
				e->outputs[i-e->num_inputs] = e->nodes + node_iter;
			else
				e->outputs[i-e->num_inputs] = 0;
		}

		if ( n->used ) {
			n->index = node_iter++;
			if ( i < p->num_inputs )
				e->num_inputs++;
			else if ( i < p->num_inputs+p->num_outputs )
				e->num_outputs++;
		}
	}
	
	// Group links by target node
	pLink *l;
	eLink *next_link=e->links;
	eNode *next_node=e->nodes;
	double stretch = 3/(1-params->expression_thresholds[0]), offset = params->expression_thresholds[0] * stretch;
	int l_block, l_index;
	block = -1;
	for ( i=0; i<p->num_nodes; i++ ) {
		index = i % BLOCKSIZE_NODES;
		if ( ! index )
			block++;
		n = &p->p_nodes[block][index];
		if ( n->used ) {
			next_node->num_inputs = 0;
			l_block = -1;
			for ( j=0; j<p->num_links; j++ ) {
				l_index = j % BLOCKSIZE_LINKS;
				if ( ! l_index )
					l_block++;
				l = &p->p_links[l_block][l_index];
				if ( l->used && l->to == n) {
					next_link->w = l->r[0]*stretch - offset;
					next_link->from = e->nodes + l->from->index;
					next_node->num_inputs++;
					next_link++;
				}
			}
			next_node++;
		}
	}

	return err;
}

