#ifndef _NETWORK_H
#define _NETWORK_H

#include "params.h"
#include "extract.h"
#include "cppn.h"

#ifndef BLOCKSIZE_NODES
#define BLOCKSIZE_NODES 256
#endif

#ifndef BLOCKSIZE_LINKS
#define BLOCKSIZE_LINKS 1024
#endif

// *** Full network structure for extraction (p=positional)
typedef struct pNode {
	double x[DIMENSIONS];
	unsigned char used;
} pNode;

typedef struct pLink {
	pNode *from, *to;
	double r[N_OUTPUTS];
	unsigned char used;
} pLink;

typedef struct pNetwork {
	unsigned int num_nodes, num_node_blocks;
	pNode **p_nodes;
	int num_inputs, num_outputs;
	
	unsigned int num_links, num_link_blocks;
	pLink **p_links;
} pNetwork;


// *** Constructor
int create_pNetwork( pNetwork *n );

// *** Destructor
void delete_pNetwork( pNetwork *n );

// *** Public

// Build a pNetwork in the substrate space (]-0.5, 0.5[ ^ DIMENSIONS), using the information encapsulated within cppn.
// Links are built starting from the input nodes in up to params->max_network_depth layers, which are then connected to the output nodes.
// Both input and output may lie within or outside the substrate space.
// The resulting pNetwork may contain references to links and nodes that aren't implied in input-to-output structure (::used = 0).
// The function compiling the network for evaluation will have to deal with these according to its own paradigm.
int build_pNetwork( pNetwork *net, CPPN *cppn, struct NEAT_Params *params, int num_inputs, const pNode *inputs, int num_outputs, const pNode *outputs );

// Lazily prepare the pNetwork to receive a new build
void reset_pNetwork( pNetwork *net );

// *** Internal

// Form a new connection. Called by the extraction algorithm once a point in the hypercube is deemed worthy of expression.
// If eparams->outgoing == 0 and the target link is inside the substrate space, link duplicity is checked. Otherwise, it is assumed
// that the connection is definitely new, and only node duplicity is considered.
int connect_pNet( BinLeaf *leaf, struct Extraction_Params *eparams );

int add_pNode( pNetwork *n, const double x[DIMENSIONS], pNode **result );
int add_pLink( pNetwork *n, pNode *from, pNode *to, pLink **result );

// Recursively find all nodes that connect to n, marking them and the implied links as used.
void backtrack( pNetwork *net, pNode *n );

#endif

