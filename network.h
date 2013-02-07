#ifndef _NETWORK_H
#define _NETWORK_H

#include "params.h"
#include "extract.h"

// *** Full network structure for extraction (p=positional)
typedef struct pNode {
	eNode *n;
	double x[DIMENSIONS];
} pNode;

typedef struct pLink {
	eLink *l;
	pNode *from, *to;
} pLink;

typedef struct pNetwork {
	unsigned int num_nodes, num_node_blocks;
	pNode **p_nodes;
	eNode **e_nodes;
	
	unsigned int num_links, num_link_blocks;
	pLink **p_links;
	eLink **e_links;
} pNetwork;


// *** Constructor
int create_pNetwork( pNetwork *n );

// *** Destructor
void delete_pNetwork( pNetwork *n , int retain_eNet );

// *** Function prototype - use in specific network implementations
// This function is called by connect_pNet once the pNet connection is established and should initialise eNode and eLink data.
int connect_eNet( pNode *source_pnode, pNode *target_pnode, pLink *plink, struct BinLeaf *leaf, struct Extraction_Params *eparams );

// *** Public
int connect_pNet( struct BinLeaf *leaf, struct Extraction_Params *eparams );

// *** Internal
int add_pNode( pNetwork *n, double x[DIMENSIONS], pNode **result );
int add_pLink( pNetwork *n, pNode *from, pNode *to, pLink **result );

#endif

