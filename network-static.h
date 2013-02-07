#ifndef _NETWORK_STATIC_H
#define _NETWORK_STATIC_H

#define N_OUTPUTS 1
#define BLOCKSIZE_NODES 256
#define BLOCKSIZE_LINKS 1024

// *** Minimal network structure, for evaluation
typedef struct eNode {
	double a;
} eNode;

typedef struct eLink {
	unsigned int from, to;
	double w;
} eLink;

typedef struct eNetwork {
	unsigned int num_nodes;
	eNode *nodes;
	
	unsigned int num_links;
	eLink *links;
} eNetwork;


// *** Prototype as declared in network.h -- Reminder
// int connect_eNet( pNode *source_pnode, pNode *target_pnode, pLink *plink, struct BinLeaf *leaf, struct Extraction_Params *eparams );

#endif

