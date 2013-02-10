#ifndef _NETWORK_STATIC_H
#define _NETWORK_STATIC_H

#define DIMENSIONS 2
#define N_OUTPUTS 1
#define BLOCKSIZE_NODES 256
#define BLOCKSIZE_LINKS 1024

#define _PNODE_DEF
typedef struct pNode {
	double x[DIMENSIONS];
	unsigned char used;

	// Index of the relevant eNode in eNetwork::nodes
	unsigned int index;
} pNode;

// *** Minimal network structure for serial evaluation
typedef struct eNode {
	double a;
	int num_inputs;
} eNode;

typedef struct eLink {
	double w;
	eNode *from;
} eLink;

typedef struct eNetwork {
	unsigned int num_nodes;
	eNode *nodes;

	int num_inputs, num_outputs;
	eNode **inputs, **outputs;
	
	unsigned int num_links;
	eLink *links;
} eNetwork;


#endif

