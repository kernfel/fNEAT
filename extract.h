#ifndef _EXTRACT_H
#define _EXTRACT_H

#include "params.h"
#include "cppn.h"

typedef struct BinLeaf {
	struct BinLeaf *c;
	double r[N_OUTPUTS];
	double x[DIMENSIONS];
	int level;
} BinLeaf;

struct Extraction_Params {
	BinLeaf *root;
	double ref[DIMENSIONS];
	unsigned char outgoing;
	unsigned char create_nodes;
	CPPN *cppn;
	struct NEAT_Params *params;
	struct pNetwork *net;
};

// *** Public

// Main entry point. Supply coordinates of a reference point from/to which to build connections
// Requires eparams to be populated with all of the following: ref, outgoing, cppn, params, net.
int extract_links( struct Extraction_Params *eparams );


// *** Internal

// Build a bin/quad/oct/... tree, following resolution and variance parameters
int build_tree( BinLeaf *p, struct Extraction_Params *eparams );

// Extract connections from a completed tree, depth-first
int extract_tree( BinLeaf *p, struct Extraction_Params *eparams );

// Efficiently get the CPPN readout at coordinates x
void get_activation_at( double x[DIMENSIONS], double activation[N_OUTPUTS], struct Extraction_Params *eparams );

// Calculate the variance among a leaf's child nodes
double get_binleaf_variance( BinLeaf *p );

// Prune the shrubbery to a smoking stump
void delete_tree( BinLeaf *p );

#endif

