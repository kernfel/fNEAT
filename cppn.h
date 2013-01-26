#ifndef _CPPN_H
#define _CPPN_H

typedef struct CPPN_Link {
	unsigned int innov_id;
	
	int from, to;			// ids of the source/target nodes.
	double weight;
	
	int is_disabled;
} CPPN_Link;

typedef struct CPPN_Node {
	enum CPPNFunc func;
	double activation;
} CPPN_Node;

typedef struct CPPN {
	CPPN_Node *nodes;		// Includes input, output and hidden nodes, in this order
	int num_inputs, num_outputs, num_hidden;
	
	CPPN_Link *links;		// Links, sorted by innovation number
	CPPN_Link **links_nodesort;	// Pointers to the above, sorted by postsynaptic node id
	int num_links;
} CPPN;

// Node insertion summary, contains innov_id's of the deactivated and the inserted links
typedef struct Node_Innovation {
	unsigned int replaced_link;
	unsigned int link_in, link_out;
} Node_Innovation;

// Link insertion summary
typedef struct Link_Innovation {
	unsigned int innov_id;
	int type;	// 0: input to output node, from/to carry node indices
			// 1: hidden to output
			// 2: input to hidden
			// 3: hidden to hidden, from/to carry lowest link innov_id's at the respective nodes
	unsigned int from, to;
} Link_Innovation;


// Constructors
int create_CPPN(	CPPN *net,
			int num_outputs,		// Number of output values
			enum CPPNFunc *output_funcs,	// Output activation functions, eg sigmoid to normalise
			int *outputs_linked,		// Boolean array defining those outputs that will be fully linked with the net's inputs
			int create_disabled_links,	// Boolean. If true, will fully connect all outputs, but disable those not indicated by outputs_linked
			struct NEAT_Params *parameters );

int clone_CPPN( CPPN *net, const CPPN *original );

// Destructor
void delete_CPPN( CPPN *trash );

// ** Public methods

// Mutate net according to the probabilities given in parameters
int mutate_CPPN( CPPN *net, struct NEAT_Params *parameters, Node_Innovation *ni, Link_Innovation *li );

// Activate net to read its (final) output value at coords.
// Note that neither distance nor bias should be added to coords!
// Returns the cumulated activation deltas of the last activation iteration.
double read_CPPN( CPPN *net, const struct NEAT_Params *parameters, double *coords, double *output );

// Determine genetic distance
double get_genetic_distance( CPPN *net1, CPPN *net2, const struct NEAT_Params *parameters );

// ** Private methods

// Set possible_sources[i]=0 for all i where net->nodes[i] is reachable by following any forward links from net->nodes[target_id].
// Returns the number of nodes excluded in this way, accounting for any previously excluded in possible_sources.
// Warning: May recurse indefinitely on a recurrent net!
int CPPN_exclude_recurrent_links( const CPPN *net, const struct NEAT_Params *params, int target_id, int *possible_sources );

// Insert a link into net, maintaining all links within net sorted by postsynaptic node id
int CPPN_insert_link( CPPN *net, struct NEAT_Params *params, int from, int to, double weight, int is_disabled, int no_realloc );

#endif

