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
	
	CPPN_Link *links;		// Links, sorted by innovation id
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


// ** Constructors
int create_CPPN( CPPN *net, struct NEAT_Params *parameters );

// Copy constructor
int clone_CPPN( CPPN *net, const CPPN *original );

// Allocate storage for a new CPPN guided by the respective num_* values
// Allocation aside, no initialisation takes place.
int allocate_CPPN( CPPN *net );

// ** Destructor
void delete_CPPN( CPPN *trash );

// ** Public methods

// Randomise all weights uniformly within +-params->random_weights_range
void randomise_CPPN_weights( CPPN *net, struct NEAT_Params *params );

// Mutate net according to the probabilities given in parameters
int mutate_CPPN( CPPN *net, struct NEAT_Params *parameters, Node_Innovation *ni, Link_Innovation *li );

// Cross mate into net, keeping all of net's weight values where applicable
int crossover_CPPN( CPPN *net, const CPPN *mate, struct NEAT_Params *params );

// Activate net to read its (final) output value at coords.
// Note that neither distance nor bias should be added to coords!
// Returns the cumulated activation deltas of the last activation iteration.
double read_CPPN( CPPN *net, const struct NEAT_Params *parameters, double *coords, double *output );

// Determine genetic distance
double get_genetic_distance( const CPPN *net1, const CPPN *net2, const struct NEAT_Params *parameters );

// ** Private methods

// Set possible_sources[i]=0 for all i where net->nodes[i] is reachable by following any forward links from net->nodes[target_id].
// Returns the number of nodes excluded in this way, accounting for any previously excluded in possible_sources.
// Warning: May recurse indefinitely on a recurrent net!
int CPPN_exclude_recurrent_links( const CPPN *net, const struct NEAT_Params *params, int target_id, int *possible_sources );

// Insert a link into net, maintaining all links within net sorted by postsynaptic node id
int CPPN_insert_link( CPPN *net, struct NEAT_Params *params, int from, int to, double weight, int is_disabled, unsigned int innov_id );

// Change the innov_id of a link and update its place in net->links_innovsort.
// link must be a pointer to the relevant element of net->links_innovsort.
int CPPN_update_innov_id( CPPN *net, unsigned int old_id, unsigned int new_id );

// Calculate activation value from function and input
double CPPN_func( enum CPPNFunc fn, double x );

#endif

