#ifndef _CPPN_H
#define _CPPN_H

#define CFL_USE_DIST		(1 << 0)
#define CFL_USE_BIAS		(1 << 1)
#define CFL_ALLOW_RECURRENCE	(1 << 2)
#define CFL_ALLOW_O_TO_O	(1 << 3)
#define CFL_SQUARE_DIST		(1 << 4)

enum CPPNFunc {
	CF_GAUSS,
	CF_SIGMOID,
	CF_SINE,
	CF_LINEAR,
	CF_STEP,
	CF_ABS
};

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

typedef struct CPPN_Params {
	int num_dimensions;		// Dimensionality of the substrate space
	enum CPPNFunc *allowed_funcs;
	int num_allowed_funcs;
	
	unsigned int innov_counter;
	
	double	add_link_prob,
		add_node_prob,
		change_weight_prob,	// Change link weights by a maximum of +/- change_weight_rate
		change_weight_rate,
		enable_link_prob;	// Enable previously disabled links
	
	int num_activations;		// Maximum number of iterations through the nodes to fully activate the net.
					// Has no effect if ~CFL_ALLOW_RECURRENCE
	
	unsigned int flags;
} CPPN_Params;


// Constructors
int create_CPPN(	CPPN *net,
			int num_outputs,		// Number of output values
			enum CPPNFunc *output_funcs,	// Output activation functions, eg sigmoid to normalise
			int *outputs_linked,		// Boolean array defining those outputs that will be fully linked with the net's inputs
			int create_disabled_links,	// Boolean. If true, will fully connect all outputs, but disable those not indicated by outputs_linked
			CPPN_Params *parameters );

int clone_CPPN( CPPN *net, const CPPN *original );

// Destructor
void delete_CPPN( CPPN *trash );

// ** Public methods

// Mutate net according to the probabilities given in parameters
int mutate_CPPN( CPPN *net, CPPN_Params *parameters );

// Activate net to read its (final) output value at coords.
// Note that neither distance nor bias should be added to coords!
// Returns the cumulated activation deltas of the last activation iteration.
double read_CPPN( CPPN *net, const CPPN_Params *parameters, double *coords, double *output );

// ** Private methods

// Set possible_sources[i]=0 for all i where net->nodes[i] is reachable by following any forward links from net->nodes[target_id].
// Returns the number of nodes excluded in this way, accounting for any previously excluded in possible_sources.
// Warning: May recurse indefinitely on a recurrent net!
int CPPN_exclude_recurrent_links( const CPPN *net, const CPPN_Params *params, int target_id, int *possible_sources );

// Insert a link into net, maintaining all links within net sorted by postsynaptic node id
int CPPN_insert_link( CPPN *net, CPPN_Params *params, int from, int to, double weight, int is_disabled, int no_realloc );

#endif

