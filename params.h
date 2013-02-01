#ifndef _PARAMS_H
#define _PARAMS_H


#define CFL_USE_DIST		(1 << 0) // Use calculated distance input
#define CFL_USE_BIAS		(1 << 1) // Use a constant bias (1) input
#define CFL_ALLOW_RECURRENCE	(1 << 2) // Allow recurrent connections, including loops with multiple nodes
#define CFL_ALLOW_O_TO_O	(1 << 3)
#define CFL_SQUARE_DIST		(1 << 4) // Feed the squared distance, rather than the linear one, into the distance input
#define CFL_MASK_INIT_UNLINKED	(1 << 5) // Fully link the network at init, but disable links not selected through params.initially_linked_outputs
					// to facilitate their subsequent discovery.
#define CFL_NO_DISTCALC_NORM	(1 << 6) // Assume "small" network, calculate genetic distance without normalising by link count

enum CPPNFunc {
	CF_GAUSS,
	CF_SIGMOID,
	CF_SINE,
	CF_LINEAR,
	CF_STEP,
	CF_ABS
};

struct NEAT_Params {
// Flags, see above
	unsigned int flags;

// Parameters related to Hyper
	int num_dimensions;		// Dimensionality of the substrate space
	int num_outputs;		// Number of output values in the CPPN
	enum CPPNFunc *output_funcs;	// Output activation functions
	int *initially_linked_outputs;	// Output nodes to fully connect with inputs at initialisation

// Parameters related to CPPN function
	enum CPPNFunc *allowed_funcs;
	int num_allowed_funcs;

	int num_activations;		// Maximum number of iterations through the nodes to fully activate the net.
					// Has no effect if ~CFL_ALLOW_RECURRENCE

// Parameters related to the NEAT algorithm proper
	int population_size;
	int extinction_threshold;	// Minimum number of offspring a species must have to survive
	double	survival_quota,		// Percentage of members that are allowed to reproduce
		speciation_threshold,	// Genetic distance threshold
		disjoint_factor,	// Factors for determining genetic distance
		excess_factor,
		weight_factor;

// Parameters related to mutations
	double	add_link_prob,
		add_node_prob,
		change_weight_prob,	// Change link weights by a maximum of +/- change_weight_rate
		change_weight_rate,
		enable_link_prob,	// Enable previously disabled links
		crossover_prob;

// UID counters
	unsigned int innov_counter;
	unsigned int species_counter;
};


#endif

