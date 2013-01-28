#ifndef _PARAMS_H
#define _PARAMS_H


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

struct NEAT_Params {
// Flags, see above
	unsigned int flags;

// Parameters related to Hyper
	int num_dimensions;		// Dimensionality of the substrate space

// Parameters related to the NEAT algorithm proper
	int population_size;
	int extinction_threshold;	// Minimum number of members a species must have to survive
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

// Parameters related to CPPN function
	enum CPPNFunc *allowed_funcs;
	int num_allowed_funcs;

	int num_activations;		// Maximum number of iterations through the nodes to fully activate the net.
					// Has no effect if ~CFL_ALLOW_RECURRENCE

// UID counters
	unsigned int innov_counter;
	unsigned int species_counter;
};


#endif

