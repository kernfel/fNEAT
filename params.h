#ifndef _PARAMS_H
#define _PARAMS_H

// Default network implementation. To use other implementations, compile with the appropriate "-include file.h" argument.
#ifndef N_OUTPUTS
#include "network-static.h"
#endif

//#define VERBOSE 1

// *** CPPN flags
#define CFL_USE_DIST			(1 << 0) // Use calculated distance input
#define CFL_SQUARE_DIST			(1 << 1) // Feed the squared distance, rather than the linear one, into the distance input
#define CFL_USE_BIAS			(1 << 2) // Use a constant bias (1) input
#define CFL_ALLOW_RECURRENCE		(1 << 3) // Allow recurrent connections, i.e. loops with multiple nodes
#define CFL_ALLOW_OUTPUT_TO_HIDDEN	(1 << 4) // Allow output nodes to link back into hidden nodes
#define CFL_ALLOW_OUTPUT_TO_OUTPUT	(1 << 5) // Allow output nodes to feed into other outputs (and themselves, if recurrence is allowed)
#define CFL_ALLOW_OUTPUT_TO_SELF	(1 << 6) // Allow self-recurrent links on output nodes regardless of out2out flag
						// Has no effect if recurrence is disabled.
#define CFL_MASK_INIT_UNLINKED		(1 << 7) // Fully link the network at init, but disable links not selected through params.initially_linked_outputs
						// to facilitate their subsequent discovery.
#define CFL_NO_DISTCALC_NORM		(1 << 8) // Assume "small" network, calculate genetic distance without normalising by link count

// *** Extraction flags
#define EFL_NO_BACKTRACKING		(1 << 10) // Don't specially mark substrate nodes that don't reach both an input and an output node 

// *** Error codes
#define E_NO_OFFSPRING (-10)

// HyperNEAT's substrate space dimensionality
#ifndef DIMENSIONS
#define DIMENSIONS 2
#endif

// Number of output values in the CPPN
#ifndef N_OUTPUTS
#define N_OUTPUTS 1
#endif

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

// Parameters related to network extraction
	int	min_resolution,		// Minimum resolution (i.e. number of binary subdivisions) of the substrate space to explore
		max_resolution,
		max_network_depth;	// Number of hidden layers to explore in the substrate space

	double	variance_threshold,	// Min variance among the children (quadrants etc) of a given spatial region required to continue exploration
		output_variance_weight[N_OUTPUTS],	// Relative weights of the CPPN outputs for variance calculation and band pruning
			// Note, the variance threshold is checked against the weighted mean of variances for each CPPN output value.

		band_threshold,		// Min gradient of a quadrant relative to its neighbours required to warrant link expression
			// Note, the band threshold is checked against the steepest gradient and disregards flatter axes.
		output_bandpruning_weight[N_OUTPUTS],

		expression_thresholds[N_OUTPUTS];	// Min absolute activation of the given CPPN output to warrant link expression

// Parameters related to HyperNEAT function
	enum CPPNFunc output_funcs[N_OUTPUTS];	// Output activation functions
	int initially_linked_outputs[N_OUTPUTS];	// Output nodes to fully connect with inputs at initialisation

// Parameters related to CPPN function
	enum CPPNFunc *allowed_funcs;
	int num_allowed_funcs;

	int num_activations;		// Maximum number of iterations through the nodes to fully activate the net.
					// Has no effect if ~CFL_ALLOW_RECURRENCE

// Parameters related to the NEAT algorithm proper
	int	population_size,
		extinction_threshold,	// Minimum number of offspring a species must have to survive
		champion_threshold,	// Minimum number of members a species must have to nominate an unmutated champion
		target_num_species;	// Number of species to aim for. Set to 0 to disable dynamic speciation

	double	survival_quota,		// Percentage of members that are allowed to reproduce
		speciation_threshold,	// Genetic distance threshold
		d_speciation_threshold,	// Rate of change (up or down) to the speciation_threshold in dynamic speciation mode
		max_speciation_threshold,	// Upper bound for dynamic speciation
		disjoint_factor,	// Factors for determining genetic distance
		excess_factor,
		weight_factor;
	
	int	stagnation_age_threshold;	// Number of generations a species is allowed to stagnate without penalty
	double	stagnation_score_threshold,	// Amount of change needed to escape the stagnation guillotine
		stagnation_penalty;

// Parameters related to mutations
	double	add_link_prob,
		add_node_prob,
		enable_link_prob,	// Enable previously disabled links
		disable_link_prob,
		crossover_prob,
		interspecies_mating_prob,
		
		mutate_weights_prob,		// Probability that the network's weights are mutated (1)
		perturb_weights_proportion,	// Proportion of links whose weights are perturbed, given (1)
		perturb_weights_range,		// Upper/lower bound (+/- ~) of weight perturbation
		reassign_weights_proportion,	// Proportion of links that are assigned random weights, given (1)
		random_weights_range;		// Upper/lower bound (+/- ~) for randomly assigned weights

// UID counters -- initialise to 0
	unsigned int innov_counter;
	unsigned int species_counter;
	unsigned int individual_counter;
};


#endif

