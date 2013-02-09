#include <math.h>
#include <string.h>

#include "util.h"
#include "params.h"
#include "network.h"

#include "extract.h"

static const int CLUSTERSIZE = 1<<DIMENSIONS;

int extract_links( struct Extraction_Params *eparams ) {
	int err=0;
	
	BinLeaf root = {0};
	eparams->root = &root;
	
	if (( err = build_tree( &root, eparams ) )) {
		goto cleanup;
	}
	
	if (( err = extract_tree( &root, eparams ) )) {
		goto cleanup;
	}

cleanup:
	delete_tree( &root );
	eparams->root = 0;
	return err;
}

int build_tree( BinLeaf *p, struct Extraction_Params *eparams ) {
	int i, j, err=0;

	// Divide the parent
	if (( err = Malloc( p->c, CLUSTERSIZE*sizeof *p->c ) ))
		return err;

	double w = 1.0 / (1 << (p->level+2));
	for ( i=0; i<CLUSTERSIZE; i++ ) {
		p->c[i].c = 0;
		p->c[i].level = p->level+1;
		for ( j=0; j<DIMENSIONS; j++ ) {
			// Adjust child coords to be in the center of the quadrant/octant
			p->c[i].x[j] = p->x[j] + ( (i>>j & 1) ? (-1) : 1 ) * w;
		}

		// Populate the new child with CPPN data
		if ( eparams->outgoing )
			read_CPPN( eparams->cppn, eparams->params, eparams->ref, p->c[i].x, p->c[i].r );
		else
			read_CPPN( eparams->cppn, eparams->params, p->c[i].x, eparams->ref, p->c[i].r );
	}

	// Recurse
	if ( p->level < eparams->params->min_resolution \
	 || ( p->level < eparams->params->max_resolution && get_binleaf_variance(p, eparams->params) > eparams->params->variance_threshold ) ) {
		for ( i=0; i<CLUSTERSIZE; i++ ) {
			if (( err = build_tree( &p->c[i], eparams ) ))
				return err;
		}
	}

	return err;
}

int extract_tree( BinLeaf *p, struct Extraction_Params *eparams ) {
	int i, j, err=0;
	// If the children c of p do not have children gc, that means that either:
	// (a) c weren't recursed into because the variance threshold wasn't met
	//	=> c's are not different enough to warrant expressing
	//	=> express p and move on
	// (b) c were at highest resolution
	//	=> check variance over all c to decide whether to express them or not
	// Since variance is not calculated recursively, this process may leave a few fragments from min_resolution.
	if ( p->c[0].c \
	 || ( p->level == eparams->params->max_resolution && get_binleaf_variance(p, eparams->params) > eparams->params->variance_threshold ) ) {
		for ( i=0; i<CLUSTERSIZE; i++ ) {
			if (( err = extract_tree( &p->c[i], eparams ) ))
				return err;
		}
	} else {
		// Band pruning: See whether I'm in a slope of some sort
		double w = 1.0 / (1<<p->level);
		double dband=0;
		for ( i=0; i<DIMENSIONS; i++ ) {
			double neighbour[DIMENSIONS], left[N_OUTPUTS], right[N_OUTPUTS];
			for ( j=0; j<DIMENSIONS; j++ )
				neighbour[j] = p->x[j];

			// Left neighbour
			neighbour[i] = p->x[i] - w;
			get_activation_at( neighbour, left, eparams );
			// Right neighbour
			neighbour[i] = p->x[i] + w;
			get_activation_at( neighbour, right, eparams );
			
			// Calculate weighted difference
			double dleft=0, dright=0;
			for ( j=0; j<N_OUTPUTS; j++ ) {
				dleft += eparams->params->output_bandpruning_weight[j] * fabs(p->r[j] - left[j]);
				dright += eparams->params->output_bandpruning_weight[j] * fabs(p->r[j] - right[j]);
			}
			double ddim = dleft < dright ? dleft : dright;
			if ( ddim > dband )
				dband = ddim;
		}

		// Add a new connection
		if ( dband > eparams->params->band_threshold ) {
			if (( err = connect_pNet( p, eparams ) ))
				return err;
		}
	}

	return err;
}

void get_activation_at( double x[DIMENSIONS], double r[N_OUTPUTS], struct Extraction_Params *eparams ) {
	int i, read_new=0;
	BinLeaf *b = eparams->root;
	// Check whether x lies within root's bounding box at all
	double w = 1.0 / (1<<(b->level+1));
	for ( i=0; i<DIMENSIONS; i++ ) {
		if ( x[i] > b->x[i]+w || x[i] < b->x[i]-w ) {
			read_new = 1;
			break;
		}
	}
	
	if ( !read_new ) {
		// Drill down into the tree
		while (1) {
			// b is at the requested coordinates, return its readout.
			// No need to check for eparams->outgoing, since the tree was constructed with the same parameters
			if ( ! memcmp(x, b->x, DIMENSIONS*sizeof *x) ) {
				memcpy( r, b->r, N_OUTPUTS*sizeof *r );
				return;
			}
			
			// Determine the appropriate child leaf - 0 for left, 1 for right, dim 0 is right-most bit
			int idx=0;
			for ( i=0; i<DIMENSIONS; i++ ) {
				if ( x[i] > b->x[i] )
					idx |= 1 << i;
			}

			// b has children, c[idx] is closest to the requested coords, check that out
			if ( b->c )
				b = &b->c[idx];

			// b has no children, compute the readout afresh
			else
				break;
		}
	}

	if ( eparams->outgoing )
		read_CPPN( eparams->cppn, eparams->params, eparams->ref, x, r );
	else
		read_CPPN( eparams->cppn, eparams->params, x, eparams->ref, r );
}

double get_binleaf_variance( BinLeaf *p, struct NEAT_Params *params ) {
	int i,j;
	double sum_of_variances=0;
	for ( j=0; j<N_OUTPUTS; j++ ) {
		double sum=0, mean, sum_of_deviations=0;
		for ( i=0; i<CLUSTERSIZE; i++ ) {
			sum += p->c[i].r[j];
		}
		mean = sum/CLUSTERSIZE;
		for ( i=0; i<CLUSTERSIZE; i++ ) {
			sum_of_deviations += fabs( mean - p->c[i].r[j] );
		}

		sum_of_variances += sum_of_deviations/CLUSTERSIZE * params->output_variance_weight[j];
	}
	
	return sum_of_variances/N_OUTPUTS;
}

void delete_tree( BinLeaf *p ) {
	int i;
	if ( ! p->c )
		return;
	for ( i=0; i<CLUSTERSIZE; i++ )
		delete_tree( &p->c[i] );
	free( p->c );
	p->c = 0;
}

