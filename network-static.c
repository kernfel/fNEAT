#include "network-static.h"
#include "network.h"
#include "extract.h"

int connect_eNet( pNode *source_pnode, pNode *target_pnode, pLink *plink, BinLeaf *leaf, struct Extraction_Params *eparams ) {
	source_pnode->n->a = 0;
	target_pnode->n->a = 0;
	plink->l->w = leaf->r[0];
	return 0;
}

