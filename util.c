#include <stdlib.h>

#include "util.h"

int Malloc( void *p, size_t s ) {
	p = malloc( s );
	return p ? 0 : E_ALLOC;
}

int Calloc( void *p, size_t n, size_t s ) {
	p = calloc( n, s );
	return p ? 0 : E_ALLOC;
}

int Realloc( void *p, size_t s ) {
	void *q = realloc( p, s );
	if ( ! q )
		return E_ALLOC;
	p = q;
	return 0;
}

