#ifndef _UTIL_H
#define _UTIL_H

#define E_ALLOC (-1)
#define E_ASSERT (-2)

// Safe allocation: Returns 0 or E_ALLOC on failure
#define Malloc(p,s) Malloc_((void**)&p,s)
#define Calloc(p,n,s) Calloc_((void**)&p,n,s)
#define Realloc(p,s) Realloc_((void**)&p,s)

int Malloc_( void **p, size_t s );
int Calloc_( void **p, size_t n, size_t s );
int Realloc_( void **p, size_t s );

#define max(a,b) ((a)>(b)?(a):(b))

#ifdef DEBUG
#include <stdio.h>

#define ASSERT(foo) do { \
	if(!(foo)) { \
		fprintf( stderr, "[Debug] <" __FILE__ ":%d> Assertion failed: " #foo, __LINE__ ); \
		return E_ASSERT; \
	} \
} while(0)

#else // DEBUG

#define ASSERT(foo)

#endif // DEBUG

#endif

