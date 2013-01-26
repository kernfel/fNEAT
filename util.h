#ifndef _UTIL_H
#define _UTIL_H

#define E_ALLOC (-1)

// Safe allocation: Returns 0 or E_ALLOC on failure
int Malloc( void *p, size_t s );
int Calloc( void *p, size_t n, size_t s );
int Realloc( void *p, size_t s );

#define max(a,b) ((a)>(b)?(a):(b))

#endif

