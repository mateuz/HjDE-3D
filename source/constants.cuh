#ifndef _CONSTANTS_H
#define _CONSTANTS_H

/*
 * | ---------------------------
 * | Data Type | size (byte)   |
 * | ---------------------------
 * | short     |      2        |
 * | int       |      4        |
 * | uint      |      4        |
 * | float     |      4        |
 * | double    |      8        |
 * | ---------------------------
 * | Total     |   182 / 65536 |
 * | ---------------------------
 */

typedef struct {
    float x_min;
    float x_max;
    uint n_dim;
    uint ps;
} Configuration;

extern __constant__ Configuration params;
extern __constant__ float F_Lower;
extern __constant__ float F_Upper;
extern __constant__ float T;
extern __constant__ char S_AB[150];
extern __constant__ int PL;

#endif
