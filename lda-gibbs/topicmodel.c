/*---------------------------------------------------
* file:    topicmodel.c
* purpose: run standard topic model
* inputs:  T (number of topics), iter (number of iterations), docword.txt
* outputs: Nwt.txt, Ndt.txt, z.txt
* author:  newman@uci.edu
*-------------------------------------------------*/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <assert.h>
#include "tlib.h"

/*==========================================
* main
*========================================== */
int main(int argc, char* argv[])
{
	int T; // number of topics
	int W; // number of unique words
	int D; // number of docs
	int N; // number of words in corpus

	int i, iter, seed;
	int *w, *d, *z, *order;
	double **Nwt, **Ndt, *Nt;
	double alpha, beta;
	char * file_name = NULL;
	char * data_format = NULL;

	if (argc < 6) {
		fprintf(stderr, "usage: %s T iter seed data_format input_file \n", argv[0]);
		exit(-1);
	}
	T    = atoi(argv[1]); assert(T>0);
	iter = atoi(argv[2]); assert(iter>0);
	seed = atoi(argv[3]); assert(seed>0);
	data_format = argv[4];
	file_name = argv[5];

	// reads the total number of instances
	if (!strcmp(data_format, "sm"))
		N = countN(file_name);
	else
		N = countN_ldac(file_name);
	w = ivec(N);
	d = ivec(N);
	z = ivec(N);
	if (!strcmp(data_format, "sm"))
		read_dw(file_name, d, w, &D, &W);
	else
		read_ldac(file_name, d, w, &D, &W);
	Nwt = dmat(W,T);
	Ndt = dmat(D,T);
	Nt  = dvec(T);  

	alpha = 0.05 * N / (D * T);
	beta  = 0.01;

	printf("format     = %s\n", data_format);
	printf("file name  = %s\n", file_name);
	printf("seed       = %d\n", seed);
	printf("N          = %d\n", N);
	printf("W          = %d\n", W);
	printf("D          = %d\n", D);
	printf("T          = %d\n", T);
	printf("iter       = %d\n", iter);
	printf("alpha      = %f\n", alpha);
	printf("beta       = %f\n", beta);

	srand48(seed);
	randomassignment_d(N, T, w, d, z, Nwt, Ndt, Nt);
	order = randperm(N);

	add_smooth_d(D, T, Ndt, alpha);
	add_smooth_d(W, T, Nwt, beta);
	add_smooth1d(T, Nt, W * beta);

	for (i = 0; i < iter; i++) {
		sample_chain_d(N,W,T,w,d,z,Nwt,Ndt,Nt,order);
		printf("iter %d \n", i);
	}

	printf("In-Sample Perplexity = %.2f\n", pplex_d(N, W, T, w, d, Nwt, Ndt));

	add_smooth_d(D, T, Ndt, -alpha);
	add_smooth_d(W, T, Nwt,-beta);
	add_smooth1d(T, Nt, -W * beta);

	write_sparse_d(W, T, Nwt, "Nwt.txt");
	write_sparse_d(D, T, Ndt, "Ndt.txt");
	write_ivec(N, z, "z.txt");

	return 0;
}