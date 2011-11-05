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
#include <sys/time.h>
#include <time.h>
#include "tlib.h"

// Current time
double current_time(void) {
	struct timeval time;
	gettimeofday(&time, NULL);
	return time.tv_sec + 1.0e-6 * time.tv_usec;
}

/*==========================================
* main
*========================================== */
int main(int argc, char* argv[])
{
	int T; // number of topics
	int W; // number of unique words
	int D; // number of docs
	int N; // number of words in corpus
	int prior_W;
	int prior_T;

	int i, iter, seed, j;
	int *w, *d, *z, *order;
	double **Nwt, **Ndt, *Nt, **prior_Nwt = NULL;
	double alpha, beta;
	char * file_name = NULL;
	char * data_format = NULL;
	char * prior_beta_file = NULL;
	int prior_flg = 0;
	double start_time = 0.0, total_time = 0.0, period = 0.0;
	int incremental_gibbs = 0;
	char* output_prefix = NULL;
	char name[500];
	FILE *fp = NULL;


	if (argc < 7) {
		fprintf(stderr, "usage: %s T iter seed data_format input_file output_prefix prior_beta(optional) incremental_gibbs (optional)\n", argv[0]);
		exit(-1);
	}

	T    = atoi(argv[1]); assert(T>0);
	iter = atoi(argv[2]); assert(iter>0);
	seed = atoi(argv[3]); assert(seed>0);
	data_format = argv[4];
	file_name = argv[5];
	output_prefix = argv[6];

	// Reads prior beta provided
	if (argc >= 8){
		prior_flg = 1;
		prior_beta_file = argv[7];
		prior_Nwt = read_sparse(prior_beta_file, &prior_W, &prior_T);
		assert(T == prior_T);

		// Case: online and one document at a time
		if (argc == 9){
			incremental_gibbs = atoi(argv[8]);
			assert(incremental_gibbs  == 0 || incremental_gibbs == 1);
		}
	}

    if (output_prefix == NULL)
    	output_prefix = "lda";

    sprintf(name, "%s_model_info", output_prefix);
	fp = fopen(name, "w");assert(fp);


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

	// The vocabulary size should be the same
	// We assume that there won't be any new
	// terms in the online document (that
	// should be removed during pre-processing step)
	if (prior_flg) W = prior_W;

	printf("format     = %s\n", data_format);
	printf("file name  = %s\n", file_name);
	printf("seed       = %d\n", seed);
	printf("N          = %d\n", N);
	printf("W          = %d\n", W);
	printf("D          = %d\n", D);
	printf("T          = %d\n", T);
	printf("iterations = %d\n", iter);

	fprintf(fp, "format     = %s\n", data_format);
	fprintf(fp, "file name  = %s\n", file_name);
	fprintf(fp, "seed       = %d\n", seed);
	fprintf(fp, "N          = %d\n", N);
	fprintf(fp, "W          = %d\n", W);
	fprintf(fp, "D          = %d\n", D);
	fprintf(fp, "T          = %d\n", T);
	fprintf(fp, "iterations = %d\n", iter);

	init_random_generator();

	if (incremental_gibbs){ // -- online with one document at a time (handles only LDA-C format)

		int ii = 0, di = 0, NN = 0, nn = 0;
		int * zid, * wid, * did, * doc_counts;
		double * theta;

		doc_counts = ivec(D);
		read_doc_word_counts_ldac(file_name, doc_counts);

		for (di = 0; di < D; di++){

			NN = doc_counts[di];
			wid = ivec(NN);
			did = ivec(NN);
			zid = ivec(NN);

			Nwt = dmat(W, T);
			// Ndt = dmat(D, T);
			theta = dvec(T);
			Nt  = dvec(T);

			for (i = 0; i < NN; i++){
				wid[i] = w[nn];
				did[i] = d[nn];
				nn++;
			}

			alpha = 0.05 * N / (D * T);
			beta  = 0.01;

			printf("Sampling doc #%d counts #%d ", di+1, NN);
			fprintf(fp, "Sampling doc #%d counts #%d ", di+1, NN);
			start_time = current_time();

			randomassignment_for_doc(NN, T, wid, did, zid, Nwt, theta, Nt);
			order = randperm(NN);

			// add_smooth_d(D, T, Ndt, alpha);
			add_smooth_d(W, T, Nwt, beta);
			add_smooth1d(T, theta, alpha);
			add_smooth1d(T, Nt, W * beta);

			for (i = 0; i < iter; i++)
				sample_doc_with_prior(NN, W, T, wid, did, zid, Nwt, theta, Nt, order, prior_Nwt);

			for (i = 0; i < NN; i++){
				z[ii] = zid[i];
				ii++;
			}

			period = current_time() - start_time;
			total_time += period;

			printf(" processing time: %.5fs\n", period);
			fprintf(fp, " processing time: %.5fs\n", period);

		}

		// updates global variables
		Nwt = dmat(W, T);
		Ndt = dmat(D, T);
		for (i = 0; i < N; i++){
			Nwt[w[i]][z[i]]++;
			Ndt[d[i]][z[i]]++;
		}


	} // -- end online with one document at time

	else { // -- modifications to handle lda-c format and prior document data

		Nwt = dmat(W,T);
		Ndt = dmat(D,T);
		Nt  = dvec(T);

		alpha = 0.05 * N / (D * T);
		beta  = 0.01;

		printf("alpha      = %f\n", alpha);
		printf("beta       = %f\n", beta);

		// srand48(seed);

		randomassignment_d(N, T, w, d, z, Nwt, Ndt, Nt);
		order = randperm(N);

		add_smooth_d(D, T, Ndt, alpha);
		add_smooth_d(W, T, Nwt, beta);
		add_smooth1d(T, Nt, W * beta);

		for (i = 0; i < iter; i++) {

			start_time = current_time();

			if (!prior_flg)
				sample_chain_d(N,W,T,w,d,z,Nwt,Ndt,Nt,order);
			else
				sample_chain_with_prior(N, W, T, w, d, z, Nwt, Ndt, Nt, order, prior_Nwt);

			period = current_time() - start_time;
			total_time += period;

			// printf("iter %d time: %.5fs perplexity: %.2f\n", i, period, pplex_d(N, W, T, w, d, Nwt, Ndt));
			printf("iter %d time: %.5fs\n", i, period);
			fprintf(fp, "iter %d time: %.5fs\n", i, period);

		}

	} // -- not online (incremental)


	free_random_generator();

	// Calculates the effective beta (for online learning) TODO not sure!!
	if (prior_flg){
		for (i = 0; i < W; i++)
			for (j = 0; j < T; j++)
				Nwt[i][j] += prior_Nwt[i][j];
	}

	sprintf(name, "%s_Nwt_full", output_prefix);
	write_matrix_transpose(W, T, Nwt, name);
	sprintf(name, "%s_Ndt_full", output_prefix);
	write_matrix_transpose(D, T, Ndt, name);
	sprintf(name, "%s_Nwt", output_prefix);
	write_sparse_d(W, T, Nwt, name);
	sprintf(name, "%s_Ndt", output_prefix);
	write_sparse_d(D, T, Ndt, name);
	sprintf(name, "%s_z", output_prefix);
	write_ivec(N, z, name);

	double pp = pplex_d(N, W, T, w, d, Nwt, Ndt);
	printf("In-sample perplexity for %d documents = %.2f time taken = %.5fs\n", D, pp, total_time);
	fprintf(fp, "In-sample perplexity for %d documents = %.2f time taken = %.5fs\n", D, pp, total_time);

	fclose(fp);

	return 0;
}
