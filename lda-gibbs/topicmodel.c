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

	int i, iter, burn_in_period, j;
	int *w, *d, *z, *order;
	double **Nwt, **Ndt, *Nt, **prior_Nwt = NULL, **burn_in_Nwt, **burn_in_Ndt;
	double alpha, beta;
	char * file_name = NULL;
	char * data_format = NULL;
	char * prior_beta_file = NULL;
	double start_time = 0.0, total_time = 0.0, period = 0.0;
	char* output_prefix = NULL;
	char name[500];
	FILE *fp = NULL;
	double divider = 0.0;
	int algorithm = 0;


	if (argc < 7) {
		fprintf(stderr, "usage: %s T iter burn_in_period data_format input_file output_prefix algorithm (optional) prior_beta(optional)\n", argv[0]);
		exit(-1);
	}

	T    = atoi(argv[1]); assert(T>0);
	iter = atoi(argv[2]); assert(iter>0);
	burn_in_period = atoi(argv[3]); assert(burn_in_period > 0 && burn_in_period < iter);
	data_format = argv[4];
	file_name = argv[5];
	output_prefix = argv[6];

	// Reads prior beta provided
	if (argc >= 8){
		algorithm = atoi(argv[7]);
		assert(algorithm  == 0 || algorithm == 1 || algorithm == 2 || algorithm == 3);
		prior_beta_file = argv[8];
		prior_Nwt = read_sparse(prior_beta_file, &prior_W, &prior_T);
		printf("topics %d\n", prior_T);
		assert(T == prior_T);
	}

    if (output_prefix == NULL)
    	output_prefix = "lda";

    sprintf(name, "%s_collapsed", output_prefix);
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
	if (algorithm > 0) W = prior_W;

	printf("format     = %s\n", data_format);
	printf("file name  = %s\n", file_name);
	printf("N          = %d\n", N);
	printf("W          = %d\n", W);
	printf("D          = %d\n", D);
	printf("T          = %d\n", T);
	printf("iterations = %d\n", iter);
	printf("burn in    = %d\n", burn_in_period);

	init_random_generator();

	if (algorithm == 3){ // -- online with one document at a time (handles only LDA-C format) TODO: fix the burn in samples case

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
		burn_in_Nwt = dmat(W,T);
		burn_in_Ndt = dmat(D,T);

		alpha = 1.0; // 0.05 * N / (D * T);
		beta  = 1.0; // 0.01;

		printf("alpha      = %f\n", alpha);
		printf("beta       = %f\n", beta);

		// srand48(seed);

		randomassignment_d(N, T, w, d, z, Nwt, Ndt, Nt);
		order = randperm(N);

		add_smooth_d(D, T, Ndt, alpha);
		add_smooth_d(W, T, Nwt, beta);
		add_smooth1d(T, Nt, W * beta);
		if (algorithm == 2)
			add_smooth_d(W, T, prior_Nwt, beta);

		fprintf(fp, "iter,time,perplexity\n");


		for (i = 0; i < iter; i++) {

			start_time = current_time();

			if (algorithm == 0) // regular Gibbs sampling
				sample_chain_d(N, W, T, w, d, z, Nwt, Ndt, Nt, order);

			else if (algorithm == 1) // batch Gibbs sampling with prior beta
				sample_chain_with_prior(N, W, T, w, d, z, Nwt, Ndt, Nt, order, prior_Nwt);

			else if (algorithm == 2) // fixed beta case
				sample_chain_with_fixed_beta(N, W, T, w, d, z, Nwt, Ndt, Nt, order, prior_Nwt);


			period = current_time() - start_time;
			total_time += period;

			// printf("iter %d time: %.5fs perplexity: %.2f\n", i, period, pplex_d(N, W, T, w, d, Nwt, Ndt));
			printf("iter %d time: %.5fs\n", i+1, period);
			fprintf(fp, "%d,%.5f,%.5f\n", i+1, period, pplex_d(N, W, T, w, d, Nwt, Ndt));


			if (i >= burn_in_period){
				append_dmat(burn_in_Nwt, Nwt, W, T);
				append_dmat(burn_in_Ndt, Ndt, D, T);
				divider += 1.0;
			}

		}



	} // -- not online (incremental)


	free_random_generator();

	// Takes mean for the burn_in_samples
	assert(divider > 0);
	div_scalar_dmat(burn_in_Nwt, divider, W, T);
	div_scalar_dmat(burn_in_Ndt, divider, D, T);

	// Calculates the effective beta (for online learning) TODO not sure!!
	if (algorithm == 1 || algorithm == 3){
		for (i = 0; i < W; i++)
			for (j = 0; j < T; j++){
				Nwt[i][j] += prior_Nwt[i][j];
				burn_in_Nwt[i][j] += prior_Nwt[i][j];
			}
	}

	sprintf(name, "%s_Nwt_sparse", output_prefix);
	write_matrix_transpose(W, T, Nwt, name);
	sprintf(name, "%s_Ndt_sparse", output_prefix);
	write_matrix_transpose(D, T, Ndt, name);
	sprintf(name, "%s_Nwt_mean_sparse", output_prefix);
	write_matrix_transpose(W, T, burn_in_Nwt, name);
	sprintf(name, "%s_Ndt_mean_sparse", output_prefix);
	write_matrix_transpose(D, T, burn_in_Ndt, name);
//	sprintf(name, "%s_collapsed_z", output_prefix);
//	write_ivec2(N, z, name, "w");
	sprintf(name, "%s_Nwt", output_prefix);
	write_sparse_d(W, T, Nwt, name);
	sprintf(name, "%s_Ndt", output_prefix);
	write_sparse_d(D, T, Ndt, name);
	sprintf(name, "%s_Nwt_mean", output_prefix);
	write_sparse_d(W, T, burn_in_Nwt, name);
	sprintf(name, "%s_Ndt_mean", output_prefix);
	write_sparse_d(D, T, burn_in_Ndt, name);


	printf("Total execution time = %.5fs\n", total_time);

	double pp = pplex_d(N, W, T, w, d, Nwt, Ndt);
	printf("In-sample perplexity for %d documents (based on the last sample) = %.2f\n", D, pp);

	double pp2 = pplex_d(N, W, T, w, d, burn_in_Nwt, burn_in_Ndt);
	printf("In-sample perplexity for %d documents (based on %d burn in samples) = %.2f\n", D, (int)divider, pp2);
//	fprintf(fp, "In-sample perplexity for %d documents = %.2f time taken = %.5fs\n", D, pp, total_time);

	fclose(fp);

	return 0;
}
