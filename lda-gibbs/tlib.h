/*---------------------------------------------------
 * file:    tlib.h
 * purpose: header file for tlib.c
 * usage:   include in *.c
 * author:  newman@uci.edu
 *-------------------------------------------------*/

#ifndef _TLIB_H_
#define _TLIB_H_
 
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <math.h>
#include <time.h>
#include <gsl/gsl_rng.h>
#include <gsl/gsl_randist.h>
#include <unistd.h>

#define MAX(a,b) ((a) > (b) ? (a) : (b))
#define MIN(a,b) ((a) < (b) ? (a) : (b))
#define SQR(a)   ((a) * (a))

int **imat(int nr, int nc);
void free_imat(int **x);
double **dmat(int nr, int nc);
void free_dmat(double **x);
float **fmat(int nr, int nc);
void free_fmat(float **x);
int *ivec(int n);
double *dvec(int n);
float *fvec(int n);
int imax(int n, int *x);
double dmax(int n, double *x);
int imin(int n, int *x);
double dmin(int n, double *x);
int isum(int n, int *x);
double dsum(int n, double *x);
double ddot(int n, double *x, double *y);
int countlines(char *fname);
int countN(char *fname);
void isort(int n, int *x, int direction, int *indx);
int *dsort(int n, double *x);
void read_docwordcountbin(int NNZ, int *w, int *d, int *c, char *fname);
int countNNZ(int nr, int nc, int **x);
int countNNZ_d(int nr, int nc, double **x);
void write_sparse(int nr, int nc, int **x, char *fname);
void write_sparse_d(int nr, int nc, double **x, char *fname);
void write_sparsebin(int nr, int nc, int **x, char *fname);
double **read_sparse(char *fname, int *nr_, int *nc_);
void read_docID_wordID(char *fname, int *d, int *w);
int **read_sparse_trans(char *fname, int *nr_, int *nc_);
void read_dw(char *fname, int *d, int *w, int *D, int *W);
void fill_Nd(int N, int *d, int *Nd);
void read_dwc(char *fname, int *d, int *w, int *c, int *D, int *W);
int read_NNZbin(char *fname);
int read_NNZ(char *fname);
void read_sparsebin(char *fname, int *col1, int *col2, int *col3);
// void write_ivec (int n, int *x, char *fname);
void write_ivec (int n, int *x, char *fname, char *type);
void write_dvec (int n, double *x, char *fname);
void read_ivec (int n, int *x, char *fname);
void read_dvec (int n, double *x, char *fname);
int *randperm(int n);
void randomassignment(int N, int T, int *w, int *d, int *z, int **Nwt, int **Ndt, int *Nt);
void randomassignment_d(int N, int T, int *w, int *d, int *z, double **Nwt, double **Ndt, double *Nt);
void randomassignmentC(int NNZ, int T, int *w, int *d, int *c, int *z, int **Nwt, int **Ndt, int *Nt);
void randomassignmentC_d(int NNZ, int T, int *w, int *d, int *c, int *z, double **Nwt, double **Ndt, double *Nt);
void randomassignment_rank(int N, int T, int *w, int *d, int *drank, int *z, int **Nwt, int **Ndt, int *Nt);
void assignment(int N, int *w, int *d, int *z, int **Nwt, int **Ndt, int *Nt);
void randomassignment_2layer(int N, int T, int S, int *w, int *d, int *z, int *y, int **Nwt, int **zy, int **Ndt, int *Nt, int *ytot);
void randomassignment2(int N, int T, int *d, int *z, int **Ndt);
void sample_chain (int N, int W, int T, double alpha, double beta, int *w, int *d, int *z, int **Nwt, int **Ndt, int *Nt, int *order);
void sample_chain_d (int N, int W, int T, int *w, int *d, int *z, double **Nwt, double **Ndt, double *Nt, int *order);
void sample_chain_d_icm (int N, int W, int T, int *w, int *d, int *z, double **Nwt, double **Ndt, double *Nt, int *order);
void sample_chain_alpha (int N, int W, int T, double *alpha, double beta, int *w, int *d, int *z, int **Nwt, int **Ndt, int *Nt, int *order);
void sample_chain_rank (int N, int W, int T, double alpha, double beta, int *w, int *d, int *drank, int *z, int **Nwt, int **Ndt, int *Nt, int *order);
void sample_chain0 (int N, int W, int T, double alpha, double beta, int *w, int *d, int *z, int **Nwt, int **Ndt, int *Nt);
void sample_chain_2layer (int N, int W, int T, int S, double alpha, double beta, double gamma, int *w, int *d, int *z, int *y, int **Nwt, int **zy, int **Ndt, int *Nt, int *ytot);
void resample_chain (int N, int W, int T, double alpha, double beta, int *w, int *d, int *z, int **Nwt, int **Ndt, int *Nt);
void oversample_Ndt (int N, int W, int T, double alpha, double beta, int *w, int *d, int *z, int **Nwt, int **Ndt, int *Nt);
void loglike (int N, int W, int D, int T, double alpha, double beta, int *w, int *d, int **Nwt, int **Ndt, int *Nt, int *Nd);
void chk_sum (int n, int T, int **x, int *sumx);
void chk_sum_d (int n, int T, double **x, double *sumx);
void getNt (int n, int T, int **x, int *Nt);
void add_smooth_d (int n, int T, double **x, double smooth);
void add_smooth1d (int T, double *x, double smooth);
void set_smooth_d (int n, int T, double **x, double smooth);
void set_smooth1d (int T, double *x, double smooth);
double pplex(int N, int W, int T, double alpha, double beta, int *w, int *d, int **Nwt, int **Ndt);
double pplex_d(int N, int W, int T, int *w, int *d, double **Nwt, double **Ndt);

void read_ldac(char *fname, int *d, int *w, int *D, int *W) ;
int countN_ldac(char *fname);
void sample_chain_with_prior (int N, int W, int T, int *w, int *d, int *z, double **Nwt, double **Ndt, double *Nt, int *order, double **prior_Nwt);
void sample_chain_with_fixed_beta (int N, int W, int T, int *w, int *d, int *z, double **Nwt, double **Ndt, double *Nt, int *order, double **prior_Nwt);
void write_matrix(int nr, int nc, double **x, char *fname);
void write_matrix_transpose(int nr, int nc, double **x, char *fname);
void read_doc_word_counts_ldac(char *fname, int *d);
void sample_doc_with_prior (int N, int W, int T, int *w, int *d, int *z, double **Nwt, double *Ndt, double *Nt, int *order, double **prior_Nwt);
void randomassignment_for_doc(int N, int T, int *w, int *d, int *z, double **Nwt, double *Ndt, double *Nt);
void init_random_generator();
void free_random_generator();
double sample_uniform();
unsigned int sample_uniform_int(unsigned int K);
void div_scalar_dmat(double **x, double div, int nr, int nc); // divides mat x by double div
void append_dmat(double **x, double **y, int nr, int nc); // appends mat y to x
void write_ivec2 (int n, int *x, char *fname, char *type);

#endif
