/*
 * LDAPPMBase.h
 *
 *  Created on: Mar 28, 2011
 *      Author: Clint P. George
 */

#ifndef LDAPPMBASE_H_
#define LDAPPMBASE_H_
#include <iostream>
#include <assert.h>
#include <armadillo>
#include <math.h>
#include <map>
#include <fstream>
#include <gsl/gsl_rng.h>
#include <gsl/gsl_randist.h>
#include <gsl/gsl_sf.h>

using namespace std;
using namespace arma;


class LDAPPMBase {

private:

	gsl_rng *random_num_generator_;				// random number generator
	unsigned long int random_seed_; 			// seed

	void init_random_generator();
	void free_random_generator();
	void read_gibbs_data_file(string file_name);
	void read_ldac_data_file(string file_name);
	void read_ldac_data_file2(string file_name);
	size_t count_gibbs_data_file_lines(string file_name);

public:

	size_t vocabulary_size_;
	size_t num_documents_;
	size_t num_word_instances_;
	size_t num_topics_;
	size_t max_iterations_;

	LDAPPMBase(size_t max_iterations,
			string data_file,
			string data_format,
			string vocab_file);
	virtual ~LDAPPMBase();
	void print_metadata();
	void set_verbose(size_t value);
	void print_message(string msg);
	template <typename T>
	std::string to_string(T const& value);

protected:

	string data_format_;
	size_t verbose_;
	uvec word_ids_;
	uvec document_indices_;
	uvec document_ids_;

	vector < vector < size_t > > document_word_indices_;
	vector < vector < size_t > > document_unique_words_;
	vector < size_t > document_lengths_;
	vector < map <size_t, size_t> > document_word_counts_;

	double sample_beta(double a, double b);
	size_t sample_multinomial(vec theta);
	size_t sample_multinomial2 (vec theta);
	size_t sample_uniform_int(size_t K);
	double sample_uniform();
	rowvec sample_dirichlet_row_vec(size_t num_elements, rowvec alpha);
	rowvec sample_dirichlet_row_vec2(size_t num_elements, rowvec alpha);
	vec sample_dirichlet_col_vec(size_t num_elements, vec alpha);
	vec sample_dirichlet_col_vec2(size_t num_elements, vec alpha);
	vec sample_stick_breaking_prior(size_t num_elements, double a, double b);
	double log_gamma(double x);
	vec log_gamma_vec(vec x_vec);
	rowvec log_gamma_rowvec(rowvec x_vec);
	vec calc_topic_counts (uvec Z_vec, size_t num_topics);
	uvec find_mode(umat accum_matrix);
	u32 mode(urowvec data, int size);


};


#endif /* LDAPPMBASE_H_ */
