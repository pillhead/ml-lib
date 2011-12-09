/*
 * LDAPPMModel.h
 *
 *  Created on: Mar 22, 2011
 *      Author: Clint P. George
 */

#ifndef LDAPPMMODEL_H_
#define LDAPPMMODEL_H_

#include <string>
#include <fstream>
#include <assert.h>
#include <armadillo>

#include "LDAPPMBase.h"
#include "Timer.h"

using namespace std;
using namespace arma;


class LDAPPMModel : public LDAPPMBase {

private:

	umat z_bp_; 								// for burn in period
	uvec z_;
	uvec z_mode_; // for burn in period

	mat prior_beta_counts_;
	mat post_beta_counts_;
	mat theta_counts_;
	mat beta_counts_;
	mat theta_sample_;
	mat beta_sample_;
	mat theta_samples_mean_; // for burn in period
	mat beta_samples_mean_; // for burn in period
	mat theta_counts_mean_; // for burn in period
	mat beta_counts_mean_; // for burn in period

	void init_theta();
	void init_beta();
	void init_z();
	void init_beta(string beta_counts_file);

	vec calc_partition_counts(vector<size_t> idx_z_ids);

public:

	size_t burn_in_period_;
	double alpha_;
	double eta_;

	LDAPPMModel(size_t num_topics,
			size_t max_iterations,
			size_t burn_in_period,
			double alpha,
			double eta,
			string data_file,
			string data_format,
			string vocab_file);
	LDAPPMModel(size_t num_topics,
			size_t max_iterations,
			size_t burn_in_period,
			double alpha,
			double eta,
			string data_file,
			string data_format,
			string vocab_file,
			string beta_file);
	virtual ~LDAPPMModel();

	void run_gibbs(string output_prefix);
	void run_biased_gibbs (string output_prefix);
	void run_collapsed_beta_gibbs (string output_prefix);
	void run_fixed_beta_gibbs (string output_prefix);
	void run_incremental_gibbs(string output_prefix);
	void run_gibbs_with_KL(string output_prefix);
	void save_state(string state_name);
	double calc_corpus_perplexity();
	double calc_corpus_perplexity2();
	double calc_corpus_perplexity3();
//	double calc_ln_corpus_partition_probality();
	double calc_ln_corpus_partition_probality2();

};




#endif /* LDAPPMMODEL_H_ */
