/*
 * LDAPPMTopicSearch.h
 *
 *  Created on: Mar 28, 2011
 *      Author: Clint P. George
 */

#ifndef TOPICSEARCH_H_
#define TOPICSEARCH_H_

#include "LDAPPMBase.h"
#include "Timer.h"

class TopicSearch : public LDAPPMBase {

private:

	uvec initial_z_;
	mat ln_init_beta_sample_;
	vec alpha_vec_;

	mat init_topical_Multinomial_probabilities();
	mat init_base_transition_matrix2();
	void read_beta_matrix(string beta_file);
	void read_beta_counts(string beta_file);
	void init_z();
	void init_z_with_beta();
	double calc_partition_probality(vector <size_t> doc_idx, uvec Z);
	long double calc_transition_probability(size_t num_words, uvec Z, uvec Z_prime, mat pdf);
	long double calc_hybrid_random_walk_transition_probability(size_t num_words, uvec Z, uvec Z_prime, mat pdf, double random_walk_prob, size_t random_walk_words);


public:

	size_t spacing;
	mat init_beta_sample_;
	mat theta_counts_;
	uvec sampled_z_;
	mat beta_counts_;
	size_t burn_in_period_;
	mat theta_counts_final_;
	mat beta_counts_final_;

	TopicSearch(string data_file,
			string data_format,
			string vocab_file,
			string beta_file,
			double alpha,
			size_t max_iterations,
			size_t spacing,
			size_t burn_in_period);

	TopicSearch(string data_file,
			string data_format,
			string vocab_file,
			string beta_file,
			string theta_file,
			size_t max_iterations,
			size_t spacing);

	virtual ~TopicSearch();
	void run_hybrid_random_walk(double random_walk_prob, double percent_random_walk);
	void run_hybrid_random_walk_simulated_annealing(vec iter_temperature, double random_walk_prob, double percent_random_walk);
	void run_hybrid_random_walk_simulated_annealing_uniform(vec iter_temperature, double random_walk_prob, double percent_random_walk);
	void run_hybrid_random_walk_simulated_annealing(double init_temperature, double final_temperature, double random_walk_prob, double percent_random_walk);
	void save_state(string state_name);
	void save_theta(string state_name);

};



#endif /* TOPICSEARCH_H_ */
