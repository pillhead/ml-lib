/*
 * TopicSearch.cpp
 *
 *  Created on: Mar 28, 2011
 *      Author: Clint P. George and Taylor Glenn
 */

#include "TopicSearch.h"

TopicSearch::TopicSearch(string data_file,
		string data_format,
		string vocab_file,
		string beta_file,
		double alpha,
		size_t max_iterations,
		size_t spacing,
		size_t burn_in_period) : LDAPPMBase(max_iterations,
		data_file,
		data_format,
		vocab_file) {

	this->spacing = spacing;
	this->read_beta_counts(beta_file);
	this->alpha_vec_ = ones<vec>(this->num_topics_) * alpha;
	this->theta_counts_ = zeros<mat>(this->num_topics_, this->num_documents_);
	this->beta_counts_ = zeros<mat>(this->num_topics_, this->vocabulary_size_);
	this->theta_counts_final_ = zeros<mat>(this->num_topics_, this->num_documents_);
	this->beta_counts_final_ = zeros<mat>(this->num_topics_, this->vocabulary_size_);
	this->burn_in_period_ = burn_in_period;

	// TODO: broken when run --algorithm ts_hrw --data /home/clint/Dropbox/TREC/query/1/1.supertweets --vocab /home/clint/Dropbox/TREC/query/1/1.vocabulary --data_format ldac --saved_beta /home/clint/Dropbox/TREC/batch/1/hdp-topics.dat --max_iter 50 --burn_in 40 --output_prefix ts --output_dir /home/clint/Dropbox/TREC/query/1

}



TopicSearch::~TopicSearch() {

}

void TopicSearch::save_state(string state_name){

	this->beta_counts_.save(state_name + "_beta_counts.dat", raw_ascii);
	this->theta_counts_.save(state_name + "_theta_counts.dat", raw_ascii);

	// Generates Dirichlet samples for Theta
	mat theta_sample = this->theta_counts_;
	for(size_t d = 0; d < this->num_documents_; d++)
		theta_sample.col(d) = sample_dirichlet_col_vec(this->num_topics_, theta_sample.col(d) + this->alpha_vec_ + 1);
	theta_sample.save(state_name + "_theta_samples.dat", raw_ascii);

	// Generates Dirichlet samples for Beta
	mat beta_sample = this->beta_counts_;
	for(size_t k = 0; k < this->num_topics_; k++)
		beta_sample.row(k) = sample_dirichlet_row_vec(this->vocabulary_size_, beta_sample.row(k));
	beta_sample.save(state_name + "_beta_samples.dat", raw_ascii);


	// If we store the last Z sample
	this->beta_counts_final_.save(state_name + "_beta_counts_last.dat", raw_ascii);
	this->theta_counts_final_.save(state_name + "_theta_counts_last.data", raw_ascii);

	// Generates Dirichlet samples for Beta
	beta_sample = beta_counts_final_;
	for(size_t k = 0; k < this->num_topics_; k++)
		beta_sample.row(k) = sample_dirichlet_row_vec(this->vocabulary_size_, beta_sample.row(k));
	beta_sample.save(state_name + "_beta_samples_last.dat", raw_ascii);

	// Generates Dirichlet samples for Theta
	theta_sample = theta_counts_final_;
	for(size_t d = 0; d < this->num_documents_; d++)
		theta_sample.col(d) = sample_dirichlet_col_vec(this->num_topics_, theta_sample.col(d) + this->alpha_vec_ + 1);
	theta_sample.save(state_name + "_theta_samples_last.dat", raw_ascii);

}


void TopicSearch::save_theta(string state_name){

	mat theta_counts = trans(this->theta_counts_);
	theta_counts.save(state_name + "_theta_counts", raw_ascii);

	// Generates Dirichlet samples for Theta
	mat theta_sample = this->theta_counts_;
	for(size_t d = 0; d < this->num_documents_; d++)
		theta_sample.col(d) = sample_dirichlet_col_vec(this->num_topics_, theta_sample.col(d) + this->alpha_vec_ + 1);
	theta_sample = trans(theta_sample);
	theta_sample.save(state_name + "_theta", raw_ascii);

}


void TopicSearch::read_beta_matrix(string beta_file){

	// assumes that data file is in the armadillo format
	this->init_beta_sample_.load(beta_file);
	this->ln_init_beta_sample_ = log(this->init_beta_sample_);
	this->num_topics_ = this->init_beta_sample_.n_rows; // sets the number of topics from the learned model
}

void TopicSearch::read_beta_counts(string beta_file){

	// assumes that data file is in the armadillo format
	this->init_beta_sample_.load(beta_file);
	this->num_topics_ = this->init_beta_sample_.n_rows; // sets the number of topics from the learned model

	this->init_beta_sample_ += 1; // smoothing
	for(size_t k = 0; k < this->num_topics_; k++)
		this->init_beta_sample_.row(k) = sample_dirichlet_row_vec(
				this->vocabulary_size_, this->init_beta_sample_.row(k));

	this->ln_init_beta_sample_ = log(this->init_beta_sample_);

}



void TopicSearch::init_z(){

	this->initial_z_ = zeros<uvec>(this->num_word_instances_);
	this->sampled_z_ = zeros<uvec>(this->num_word_instances_);

	for(size_t j = 0; j < this->num_word_instances_; j++)
		this->initial_z_(j) = sample_uniform_int(this->num_topics_);

}

void TopicSearch::init_z_with_beta(){

	this->initial_z_ = zeros<uvec>(this->num_word_instances_);
	this->sampled_z_ = zeros<uvec>(this->num_word_instances_);

	for(size_t j = 0; j < this->num_word_instances_; j++)
		this->initial_z_(j) = sample_multinomial(this->init_beta_sample_.col(this->word_ids_(j)));

}


mat TopicSearch::init_topical_Multinomial_probabilities(){

	mat mprob = ones<mat>(this->num_topics_, this->num_topics_) * 1e-24;

	for (size_t i = 0; i < this->num_topics_; i++){
		mprob(i, i) *= 1.75;
		if (i > 0) mprob(i-1, i) *= 1.25;
		if (i < this->num_topics_ - 1) mprob(i+1, i) *= 1.25;
	}

	// normalize over columns
	rowvec col_sum = sum(mprob, 0);
	for (size_t i = 0; i < mprob.n_cols; i++)
		mprob.col(i) /= col_sum(i);

	return mprob;
}



void TopicSearch::run_hybrid_random_walk_simulated_annealing(
		double init_temperature,
		double final_temperature,
		double random_walk_prob,
		double percent_random_walk) {

	init_temperature = max(init_temperature, 1.0);
	vec iter_temperature = zeros<vec> (this->max_iterations_);
	for (size_t iter = 0; iter < this->max_iterations_; iter++)
		iter_temperature(iter) = max(
				final_temperature,
				init_temperature * pow(final_temperature / init_temperature,
						((iter + 1) / this->max_iterations_)));

	run_hybrid_random_walk_simulated_annealing_uniform(
			iter_temperature,
			random_walk_prob,
			percent_random_walk);

}

void TopicSearch::run_hybrid_random_walk(
		double random_walk_prob,
		double percent_random_walk){

	vec iter_temperature = ones<vec> (this->max_iterations_);

	run_hybrid_random_walk_simulated_annealing_uniform(
			iter_temperature,
			random_walk_prob,
			percent_random_walk);

}

void TopicSearch::run_hybrid_random_walk_simulated_annealing(
		vec iter_temperature,
		double random_walk_prob,
		double percent_random_walk){

	size_t accepted_Z_instances;
	bool valid_burn_in_period;

    Timer timer = Timer();
	timer.restart_time();

	init_z();
	mat multinomial_prob = init_topical_Multinomial_probabilities();

	if (this->verbose_ >= 1)
		cout << "Multinomial probabilities: " << endl << multinomial_prob;

	if (this->burn_in_period_ > 0 && this->burn_in_period_ < this->max_iterations_){
		valid_burn_in_period = true;
		accepted_Z_instances = ceil((this->max_iterations_ - this->burn_in_period_) / this->spacing);
	}
	else {
		valid_burn_in_period = false;
		accepted_Z_instances = ceil(this->max_iterations_ / this->spacing);
	}

	for (size_t d = 0; d < this->num_documents_; d++){ // START For each document

		size_t num_words = this->document_lengths_[d];
		umat accepted_Z = zeros<umat>(num_words, accepted_Z_instances);
		uvec proposed_Z = zeros<uvec>(num_words);
		uvec current_Z = zeros<uvec>(num_words);
		size_t acceptance_count = 0;
		size_t count = 0;
		size_t random_walk_count = ceil((percent_random_walk / 100) * num_words);
		uvec sampled_z;
		uvec sampled_z2;

		vector <size_t> word_indices = this->document_word_indices_[d];
		for (size_t n = 0; n < num_words; n++)
			current_Z(n) = this->initial_z_(word_indices[n]);
		long double ppZ = calc_partition_probality(word_indices, current_Z);

		for (size_t iter = 0; iter < this->max_iterations_; iter++){ // START TOPIC SEARCH

			if (this->sample_uniform() <= random_walk_prob){ // do random walk from the previous state
				proposed_Z = current_Z;
				for (size_t s = 0; s < random_walk_count; s++){
					size_t idx = sample_uniform_int(num_words); // selects a word at random
					while(1){
						size_t topic = sample_uniform_int(this->num_topics_);
						if (topic != current_Z(idx)){
							proposed_Z(idx) = topic;
							break;
						}
					}
				}
			}
			else { // do sample from the topic specific Multinomial
				for (size_t i = 0; i < num_words; i++)
					proposed_Z(i) = sample_multinomial(multinomial_prob.col(current_Z(i)));
			}

			long double ppZ_prime = calc_partition_probality(word_indices, proposed_Z);

			long double tpZ_prime = calc_hybrid_random_walk_transition_probability(
					num_words, proposed_Z, current_Z, multinomial_prob, random_walk_prob, random_walk_count);
			long double tpZ = calc_hybrid_random_walk_transition_probability(
					num_words, current_Z, proposed_Z, multinomial_prob, random_walk_prob, random_walk_count);
			long double p_ratio = ppZ_prime - ppZ;
			long double q_ratio = tpZ_prime - tpZ;
			double acceptance_probability = min(1.0L, pow(exp(p_ratio + q_ratio), (1 / iter_temperature(iter)))); // MH acceptance probability

			if (this->sample_uniform() <= acceptance_probability){
				ppZ = ppZ_prime; // To avoid re-calculation
				current_Z = proposed_Z;
				acceptance_count++;

				if (this->verbose_ >= 1){

					cout << "doc " << d + 1 << " iter " << iter + 1;
					cout << " accepted";
					cout << " [a.p. = " << pow(exp(p_ratio + q_ratio), (1 / iter_temperature(iter)))
							// << " t.ratio = " << exp(q_ratio) << " p.ratio = " << exp(p_ratio)
							<< " ln P(z') = " << ppZ_prime << " ln P(z) = " << ppZ
							<< " ln T(z',z) = " << tpZ_prime << " ln T(z,z') = " << tpZ
							<< " a.count = " << acceptance_count << " ]" << endl;
				}

			}

			if ((iter % this->spacing == 0)
					&& (!valid_burn_in_period || (valid_burn_in_period
							&& (this->burn_in_period_ < iter)))) {
				accepted_Z.col(count) = current_Z;
				count++;
			}

		} // END TOPIC SEARCH


		// Saves the results to the class variables
		sampled_z = find_mode(accepted_Z);
		sampled_z2 = accepted_Z.col(count - 1);
//		for (size_t n = 0; n < num_words; n++){
//			this->sampled_z_(word_indices[n]) = sampled_z(n);
//			this->beta_counts_(this->sampled_z_(word_indices[n]), this->word_ids_(word_indices[n])) += 1;
//			this->beta_counts_final_(sampled_z2(n), this->word_ids_(word_indices[n])) += 1;
//		}

		// Calculates theta counts
		this->theta_counts_.col(d) = calc_topic_counts(sampled_z, this->num_topics_);
		this->theta_counts_final_.col(d) = calc_topic_counts(sampled_z2, this->num_topics_);

		// Resets all used data structures
		current_Z.reset();
		proposed_Z.reset();
		accepted_Z.reset();
		sampled_z.reset();
		sampled_z2.reset();

	} // END For each document

	if (this->verbose_ >= 1)
		cout << endl << "Total time taken: " << timer.get_time() << "s" << endl;
}

// random walk with uniform candidate

void TopicSearch::run_hybrid_random_walk_simulated_annealing_uniform(
		vec iter_temperature,
		double random_walk_prob,
		double percent_random_walk){

	size_t accepted_Z_instances;
	bool valid_burn_in_period;

    Timer timer = Timer();
	timer.restart_time();



	init_z();

	if (this->burn_in_period_ > 0 && this->burn_in_period_ < this->max_iterations_){
		valid_burn_in_period = true;
		accepted_Z_instances = ceil((this->max_iterations_ - this->burn_in_period_) / this->spacing);
	}
	else {
		valid_burn_in_period = false;
		accepted_Z_instances = ceil(this->max_iterations_ / this->spacing);
	}



	for (size_t d = 0; d < this->num_documents_; d++){ // START For each document

		size_t num_words = this->document_lengths_[d];
		umat accepted_Z = zeros<umat>(num_words, accepted_Z_instances);
		vec accepted_Z_pp = zeros<vec>(accepted_Z_instances);
		uvec proposed_Z = zeros<uvec>(num_words);
		uvec current_Z = zeros<uvec>(num_words);
		size_t acceptance_count = 0;
		size_t count = 0;
		size_t random_walk_count = ceil((percent_random_walk / 100) * num_words);
		uvec sampled_z;
		uvec sampled_z2;
		vector <size_t> word_indices = this->document_word_indices_[d];

		for (size_t n = 0; n < num_words; n++)
			current_Z(n) = this->initial_z_(word_indices[n]);

		long double ppZ = calc_partition_probality(word_indices, current_Z);

		for (size_t iter = 0; iter < this->max_iterations_; iter++){ // START TOPIC SEARCH

			if (this->sample_uniform() <= random_walk_prob){ // do random walk from the previous state
				proposed_Z = current_Z;
				for (size_t s = 0; s < random_walk_count; s++){
					size_t idx = sample_uniform_int(num_words); // selects a word at random
					while(1){
						size_t topic = sample_uniform_int(this->num_topics_);
						if (topic != current_Z(idx)){
							proposed_Z(idx) = topic;
							break;
						}
					}
				}
			}
			else { // do sample from a uniform
				for (size_t i = 0; i < num_words; i++)
					proposed_Z(i) = sample_uniform_int(this->num_topics_);
			}

			long double ppZ_prime = calc_partition_probality(word_indices, proposed_Z);
			long double p_ratio = ppZ_prime - ppZ;
			double acceptance_probability = min(1.0L, pow(exp(p_ratio), (1 / iter_temperature(iter)))); // MH acceptance probability

			if (this->sample_uniform() <= acceptance_probability){
				current_Z = proposed_Z;
				ppZ = ppZ_prime;
				acceptance_count++;
				if (this->verbose_ >= 1){
					cout << "doc " << d + 1 << " iter " << iter + 1;
					cout << " accepted";
					cout << " [a.p. = " << pow(exp(p_ratio), (1 / iter_temperature(iter)))
							<< " ln P(z') = " << ppZ_prime << " ln P(z) = " << ppZ
							<< " a.count = " << acceptance_count << " ]" << endl;
				}
			}

			if (((iter + 1) % this->spacing == 0)
					&& (!valid_burn_in_period || (valid_burn_in_period && (this->burn_in_period_ < iter)))) {
				accepted_Z.col(count) = current_Z;
				accepted_Z_pp(count) = ppZ_prime;
				count++;
			}

		} // END TOPIC SEARCH


		// Saves the results to the class variable
		sampled_z = find_mode(accepted_Z);
		sampled_z2 = accepted_Z.col(count - 1);
//		for (size_t n = 0; n < num_words; n++){
//			this->sampled_z_(word_indices[n]) = sampled_z(n);
//			this->beta_counts_(this->sampled_z_(word_indices[n]), this->word_ids_(word_indices[n])) += 1;
//			this->beta_counts_final_(sampled_z2(n), this->word_ids_(word_indices[n])) += 1;
//		}

		// Calculates theta counts
		this->theta_counts_.col(d) = calc_topic_counts(sampled_z, this->num_topics_);
		this->theta_counts_final_.col(d) = calc_topic_counts(sampled_z2, this->num_topics_);


		// Resets all used data structures
		current_Z.reset();
		proposed_Z.reset();
		accepted_Z.reset();
		sampled_z.reset();
		sampled_z2.reset();

	} // END For each document


	if (this->verbose_ >= 1)
		cout << endl << "Total time taken: " << timer.get_time() << "s" << endl;

}



/*
 * This function calculates partition probability for
 * a document.
 *
 * Ref: LDA production partition model by George Casella
 *
 */
double TopicSearch::calc_partition_probality(
		vector<size_t> word_indices,
		uvec Z) {

	double partition_probability = 0.0;
	mat beta_counts = zeros<mat> (this->num_topics_, this->vocabulary_size_);

	// Calculate m_ji' s
	for (size_t n = 0; n < word_indices.size(); n++)
		beta_counts(Z(n), this->word_ids_(word_indices[n])) += 1;

	// Calculate partition counts from  m_ji' s; i' = 1 ... V
	vec partition_counts = sum(beta_counts, 1); // sums over rows

	// ln_gamma (n_j + alpha_j + 1)
	vec ln_gamma_j = log_gamma_vec(partition_counts + this->alpha_vec_); // ln_gamma (n_j + alpha_j)

	// ln a_j = \sum_i' (m_ji' * ln beta_ji')
	vec ln_a_j = sum(beta_counts % this->ln_init_beta_sample_, 1); // sums over rows i.e. over i' s

	partition_probability = accu(ln_gamma_j + ln_a_j); // sum over all j s  - ln_gamma_K
	return partition_probability;
}


long double TopicSearch::calc_hybrid_random_walk_transition_probability(
		size_t num_words,
		uvec Z,
		uvec Z_prime,
		mat pdf,
		double random_walk_prob,
		size_t random_walk_count) {

	long double transition_probability = 0.0;
	long double b = 0.0;

	for (size_t i = 0; i < num_words; i++)
		b += log(pdf(Z_prime(i), Z(i)));
	b = exp(b);

	transition_probability = (long double) ((random_walk_prob
			* random_walk_count) / (this->num_topics_ - 1.0L))
			+ (long double) (1.0L - random_walk_prob) * b;

	return log(transition_probability);
}

