/*
 * LDAPPMModel.cpp
 *
 *  Created on: Mar 22, 2011
 *      Author: Clint P. George
 * Description: Implements the Dirichlet Allocation Product
 * 				Partition Model - full Gibbs sampler and
 * 				online batch and incremental Gibbs samplers
 */

#include "LDAPPMModel.h"

/**
 * Implements the class constructor
 *
 */
LDAPPMModel::LDAPPMModel(size_t num_topics,
		size_t max_iterations,
		size_t burn_in_period,
		double alpha,
		double eta,
		string data_file,
		string data_format,
		string vocab_file) : LDAPPMBase(max_iterations,
		data_file,
		data_format,
		vocab_file) {

	this->num_topics_ = num_topics;
	this->burn_in_period_ = burn_in_period;
	this->alpha_ = alpha;
	this->eta_ = eta;

	this->init_z();
	this->init_theta();
	this->init_beta();

}

/**
 * Implements the class constructor: when we
 * provide beta counts file to initialize the
 * model
 *
 */
LDAPPMModel::LDAPPMModel(size_t num_topics,
		size_t max_iterations,
		size_t burn_in_period,
		double alpha,
		double eta,
		string data_file,
		string data_format,
		string vocab_file,
		string beta_file) : LDAPPMBase(max_iterations,
		data_file,
		data_format,
		vocab_file) {

	this->num_topics_ = num_topics;
	this->burn_in_period_ = burn_in_period;
	this->alpha_ = alpha;
	this->eta_ = eta;

	this->init_z();
	this->init_theta();
	this->init_beta(beta_file); // uses a prior beta

}



/**
 * Implements class destructor
 */

LDAPPMModel::~LDAPPMModel() {

}

void LDAPPMModel::init_z(){

	this->z_ = zeros<uvec>(this->num_word_instances_);
	if (this->burn_in_period_ > 0){
		this->z_bp_ = zeros<umat>(this->num_word_instances_,
				(this->max_iterations_ - this->burn_in_period_));
		this->z_mode_ = this->z_;
	}

	for(size_t j = 0; j < this->num_word_instances_; j++)
		this->z_(j) = this->sample_uniform_int(this->num_topics_);

}

void LDAPPMModel::init_theta(){

	vec alpha = ones<vec>(this->num_topics_) * this->alpha_;

	this->theta_counts_ = zeros(this->num_topics_, this->num_documents_);
	this->theta_sample_ = zeros(this->num_topics_, this->num_documents_);
	if (this->burn_in_period_ > 0){
		this->theta_samples_mean_ = zeros(this->num_topics_, this->num_documents_);
		this->theta_counts_mean_ = zeros(this->num_topics_, this->num_documents_);
	}

}

void LDAPPMModel::init_beta(){

	this->beta_counts_ = zeros(this->num_topics_, this->vocabulary_size_);
	this->prior_beta_counts_ = zeros(this->num_topics_, this->vocabulary_size_); // zero prior

	if (this->burn_in_period_ > 0){
		this->beta_samples_mean_ = zeros(this->num_topics_, this->vocabulary_size_);
		this->beta_counts_mean_ = zeros(this->num_topics_, this->vocabulary_size_);
	}

	for(size_t i = 0; i < this->num_word_instances_; i++)
		this->beta_counts_(this->z_(i), this->word_ids_(i)) += 1;

}


void LDAPPMModel::init_beta(string beta_counts_file){

	this->prior_beta_counts_.load(beta_counts_file);
	assert(this->prior_beta_counts_.n_cols == this->vocabulary_size_);
	assert(this->prior_beta_counts_.n_rows == this->num_topics_);

	this->beta_counts_ = zeros(this->num_topics_, this->vocabulary_size_);

	if (this->burn_in_period_ > 0){
		this->beta_samples_mean_ = zeros(this->num_topics_, this->vocabulary_size_);
		this->beta_counts_mean_ = zeros(this->num_topics_, this->vocabulary_size_);
	}

	for(size_t i = 0; i < this->num_word_instances_; i++)
		this->beta_counts_(this->z_(i), this->word_ids_(i)) += 1;
}

/**
 * Saves Gibbs sampler states to ASCII text
 * (MATLAB/R compatible)
 */

void LDAPPMModel::save_state(string state_name){

	if (this->burn_in_period_ > 0){ // handles burn in period
//		this->beta_samples_mean_.save(state_name + "_beta_samples_mean", raw_ascii);
//		this->theta_samples_mean_.save(state_name + "_theta_samples_mean", raw_ascii);
		this->beta_counts_mean_.save(state_name + "_beta_counts_mean", raw_ascii);
		this->theta_counts_mean_.save(state_name + "_theta_counts_mean", raw_ascii);
	}
	else {
		mat post_beta_counts = this->prior_beta_counts_ + this->beta_counts_;
		post_beta_counts.save(state_name + "_beta_counts", raw_ascii);
//		this->beta_sample_.save(state_name + "_beta_samples", raw_ascii);
		this->theta_counts_.save(state_name + "_theta_counts", raw_ascii);
//		this->theta_sample_.save(state_name + "_theta_samples", raw_ascii);
	}
	this->beta_counts_.save(state_name + "_beta_counts_last", raw_ascii);
	this->theta_counts_.save(state_name + "_theta_counts_last", raw_ascii);
}


vec LDAPPMModel::calc_partition_counts (vector<size_t> indices){

	register unsigned int i;
	register unsigned int num_elements = indices.size();
	vec partition_counts = zeros<vec>(this->num_topics_);

	for (i = 0; i < num_elements; i++)
		partition_counts(this->z_(indices[i])) += 1;

	return partition_counts;
}


/**
 *
 * Run Gibbs Sampler for the LDA full Gibbs sampler and batch online Gibbs sampler
 *
 */

void LDAPPMModel::run_gibbs (string output_prefix)
{
	vector < size_t > word_indices;
	vec theta_d;
	size_t bp_count = 0;
	size_t document_length = 0;
	Timer t = Timer();
	double total_time_taken = 0.0, period = 0.0;
	double pp = 0.0, ln_pp;
	register unsigned int i = 0, iter = 0, k = 0, d = 0;
	ofstream myfile;
	string str = output_prefix + "_full";
	myfile.open(str.c_str());

	myfile << "iter,time,dirichlet_time,perplexity,partition_prob" << endl;


	Timer t1 = Timer();
	double dirichlet_time;

	// START GIBBS ITERATIONS

	for (iter = 0; iter < this->max_iterations_; iter++){

		t.restart_time();
		dirichlet_time = 0.0;

		if (this->verbose_ >= 1) cout << "iter #" << iter + 1;
		myfile << iter + 1 << ",";

		for (d = 0; d < this->num_documents_; d++){ // for each document

			word_indices = this->document_word_indices_[d];
			document_length = this->document_lengths_[d];

			t1.restart_time();

			// Gibbs sampling for beta

			this->beta_sample_ = this->prior_beta_counts_ + this->beta_counts_ + this->eta_;
			for(k = 0; k < this->num_topics_; k++)
				this->beta_sample_.row(k) = sample_dirichlet_row_vec(this->vocabulary_size_, this->beta_sample_.row(k));


			// Gibbs sampling for theta

			this->theta_counts_.col(d) = calc_partition_counts(word_indices); // updates theta counts
			this->theta_sample_.col(d) = sample_dirichlet_col_vec(this->num_topics_, this->theta_counts_.col(d) + this->alpha_);

			dirichlet_time += t1.get_time();



			// Gibbs sampling for each document-word-instance

			for(i = 0; i < document_length; i++)
				this->beta_counts_(this->z_(word_indices[i]), this->word_ids_(word_indices[i])) -= 1; // Excludes the document d's word-topic counts


			//  Samples topics Z (word topic selection)
			for(i = 0; i < document_length; i++)
				this->z_(word_indices[i]) = sample_multinomial(this->theta_sample_.col(d) % this->beta_sample_.col(this->word_ids_(word_indices[i])));

			for(i = 0; i < document_length; i++)
				this->beta_counts_(this->z_(word_indices[i]), this->word_ids_(word_indices[i])) += 1; // updates beta counts

		}


		// Handles burn in period

		if (this->burn_in_period_ > 0 && iter > this->burn_in_period_){
			this->z_bp_.col(bp_count) = this->z_;
//			this->beta_samples_mean_ += this->beta_sample_;
//			this->theta_samples_mean_ += this->theta_sample_;
			this->beta_counts_mean_ += (this->prior_beta_counts_ + this->beta_counts_); // TODO need to make sure whether it's right
			this->theta_counts_mean_ += this->theta_counts_;

			bp_count++;
		}

		period = t.get_time();
		total_time_taken += period;
		pp = calc_corpus_perplexity2();
		ln_pp = calc_ln_corpus_partition_probality2();

		if (this->verbose_ >= 1) {
			cout << " time: " << period;
			cout << " dirichlet: " << dirichlet_time;
			cout << " perplexity: " << pp;
			cout << " log partition prob:" << ln_pp << "\n";
		}

		myfile << period << "," << dirichlet_time << "," << pp << "," << ln_pp << "\n";
	}

	// END GIBBS ITERATIONS

	// Averages for burn in period
	if (this->burn_in_period_ > 0) {
//		this->beta_samples_mean_ /= (bp_count + 1);
//		this->theta_samples_mean_ /= (bp_count + 1);
		this->beta_counts_mean_ /= (bp_count + 1);
		this->theta_counts_mean_ /= (bp_count + 1);
		this->z_mode_ = find_mode(this->z_bp_);
	}


	if (this->verbose_ >= 1){
		cout << endl << "Total execution time: " << total_time_taken << "s" << endl;
//		cout << "Log partition probability:" << calc_ln_corpus_partition_probality() << "\n";
		if (this->burn_in_period_ > 0)
			cout << "Model perplexity: " << calc_corpus_perplexity() << endl;
	}

	myfile.close();
}

void LDAPPMModel::run_biased_gibbs (string output_prefix)
{
	vector < size_t > word_indices;
	vector < size_t > unique_words;
	mat beta_sample;
	vec theta_d;
	vec beta_remain;
	rowvec beta_row;
	size_t bp_count = 0;
	size_t document_length = 0;
	size_t num_unique_words = 0;
	Timer t = Timer();
	double total_time_taken = 0.0, period = 0.0;
	double pp = 0.0, ln_pp = 0.0;
	register unsigned int i = 0, iter = 0, k = 0, d = 0;
	ofstream myfile;
	string str = output_prefix + "_full";
	myfile.open(str.c_str());

	myfile << "iter,time,dirichlet_time,perplexity,partition_prob" << endl;


	Timer t1 = Timer();
	double dirichlet_time;

	// START GIBBS ITERATIONS

	for (iter = 0; iter < this->max_iterations_; iter++){

		t.restart_time();
		dirichlet_time = 0.0;

		if (this->verbose_ >= 1) cout << "iter #" << iter + 1;
		myfile << iter + 1 << ",";

		for (d = 0; d < this->num_documents_; d++){ // for each document

			word_indices = this->document_word_indices_[d];
			document_length = this->document_lengths_[d];
			unique_words = this->document_unique_words_[d];
			num_unique_words = unique_words.size();

			beta_sample = zeros<mat>(this->num_topics_, num_unique_words + 1);
			t1.restart_time();

			// Gibbs sampling for beta (with selected samples)

			this->beta_sample_ = this->prior_beta_counts_ + this->beta_counts_ + this->eta_;
			for (i = 0; i < num_unique_words; i++)
				beta_sample.col(i) = this->beta_sample_.col(unique_words[i]);

			beta_remain = sum(this->beta_sample_, 1) - sum(beta_sample, 1);
			for(k = 0; k < this->num_topics_; k++){
				beta_row = beta_sample.row(k);
				beta_row(num_unique_words) = beta_remain(k);
				beta_sample.row(k) = sample_dirichlet_row_vec(num_unique_words + 1, beta_row); // This incorporates to reduce errors
			}
			for (i = 0; i < num_unique_words; i++)
				this->beta_sample_.col(unique_words[i]) = beta_sample.col(i);


			// Gibbs sampling for theta

			this->theta_counts_.col(d) = calc_partition_counts(word_indices); // updates theta counts
			this->theta_sample_.col(d) = sample_dirichlet_col_vec(this->num_topics_, this->theta_counts_.col(d) + this->alpha_);

			dirichlet_time += t1.get_time();

			// Gibbs sampling for each document-word-instance

			for(i = 0; i < document_length; i++)
				this->beta_counts_(this->z_(word_indices[i]), this->word_ids_(word_indices[i])) -= 1; // Excludes the document d's word-topic counts


			//  Samples topics Z (word topic selection)
			for(i = 0; i < document_length; i++)
				this->z_(word_indices[i]) = sample_multinomial(this->theta_sample_.col(d) % this->beta_sample_.col(this->word_ids_(word_indices[i])));

			for(i = 0; i < document_length; i++)
				this->beta_counts_(this->z_(word_indices[i]), this->word_ids_(word_indices[i])) += 1; // updates beta counts



		}


		// Handles burn in period

		if (this->burn_in_period_ > 0 && iter > this->burn_in_period_){
			this->z_bp_.col(bp_count) = this->z_;
			this->beta_counts_mean_ += (this->prior_beta_counts_ + this->beta_counts_); // TODO need to make sure whether it's right
			this->theta_counts_mean_ += this->theta_counts_;

			bp_count++;
		}

		period = t.get_time();
		total_time_taken += period;
		pp = calc_corpus_perplexity2();
		ln_pp = calc_ln_corpus_partition_probality2();

		if (this->verbose_ >= 1) {
			cout << " time: " << period;
			cout << " dirichlet: " << dirichlet_time;
			cout << " perplexity: " << pp;
			cout << " log partition prob:" << ln_pp << "\n";
		}

		myfile << period << "," << dirichlet_time << "," << pp << ","  << ln_pp << "\n";
	}

	// END GIBBS ITERATIONS

	// Averages for burn in period
	if (this->burn_in_period_ > 0) {
		this->beta_counts_mean_ /= (bp_count + 1);
		this->theta_counts_mean_ /= (bp_count + 1);
		this->z_mode_ = find_mode(this->z_bp_);
	}


	if (this->verbose_ >= 1){
		cout << endl << "Total execution time: " << total_time_taken << "s" << endl;
		if (this->burn_in_period_ > 0)
			cout << "Model perplexity: " << calc_corpus_perplexity() << endl;
	}

	myfile.close();
}


void LDAPPMModel::run_collapsed_beta_gibbs (string output_prefix)
{
	vector < size_t > word_indices;
	vector < size_t > unique_words;
	mat beta_sample;
	vec theta_d;
	vec beta_remain;
	rowvec beta_row;
	size_t bp_count = 0;
	size_t document_length = 0;
	size_t num_unique_words = 0;
	Timer t = Timer();
	double total_time_taken = 0.0, period = 0.0;
	double pp = 0.0, ln_pp = 0.0, pp2 = 0.0;
	register unsigned int i = 0, iter = 0, d = 0;
	ofstream myfile;
	string str = output_prefix + "_full";
	myfile.open(str.c_str());

	myfile << "iter,time,dirichlet_time,perplexity,pp_perplexity,partition_prob" << endl;


	Timer t1 = Timer();
	double dirichlet_time;

	// START GIBBS ITERATIONS

	for (iter = 0; iter < this->max_iterations_; iter++){

		t.restart_time();
		dirichlet_time = 0.0;

		if (this->verbose_ >= 1) cout << "iter #" << iter + 1;
		myfile << iter + 1 << ",";

		for (d = 0; d < this->num_documents_; d++){ // for each document

			word_indices = this->document_word_indices_[d];
			document_length = this->document_lengths_[d];
			unique_words = this->document_unique_words_[d];
			num_unique_words = unique_words.size();

			beta_sample = zeros<mat>(this->num_topics_, num_unique_words + 1);
			t1.restart_time();

			// No Gibbs sampling for beta

			this->beta_sample_ = this->prior_beta_counts_ + this->beta_counts_ + this->eta_;
			for(register unsigned int k = 0; k < this->num_topics_; k++)
				this->beta_sample_.row(k) /= sum(this->beta_sample_.row(k));


			// Gibbs sampling for theta

			this->theta_counts_.col(d) = calc_partition_counts(word_indices); // updates theta counts
			this->theta_sample_.col(d) = sample_dirichlet_col_vec(this->num_topics_, this->theta_counts_.col(d) + this->alpha_);

			dirichlet_time += t1.get_time();

			// Gibbs sampling for each document-word-instance

			for(i = 0; i < document_length; i++)
				this->beta_counts_(this->z_(word_indices[i]), this->word_ids_(word_indices[i])) -= 1; // Excludes the document d's word-topic counts


			//  Samples topics Z (word topic selection)
			for(i = 0; i < document_length; i++)
				this->z_(word_indices[i]) = sample_multinomial(this->theta_sample_.col(d) % this->beta_sample_.col(this->word_ids_(word_indices[i])));

			for(i = 0; i < document_length; i++)
				this->beta_counts_(this->z_(word_indices[i]), this->word_ids_(word_indices[i])) += 1; // updates beta counts



		}


		// Handles burn in period

		if (this->burn_in_period_ > 0 && iter > this->burn_in_period_){
			this->z_bp_.col(bp_count) = this->z_;
			this->beta_counts_mean_ += (this->prior_beta_counts_ + this->beta_counts_); // TODO need to make sure whether it's right
			this->theta_counts_mean_ += this->theta_counts_;

			bp_count++;
		}

		period = t.get_time();
		total_time_taken += period;
		pp = calc_corpus_perplexity2();
		pp2 = calc_corpus_perplexity3();
		ln_pp = calc_ln_corpus_partition_probality2();

		if (this->verbose_ >= 1) {
			cout << " time: " << period;
			cout << " dirichlet: " << dirichlet_time;
			cout << " perplexity: " << pp;
			cout << " new perplexity: " << pp2;
			cout << " log partition prob:" << ln_pp << "\n";
		}

		myfile << period << "," << dirichlet_time << "," << pp << ","  << pp2 << "," << ln_pp << "\n";
	}

	// END GIBBS ITERATIONS

	// Averages for burn in period
	if (this->burn_in_period_ > 0) {
		this->beta_counts_mean_ /= (bp_count + 1);
		this->theta_counts_mean_ /= (bp_count + 1);
		this->z_mode_ = find_mode(this->z_bp_);
	}


	if (this->verbose_ >= 1){
		cout << endl << "Total execution time: " << total_time_taken << "s" << endl;
		if (this->burn_in_period_ > 0)
			cout << "Model perplexity: " << calc_corpus_perplexity() << endl;
	}

	myfile.close();
}


void LDAPPMModel::run_fixed_beta_gibbs (string output_prefix)
{
	vector < size_t > word_indices;
	mat beta_sample;
	mat prior_beta;
	vec theta_d;
	vec beta_remain;
	rowvec beta_row;
	size_t bp_count = 0;
	size_t document_length = 0;
	Timer t = Timer();
	double total_time_taken = 0.0, period = 0.0;
	double pp = 0.0, ln_pp = 0.0;
	register unsigned int i = 0, iter = 0, d = 0;
	ofstream myfile;
	string str = output_prefix + "_full";
	myfile.open(str.c_str());

	myfile << "iter,time,dirichlet_time,perplexity,partition_prob" << endl;


	Timer t1 = Timer();
	double dirichlet_time;

	// No Gibbs sampling for beta

	prior_beta = this->prior_beta_counts_ + this->eta_;
	for(register unsigned int k = 0; k < this->num_topics_; k++)
		prior_beta.row(k) /= sum(prior_beta.row(k));



	// START GIBBS ITERATIONS

	for (iter = 0; iter < this->max_iterations_; iter++){

		t.restart_time();
		dirichlet_time = 0.0;

		if (this->verbose_ >= 1) cout << "iter #" << iter + 1;
		myfile << iter + 1 << ",";

		for (d = 0; d < this->num_documents_; d++){ // for each document

			word_indices = this->document_word_indices_[d];
			document_length = this->document_lengths_[d];

			t1.restart_time();

			// Gibbs sampling for theta

			this->theta_counts_.col(d) = calc_partition_counts(word_indices); // updates theta counts
			this->theta_sample_.col(d) = sample_dirichlet_col_vec(this->num_topics_, this->theta_counts_.col(d) + this->alpha_);

			dirichlet_time += t1.get_time();

			// Gibbs sampling for each document-word-instance

			for(i = 0; i < document_length; i++)
				this->beta_counts_(this->z_(word_indices[i]), this->word_ids_(word_indices[i])) -= 1; // Excludes the document d's word-topic counts


			//  Samples topics Z (word topic selection)
			for(i = 0; i < document_length; i++)
				this->z_(word_indices[i]) = sample_multinomial(this->theta_sample_.col(d) % prior_beta.col(this->word_ids_(word_indices[i])));

			for(i = 0; i < document_length; i++)
				this->beta_counts_(this->z_(word_indices[i]), this->word_ids_(word_indices[i])) += 1; // updates beta counts

		}


		// Handles burn in period

		if (this->burn_in_period_ > 0 && iter > this->burn_in_period_){
			this->z_bp_.col(bp_count) = this->z_;
			this->beta_counts_mean_ += this->beta_counts_;
			this->theta_counts_mean_ += this->theta_counts_;

			bp_count++;
		}

		period = t.get_time();
		total_time_taken += period;
		pp = calc_corpus_perplexity2();
		ln_pp = calc_ln_corpus_partition_probality2();

		if (this->verbose_ >= 1) {
			cout << " time: " << period;
			cout << " dirichlet: " << dirichlet_time;
			cout << " perplexity: " << pp;
			cout << " log partition prob:" << ln_pp << "\n";
		}

		myfile << period << "," << dirichlet_time << "," << pp << ","  << ln_pp << "\n";
	}

	// END GIBBS ITERATIONS

	// Averages for burn in period
	if (this->burn_in_period_ > 0) {
		this->beta_counts_mean_ /= (bp_count + 1);
		this->theta_counts_mean_ /= (bp_count + 1);
		this->z_mode_ = find_mode(this->z_bp_);
	}


	if (this->verbose_ >= 1){
		cout << endl << "Total execution time: " << total_time_taken << "s" << endl;
		if (this->burn_in_period_ > 0)
			cout << "Model perplexity: " << calc_corpus_perplexity() << endl;
	}

	myfile.close();
}



// Runs Gibbs sampler for all documents in the corpus in an incremental basis

void LDAPPMModel::run_incremental_gibbs (string output_prefix)
{
	vector < size_t > word_indices;
	vec theta_d;
	size_t bp_count = 0;
	mat beta_counts = zeros(this->num_topics_, this->vocabulary_size_);
	mat doc_beta_counts = zeros(this->num_topics_, this->vocabulary_size_);
	mat beta_sample;
	Timer t = Timer();
	long double total_time_taken = 0.0;
	double pp = 0.0;
	double period = 0.0;
	ofstream myfile;
	string str = output_prefix + "_full";
	myfile.open(str.c_str());

	myfile << "Gibbs sampling" << endl;

	this->beta_counts_.fill(0);

	for (size_t d = 0; d < this->num_documents_; d++){ // for each document

		t.restart_time();

		if (this->verbose_ >= 1) cout << "doc #" << d + 1;
		myfile << "doc #" << d + 1;

		word_indices = this->document_word_indices_[d];
		bp_count = 0;
		doc_beta_counts.fill(0);

		// START GIBBS ITERATIONS

		for (size_t iter = 0; iter < this->max_iterations_; iter++){

			// Gibbs sampling for beta

			beta_counts.fill(0);
			for(size_t i = 0; i < this->document_lengths_[d]; i++)
				beta_counts(this->z_(word_indices[i]), this->word_ids_(word_indices[i])) += 1;

			beta_sample = this->prior_beta_counts_ + beta_counts + this->eta_;
			for(size_t k = 0; k < this->num_topics_; k++)
				beta_sample.row(k) = sample_dirichlet_row_vec(this->vocabulary_size_, beta_sample.row(k));


			// Gibbs sampling for theta
			this->theta_counts_.col(d) = calc_partition_counts(word_indices);
			this->theta_sample_.col(d) = sample_dirichlet_col_vec(this->num_topics_, this->theta_counts_.col(d) + this->alpha_);


			//  Gibbs sampling for Z (word topic selection)
			for(size_t i = 0; i < this->document_lengths_[d]; i++)
				this->z_(word_indices[i]) = sample_multinomial(this->theta_sample_.col(d) % beta_sample.col(this->word_ids_(word_indices[i])));


			// Handles burn in period

			if (this->burn_in_period_ > 0 && iter > this->burn_in_period_){
				for (size_t di = 0; di < this->document_lengths_[d]; di++)
					this->z_bp_(word_indices[di], bp_count) = this->z_(word_indices[di]);
				this->theta_samples_mean_.col(d) += this->theta_sample_.col(d);
				doc_beta_counts += beta_counts;
				bp_count++;
			}

		}

		// END GIBBS ITERATIONS

		period = t.get_time();
		total_time_taken += period;

		if (this->verbose_ >= 1) cout << " exec time: " << period << "\n";
		myfile << " exec time: " << period << "\n";

		// Handles burn in period for all a document
		if (this->burn_in_period_ > 0){
			// Finds mode and updates the prior beta counts
			for (size_t di = 0; di < this->document_lengths_[d]; di++){
				size_t z_mode = this->mode(this->z_bp_.row(word_indices[di]), bp_count);
				this->z_mode_(word_indices[di]) = z_mode;
			}
			this->beta_counts_mean_ += (doc_beta_counts / bp_count); // adds document
			this->theta_samples_mean_.col(d) /= bp_count;
		}
		else {
			for (size_t di = 0; di < this->document_lengths_[d]; di++)
				this->beta_counts_(this->z_(word_indices[di]), this->word_ids_(word_indices[di])) += 1; // considers only the final Z
		}

	} // for each document

	// Handles burn in for all documents
	if (this->burn_in_period_ > 0){

		this->beta_counts_mean_ += this->prior_beta_counts_; // aggregate with prior; check whether it's invalid TODO
		// Here the beta matrix calculation is different from
		// the full or online (batch) Gibbs sampler;
		for(size_t k = 0; k < this->num_topics_; k++)
			this->beta_samples_mean_.row(k) = sample_dirichlet_row_vec(
					this->vocabulary_size_,
					this->beta_counts_mean_.row(k) + this->eta_);

//		this->beta_counts_ = this->beta_counts_mean_;
//		this->beta_sample_ = this->beta_samples_mean_;

	}
	else {

		this->beta_counts_ += this->prior_beta_counts_;

		for(size_t k = 0; k < this->num_topics_; k++)
			this->beta_sample_.row(k) = sample_dirichlet_row_vec(
					this->vocabulary_size_,
					this->beta_counts_.row(k) + this->eta_);

	}

	myfile << endl << "Total execution time: " << total_time_taken << "s" << endl;
	if (this->burn_in_period_ > 0){
		pp = calc_corpus_perplexity();
		myfile << "Model perplexity: " << pp  << endl;
	}

	if (this->verbose_ >= 1){
		cout << endl << "Total execution time: " << total_time_taken << "s" << endl;
		if (this->burn_in_period_ > 0) 	cout << "Model perplexity: " << pp << endl;
	}

}



void LDAPPMModel::run_gibbs_with_KL(string output_prefix)
{
	vector < size_t > word_indices;
	vector < size_t > unique_words;
	vec theta_d;
	size_t bp_count = 0;
	mat prob_in = zeros<mat>(this->num_topics_, this->vocabulary_size_);
	mat prob_ex = zeros<mat>(this->num_topics_, this->vocabulary_size_);
	Timer t = Timer();
	long double total_time_taken = 0.0;

	// START GIBBS ITERATIONS

	for (size_t iter = 0; iter < this->max_iterations_; iter++){

		if (this->verbose_ >= 1){
			t.restart_time();
			cout << "iter #" << iter + 1;
		}

		for (size_t d = 0; d < this->num_documents_; d++){ // for each document

			word_indices = this->document_word_indices_[d];
			unique_words = this->document_unique_words_[d];

			// Gibbs sampling for beta
			this->beta_sample_ = this->prior_beta_counts_ + this->beta_counts_ + this->eta_;
			for(size_t k = 0; k < this->num_topics_; k++)
				this->beta_sample_.row(k) = sample_dirichlet_row_vec(this->vocabulary_size_, this->beta_sample_.row(k));

			// Gibbs sampling for theta
			this->theta_counts_.col(d) = calc_partition_counts(word_indices); // updates theta counts
			this->theta_sample_.col(d) = sample_dirichlet_col_vec(this->num_topics_, this->theta_counts_.col(d) + this->alpha_);


			// Gibbs sampling for each document-word-instance

			for(size_t i = 0; i < this->document_lengths_[d]; i++)
				this->beta_counts_(this->z_(word_indices[i]), this->word_ids_(word_indices[i])) -= 1; // Excludes the document d's word-topic counts

			//  Samples topics Z (word topic selection)
			for(size_t i = 0; i < this->document_lengths_[d]; i++)
				this->z_(word_indices[i]) = sample_multinomial(this->theta_sample_.col(d) % this->beta_sample_.col(this->word_ids_(word_indices[i])));

			for(size_t i = 0; i < this->document_lengths_[d]; i++)
				this->beta_counts_(this->z_(word_indices[i]), this->word_ids_(word_indices[i])) += 1; // updates beta counts

		}


		// Handles burn in period

		if (this->burn_in_period_ > 0 && iter > this->burn_in_period_){
			this->z_bp_.col(bp_count) = this->z_;
			this->beta_samples_mean_ += this->beta_sample_;
			this->theta_samples_mean_ += this->theta_sample_;
			this->beta_counts_mean_ += this->prior_beta_counts_ + this->beta_counts_;
			this->theta_counts_mean_ += this->theta_counts_;

			for (size_t ii = 0; ii < this->num_word_instances_; ii++){
				for (size_t tt = 0; tt < this->num_topics_; tt++){
					if (this->z_(ii) ==  tt)
						prob_in(tt, this->word_ids_(ii)) += 1;
					else
						prob_ex(tt, this->word_ids_(ii)) += 1;
				}
			}

			bp_count++;
		}


		if (this->verbose_ == 2){
			cout << " model perplexity: " << calc_corpus_perplexity();
//			cout << " ln partition prob: " << calc_ln_corpus_partition_probality();
		}

		if (this->verbose_ >= 1){
			total_time_taken += t.get_time();
			cout << " time: " << t.get_time() << "\n";
		}
	}

	// END GIBBS ITERATIONS

	// Averages for burn in period
	if (this->burn_in_period_ > 0) {
		this->beta_samples_mean_ /= bp_count;
		this->theta_samples_mean_ /= bp_count;
		this->beta_counts_mean_ /= bp_count;
		this->theta_counts_mean_ /= bp_count;
		this->z_mode_ = find_mode(this->z_bp_);

		prob_in /= bp_count;
		prob_ex / bp_count;
	}
	else {
		for (size_t ii = 0; ii < this->num_word_instances_; ii++){
			for (size_t tt = 0; tt < this->num_topics_; tt++){
				if (this->z_(ii) ==  tt)
					prob_in(tt, this->word_ids_(ii)) += 1;
				else
					prob_ex(tt, this->word_ids_(ii)) += 1;
			}
		}
	}

	if (this->verbose_ >= 1)
		cout << endl << "Total execution time: "
		<< total_time_taken << "s" << endl;

	prob_in.save(output_prefix + "_prob_in.dat", raw_ascii);
	prob_ex.save(output_prefix + "_prob_ex.dat", raw_ascii);

}

/**
 * Calculates the corpus perplexity (uses beta and theta mean)
 *
 * Ref: LDA Gibbs implementation by David Newman
 *
 */

double LDAPPMModel::calc_corpus_perplexity() {

	double perplexity, ln_likelihood = 0, p1, p2, Z, prob_wd;
	vec partition_counts = zeros<vec> (this->num_topics_);

	// Calculates number of words assigned to each topic
	for (size_t t = 0; t < this->num_topics_; t++)
		partition_counts(t) = accu(this->beta_counts_mean_.row(t) + this->eta_);

	// partition_counts += 1e-24;

	for (size_t i = 0; i < this->num_word_instances_; i++) {

		Z = prob_wd = 0;

		for (size_t t = 0; t < this->num_topics_; t++) {
			p1 = this->beta_counts_mean_(t, this->word_ids_(i)) + this->eta_;
			p2 = this->theta_counts_mean_(t, this->document_indices_(i)) + this->alpha_;
			Z += p2;
			prob_wd += p1 * p2 / partition_counts(t);
		}

		ln_likelihood += log(prob_wd / Z);
	}

	perplexity = exp(-ln_likelihood / this->num_word_instances_);

	return perplexity;
}

double LDAPPMModel::calc_corpus_perplexity2() {

	double perplexity, ln_likelihood = 0, p1, p2, Z, prob_wd;
	vec partition_counts = zeros<vec> (this->num_topics_);

	// Calculates number of words assigned to each topic
	for (size_t t = 0; t < this->num_topics_; t++)
		partition_counts(t) = accu(this->beta_counts_.row(t) + this->eta_);

	// partition_counts += 1e-24;

	for (size_t i = 0; i < this->num_word_instances_; i++) {

		Z = prob_wd = 0;

		for (size_t t = 0; t < this->num_topics_; t++) {
			p1 = this->beta_counts_(t, this->word_ids_(i)) + this->eta_;
			p2 = this->theta_counts_(t, this->document_indices_(i)) + this->alpha_;
			Z += p2;
			prob_wd += p1 * p2 / partition_counts(t);
		}

		ln_likelihood += log(prob_wd / Z);
	}

	perplexity = exp(-ln_likelihood / this->num_word_instances_);

	return perplexity;
}


//
///*
// * This function calculates partition
// * probability for the corpus.
// *
// */
//double LDAPPMModel::calc_ln_corpus_partition_probality() {
//
//	double partition_probability = 0.0;
//
//	// Calculate partition counts from  m_ji' s; i' = 1 ... V
//	vec partition_counts = sum(this->beta_counts_mean_, 1); // sums over rows
//
//	// ln_gamma (n_j + alpha_j)
//	vec ln_gamma_Nj = log_gamma_vec(partition_counts + this->alpha_);
//
//	vec ln_gamma_Mj = zeros<vec> (this->num_topics_);
//	for (size_t t = 0; t < this->num_topics_; t++)
//		ln_gamma_Mj(t) = sum(log_gamma_rowvec(this->beta_counts_mean_.row(t) + this->eta_));
//
//	partition_probability = accu(ln_gamma_Nj + ln_gamma_Mj); // sum over all j s
//
//	return partition_probability;
//}


/*
 * This function calculates partition
 * probability for the corpus.
 *
 */
double LDAPPMModel::calc_ln_corpus_partition_probality2() {

	double partition_probability = 0.0;
	vec ln_gamma_Nj = zeros<vec> (this->num_topics_);
	vec ln_gamma_Mj = zeros<vec> (this->num_topics_);

	// \sum_{d = 0}^{D} ln_gamma (n_j,d + alpha_j)
	for (size_t d = 0; d < this->num_documents_; d++)
		ln_gamma_Nj += log_gamma_vec(this->theta_counts_.col(d) + this->alpha_);

	// \sum_{t = 0}^{V} ln_gamma (\sum_{d=0}^{D}m_{jt,d} + alpha_j)
	for (size_t t = 0; t < this->vocabulary_size_; t++)
		ln_gamma_Mj += log_gamma_vec(this->beta_counts_.col(t) + this->eta_);

	partition_probability = accu(ln_gamma_Nj + ln_gamma_Mj); // sum over all topics

	return partition_probability;

}


double LDAPPMModel::calc_corpus_perplexity3() {
	return exp(-calc_ln_corpus_partition_probality2() / this->num_word_instances_);
}

