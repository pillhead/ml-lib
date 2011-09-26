/*
 * LDAPPMBase.cpp
 *
 *  Created on: Mar 22, 2011
 *      Author: Clint P. George
 */

#include "LDAPPMBase.h"


LDAPPMBase::LDAPPMBase(size_t max_iterations,
		string data_file,
		string data_format,
		string vocab_file) {

	this->init_random_generator();

	this->max_iterations_ = max_iterations;
	this->vocabulary_size_ = count_gibbs_data_file_lines(vocab_file);

	this->data_format_ = data_format;
	if (!this->data_format_.compare("gibbs")){
		this->num_word_instances_ = count_gibbs_data_file_lines(data_file);
		read_gibbs_data_file(data_file);
	}
	else if (!this->data_format_.compare("ldac")){
		read_ldac_data_file(data_file);
	}

}

LDAPPMBase::~LDAPPMBase() {

	this->free_random_generator();

//	if (this->log_file_.is_open())
//		this->log_file_.close();
}

void LDAPPMBase::print_metadata(){

	if (this->verbose_ == 1){
		cout << "\nData set information:\n\n";
		cout << "Total word instances       = " << this->num_word_instances_ << endl;
		cout << "Vocabulary size            = " << this->vocabulary_size_ << endl;
		cout << "Number of documents        = " << this->num_documents_ << endl << endl;
	}
}

void LDAPPMBase::set_verbose(size_t value){
	this->verbose_ = value;
}

void LDAPPMBase::print_message(string msg){
	if (this->verbose_ >= 1)
		cout << msg;
}

template <typename T>
std::string LDAPPMBase::to_string(T const& value) {
    stringstream sstr;
    sstr << value;
    return sstr.str();
}


void LDAPPMBase::init_random_generator(){

	random_num_generator_ = gsl_rng_alloc (gsl_rng_ranlux);     		// pick random number generator
	random_seed_ = time (NULL) * getpid();								// seed

	gsl_rng_set (random_num_generator_, random_seed_);                  // sets seed

}


void LDAPPMBase::free_random_generator(){

	gsl_rng_free(random_num_generator_);

}


vec LDAPPMBase::sample_stick_breaking_prior(
		size_t num_elements,
		double a, double b){

	vec stick_breaking_prior = zeros<vec>(num_elements);
	double theta_element = 0;

	for (size_t i = 0; i < num_elements; i++){

		theta_element = gsl_ran_beta(random_num_generator_, a, b);

		if (i == 0) {
			stick_breaking_prior(i) = theta_element;
			continue;
		}

		for (size_t c = 0; c < i - 1; c++)
			theta_element *= (1 - gsl_ran_beta(random_num_generator_, a, b));

		stick_breaking_prior(i) = theta_element;

	}

	return stick_breaking_prior;
}


double LDAPPMBase::sample_beta(double a, double b){
	return gsl_ran_beta(random_num_generator_, a, b);
}

size_t LDAPPMBase::sample_uniform_int(size_t K){
	return (size_t) gsl_rng_uniform_int(random_num_generator_, K);
}

double LDAPPMBase::sample_uniform(){
	return gsl_rng_uniform(random_num_generator_);
}

double LDAPPMBase::log_gamma(double x){
	return gsl_sf_lngamma (x);
}

vec LDAPPMBase::log_gamma_vec(vec x_vec){

	vec lgamma_vec = zeros<vec>(x_vec.n_elem);

	for (size_t i = 0; i < x_vec.n_elem; i++)
		lgamma_vec(i) = gsl_sf_lngamma(x_vec(i));

	return lgamma_vec;

}

rowvec LDAPPMBase::log_gamma_rowvec(rowvec x_vec){

	rowvec lgamma_vec = zeros<rowvec>(x_vec.n_elem);

	for (size_t i = 0; i < x_vec.n_elem; i++)
		lgamma_vec(i) = gsl_sf_lngamma(x_vec(i));

	return lgamma_vec;

}

size_t LDAPPMBase::sample_multinomial (vec theta) {

	register unsigned int i,t;
	register unsigned int num_elements = theta.n_elem;
	unsigned int z[num_elements];
	double *theta_d = new double[num_elements]();

	for (t = 0; t < num_elements; t++)
		theta_d[t] = theta(t);

	gsl_ran_multinomial (random_num_generator_,
			num_elements, 1, theta_d, z);

	delete [] theta_d;

	for (i = 0; i < num_elements; i++)
		if (z[i] == 1) return i;

	assert(i != num_elements); // if multinomial fails

	return 0;

}


vec LDAPPMBase::sample_dirichlet_col_vec (size_t num_elements, vec alpha){

	vec dirichlet_sample = zeros<vec>(num_elements);
	double theta[num_elements];
	double *alpha_d = new double[num_elements]();

	for (register unsigned int v = 0; v < num_elements; v++)
		alpha_d[v] = alpha(v);

	gsl_ran_dirichlet (random_num_generator_, num_elements, alpha_d, theta);

	delete []alpha_d;

	for (register unsigned int i = 0; i < num_elements; i++)
		dirichlet_sample(i) = theta[i];

	return dirichlet_sample;

}

rowvec LDAPPMBase::sample_dirichlet_row_vec (size_t num_elements, rowvec alpha){

	rowvec dirichlet_sample = zeros<rowvec>(num_elements);
	double theta[num_elements];
	double *alpha_d = new double[num_elements]();

	for (register unsigned int v = 0; v < num_elements; v++)
		alpha_d[v] = alpha(v);

	gsl_ran_dirichlet (random_num_generator_, num_elements, alpha_d, theta);

	delete []alpha_d;

	for (register unsigned int i = 0; i < num_elements; i++)
		dirichlet_sample(i) = theta[i];

	return dirichlet_sample;

}

rowvec LDAPPMBase::sample_dirichlet_row_vec (rowvec alpha, vector <size_t> indices){

	rowvec dirichlet_sample = ones<rowvec>(alpha.n_elem) * 1e-15;
	register unsigned int sel_num_elements = indices.size();
	double theta[sel_num_elements];
	double *alpha_d = new double[sel_num_elements]();

	for (register unsigned int v = 0; v < sel_num_elements; v++)
		alpha_d[v] = alpha(indices[v]);

	gsl_ran_dirichlet (random_num_generator_, sel_num_elements, alpha_d, theta);

	delete []alpha_d;

	for (register unsigned int i = 0; i < sel_num_elements; i++)
		dirichlet_sample(indices[i]) = theta[i];

	return dirichlet_sample;

}


/**
 * Counts number of lines in a given file
 */

size_t LDAPPMBase::count_gibbs_data_file_lines (string file_name) {

	size_t lines = 0;
	string temp = "";

	ifstream in(file_name.c_str());
	if (in.is_open())
		while(getline(in, temp))
			lines++;
	in.close();

	return lines;
}


/**
 * Reads data from a given data file in Gibbs format.
 *
 * Data format:
 * 		Each data record (line) is in this format [doc id] [word id],
 * 		which represents a word instance in the corpus.
 *
 * Note:
 * 		We assume vocabulary ids start from 1 (since the arrays in
 * 		C/C++ starts from 0, we need a correction as OFFSET)
 *
 */

void LDAPPMBase::read_gibbs_data_file (string file_name){

	// This method should be called after setting this value
	assert(this->num_word_instances_ > 0);

	unsigned int wid = 0;
	unsigned int did = 0;
	size_t OFFSET = 1;
	size_t line = 0;
	size_t previous_did = 0;
	size_t document_length = 0;
	vector <size_t> idx_words;
	vector <size_t> unique_words;
	map <size_t, size_t> word_counts;

	FILE *fp = fopen(file_name.c_str() ,"r"); assert(fp);

	this->document_indices_ = zeros<uvec>(this->num_word_instances_);
	this->word_ids_ = zeros<uvec>(this->num_word_instances_);
	this->num_documents_ = 0;

	while(fscanf(fp, "%u%u", &did, &wid) != EOF){

		if (previous_did != did && line > 0){
			this->document_word_indices_.push_back(idx_words);
			this->document_lengths_.push_back(document_length);
			this->document_word_counts_.push_back(word_counts);
			this->document_unique_words_.push_back(unique_words);
			this->num_documents_++;
			idx_words.clear();
			word_counts.clear();
			unique_words.clear();
			document_length = 0;
		}

		this->word_ids_[line] = wid - OFFSET;
		this->document_indices_[line] = did - OFFSET;
		idx_words.push_back(line);

		if (word_counts.find(wid - OFFSET) == word_counts.end()){ // observed at first time
			word_counts[wid - OFFSET] = 1;
			unique_words.push_back(wid - OFFSET);
		}
		else
			word_counts[wid - OFFSET] += 1;

		previous_did = did;
		document_length++;
		line++;
	}

	fclose(fp);

	// Adds the last document's indices
	if (0 < line && line <= this->num_word_instances_){
		this->document_word_indices_.push_back(idx_words);
		this->document_lengths_.push_back(document_length);
		this->document_word_counts_.push_back(word_counts);
		this->document_unique_words_.push_back(unique_words);
		this->num_documents_++;
	}

}


/**
 * Reads data from a given data file which is LDA-C format.
 *
 * Data format:
 * 		Each line is this format [unique document words] [vocabulary id]:[count]
 *
 * Note:
 * 		We assume vocabulary ids start from 1 (since the arrays in
 * 		C/C++ starts from 0, we need a correction as OFFSET)
 *
 */

void LDAPPMBase::read_ldac_data_file(string file_name) {

	int OFFSET = 0; // represents whether the word IDs start with 0
	int length, count, word, n, ni, ret;
	size_t num_word_instances = 0;
	size_t document_length = 0;
	vector <size_t> word_indices;
	vector <size_t> unique_words;
	vector <size_t> document_ids;
	vector <size_t> word_ids;
	map <size_t, size_t> word_counts;
	unsigned int did = 0;

	// reading the data
	cout << "\nReading data from " << file_name << endl;

	FILE * fileptr = fopen(file_name.c_str(), "r"); assert(fileptr);
	this->num_documents_ = 0;

	while ((fscanf(fileptr, "%10d", &length) != EOF)) {

		for (n = 0; n < length; n++) {
			ret = fscanf(fileptr, "%10d:%10d", &word, &count);assert(ret);
			word = word - OFFSET;
			word_counts[word] = count;
			unique_words.push_back(word);

			for (ni = 0; ni < count; ni++){
				word_indices.push_back(num_word_instances);
				word_ids.push_back(word);
				document_ids.push_back(did);
				num_word_instances++; // increments word instances
			}
			document_length += count;
		}

		this->document_lengths_.push_back(document_length);
		this->document_word_indices_.push_back(word_indices);

		this->document_word_counts_.push_back(word_counts);
		this->document_unique_words_.push_back(unique_words);
		this->num_documents_++;

		word_indices.clear();
		word_counts.clear();
		unique_words.clear();
		document_length = 0;

		did++; // increments document ids

	}

	fclose(fileptr);

	this->num_word_instances_ = num_word_instances; // set the total number of instances
	this->document_indices_ = zeros<uvec>(this->num_word_instances_);
	this->word_ids_ = zeros<uvec>(this->num_word_instances_);

	for (size_t i = 0; i < this->num_word_instances_; i++){
		this->document_indices_(i) = document_ids[i];
		this->word_ids_(i) = word_ids[i];
	}

	document_ids.empty();
	word_ids.empty();

	cout << "Number of documents   : " << did << endl;
	cout << "Number of total words : " << num_word_instances << endl << endl;

}


void LDAPPMBase::read_ldac_data_file2(string file_name) {

	int OFFSET = 0; // represents whether the word IDs start with 0
	int length, count, word, n, ni, ret, document_id;
	size_t num_word_instances = 0;
	size_t document_length = 0;
	vector <size_t> word_indices;
	vector <size_t> unique_words;
	vector <size_t> document_indices;
	vector <size_t> document_ids;
	vector <size_t> word_ids;
	map <size_t, size_t> word_counts;
	unsigned int did = 0;

	// reading the data
	cout << "\nReading data from " << file_name << endl;

	FILE * fileptr = fopen(file_name.c_str(), "r"); assert(fileptr);
	this->num_documents_ = 0;

	while ((fscanf(fileptr, "%10d %10d", &document_id, &length) != EOF)) {

		for (n = 0; n < length; n++) {
			ret = fscanf(fileptr, "%10d:%10d", &word, &count);assert(ret);
			word = word - OFFSET;
			word_counts[word] = count;
			unique_words.push_back(word);

			for (ni = 0; ni < count; ni++){
				word_indices.push_back(num_word_instances);
				word_ids.push_back(word);
				document_indices.push_back(did);
				document_ids.push_back(document_id);
				num_word_instances++; // increments word instances
			}
			document_length += count;
		}

		this->document_lengths_.push_back(document_length);
		this->document_word_indices_.push_back(word_indices);

		this->document_word_counts_.push_back(word_counts);
		this->document_unique_words_.push_back(unique_words);
		this->num_documents_++;

		word_indices.clear();
		word_counts.clear();
		unique_words.clear();
		document_length = 0;

		did++; // increments document ids

	}

	fclose(fileptr);

	this->num_word_instances_ = num_word_instances; // set the total number of instances
	this->document_indices_ = zeros<uvec>(this->num_word_instances_);
	this->word_ids_ = zeros<uvec>(this->num_word_instances_);
	this->document_ids_ = zeros<uvec>(this->num_word_instances_);

	for (size_t i = 0; i < this->num_word_instances_; i++){
		this->document_indices_(i) = document_indices[i];
		this->word_ids_(i) = word_ids[i];
		this->document_ids_(i) = document_ids[i];
	}

	document_indices.empty();
	word_ids.empty();
	document_ids.empty();

	cout << "Number of documents   : " << did << endl;
	cout << "Number of total words : " << num_word_instances << endl << endl;

}


vec LDAPPMBase::calc_topic_counts (uvec Z_vec, size_t num_topics){

	register unsigned int num_elements = Z_vec.n_elem;
	vec partition_counts = zeros<vec>(num_topics);

	for (register unsigned int i = 0; i < num_elements; i++)
		partition_counts(Z_vec(i)) += 1;

	return partition_counts;
}

/*
 * Finds mode for a given [1 X D] row vector
 *
 */

u32 LDAPPMBase::mode(urowvec data, int size) {
	register int t, w;
	u32 md, oldmd;
	int count, oldcount;

	oldmd = 0;
	oldcount = 0;
	for (t = 0; t < size; t++) {
		md = data(t);
		count = 1;
		for (w = t + 1; w < size; w++)
			if (md == data(w))
				count++;
		if (count > oldcount) {
			oldmd = md;
			oldcount = count;
		}
	}
	return oldmd;
}

/*
 * Finds row-wise modes for a [N X D] matrix,
 * and returns [N X 1] vector
 *
 */

uvec LDAPPMBase::find_mode(umat accum_matrix){

	register unsigned int num_rows = accum_matrix.n_rows;
	register unsigned int num_cols = accum_matrix.n_cols;
	uvec topics_mode = zeros<uvec>(num_rows);

	for (register unsigned int i = 0; i < num_rows; i++)
		topics_mode(i) = mode(accum_matrix.row(i), num_cols);

	return topics_mode;
}
