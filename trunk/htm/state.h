#ifndef STATE_H
#define STATE_H
#include "Timer.h"
#include "corpus.h"

#include <map>

/**
 * For the fixed parameters used in the HDP learning/testing
 */
class parameters {

public:

	double gamma_a; // top level DP shape parameter
	double gamma_b; // top level DP scale parameter
	double alpha_a; // bottom level DP shape parameter
	double alpha_b; // bottom level DP scale parameter
	int max_iter; // max number of Gibbs iterations
	int save_lag;
	int num_restricted_scans;

	bool sample_hyperparameter;
	bool split_merge_sampler;

	const char * output_directory;
	const char * output_prefix;

public:

	void setup_parameters(double _gamma_a, double _gamma_b, double _alpha_a,
			double _alpha_b, int _max_iter, int _save_lag,
			int _num_restricted_scans, bool _sample_hyperparameter,
			bool _split_merge_sampler, const char * _output_directory,
			const char * _output_prefix);
};

typedef vector<int> int_vec; // define the vector of int
typedef vector<double> double_vec; // define the vector of double
typedef map<int, int> word_stats;
enum ACTION { SPLIT, MERGE };
enum DOCUMENT_STATUS { NEW, PROCESSED };


/// word info structure used in the main class
struct word_info {

public:
	int m_word_index;
	int m_table_assignment;

};




class doc_state {

public:
	int m_doc_id; // document id
	int m_doc_length; // document length
	int m_num_tables; // number of tables in this document, m_j.
	word_info * m_words; // words' information
	DOCUMENT_STATUS m_doc_status;

	int_vec m_table_to_topic; // table index to topic index, i.e., disk k assigned to table t ??
	int_vec m_word_counts_by_t; // word counts for each table, i.e., n_jt
	vector<word_stats> m_word_stats_by_t;

	bool m_is_valid;

	//vector < vector<int> > m_words_by_zi; // stores the word idx indexed by z then i

public:
	doc_state();
	virtual ~doc_state();

public:
	void setup_state_from_doc(const document * doc);
	void free_doc_state();

};



class hdp_state {

public:

	int m_size_vocab; // vocabulary size (V)
	int m_total_words; // total number of word instances (N)
	int m_num_docs; // number of documents in the corpus (D)


	doc_state** m_doc_states; /// document states
	int m_num_topics; // number of topics (K)
	int m_total_num_tables; // total number of tables for all topics (m..)

	int_vec m_num_tables_by_z; // number of tables assigned for each topic (K X 1)
	int_vec m_word_counts_by_z; // number of words assigned for each topic (K X 1)
	vector<int*> m_word_counts_by_zd; // word counts for [each topic, each doc] (K X D)
	vector<int*> m_word_counts_by_zw; // word counts for [each topic, each word] (K X V)

	double m_eta; // topic Dirichlet hyper-parameter
	double m_gamma; // concentration parameter
	double m_alpha; // concentration parameter

public:
	hdp_state();
	virtual ~hdp_state();

public:
	void setup_state_from_corpus(const corpus* c);
	void allocate_initial_space();
	void free_state();
	void init_gibbs_state_using_docs();
	void init_gibbs_state_with_fixed_num_topics();
	void iterate_gibbs_state(bool remove, bool permute,	parameters* param, bool table_sampling = false);
	void sample_first_level_concentration(parameters* param);
	void sample_second_level_concentration(parameters* param);
	void sample_tables(doc_state* d_state, double_vec & q, double_vec & f);
	void sample_table_assignment(doc_state* d_state, int t, int* words, double_vec & q, double_vec & f);
	void sample_word_assignment(doc_state* d_state, int i, bool remove, double_vec & q, double_vec & f);
	void doc_state_update(doc_state* d_state, int i, int update, int k = -1);
	void compact_doc_state(doc_state* d_state, int* k_to_new_k);
	void compact_hdp_state();
	double doc_partition_likelihood(doc_state* d_state);
	double table_partition_likelihood();
	double data_likelihood();
	double joint_likelihood(parameters * param);
	void save_state(char * name);
	void save_state_ex(char * name);
	void load_state_ex(char * name);
//	void save_full_state_ex(char * name);
//	void load_full_state_ex(char * name);

	/// the followings are the functions used in the split-merge algorithm
	void copy_state(const hdp_state* state);
	ACTION select_mcmc_move(int& d0, int& d1, int& t0, int& t1);
	void doc_table_state_update(doc_state* d_state, int t, int update, int k =-1);
	double sample_table_assignment_sm(doc_state* d_state, int t, bool remove, int k0, int k1, int target_k = -1);
	void merge_two_topics(int k0, int k1);
	double split_sampling(int num_scans, int d0, int d1, int t0, int t1, hdp_state * target_state = NULL);
	friend double compute_split_ratio(const hdp_state* split_state, const hdp_state* merge_state, int k0, int k1);

	/// functions that should not be used when running the experiments
	bool state_check_sum();

protected:
	void add_new_topic();

};

#endif // STATE_H
