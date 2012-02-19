#include "hdp.h"
#include "utils.h"
#include <assert.h>

#define VERBOSE true
#define null NULL

#define PERMUTE true
#define PERMUTE_LAG 10
#define TABLE_SAMPLING true
#define NUM_SPLIT_MERGE_TRIAL 15
#define SPLIT_MERGE_MAX_ITER 1
#define SPLIT_MERGE_LAG 50

hdp::hdp() {
	m_hdp_param = NULL;
	m_state = NULL;
}

hdp::~hdp() {
	m_hdp_param = NULL;
	delete m_state;
	m_state = NULL;
}

void hdp::load(char * model_path) {
	m_state = new hdp_state();
	m_state->load_state_ex(model_path);
}

/**
 * Used for HDP test
 */
void hdp::setup_state(
		const corpus * c,
		parameters * _hdp_param) {

	m_hdp_param = _hdp_param;
	m_state->setup_state_from_corpus(c);
	m_state->allocate_initial_space();

}

/**
 * Used for HDP train
 */
void hdp::setup_state(
		const corpus * c,
		double _eta,
		int _init_topics,
		parameters * _hdp_param) {

	m_hdp_param = _hdp_param;
	m_state = new hdp_state();

	m_state->setup_state_from_corpus(c);
	m_state->allocate_initial_space();
	m_state->m_eta = _eta;
	m_state->m_num_topics = _init_topics; // initial topics given

	/// use the means of gamma distribution
	m_state->m_gamma = m_hdp_param->gamma_a * m_hdp_param->gamma_b;
	m_state->m_alpha = m_hdp_param->alpha_a * m_hdp_param->alpha_b;

}

void hdp::run() {

	double best_likelihood = 0.0;
	double likelihood = 0.0;
	int tot = 0;
	int acc = 0;
	char name[500];
	bool permute = false;
	FILE* file;
	Timer timer = Timer();
	double total_time_taken = 0.0;
	double total_time_taken_sm = 0.0;
	double time_taken = 0.0;



	// Initializes state

	printf("starting with %d topics \n", m_state->m_num_topics);

	if (m_state->m_num_topics == 0)
		m_state->iterate_gibbs_state(false, PERMUTE, m_hdp_param, TABLE_SAMPLING);
	else if (m_state->m_num_topics > 0)
		m_state->init_gibbs_state_using_docs();
	else { // m_state->m_num_topics < 0
		m_state->m_num_topics = abs(m_state->m_num_topics);
		m_state->init_gibbs_state_with_fixed_num_topics();
	}


	// Open the log file

	sprintf(name, "%s/%s_state_log", m_hdp_param->output_directory, m_hdp_param->output_prefix);
	file = fopen(name, "w");
	fprintf(file, "time iter num.topics num.tables likelihood gamma alpha split merge trial sm_time\n");

	printf("After init: %d topics \n", m_state->m_num_topics);

	best_likelihood = m_state->joint_likelihood(m_hdp_param);

	// START SAMPLING ITERATIONS

	for (int iter = 0; iter < m_hdp_param->max_iter; iter++) {

		printf("iter = %5d: ", iter+1);

		if (PERMUTE && (iter > 0) && (iter % PERMUTE_LAG == 0))
			permute = true;
		else
			permute = false;

		timer.restart_time();

		m_state->iterate_gibbs_state(true, permute, m_hdp_param, TABLE_SAMPLING); // The CRP Gibbs

		likelihood = m_state->joint_likelihood(m_hdp_param);
		time_taken = timer.get_time();
		total_time_taken += time_taken;


		printf("#topics = %4d, #tables = %4d, gamma = %.5f, alpha = %.5f, likelihood = %.5f, time = %2.2f\n",
				m_state->m_num_topics, m_state->m_total_num_tables, m_state->m_gamma, m_state->m_alpha, likelihood, time_taken);

		fprintf(file, "%8.2f %05d %04d %05d %.5f %.5f %.5f ", time_taken, iter,
				m_state->m_num_topics, m_state->m_total_num_tables, likelihood, m_state->m_gamma, m_state->m_alpha);


		// Saves state based on the criteria

		if (best_likelihood < likelihood) {
			best_likelihood = likelihood;
			sprintf(name, "%s/%s_mode", m_hdp_param->output_directory, m_hdp_param->output_prefix);
			m_state->save_state(name);
			sprintf(name, "%s/%s_mode_bin", m_hdp_param->output_directory, m_hdp_param->output_prefix);
			m_state->save_state_ex(name);
		}

		if (m_hdp_param->save_lag != -1 && (iter % m_hdp_param->save_lag == 0)) {
			sprintf(name, "%s/%s_%05d", m_hdp_param->output_directory, m_hdp_param->output_prefix, iter);
			m_state->save_state(name);
			sprintf(name, "%s/%s_%05d_bin", m_hdp_param->output_directory, m_hdp_param->output_prefix, iter);
			m_state->save_state_ex(name);
		}




		// BEGIN SPLIT AND MERGE

		int num_split = 0, num_merge = 0;
		double sm_time_taken = 0.0;

		if (m_hdp_param->split_merge_sampler && iter < SPLIT_MERGE_MAX_ITER) { //  (iter % SPLIT_MERGE_LAG == 0)

			timer.restart_time();

			for (int num = 0; num < NUM_SPLIT_MERGE_TRIAL; num++) {

				tot++;
				hdp_state* proposed_state = new hdp_state();
				proposed_state->copy_state(m_state);

				int d0, t0, d1, t1, k0, k1;
				double prob_split, r;

				ACTION action = m_state->select_mcmc_move(d0, d1, t0, t1);
				double u = log(runiform());

				if (action == SPLIT) {

					k0 = m_state->m_doc_states[d0]->m_table_to_topic[t0];
					k1 = m_state->m_num_topics;
					prob_split = proposed_state->split_sampling(m_hdp_param->num_restricted_scans, d0, d1, t0, t1);

					r = compute_split_ratio(proposed_state, m_state, k0, k1);
					printf("like.log = %5.2lf, ", r);
					printf("scan.log = %5.2lf, ", prob_split);
					r -= prob_split;

					if (u < r) {
						acc++;
						num_split++;
						printf("ratio.log = %5.2lf/%5.2lf, split (--- A ---), ", r, u);
						printf("%d -> %d\n", m_state->m_num_topics, m_state->m_num_topics + 1);
						hdp_state* old_state = m_state;
						m_state = proposed_state;
						proposed_state = old_state;
					} else
						printf("ratio.log = %5.2lf, split (--- R ---)\n", r);


				}
				else { // action == MERGE

					hdp_state* intermediat_state = new hdp_state();
					intermediat_state->copy_state(m_state);

					prob_split = intermediat_state->split_sampling(m_hdp_param->num_restricted_scans, d0, d1, t0, t1, m_state);

					k0 = m_state->m_doc_states[d0]->m_table_to_topic[t0];
					k1 = m_state->m_doc_states[d1]->m_table_to_topic[t1];
					proposed_state->merge_two_topics(k0, k1);

					r = -compute_split_ratio(m_state, proposed_state, k0, k1);
					printf("like.log = %5.2lf, ", r);
					printf("scan.log = %5.2lf, ", prob_split);
					r += prob_split;

					if (u < r) {
						acc++;
						num_merge++;
						printf("ratio.log = %5.2lf/%5.2lf, merge (--- A ---), ", r, u);
						printf("%d -> %d\n", m_state->m_num_topics, m_state->m_num_topics - 1);
						hdp_state* old_state = m_state;
						m_state = proposed_state;
						proposed_state = old_state;
					}
					else
						printf("ratio.log = %5.2lf, merge (--- R ---)\n", r);


					delete intermediat_state;
				}

				delete proposed_state;

			}

			sm_time_taken = timer.get_time();
			total_time_taken_sm += sm_time_taken;
			printf("split-merge time = %2.2f\n", sm_time_taken);
		}

		// END SPLIT AND MERGE


		fprintf(file, "%d %d %d %2.2f\n", num_split, num_merge, NUM_SPLIT_MERGE_TRIAL, sm_time_taken);
		fflush(file);

	}

	// END SAMPLING ITERATIONS


	// Saves the final state
	sprintf(name, "%s/%s", m_hdp_param->output_directory, m_hdp_param->output_prefix);
	m_state->save_state(name);

	fclose(file);

	printf("\nTotal time taken for Gibbs sampling: %.2f\n", total_time_taken);
	printf("Total time taken for SM sampling: %.2f\n", total_time_taken_sm);

	if (m_hdp_param->split_merge_sampler)
		printf("acceptance rate: %0.2f\n", 100.0 * (double) acc / tot);

}

void hdp::run_test() {

	Timer timer = Timer();
	double total_time_taken = 0.0;
	double time_taken = 0.0;
	bool permute = false;
	double likelihood, best_likelihood;

	double old_likelihood = m_state->table_partition_likelihood() + m_state->data_likelihood();

	m_state->iterate_gibbs_state(false, PERMUTE, m_hdp_param, TABLE_SAMPLING); //init the state

	printf("starting with %d topics \n", m_state->m_num_topics);

	char name[500];
	sprintf(name, "%s/%s_test.log", m_hdp_param->output_directory, m_hdp_param->output_prefix);
	FILE* file = fopen(name, "w");
	fprintf(file, "time iter num.topics num.tables likelihood\n");


	for (int iter = 0; iter < m_hdp_param->max_iter; iter++) {

		printf("iter = %5d, ", iter);

		if (PERMUTE && (iter > 0) && (iter % PERMUTE_LAG == 0))
			permute = true;
		else
			permute = false;

		timer.restart_time();

		m_state->iterate_gibbs_state(true, permute, m_hdp_param, TABLE_SAMPLING);

		time_taken = timer.get_time();
		total_time_taken += time_taken;

		likelihood = m_state->joint_likelihood(m_hdp_param) - old_likelihood;

		printf("#topics = %4d, #tables = %4d, likelihood = %.5f, time = %2.2f\n",
				m_state->m_num_topics, m_state->m_total_num_tables, likelihood, time_taken);
		fprintf(file, "%8.2f %05d %04d %05d %.5f\n", time_taken, iter,
				m_state->m_num_topics, m_state->m_total_num_tables, likelihood);

		if (m_hdp_param->save_lag != -1 && (iter % m_hdp_param->save_lag == 0)) {
			sprintf(name, "%s/%s_test-%05d", m_hdp_param->output_directory, m_hdp_param->output_prefix, iter);
			m_state->save_state(name);
			sprintf(name, "%s/%s_test-%05d.bin", m_hdp_param->output_directory, m_hdp_param->output_prefix, iter);
			m_state->save_state_ex(name);
		}

		if (iter == 0 || best_likelihood < likelihood) {
			best_likelihood = likelihood;
			sprintf(name, "%s/%s_test_mode", m_hdp_param->output_directory, m_hdp_param->output_prefix);
			m_state->save_state(name);
			sprintf(name, "%s/%s_test_mode.bin", m_hdp_param->output_directory, m_hdp_param->output_prefix);
			m_state->save_state_ex(name);
		}
	}
	fclose(file);


	printf("\nTotal time taken for sampling: %.2f\n", total_time_taken);

}
