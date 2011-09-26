//============================================================================
// Name        : ldappm_cpp.cpp
// Author      : Clint P. George
// Version     : 0.3
// Copyright   : 
// Description : This file is an interface to the DA product partition
// 				 model topic learning and inference algorithms
//============================================================================

#include <iostream>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <dirent.h>
#include <time.h>
#include <sys/stat.h>
#include <sys/types.h>

#include "LDAPPMModel.h"
#include "TopicSearch.h"

/**
*
* check if directory exisits
*/
bool is_dir_exists(string directory)
{
    struct stat st;
    if(stat(directory.c_str(), &st) == 0)
        return true;
    return false;
}

void print_usage_and_exit()
{
    cout << "\nC++ implementation of Gibbs sampling (full Gibbs sampler) and Hybrid Metropolis search for LDA.\n";
    cout << "usage:\n";
    cout << "\n  general parameters:\n";
    cout << "      --algorithm (mandatory):\n";
    cout << "                 lda              - LDA Gibbs sampling with fixed number of topics \n"
			"                 lda_oi           - online (incremental) LDA Gibbs sampling  \n"
			"                 lda_ob           - online (batch) LDA Gibbs sampling  \n"
    		"                 ts_hrw           - topic search based on hybrid Metropolis search \n";
    cout << "      --data (mandatory):         data file (for format see README)\n";
    cout << "      --vocab (mandatory):        vocabulary file (for format see README) \n";
    cout << "      --max_iter:                 the max number of iterations, default 100\n";
    cout << "      --burn_in:                  the burn in period, default 90\n";
    cout << "      --topics:                   the given number of topics\n";
    cout << "      --alpha:                    hyper-parameter (symmetric) for document topic distributions, default 1.0\n";
    cout << "      --eta:                      hyper-parameter (symmetric) for topic distributions, default 1.0\n";
    cout << "      --data_format (mandatory):  [gibbs] - Gibbs format, [ldac] - LDA-C format reference Blei's LDA implementation\n";
    cout << "      --output_prefix:            output files prefix\n";
    cout << "      --output_dir:               output files folder\n";
    cout << "      --verbose:                  [0] - do not display log, [1] - display log to standard output, default 1\n";

    cout << "\n  topic search parameters:\n";
    cout << "      --saved_beta:               saved beta samples\n";
    cout << "      --spacing:                  spacing between independent samples\n";
    cout << "      --init_temp:                initial temperature for simulated annealing\n";
    cout << "      --cool_temp:                cool down temperature for simulated annealing\n";
    cout << "      --rw_prob:                  random walk probability, default 0.7\n";
    cout << "      --perc_rw:                  percentage of data points selected for random walk, default 1%\n";

    cout << "\n  online (batch / incremental) LDA parameters:\n";
    cout << "      --saved_beta_counts:        used for Gibbs sampler based online learning\n";

    cout << "\nexamples:\n";
    cout << " ./ldappm --algorithm lda --data synth200d400w.documents --vocab synth200d400w.vocab --data_format ldac --output_prefix lda --output_dir synth200d400w --topics 10 --max_iter 100 --burn_in 90\n";
    cout << " ./ldappm --algorithm ts_hrw --data synth200d400w.documents --vocab synth200d400w.vocab --saved_beta lda_beta_samples_mean.dat  --data_format ldac --output_prefix hrw --output_dir synth200d400w --topics 10 --max_iter 2500 --burn_in 2000\n";
    cout << endl << endl;
    exit(0);
}


int main(int argc, char** argv)
{

	if (argc < 2
    		|| !strcmp(argv[1], "-help")
    		|| !strcmp(argv[1], "--help")
    		|| !strcmp(argv[1], "-h")
    		|| !strcmp(argv[1], "--usage")){
        print_usage_and_exit();
    }

	string output_dir = "";
    string algorithm = "";
	string data_file = "";
	string vocab_file = "";
	string saved_beta_counts = "";
	string saved_beta = "";
	string data_format = "";
	string output_prefix = "lda";
	size_t topic_count = 0;
	size_t max_iter = 0;
	size_t burn_in_period = 0;
	size_t spacing = 10;
	size_t init_temp = 0;
	size_t cool_temp = 0;
	double alpha = 1;
	double eta = 1;
	double random_walk_prob = 0.70;
	double percent_random_walk = 1.0;
	size_t verbose = 1;


    for (int i = 1; i < argc; i++){

        if (!strcmp(argv[i], "--algorithm"))        algorithm = argv[++i];
        else if (!strcmp(argv[i], "--data"))        data_file = argv[++i];
        else if (!strcmp(argv[i], "--vocab"))       vocab_file = argv[++i];
        else if (!strcmp(argv[i], "--max_iter"))    max_iter = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--burn_in"))     burn_in_period = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--topics"))      topic_count = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--alpha"))       alpha = atof(argv[++i]);
        else if (!strcmp(argv[i], "--eta"))         eta = atof(argv[++i]);
        else if (!strcmp(argv[i], "--spacing"))     spacing = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--saved_beta_counts")) saved_beta_counts = argv[++i];
        else if (!strcmp(argv[i], "--saved_beta"))  saved_beta = argv[++i];
        else if (!strcmp(argv[i], "--init_temp"))   init_temp = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--cool_temp"))   cool_temp = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--verbose"))     verbose = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--rw_prob"))     random_walk_prob = atof(argv[++i]);
        else if (!strcmp(argv[i], "--perc_rw"))     percent_random_walk = atof(argv[++i]);
        else if (!strcmp(argv[i], "--data_format")) data_format = argv[++i];
        else if (!strcmp(argv[i], "--output_prefix")) output_prefix = argv[++i];
        else if (!strcmp(argv[i], "--output_dir"))  output_dir = argv[++i];
        else {
            cout << argv[i] << "is an unknown parameter, exit\n";
            exit(0);
        }

    }

    if (!algorithm.compare("")
    		|| !data_file.compare("")
    		|| !vocab_file.compare("")
    		|| !data_format.compare("")) {
        cout << "Algorithm, data file, vocabulary file, and data format are mandatory!\n";
        exit(0);
    }

    if (verbose == 1) {
        cout << "\nProgram starts with following parameters: " << endl << endl;
        cout << "    algorithm:          = " << algorithm << endl;
        cout << "    data file:          = " << data_file << endl;
        cout << "    vocabulary file:    = " << vocab_file << endl;
        cout << "    max_iter            = " << max_iter << endl;
        cout << "    burn in period      = " << burn_in_period << endl;
        cout << "    topics              = " << topic_count << endl;
        cout << "    alpha               = " << alpha << endl;
        cout << "    eta                 = " << eta << endl;
        cout << "    data format         = " << data_format << endl;
        cout << "    output directory    = " << output_dir << endl;
        cout << "    verbose             = " << verbose << endl;

        if (!algorithm.compare("lda_oi") || !algorithm.compare("lda_ob"))
        	cout << "    saved_beta_counts   = " << saved_beta_counts << endl;

        if (!algorithm.compare("ts_hrw")){
			cout << "    spacing             = " << spacing << endl;
			cout << "    init. temp.         = " << init_temp << endl;
			cout << "    cool-down temp.     = " << cool_temp << endl;
			cout << "    saved_beta          = " << saved_beta << endl;
			cout << "    random walk prob.   = " << random_walk_prob << endl;
			cout << "    % of random walk    = " << percent_random_walk << endl;
        }
    }

    if (!is_dir_exists(output_dir))
        mkdir(output_dir.c_str(), S_IRUSR | S_IWUSR | S_IXUSR);

    if (output_dir.compare("") && is_dir_exists(output_dir))
    	output_prefix = output_dir + "/" + output_prefix; // prefix with directory

	if (!algorithm.compare("lda")){

		LDAPPMModel mdl = LDAPPMModel(
				topic_count,
				max_iter,
				burn_in_period,
				alpha,
				eta,
				data_file,
				data_format,
				vocab_file);

		mdl.set_verbose(verbose);
		mdl.print_metadata();
		if (verbose >= 1)
			cout << "\nThe full Gibbs sampler - iterations\n================================\n";
		mdl.run_gibbs();
		mdl.save_state(output_prefix);

	}
	else if (!algorithm.compare("lda_ob")){

		if (!saved_beta_counts.compare("")) {
	        cout << endl << "saved beta-counts file is mandatory!\n";
	        exit(1);
		}


		LDAPPMModel mdl = LDAPPMModel(
				topic_count,
				max_iter,
				burn_in_period,
				alpha,
				eta,
				data_file,
				data_format,
				vocab_file,
				saved_beta_counts);
		mdl.set_verbose(verbose);
		mdl.print_metadata();
		mdl.print_message("\nThe full Gibbs sampler with a prior Beta (batch)\n ================================\n");
		mdl.run_gibbs();
		mdl.save_state(output_prefix);
	}
	else if (!algorithm.compare("lda_oi")){

		if (!saved_beta_counts.compare("")) {
	        cout << endl << "saved beta-counts file is mandatory!\n";
	        exit(1);
		}

		LDAPPMModel mdl = LDAPPMModel(
				topic_count,
				max_iter,
				burn_in_period,
				alpha, eta,
				data_file,
				data_format,
				vocab_file,
				saved_beta_counts);
		mdl.set_verbose(verbose);
		mdl.print_message("\nThe full Gibbs sampler with a prior Beta (Incremental)\n ================================\n");
		mdl.run_incremental_gibbs();
		mdl.save_state(output_prefix);

	}
	else if (!algorithm.compare("ts_hrw")){

		if (!saved_beta.compare("")) {
	        cout << endl << "saved beta file is mandatory!\n";
	        exit(0);
		}

		TopicSearch ts_model  = TopicSearch(
				data_file,
				data_format,
				vocab_file,
				saved_beta,
				alpha,
				max_iter,
				spacing,
				burn_in_period);



		if (init_temp > 0 && cool_temp > 0 && init_temp > cool_temp){
			ts_model.print_message("\nTopic Search -- Hybrid Random Walk ( Simulated Annealing )\n ===========================================\n");
			ts_model.run_hybrid_random_walk_simulated_annealing(
					init_temp,
					cool_temp,
					random_walk_prob,
					percent_random_walk);
			ts_model.save_theta(output_prefix);
		}
		else {
			ts_model.print_message("\nTopic Search -- Hybrid Random Walk\n================================\n");
			ts_model.run_hybrid_random_walk(
					random_walk_prob,
					percent_random_walk);
			ts_model.save_theta(output_prefix);
		}

	}

	return 0;
}




