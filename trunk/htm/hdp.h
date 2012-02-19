#ifndef HDP_H
#define HDP_H

#include "state.h"


/// implement the Chinese restaurant franchise algorithm with split and merge
class hdp {
public:
	/// fixed parameters
	parameters * m_hdp_param;

	/// sampling state
	hdp_state * m_state;

public:
	hdp();
	virtual ~hdp();
public:
	void run();
	void run_test();

	void setup_state(const corpus * c, double _eta, int _init_topics, parameters * _hdp_param);
	void setup_state(const corpus * c, parameters * _hdp_param);
	void load(char * model_path);

};

#endif // HDP_H
