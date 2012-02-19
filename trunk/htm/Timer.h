#ifndef _TIMER_H_
#define _TIMER_H_

#include <sys/time.h>
#include <time.h>

/*
 * This class implements a timer on Linux with high
 * precision. The timer is based on gettimeofday().
 *
 * Created By : Girish R.
 *
 */

class Timer {

private:
	double start_time; // the start time in seconds (high precision)

	// Current time
	double current_time(void) {
		struct timeval time;
		gettimeofday(&time, NULL);
		return time.tv_sec + 1.0e-6 * time.tv_usec;
	}

public:
	Timer(void) {
		start_time = 0.0;
	}

	// To start the timer
	void restart_time(void) {
		start_time = current_time();
	}

	// To get the current time
	double get_time(void) {
		return current_time() - start_time;
	}

};

#endif // _TIMER_H_
