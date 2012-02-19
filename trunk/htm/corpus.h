#ifndef _CORPUS_H
#define	_CORPUS_H

#include <vector>
#include <cstddef>
using namespace std;

class document {
public:
	int * words;
	int * counts;
	int length;
	int total;
	int id;
public:
	document();
	document(int len);
	~document();
};

class corpus {
public:
	corpus();
	corpus(int _size_vocabulary, int _total_num_words, int _num_docs);
	~corpus();
	void read_data(const char * filename);
public:
	int size_vocab;
	int total_words;
	int num_docs;
	vector<document*> docs;
};

#endif	/* _CORPUS_H */

