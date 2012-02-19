// part of this code in this file is taken from
// the LDA-C code from prof. David Blei.
// reading the file with lda-c format
//
#include <stdlib.h>
#include <stdio.h>
#include "corpus.h"

document::document() {
	words = NULL;
	counts = NULL;
	length = 0;
	total = 0;
	id = -1;
}

document::document(int len) {
	length = len;
	words = new int[length];
	counts = new int[length];
	total = 0;
	id = -1;
}

document::~document() {
	if (words != NULL) {
		delete[] words;
		delete[] counts;
		length = 0;
		total = 0;
		id = -1;
	}
}

corpus::corpus() {
	size_vocab = 0;
	total_words = 0;
	num_docs = 0;
}

corpus::corpus(int _size_vocabulary, int _total_num_words, int _num_docs) {
	size_vocab = _size_vocabulary;
	total_words = _total_num_words;
	num_docs = _num_docs;
}

corpus::~corpus() {
	for (int i = 0; i < num_docs; i++) {
		document * doc = docs[i];
		delete doc;
	}
	docs.clear();

	size_vocab = 0;
	num_docs = 0;
	total_words = 0;
}

void corpus::read_data(const char * filename) {

	int OFFSET = 0; // represents whether the word IDs start with 0
	FILE * fileptr;
	int length, count, word_id, n, nd, nw;

	nd = max(0, num_docs); // To handle online learning
	nw = max(0, size_vocab); // To handle online learning

	// reads the data
	printf("\nreading data from %s\n", filename);

	fileptr = fopen(filename, "r");
	while ((fscanf(fileptr, "%10d", &length) != EOF)) {

		document * doc = new document(length);
		for (n = 0; n < length; n++) {
			if(fscanf(fileptr, "%10d:%10d", &word_id, &count) != EOF){
				word_id = word_id - OFFSET;
				doc->words[n] = word_id;
				doc->counts[n] = count;
				doc->total += count;
				if (word_id >= nw)
					nw = word_id + 1;
			}
		}
		total_words += doc->total;
		doc->id = nd;
		docs.push_back(doc);

		nd++;

	}
	fclose(fileptr); // closes the file

	num_docs = nd;
	size_vocab = nw;

	printf("number of documents   : %d\n", nd);
	printf("number of terms       : %d\n", nw);
	printf("number of total words : %d\n\n", total_words);
}


// end of the file
