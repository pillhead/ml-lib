
*******************************************
**      LDA Product Partition Model      **
*******************************************

C++ source: 
 - ldappm.cpp                 - main C++ file 
 - Timer.cpp                  - Handles the timer calculations 
 - LDAPPMBase.cpp             - The base class for the full Gibbs sampler and topic search implementations 
 - LDAPPMModel.cpp            - Implementation of the LDA product partition model - full Gibbs sampler and online sampler (both batch and incremental)
 - TopicLearningGibbs.cpp     - Implementation of the full Gibbs sampler for learning topics (It's not a working copy) 
 - TopicSearch.cpp            - Implementation of the Hybrid-Metropolis search for the LDA topic search  


References: 
 1. Dirichlet Allocation Using Product Partition Models 
 2. Topic Learning and Inference Using Dirichlet Allocation 
    Product Partition Models and Hybrid Metropolis Search  

Dependencies: 
 1. GSL C library (http://www.gnu.org/software/gsl) 
 2. Armadillo C++ (http://arma.sourceforge.net)


C++ source compilation 
=============================
Install g++

Install GSL C Library 
 - sudo apt-get install libatlas-dev gsl-bin libgsl0-dev 

Install armadillo C++
 - follow instructions in - http://arma.sourceforge.net/download.html
 - sudo apt-get install cmake libblas-dev liblapack-dev libboost-dev
 - wget -c http://sourceforge.net/projects/arma/files/armadillo-1.2.0.tar.gz
 - tar xvfz armadillo-1.2.0.tar.gz
 - from folder armadillo-1.2.0 
       run cmake .
       run make
       run sudo make install

Guide lines to configure in eclipse CDT IDE: 
- add /usr/lib64/ to librariies 
- add /usr/include/armadillo to libraries 
- Settings -> Misc : -c -fmessage-length=0 -larmadillo -lgsl -lgslcblas -lm
- Optimization: O1


Test run 
=============================
 1. make 
 2. for help run ./ldappm --help 
 2. for test run ./ldappm --algorithm lda --data testdata/synth200d400w.docword --vocab testdata/synth200d400w.vocab --topics 10 --max_iter 100 --burn_in 90

To use a custom path:

 1. update makefile 
 2. setenv LD_LIBRARY_PATH /home/cgeorge/lib/gsl/lib:/home/cgeorge/lib/armadillo/usr/lib64


