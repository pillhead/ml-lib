## 
## This file generates synthetic data using the LDA generative process 
## and test R and C versions of the LDA Product Partition Model 
## 
## Created On : April 09, 2011
## Created By : Clint P. George 
##


setwd("F:/Research/ml-lib/lda-r")


# Loads the necessary R pkgs 

library(MCMCpack);
library(plotrix);

# Includes the source files  

source('process_data.R');
source('utils.R');
source('top.topic.documents.R');


# Displaying the original beta (TRUTH) that is used to generate synthetic documents 

sbeta <- t(read.table('../lda-data/synth_beta.txt', header=F)); 
color2D.matplot(sbeta, c(0.6, 0), c(0, 0.9), c(0,1), xlab="words", ylab="partitions (topics)", main="Synthetic beta matrix");
		

# Emulates LDA's generative process 
# Call this function if you wanna generate 
# documents on the fly, using the known Beta  
#
alpha <- 1; 
lambda.h <- 400;
V <- 100; 
D <- 200; 
K <- 10;
ds <- LDASamples(K, D, V, alpha, lambda.h, sbeta);


## WRITES INTO FILES 

did.wid <- cbind(ds$did, ds$wid);
write.table(did.wid, file = "../lda-data/synth200d400w.docword", row.names = FALSE, col.name=FALSE);
write.table(ds$theta, file = "../lda-data/synth200d400w.theta", row.names = FALSE, col.name=FALSE);
write.table(ds$theta.counts, file = "../lda-data/synth200d400w.theta_counts", row.names = FALSE, col.name=FALSE);
write.table(1:100, file = "../lda-data/synth200d400w.vocab", row.names = FALSE, col.name=FALSE);


### ==================================================================================================================================== ### 


## RUN THE R VERSION OF LDA 

# Parameters for the LDA Gibbs sampling (R package)


## If we read from the document-words file 

V <- 100; 
D <- 250; 
total.N <- 50255
docword <- read.table(file = "../synth_docs.docword");
wid <- docword[,2]
did <- docword[,1]
max.iter <- 100;  # Maximum number of Gibbs iterations 
burn.in <- 90;  # Maximum number of Gibbs iterations 
eta <- 1; 
K <- 10;
alpha <- 0.9; 


## OR 

## If we use from the list generated from the LDA 
## generative process 

D <- ds$D;
V <- ds$V
total.N <- ds$total.N
wid <- ds$wid
did <- ds$did
max.iter <- 100;  # Maximum number of Gibbs iterations 
burn.in <- 90;  # Maximum number of Gibbs iterations 
eta <- 1; 
K <- 10;
alpha <- 0.9; 


# Fixed K 
# Rprof("lda.ppm.out");
model <- LDAPP(K, D, V, total.N, wid, did, alpha, eta, max.iter, burn.in, T); # set the last argument to TRUE if you like to visualize 
# Rprof(NULL);
# summaryRprof("lda.ppm.out"); # prints summery 

plot.doc.topics(model$theta);
plot.theta(theta);





### ==================================================================================================================================== ### 


## LOADS AND DISPLAYS THE LDA C VERSION OUTPUT 

sbeta <- t(read.table('../synth_beta.txt', header=F)); # Synthetic beta 
train.gibbs.beta <- read.table(file = "../synth200d400w/lda_beta_samples_mean.dat");
train.gibbs.theta <- read.table(file = "../synth200d400w/lda_theta_samples_mean.dat");

train.gibbs.beta.counts <- read.table(file = "../synth200d400w/lda_beta_counts_mean.dat"); 
train.gibbs.theta.counts <- read.table(file = "../synth200d400w/lda_theta_counts_mean.dat")

par(mfrow = c(2,2))
color2D.matplot(sbeta, c(1, 0), c(1, 0), c(1,0), , border=NA, xlab="vocabulary words", ylab="partitions (topics)", main="The true beta", axes=FALSE);fullaxis(1,lwd=1); fullaxis(2,lwd=1);
color2D.matplot(train.gibbs.beta, c(1, 0), c(1, 0), c(1,0), , border=NA, xlab="vocabulary words", ylab="partitions (topics)", main="Gibbs sampling - training set", axes=FALSE);fullaxis(1,lwd=1); fullaxis(2,lwd=1);
color2D.matplot(train.gibbs.beta.counts, c(1, 0), c(1, 0), c(1,0), , border=NA, xlab="vocabulary words", ylab="partitions (topics)", main="Gibbs sampling - training set", axes=FALSE);fullaxis(1,lwd=1); fullaxis(2,lwd=1);

plot.doc.topics(train.gibbs.theta);


