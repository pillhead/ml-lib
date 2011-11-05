## 
## This file generates synthetic data using the LDA generative process 
## and test R and C versions of the LDA Product Partition Model 
## 
## Created On : April 09, 2011
## Created By : Clint P. George 
##

setwd("~/Dropbox/lda-r")

# Loads the necessary R pkgs 

library(MCMCpack);
library(plotrix);

# Includes the source files  

source('process_data.R');
source('utils.R');
source('top.topic.documents.R');


# Displaying the original beta (TRUTH) that is used to generate synthetic documents 
## 10 topics 
sbeta <- t(read.table('../lda-data/synth/synth_beta.txt', header=F)); 
color2D.matplot(sbeta, c(0.6, 0), c(0, 0.9), c(0,1), xlab="words", ylab="partitions (topics)", main="Synthetic beta matrix");
		
## 2 topics 
beta <- matrix(0, nrow=2, ncol=100);
beta[1, 1:70] <- 1;
beta[2, 30:100] <- 1;
beta <- t(NormalizeTopics(t(beta)))
color2D.matplot(beta, c(0.6, 0), c(0, 0.9), c(0,1), xlab="words", ylab="partitions (topics)", main="Synthetic beta matrix");
write.table(beta, file='../lda-data/synth/synth_beta_2t', row.names=F, col.names=F)    

## 5 topics 
beta <- matrix(0, nrow=5, ncol=100);
beta[1, 1:35] <- 1;
beta[2, 15:55] <- 1;
beta[3, 30:75] <- 1;
beta[4, 40:85] <- 1;
beta[5, 65:100] <- 1;
beta <- t(NormalizeTopics(t(beta)))
color2D.matplot(beta, c(0.6, 0), c(0, 0.9), c(0,1), xlab="words", ylab="partitions (topics)", main="Synthetic beta matrix");
write.table(beta, file='../lda-data/synth/synth_beta_5.1t', row.names=F, col.names=F)    

sbeta <- read.table('../lda-data/synth/synth_beta_5t', header=F);


# Emulates LDA's generative process 
# Call this function if you wanna generate 
# documents on the fly, using the known Beta  
#
alpha <- 1; 
lambda.h <- 400;
V <- 100; 
D <- 200; 
K <- 5;
ds <- LDASamples(K, D, V, alpha, lambda.h, beta);


## WRITES INTO FILES 

did.wid <- cbind(ds$did, ds$wid);
write.table(did.wid, file = "../lda-data/synth/200d400w_5.1t.docword", row.names = FALSE, col.name=FALSE);
write.table(ds$theta, file = "../lda-data/synth/200d400w_5.1t.theta", row.names = FALSE, col.name=FALSE);
write.table(ds$theta.counts, file = "../lda-data/synth/200d400w_5.1t.theta_counts", row.names = FALSE, col.name=FALSE);
write.table(1:100, file = "../lda-data/synth/200d400w_5.1t.vocab", row.names = FALSE, col.name=FALSE);


### ==================================================================================================================================== ### 


## LOADS AND DISPLAYS THE LDA C VERSION OUTPUT 

sbeta <- t(read.table('../lda-data/synth/synth_beta.txt', header=F)); # Synthetic beta 
train.gibbs.beta <- read.table(file = "../lda-data/synth/200d400w_beta_samples_mean.dat");
train.gibbs.theta <- read.table(file = "../lda-data/synth/200d400w_theta_samples_mean.dat");

train.gibbs.beta.counts <- read.table(file = "../lda-data/synth/200d400w_beta_counts_mean.dat"); 
train.gibbs.theta.counts <- read.table(file = "../lda-data/synth/200d400w_theta_counts_mean.dat")

train.ogibbs.beta.counts <- t(read.table(file = "../lda-data/synth/200d400w_Nwt_full.txt"));
train.ogibbs.theta.counts <- t(read.table(file = "../lda-data/synth/200d400w_Ndt_full.txt"))


train.ppm.idx <- c(2, 5, 6, 8, 9, 10, 7, 3, 1, 4); # order of the topics found from the Gibbs sampler   
train.gibbs.idx <- c(2, 1, 5, 4, 7, 9, 6, 8, 10, 3); # order of the topics found from the Gibbs sampler   


par(mfrow = c(2,2))
color2D.matplot(sbeta, c(1, 0), c(1, 0), c(1,0), , border=NA, xlab="vocabulary words", ylab="partitions (topics)", main="The true beta", axes=FALSE);fullaxis(1,lwd=1); fullaxis(2,lwd=1);
color2D.matplot(train.gibbs.beta[train.ppm.idx,], c(1, 0), c(1, 0), c(1,0), , border=NA, xlab="vocabulary words", ylab="partitions (topics)", main="PPM - samples' mean", axes=FALSE);fullaxis(1,lwd=1); fullaxis(2,lwd=1);
color2D.matplot(train.gibbs.beta.counts[train.ppm.idx,], c(1, 0), c(1, 0), c(1,0), , border=NA, xlab="vocabulary words", ylab="partitions (topics)", main="PPM - counts' mean", axes=FALSE);fullaxis(1,lwd=1); fullaxis(2,lwd=1);
color2D.matplot(train.ogibbs.beta.counts[train.gibbs.idx,], c(1, 0), c(1, 0), c(1,0), , border=NA, xlab="vocabulary words", ylab="partitions (topics)", main="Collapsed Gibbs - final", axes=FALSE);fullaxis(1,lwd=1); fullaxis(2,lwd=1);


plot.doc.topics(train.gibbs.theta);


