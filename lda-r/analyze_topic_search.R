## 
## This is a test script to compare the results from topic 
## search with the original and Gibbs sampler theta and 
## Beta 
##
## Created On : May 24, 2011
## Created By : Clint P. George 
##

# To set the working dir 

# setwd('/home/clint/ldappmcpp/ldappm_cpp/src') 
# setwd('/home/clint/ldappm/src')
setwd('/home/clint/Dropbox/ldappm/src')
# setwd("F:/My Dropbox/ldappm/src")

# Loads the necessary R pkgs 

library(MCMCpack);
library(plotrix);

# Includes the source files  

source('process_data.R');
source('utils.R');
source('top.topic.documents.R');
source('top.topic.words.R');


#==============================================================================================
# To display results 
#==============================================================================================


## To KL-divergence between topic mixtures
  
calc.KL.divergence <- function(theta1, theta2)
{
	L <- ncol(theta1); 
	kld <- matrix(0, nrow = L, ncol = 1);
	for (docid in 1:L) { kld[docid] <- KLDiv(theta1[,docid], theta2[,docid]); }

	kld; 
}


## TS synth2_docs.docword synth_docs.vocab synth_bp_beta_samples.dat 1 1000 20 0

sbeta <- t(read.table('../synth_beta.txt', header=F)); # Synthetic beta 
stheta <- read.table(file = "../nips2011-synth-train.theta", header=F); # Synthetic beta
stheta.counts <- read.table(file = "../nips2011-synth-train.theta_counts", header=F); # Synthetic beta

setwd("~/Dropbox/ldappm/synth200d400w")
train.gibbs.beta <- read.table(file = "lda_lt_beta_samples.dat");
train.gibbs.theta <- read.table(file = "lda_theta_samples_mean.dat");

train.gibbs.beta.counts <- read.table(file = "lda_lt2_beta_counts.dat"); 
train.gibbs.theta.counts <- read.table(file = "lda_lt_theta_counts.dat")

test.bg.beta <- read.table(file = "../nips2011-synth-test-bg_bp_beta_samples.dat");
test.bg.theta <- read.table(file = "../nips2011-synth-test-bg_bp_theta_samples.dat");

test.ig.beta <- read.table(file = "../nips2011-synth-test-ig_bp_beta_samples.dat");
test.ig.theta <- read.table(file = "../nips2011-synth-test-ig_bp_theta_samples.dat");

ts.sa.beta <- read.table(file = "../nips2011-synth-test_ts_sa1_beta_samples.dat");
ts.sa.beta2 <- read.table(file = "../nips2011-synth-test_ts_sa1_beta_samples_last.dat");

ts.hrw.theta <- read.table(file = "../synth200d400w/hrw_theta_samples.dat");
ts.hrw.beta <- read.table(file = "../nips2011-synth/hrw_beta_samples.dat");

train.gibbs.idx <- c(4, 1, 7, 10, 3, 5, 6, 8, 2, 9); # order of the topics found from the Gibbs sampler   
train.bg.idx <- c(4, 1, 7, 10, 3, 5, 6, 8, 2, 9);


## Calculkates KL-divergence between topic mixtures

kld.ts.hrw.gen <- calc.KL.divergence(ts.hrw.theta[train.gibbs.idx, ], stheta);
kld.bg.gen <- calc.KL.divergence(test.bg.theta[train.bg.idx, ], stheta);
kld.ig.gen <- calc.KL.divergence(test.ig.theta[train.gibbs.idx, ], stheta);

## Displays KL-divergences   
par(mfrow = c(2,3))

plot(1:length(kld.bg.gen), kld.bg.gen, ylim=c(0.1,3.5), type='o', col='blue', xlab='document indices', ylab='K-L divergence', main='Online Gibbs (batch)')
plot(1:length(kld.ig.gen), kld.ig.gen, ylim=c(0.1,3.5), type='o', col='blue', xlab='document indices', ylab='K-L divergence', main='Online Gibbs (incremental)')
plot(1:length(kld.ts.hrw.gen), kld.ts.hrw.gen, ylim=c(0.1,3.5), type='o', col='blue', xlab='document indices', ylab='K-L divergence', main='Hybrid random walk')

b <- seq(from=0,to=3.5,by =0.2);

hist(kld.bg.gen, freq=T, xlim=c(0,3.5), ylim=c(0,35), breaks=b, col="lightblue", border="gray", xlab='K-L divergence', main='Online Gibbs (batch)')
hist(kld.ig.gen, freq=T, xlim=c(0,3.5), ylim=c(0,35), breaks =b, col="lightblue", border="gray", xlab='K-L divergence', main='Online Gibbs (incremental)')
hist(kld.ts.hrw.gen, freq=T, xlim=c(0,3.5), ylim=c(0,35), breaks =b, col="lightblue", border="gray", xlab='K-L divergence', main='Hybrid random walk')

dev.print(device=postscript, "theta_kld.eps", onefile=FALSE, horizontal=TRUE);


## Displays Beta matrices 

par(mfrow = c(2,2))
color2D.matplot(sbeta, c(1, 0), c(1, 0), c(1,0), , border=NA, xlab="vocabulary words", ylab="partitions (topics)", main="The true beta", axes=FALSE);fullaxis(1,lwd=1); fullaxis(2,lwd=1);
color2D.matplot(valid.beta[2:28,], c(1, 0), c(1, 0), c(1,0), , border=NA, xlab="vocabulary words", ylab="partitions (topics)", main="Gibbs sampling - training set", axes=FALSE);fullaxis(1,lwd=1); fullaxis(2,lwd=1);


color2D.matplot(train.gibbs.beta, c(1, 0), c(1, 0), c(1,0), , border=NA, xlab="vocabulary words", ylab="partitions (topics)", main="Gibbs sampling - training set", axes=FALSE);fullaxis(1,lwd=1); fullaxis(2,lwd=1);
color2D.matplot(test.bg.beta[train.bg.idx, ], c(1, 0), c(1, 0), c(1,0), , border=NA, xlab="vocabulary words", ylab="partitions (topics)", main="Online Gibbs (batch)", axes=FALSE);fullaxis(1,lwd=1); fullaxis(2,lwd=1);
color2D.matplot(test.ig.beta[train.gibbs.idx, ], c(1, 0), c(1, 0), c(1,0), , border=NA, xlab="vocabulary words", ylab="partitions (topics)", main="Online Gibbs (incremental)", axes=FALSE);fullaxis(1,lwd=1); fullaxis(2,lwd=1);
color2D.matplot(ts.sa.beta[train.gibbs.idx, ], c(1, 0), c(1, 0), c(1,0), , border=NA, xlab="vocabulary words", ylab="partitions (topics)", main="Simulated annealing", axes=FALSE);fullaxis(1,lwd=1); fullaxis(2,lwd=1);
color2D.matplot(ts.hrw.beta, c(1, 0), c(1, 0), c(1,0), , border=NA, xlab="vocabulary words", ylab="partitions (topics)", main="Hybrid random walk", axes=FALSE);fullaxis(1,lwd=1); fullaxis(2,lwd=1);

dev.print(device=postscript, "beta_matrix.eps", onefile=FALSE, horizontal=TRUE);




row.sums <- rowSums(train.gibbs.beta.counts);
valid.idx <- (row.sums > 50); #  n^th percentile
valid.beta <- train.gibbs.beta.counts[valid.idx, ];







# TEST ON THE 20 NEWS DATA 

# Go to the TEST folder 
setwd("~/ldappmcpp/test")


display.theta <- function(theta, header.text='document topic proportions')
{
  X = 1:dim(theta)[2];
  cl = c("black", "blue", "red", "chocolate4", "yellow", "green", "darkorchid", "darkorange", "darkmagenta", "cyan");
  p = c(20, 7, 8, 10, 12, 13, 14, 15, 16, 17); 
  plot(X, theta[1,], xlim=range(X), ylim=c(0,1), cex=0.5, pch=p[1], col=cl[1], xlab="document indices", ylab="topic proportions", lwd=1, main=header.text);
  for (i in 2:dim(theta)[1])
    points(X, theta[i,], col=cl[i], cex=0.8, pch=p[i], lwd=1);
}


news.gibbs.theta <- read.table(file = "nips2011-20news-train_bp_theta_samples.dat");
news.gibbs.beta <- read.table(file = "nips2011-20news-train_bp_beta_samples.dat");
news.vocab <- read.vocab("nips2011-20news-train.vocab");

colnames(news.gibbs.beta) <- news.vocab;
news.gibbs.topic.words <- top.topic.words(news.gibbs.beta, num.words=10, by.score=TRUE); # displays top topics 
news.gibbs.topic.names <- c('alt.atheism', 'rec.sport.baseball', 'rec.autos', 'sci.space');
colnames(news.gibbs.topic.words) <- news.gibbs.topic.names;

news.gibbs.topic.words



news.hrw.theta <- read.table(file = "nips2011-20news-train_ts_hrw_theta_samples.dat");
news.hrw.beta <- read.table(file = "nips2011-20news-train_ts_hrw_beta_samples.dat");
colnames(news.hrw.beta) <- news.vocab;
news.hrw.topic.words <- top.topic.words(news.hrw.beta, num.words=20, by.score=TRUE); # displays top topics 
colnames(news.hrw.topic.words) <- news.gibbs.topic.names;

news.hrw.topic.words


news.test.hrw.theta <- read.table(file = "nips2011-20news-test_ts_hrw_theta_samples.dat");
news.test.hrw.beta <- read.table(file = "nips2011-20news-test_ts_hrw_beta_samples.dat");
colnames(news.test.hrw.beta) <- news.vocab;
news.test.hrw.topic.words <- top.topic.words(news.test.hrw.beta, num.words=20, by.score=TRUE); # displays top topics 
colnames(news.test.hrw.topic.words) <- news.gibbs.topic.names;

news.test.hrw.topic.words


news.test.ig.theta <- read.table(file = "nips2011-synth-test-ig_theta_samples.dat");



par(mfrow = c(3,2))
display.theta(news.gibbs.theta, 'The full Gibbs sampler');
display.theta(news.hrw.theta, 'Hybrid random walk - training set');

kld.news.gibbs.hrw <- calc.KL.divergence(news.gibbs.theta, news.hrw.theta);
# kld.news.hrw.gibbs <- calc.KL.divergence(news.hrw.theta, news.gibbs.theta);

plot(1:length(kld.news.gibbs.hrw), kld.news.gibbs.hrw, ylim=c(0,1), type='o', col='blue', xlab='document indices', ylab='K-L divergence', main='Full Gibbs - hybrid random walk')
b <- seq(from=0,to=1,by =0.05);
hist(kld.news.gibbs.hrw, freq=T, ylim=c(0,100), breaks=b, col="lightblue", border="gray", xlab='K-L divergence', main='Full Gibbs - hybrid random walk')

display.theta(news.test.hrw.theta, 'Hybrid random walk - test set');
display.theta(news.test.ig.theta, 'Incremental Gibbs - test set');




par(mfrow = c(2,2))
color2D.matplot(sbeta, c(1, 0), c(1, 0), c(1,0), , border=NA, xlab="vocabulary words", ylab="partitions (topics)", main="The true beta", axes=FALSE);fullaxis(1,lwd=1); fullaxis(2,lwd=1);
color2D.matplot(train.gibbs.beta, c(1, 0), c(1, 0), c(1,0), , border=NA, xlab="vocabulary words", ylab="partitions (topics)", main="Gibbs sampling - training set", axes=FALSE);fullaxis(1,lwd=1); fullaxis(2,lwd=1);



nt <- t(as.vector(read.table(file='num_topics')))
hist(nt, col="lightblue", border="gray")

