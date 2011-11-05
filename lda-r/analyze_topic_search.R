## 
## This is a test script to compare the results from topic 
## search with the original and Gibbs sampler theta and 
## Beta 
##
## Created On : May 24, 2011
## Created By : Clint P. George 
##

# To set the working dir 

setwd("~/Dropbox/lda-r")
# setwd("F:/My Dropbox/lda-r")

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



## 10/27/2011


## ON THE TRAINING SET 

sbeta <- t(read.table('../lda-data/synth/synth_beta.txt', header=F)); # Synthetic beta 
stheta <- read.table(file = "../lda-data/synth/200d400w.theta_counts")
stheta0 <- read.table(file = "../lda-data/synth/200d400w.theta")

train.gibbs.beta.counts <- read.table(file = "../lda-data/synth/200d400w_1000i_beta_counts_mean")
train.gibbs.theta.counts <- read.table(file = "../lda-data/synth/200d400w_1000i_theta_counts_mean")

train.gibbs.beta.counts.last <- read.table(file = "../lda-data/synth/200d400w_1000i_beta_counts_last")
train.gibbs.theta.counts.last <- read.table(file = "../lda-data/synth/200d400w_1000i_theta_counts_last")

train.ogibbs.beta.counts <- read.table(file = "../lda-data/synth/200d400w_1000i_Nwt_full")
train.ogibbs.theta.counts <- read.table(file = "../lda-data/synth/200d400w_1000i_Ndt_full")

train.gibbs.idx <- c(6, 2, 5, 1, 10, 7, 3, 9, 8, 4); # order of the topics found from the Gibbs sampler   
train.ogibbs.idx <- c(5, 10, 9, 3, 4, 6, 7, 8, 2, 1); # order of the topics found from the Gibbs sampler   

## Calculkates the KL-divergence between the synthetic theta and estimated document topic mixtures
kld.gibbs.synth <- calc.KL.divergence(train.gibbs.theta.counts[train.gibbs.idx,], stheta);
kld.ogibbs.synth <- calc.KL.divergence(train.ogibbs.theta.counts[train.ogibbs.idx,], stheta);
kld.gibbs.synth0 <- calc.KL.divergence(train.gibbs.theta.counts[train.gibbs.idx,], stheta0);
kld.ogibbs.synth0 <- calc.KL.divergence(train.ogibbs.theta.counts[train.ogibbs.idx,], stheta0);


## Display beta 

par(mfrow = c(2,2))
color2D.matplot(sbeta, c(1, 0), c(1, 0), c(1,0), , border=NA, xlab="vocabulary words", ylab="partitions (topics)", main="The true beta", axes=FALSE);fullaxis(1,lwd=1); fullaxis(2,lwd=1);
color2D.matplot(train.gibbs.beta.counts[train.gibbs.idx,], c(1, 0), c(1, 0), c(1,0), , border=NA, xlab="vocabulary words", ylab="partitions (topics)", main="full Gibbs (mean partition-word counts)", axes=FALSE);fullaxis(1,lwd=1); fullaxis(2,lwd=1);
color2D.matplot(train.gibbs.beta.counts.last[train.gibbs.idx,], c(1, 0), c(1, 0), c(1,0), , border=NA, xlab="vocabulary words", ylab="partitions (topics)", main="full Gibbs (last sample)", axes=FALSE);fullaxis(1,lwd=1); fullaxis(2,lwd=1);
color2D.matplot(train.ogibbs.beta.counts[train.ogibbs.idx,], c(1, 0), c(1, 0), c(1,0), , border=NA, xlab="vocabulary words", ylab="partitions (topics)", main="collapsed Gibbs (last sample)", axes=FALSE);fullaxis(1,lwd=1); fullaxis(2,lwd=1);


## Displays KL-divergences   
par(mfrow = c(2,2))
plot(1:length(kld.gibbs.synth), kld.gibbs.synth, ylim=c(0,1.5), type='o', col='blue', xlab='document indices', ylab='K-L (estimated w/ observed)', main='full Gibbs (batch)')
plot(1:length(kld.ogibbs.synth), kld.ogibbs.synth, ylim=c(0,1.5), type='o', col='blue', xlab='document indices', ylab='K-L (estimated w/ observed)', main='collapsed Gibbs (batch)')
plot(1:length(kld.gibbs.synth0), kld.gibbs.synth0, ylim=c(0,0.2), type='o', col='blue', xlab='document indices', ylab='K-L (estimated w/ sampling)', main='full Gibbs (batch)')
plot(1:length(kld.ogibbs.synth0), kld.ogibbs.synth0, ylim=c(0,0.2), type='o', col='blue', xlab='document indices', ylab='K-L (estimated w/ sampling)', main='collapsed Gibbs (batch)')

## due to variance 

## ON THE TEST SET 

test.stheta <- read.table(file = "../lda-data/synth/50d400w.theta_counts")
test.stheta0 <- read.table(file = "../lda-data/synth/50d400w.theta")

test.gibbs.beta.counts <- read.table(file = "../lda-data/synth/50d400w_1000i_ob_beta_counts_mean"); 
test.gibbs.theta.counts <- read.table(file = "../lda-data/synth/50d400w_1000i_ob_theta_counts_mean")
test.oi.beta.counts <- read.table(file = "../lda-data/synth/50d400w_1000i_oi_beta_samples_mean"); 
test.oi.theta.counts <- read.table(file = "../lda-data/synth/50d400w_1000i_oi_theta_samples_mean")

test.ogibbs.beta.counts <- read.table(file = "../lda-data/synth/50d400w_1000i_ob_Nwt_full");
test.ogibbs.theta.counts <- read.table(file = "../lda-data/synth/50d400w_1000i_ob_Ndt_full")
test.ooi.beta.counts <- read.table(file = "../lda-data/synth/50d400w_1000i_oi_Nwt_full");
test.ooi.theta.counts <- read.table(file = "../lda-data/synth/50d400w_1000i_oi_Ndt_full")

test.ts.theta.counts <- read.table(file = "../lda-data/synth/50d400w_ts_theta_counts")
test.ts.theta.counts2 <- read.table(file = "../lda-data/synth/50d400w_ts_theta_counts_last")

test.ts.beta.counts <- read.table(file = "../lda-data/synth/50d400w_ts_beta_counts")
test.ts.beta.counts2 <- read.table(file = "../lda-data/synth/50d400w_ts_beta_counts_last")


## Calculkates the KL-divergence between the synthetic theta and estimated document topic mixtures
kld.test.gibbs.synth <- calc.KL.divergence(test.gibbs.theta.counts[train.gibbs.idx,], test.stheta0);
kld.test.ogibbs.synth <- calc.KL.divergence(test.ogibbs.theta.counts[train.ogibbs.idx,], test.stheta0);

kld.test.oi.synth <- calc.KL.divergence(test.oi.theta.counts[train.gibbs.idx,], test.stheta0);
kld.test.ooi.synth <- calc.KL.divergence(test.ooi.theta.counts[train.ogibbs.idx,], test.stheta0);

kld.test.ts.synth <- calc.KL.divergence(test.ts.theta.counts[train.gibbs.idx,], test.stheta0);
kld.test.ts.synth2 <- calc.KL.divergence(test.ts.theta.counts2[train.gibbs.idx,], test.stheta0);

sum(kld.test.gibbs.synth > 0.6) / 50
sum(kld.test.ogibbs.synth > 0.6) / 50

sum(kld.test.oi.synth > 0.6) / 50
sum(kld.test.ooi.synth > 0.6) / 50

sum(kld.test.ts.synth > 0.6) / 50
sum(kld.test.ts.synth2 > 0.6) / 50


## Display beta 
par(mfrow = c(4,2))
color2D.matplot(sbeta, c(1, 0), c(1, 0), c(1,0), , border=NA, xlab="vocabulary words", ylab="partitions (topics)", main="The true beta", axes=FALSE);fullaxis(1,lwd=1); fullaxis(2,lwd=1);
color2D.matplot(test.gibbs.beta.counts[train.gibbs.idx,], c(1, 0), c(1, 0), c(1,0), , border=NA, xlab="vocabulary words", ylab="partitions (topics)", main="full Gibbs (mean)", axes=FALSE);fullaxis(1,lwd=1); fullaxis(2,lwd=1);
color2D.matplot(test.ogibbs.beta.counts[train.ogibbs.idx,], c(1, 0), c(1, 0), c(1,0), , border=NA, xlab="vocabulary words", ylab="partitions (topics)", main="collapsed Gibbs (last)", axes=FALSE);fullaxis(1,lwd=1); fullaxis(2,lwd=1);
color2D.matplot(test.oi.beta.counts[train.gibbs.idx,], c(1, 0), c(1, 0), c(1,0), , border=NA, xlab="vocabulary words", ylab="partitions (topics)", main="full Gibbs (mean) - incremental", axes=FALSE);fullaxis(1,lwd=1); fullaxis(2,lwd=1);
color2D.matplot(test.ooi.beta.counts[train.ogibbs.idx,], c(1, 0), c(1, 0), c(1,0), , border=NA, xlab="vocabulary words", ylab="partitions (topics)", main="collapsed Gibbs (last) - incremental", axes=FALSE);fullaxis(1,lwd=1); fullaxis(2,lwd=1);
color2D.matplot(test.ts.beta.counts[train.gibbs.idx,], c(1, 0), c(1, 0), c(1,0), , border=NA, xlab="vocabulary words", ylab="partitions (topics)", main="topic search (samples mean)", axes=FALSE);fullaxis(1,lwd=1); fullaxis(2,lwd=1);
color2D.matplot(test.ts.beta.counts2[train.gibbs.idx,], c(1, 0), c(1, 0), c(1,0), , border=NA, xlab="vocabulary words", ylab="partitions (topics)", main="topic search (last sample)", axes=FALSE);fullaxis(1,lwd=1); fullaxis(2,lwd=1);

## Displays KL-divergences   
par(mfrow = c(3,2))
plot(1:length(kld.test.gibbs.synth), kld.test.gibbs.synth, ylim=c(0.,0.8), type='o', col='blue', xlab='document indices', ylab='K-L divergence', main='full Gibbs (batch)')
plot(1:length(kld.test.ogibbs.synth), kld.test.ogibbs.synth, ylim=c(0.,0.8), type='o', col='blue', xlab='document indices', ylab='K-L divergence', main='collapsed Gibbs (batch)')

plot(1:length(kld.test.oi.synth), kld.test.oi.synth, ylim=c(0.,0.8), type='o', col='blue', xlab='document indices', ylab='K-L divergence', main='full Gibbs (incremental)')
plot(1:length(kld.test.ooi.synth), kld.test.ooi.synth, ylim=c(0.,0.8), type='o', col='blue', xlab='document indices', ylab='K-L divergence', main='collapsed Gibbs (incremental)')

plot(1:length(kld.test.ts.synth), kld.test.ts.synth, ylim=c(0.,0.8), type='o', col='blue', xlab='document indices', ylab='K-L divergence', main='topic search (mean)')
plot(1:length(kld.test.ts.synth2), kld.test.ts.synth2, ylim=c(0.,0.8), type='o', col='blue', xlab='document indices', ylab='K-L divergence', main='topic search (last sample)')



## USING MH SEARCH 





## 10/27/2011


kld.ts.hrw.gen <- calc.KL.divergence(ts.hrw.theta[train.gibbs.idx, ], stheta);
kld.bg.gen <- calc.KL.divergence(test.bg.theta[train.bg.idx, ], stheta);
kld.ig.gen <- calc.KL.divergence(test.ig.theta[train.gibbs.idx, ], stheta);

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

