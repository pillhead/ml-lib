## This script is to analyze results from 
## HDP-LDA and LDA topic search 



# To set the source dir 
setwd('/home/clint/Dropbox/ldappm/src') 

# Loads the necessary R pkgs 
library(MCMCpack);
library(plotrix);

# Includes the source files  
source('process_data.R');
source('utils.R');
source('top.topic.documents.R');
source('top.topic.words.R');




setwd('/home/clint/Dropbox/TREC/batch/1/')
tweets.vocab <- readLines("1.vocabulary");


# LDA product partition model 

train.gibbs.beta <- read.table(file = "/home/clint/Dropbox/TREC/batch1_lda_beta_samples_mean.dat");
train.gibbs.theta <- read.table(file = "/home/clint/Dropbox/TREC/batch1_lda_theta_samples_mean.dat");
colnames(train.gibbs.beta) <- tweets.vocab;
top.topic.words(train.gibbs.beta, num.words=20, by.score=TRUE); # displays top topics

# HDP LDA 

train.hdp.beta <- read.table(file = "1.supertweets_beta");
train.hdp.wa <- read.table(file = "1.supertweets_word_assignments");


## Based on mean 
row.sums <- rowSums(train.hdp.beta);
valid.idx <- (row.sums > (mean(row.sums) * 0.1)); #  n^th percentile
train.hdp.valid.beta <- train.hdp.beta[valid.idx, ];
colnames(train.hdp.valid.beta) <- tweets.vocab;
ttw <- top.topic.words(train.hdp.valid.beta, num.words=10, by.score=TRUE); # displays top topics
write.table(ttw, file='topic_words.txt')


train.hdp.log <- read.table(file = "1.supertweets_state_log", header=T);
y <- train.hdp.log[,3];
x <- 1:length(y);
plot(x,y, xlab="Number of iterations", ylab="Number of topics", main="Number of topics vs iterations", pch=15, col="blue")
hist(y, xlab="Number of topics", ylab="Frequencies", main="topics histogram", col="lightblue", border="gray")


## Based on entropy 

eps <- 1e-10;
beta <- as.matrix(train.hdp.beta);
row.sums <- rowSums(beta);
vocab.size <- dim(beta)[2]; 
num.topics <- dim(beta)[1];

norm.beta <- matrix(0, ncol=vocab.size, nrow=num.topics)

for (k in 1:num.topics)
{
  norm.beta[,k] <- beta[,k] / row.sums[k]; 
}


topic.entropies <- -rowSums((norm.beta + eps) * log(norm.beta + eps))
par(mfrow = c(2,1))
plot(1:num.topics,topic.entropies, xlab="Number of topics", ylab="Entropy", main="Topic entropy", pch=15, col="blue")
plot(1:num.topics,log(row.sums), xlab="Number of topics", ylab="Word counts", main="Topic words counts", pch=15, col="blue")


## Based on percentile 


filter.topics("/home/clint/Dropbox/TREC/batch1_hdp_final-topics.dat", 
  "/home/clint/Dropbox/TREC/batch1_vocabulary.txt", 
  '/home/clint/Dropbox/TREC/batch1_hdp_topic_words.txt', 0.15, 40, TRUE)


filter.topics <- function(beta.file, vocab.file, output.file, percentile=0.10, num.topic.words=10, use.score=TRUE)
{
  vocab <- read.vocab(file = vocab.file);
  beta <- read.table(file = beta.file);
  
  row.sums <- rowSums(beta);
  valid.idx <- (row.sums > quantile(row.sums, c(percentile))); #  n^th percentile
  valid.beta <- beta[valid.idx, ];
  colnames(valid.beta) <- vocab;
  ttw <- top.topic.words(valid.beta, num.words=num.topic.words, by.score=use.score); # displays top topics
  
  write.table(t(ttw), file = output.file, col.names = FALSE, quote = FALSE)  
}







