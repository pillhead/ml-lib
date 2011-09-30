library(lda);
# require("ggplot2")

# wd = "F:\\My Dropbox\\datasets\\";
wd = "/home/clint/datasets/usa_tweets";


K = 10; ## Num clusters or topics 
itr = 5000; ## Num iterations
av = 1;
ev = 0.1;
bp = 4000;

setwd(wd)
documents = read.documents(filename = "20110320_usa_tweets_50K.documents");
vocab = read.vocab(filename = "20110320_usa_tweets_50K.vocab")

set.seed(8675309)

result = lda.collapsed.gibbs.sampler(documents, K, vocab, itr, alpha=av, eta=ev, burnin=bp, compute.log.likelihood=TRUE);

## Get the top 20 words in the cluster
top.words <- top.topic.words(result$topics, 10, by.score=TRUE);
top.words  


## Normalizes the topic sums 
n.t = result$topics/(rowSums(result$topics) + 1e-05);

rows = dim(n.t)[1];
cols = dim(n.t)[2];
threshold = quantile(as.vector(n.t), 0.9999); ## Takes the 99.99% quantile

for (i in 1:rows){
	s = sort(n.t[i,], decreasing = TRUE);
	s = s[s > threshold];
	print(i);
	print(s);
}



## Based on the score value 
n.t = result$topics/(rowSums(result$topics) + 1e-05);
scores = apply(n.t, 2, function(x) x * (log(x + 1e-05) - sum(log(x + 1e-05))/length(x)));

threshold = quantile(as.vector(scores), 0.9999)
rows = dim(scores)[1];
cols = dim(scores)[2];

for (i in 1:rows){
	s = sort(scores[i,], decreasing = TRUE);
	s = s[s > threshold];
	print(i);
	print(s)
}



# A list of length D. Each element of the list, say assignments[[i]] is an integer
# vector of the same length as the number of columns in documents[[i]]
# indicating the topic assignment for each word.
result$assignments[4]


# K X V matrix: each entry indicates the number of times a word (column) was
# assigned to a topic (row). The column names should correspond to the
# vocabulary words given in vocab.  
result$topics

# A length K  vector where each entry indicates the total number
# of times words were assigned to each topic.
result$topic_sums

# A K X D matrix where each entry is an integer indicating the number
# of times words in each document (column) were assigned to each topic (column).
result$document_sums


## the FULL LOG LIKELIHOOD (including the prior)
Y1 = result$log.likelihoods[1,];

## the LOG LIKELIHOOD of the observations
## conditioned on the assignments.
Y2 = result$log.likelihoods[2,];

X = 1:length(Y1);

plot(X, Y1, xlim=range(X), ylim=range(c(Y1, Y2)), type="l", col="blue", xlab="num. iterations", ylab="log.likelihood");
points(X, Y2, col="red", type="l");
# savePlot(filename = "log_likelihood", type = c("pdf"), device = dev.cur())


# Normalize theta 
cs <- colSums(result$document_sums);
theta.ol <- matrix(0, ncol=dim(result$document_sums)[1], nrow=dim(result$document_sums)[2]);
for (i in 1:dim(result$document_sums)[2]){ theta.ol[i, ] <- result$document_sums[, i] / cs[i]; }

mask <- apply(theta.ol, 2, is.na);
theta.ol[mask] <- 0.1;
write.table(theta.ol, file = "20110320_usa_tweets_50K.theta", row.names = FALSE, col.name=FALSE);

