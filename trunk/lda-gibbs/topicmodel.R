
T = 10 # number of topics

setwd('/home/clint/Dropbox/topicmodel');
setwd('F:\\My Dropbox\\topicmodel\\')

word.data <- read.table('docword.txt', skip=3, col.names=c('docid', 'termid', 'count'), header=F);
meta.data <- read.table('docword.txt', nrows=3, header=F);
N <- sum(word.data[, 3]); # number of words in corpus
D <- meta.data[1, ]; # number of docs
W <- meta.data[2, ]; # number of unique words

w <- matrix(0, 1, N);
d <- matrix(0, 1, N);

count <- 1; 

for (i in 1:dim(word.data)[1])
{
  ct <- word.data[i,3]; 
  w[1,count:(count+ct-1)] <- word.data[i, 2];
  d[1,count:(count+ct-1)] <- word.data[i, 1];
  count <- count + ct; 
}

Nwt <- matrix(0, W, T);
Ndt <- matrix(0, D, T);
Nt <- matrix(0, 1, T);

alpha <- 0.05 * N / (D * T);
beta <- 0.01;


# Initialization 

z <- sample(1:T, N, replace=T) 

for (i in 1:N){
  Nwt[w[i], z[i]] <- Nwt[w[i], z[i]] + 1;
  Ndt[d[i], z[i]] <- Ndt[d[i], z[i]] + 1;
  Nt[z[i]] <- Nt[z[i]] + 1;
}

order <- sample(1:N, N, replace=F) # similar to randperm 

# Smoothing 

Nwt <- Nwt + alpha; 
Ndt <- Ndt + alpha; 
Nt <- Nt + alpha; 




