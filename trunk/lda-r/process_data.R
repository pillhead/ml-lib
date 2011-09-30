StraightenCounts <- function(file.name, meta.lines, header=FALSE)
{
	## Straightens the counts so that we can have each record 
	## as an instance in a corpus document  	
	# 
	# Inputs: 
	#	file.name	- data file name 
	#	meta.lines	- lines that contains meta data 
 	#	header		- header exists or not 
	#	

	if (header){
		word.counts <- read.table(file.name, skip=meta.lines, header=T);
		meta.data <- read.table(file.name, nrows=meta.lines, header=T);
	}
	else 
	{
		word.counts <- read.table(file.name, skip=meta.lines, col.names=c('docid', 'termid', 'count'), header=F);
		meta.data <- read.table(file.name, nrows=meta.lines, header=F);
	}

	N <- sum(word.counts[, 3]); # number of words in a corpus	
	D <- meta.data[1, ]; # number of documents 
	W <- meta.data[2, ]; # number of unique words
	w <- array(0, c(1, N)); # word instances in a corpus 
	d <- array(0, c(1, N)); # document instances in a corpus 
	
	count <- 1; 
	
	# Straighten the counts 
	
	for (i in 1:dim(word.counts)[1])
	{
		ct <- word.counts[i, 3]; 
		w[count:(count + ct - 1)] <- word.counts[i, 2];
		d[count:(count + ct - 1)] <- word.counts[i, 1];
		count <- count + ct; 
	}
	
	list(total.N=N, D=D, V=W, wid=w, did=d);
	
}


ReadCounts <- function(wid.file, did.file)
{
	## Reads the word and document instances from the given files 	
	# 
	# Inputs: 
	#	wid.file	- word instances file 
	#	did.file	- document instances file 
	#		
	
	wids <- read.table(wid.file, header=F);
	dids <- read.table(did.file, header=F);
	
	N <- length(wids$V1); # total number of words in a corpus	
	D <- length(unique(dids$V1)); # total number of documents in a corpus	
	W <- length(unique(wids$V1)); # number of unique words in a corpus	
	
	list(total.N=N, D=D, V=W, wid=as.vector(wids$V1), did=as.vector(dids$V1));
 
}

CalcPartitionCounts <- function(Z, K)
{
  ## Calculates the partition counts 
	# 
	# Inputs: 
 	#	Z		- topic selection for each word instance
	#	K		- total topics in a corpus 
	# 
	
	Nt <- array(0, c(1, K)); 
	for (k in 1:K) Nt[k] <- sum(Z == k);

	return(Nt); 	
}

     
LDASamples <- function(K, D, V, alpha, lamda.hat, beta)
{

	## Generate words using the LDA generative process 
	# 
	# Inputs: 
	#   K - number of topics 
	#   D - number of documents 
	#   V - total number of unique words in a corpus 
	#   lamda.hat - mean of the Poisson dist. 
	#   beta - beta matrix for topic word probabilities 
	# 

	doc.N <- rpois(D, lamda.hat); 
	alpha.v <- array(1, c(1, K)) * alpha; 
	theta.counts <- matrix(0, nrow=K, ncol=D);
  theta <- matrix(0, nrow=K, ncol=D);
	did <- c();
	wid <- c(); 
	zid <- c();

	for (d in 1:D)
	{
		theta[, d] <- rdirichlet(1, alpha.v);
		did <- cbind(did, array(1, c(1, doc.N[d])) * d); # document instances 
		z_d <- c(); 
		for (i in 1:doc.N[d]){
			z_dn <- which(rmultinom(1, size=1, prob=theta[, d]) == 1);
			wid <- cbind(wid, which(rmultinom(1, size=1, beta[z_dn,]) == 1)); # word instances 
			z_d <- cbind(z_d, z_dn); 
		}
		# calculates the topic mixtures 
		Nt <- CalcPartitionCounts(z_d, K); 
		theta.counts[, d] <- Nt / sum(Nt);
		
		zid <- cbind(zid, z_d); 
	}

	list(K=K, D=D, V=V, did=as.vector(did), wid=as.vector(wid), zid=as.vector(zid), total.N=sum(doc.N), theta.counts=theta.counts, theta=theta);

}   

