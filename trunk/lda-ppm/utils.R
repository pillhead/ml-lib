GEM <- function(b) {beta(1, b);}

SB <- function(N, a, b) 
{
	theta.i <- beta(a, b); 
	
	for (j in 1:N-1){ theta.i <- theta.i * (1 - beta(a, b)); }
	
	theta.i;
}


StickBreaking <- function(K, a, b)
{
	theta <- as.vector(matrix(0, nrow=K, ncol=1));
	
	for (i in 1:K) { theta[i] <- SB(i, a, b); } 
 	
	theta; 	
}


GEM2 <- function(K, a, b)
{
	theta <- as.vector(matrix(0, nrow=K, ncol=1));
	
	for (i in 1:K) { theta[i] <- SB(i, a * b, (1-a) * b); } 
 	
	theta; 	
}


HellingerDistance <- function (d1, d2)
{
	# This function calculates the hellinger distance 
	# between two given vectors    

	# Normalizes the values 
	d1 <- d1 / sum(d1);
	d2 <- d2 / sum(d2);


	# Return value 
	0.5 * sum((sqrt(d1) - sqrt(d2))^2);

}

CosineDistance <- function (x, y)
{
	# This function calculates the cosine distance 
	# between two given vectors  

	# Normalizes the values 
	# x <- x / sum(x);
	# y <- y / sum(y);

	# Return value 
	x %*% y / sqrt(x%*%x * y%*%y)
	
}


KLDiv <- function (d1, d2)
{
	# This function calculates KL-divergence 
	# between two given vectors 

	# Normalizes the values 
	d1 <- (d1 / sum(d1)) + 0.0000000000000001;
	d2 <- (d2 / sum(d2)) + 0.0000000000000001; # to avoid divide by zero problem 
	
	# Return value 
	sum(d1 * log(d1 / d2));
}

repmat <- function(a,n,m) {kronecker(matrix(1,n,m),a)};

normalize <- function (XX, dim=1)
{
	## Normalizes the given matrix XX 
	# 
	# Inputs: 
	#	XX	- data 
	#	dim - normalizing dimension 
	
	
	if (dim == 1){
		cs <- colSums(XX);
		on <- array(1, c(1, dim(XX)[1]));
		Y <- XX / t(cs %*% on);
	}
	else {
		rs <- rowSums(XX);
		on <- array(1, c(1, dim(XX)[2]));
		Y <- XX / (rs %*% on);
	}

	return(Y);

}


normalize.theta <- function(XY)
{
	## Normalizes the value 
	cols = dim(XY)[2];
	cs = colSums(XY);
	Y = XY;  


	for (j in 1:cols)
	  Y[,j] <- Y[,j] / cs[j]; 
	

	## setting the NaN to zero 
	mask = apply(Y, 2, is.nan);
	Y[mask] = 0;
	
	Y 
}

plot.theta <- function(theta.f, normalize=FALSE)
{

  if (normalize) { Y <- normalize(theta.f, 1); }
  else { Y <- theta.f;}

  X = 1:dim(Y)[2];

  cl = c("black", "blue", "red", "yellow", "green", "chocolate4", "darkorchid", "darkorange", "darkmagenta", "cyan");
  par(mfrow = c(4,3))
  
  plot(X, Y[1,], xlim=range(X), ylim=range(Y), type="l", col=cl[1], xlab="document index", ylab="topic 1", lwd=3);
  for (i in 2:dim(theta.f)[1])
	  plot(X, Y[i,], col=cl[i], type="l", lwd=3, xlab="document index", ylab=paste("topic", i, sep=" "));

  plot(X, Y[1,], xlim=range(X), ylim=range(Y), type="l", col=cl[1], xlab="document index", ylab="topics", lwd=2, main='all topics');
  for (i in 2:dim(theta.f)[1])
	  lines(X, Y[i,], col=cl[i], type="l", lwd=2);

}


## Function to display the topic proportions for 
## each document in a corpus  
## Supports MAX 10 topics 


plot.doc.topics <- function(XY){

  cl = c("black", "blue", "red", "yellow", "green", 
		"chocolate4", "darkorchid", "darkorange", "darkmagenta", "cyan");

	Y <- NormalizeTopics(XY);

	## TODO: Need to make it a dynamic one 
	X = 1:dim(Y)[2];

	plot(X, Y[1,], 
		xlim=range(X), 
		ylim=range(Y), 
		type="l", 
		col=cl[1], 
		xlab="documents", 
		ylab="class assignments");
	for (i in 2:dim(XY)[1])
		points(X, Y[i,], col=cl[i], type="l");
		
}


NormalizeTopics <- function(XY)
{
	## Normalizes the value 
	cols = dim(XY)[2];
	rows = dim(XY)[1];
	cs = colSums(XY);
	Y = XY;  


	for (i in 1:rows){
		for (j in 1:cols){
			Y[i,j] = Y[i,j]/cs[j];
		}
	}

	## setting the NaN to zero 
	mask = apply(Y, 2, is.nan);
	Y[mask] = 0;
	
	Y 
}