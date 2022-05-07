library(dplyr)
library(rjags)

# ---- data & setup ----

# load data & get working subset
hearts = read.csv("/Users/drew/Desktop/OneDrive - University of Missouri/sp 22/STAT 4640/final project/heart_2020_cleaned.csv")
hearts = hearts[1:2000,]

# HeartDisease as response, all else as covariates
Y = as.numeric(as.factor(hearts[,1]))-1
bmi = hearts[,2] # continuous
smoking = as.factor(hearts[,3]) # T/F
alcoholdrinking = as.factor(hearts[,4]) # T/F
stroke = as.factor(hearts[,5]) # T/F
physicalhealth = hearts[,6] # 0 thru 30
mentalhealth = hearts[,7] # 0 thru 30
diffwalking = as.factor(hearts[,8]) # T/F
sex = as.factor(hearts[,9]) # Male/Female
agecategory = as.factor(hearts[,10]) # 14 levels
race = as.factor(hearts[,11]) # ?
diabetic = as.factor(hearts[,12]) # T/F/other?
physicalactivity = as.factor(hearts[,13]) # T/F
genhealth = as.factor(hearts[,14]) # ? levels
sleeptime = hearts[,15] # 0 thru 24
asthma = as.factor(hearts[,16]) # T/F
kidneydisease = as.factor(hearts[,17]) # T/F
skincancer = as.factor(hearts[,18]) # T/F
X = cbind(bmi,smoking,alcoholdrinking,stroke,physicalhealth,mentalhealth,diffwalking,sex,agecategory,race,diabetic,physicalactivity,genhealth,sleeptime,asthma,kidneydisease,skincancer)
names = colnames(hearts[,-1])

# remove missing
junk = is.na(rowSums(X))
Y = Y[!junk]
X = X[!junk,]

# standardize since we're doing penalized regression
X.scale = as.matrix(scale(X))

# train & test sets
train = sample(1:nrow(hearts),0.7*nrow(hearts))
test = (1:nrow(hearts))[-train]
Y.train = Y[train]
Y.test = Y[test]
X.train = X[train,]
X.test = X[test,]

# data into JAGS format
n = length(Y.train)
p = ncol(X.train)
data = list(Y=Y.train,X=X.train,n=n,p=p)
params = c("alpha","beta","taue","taub")
burn = 1000
n.iter = 20000
thin = 1
n.chains = 2

# ---- models ----

# standard
model_string = textConnection("model{
  # Likelihood
  for(i in 1:n) {
    Y[i] ~ dbern(q[i])
    logit(q[i]) = alpha+inprod(X[i,],beta[])
  }
  # Priors
  for(j in 1:p) {
    beta[j] ~ dnorm(0,taue)
  }
  alpha ~ dnorm(0,0.001)
  taue ~ dgamma(0.1,0.1)
  taub ~ dgamma(0.1,0.1)
}")

# ridge
ridge.model_string = textConnection("model{
  # Likelihood
  for(i in 1:n) {
    Y[i] ~ dbern(q[i])
    logit(q[i]) = alpha+inprod(X[i,],beta[])
  }
  # Priors
  for(j in 1:p) {
    beta[j] ~ dnorm(0,taue*taub)
  }
  alpha ~ dnorm(0,0.001)
  taue ~ dgamma(0.1,0.1)
  taub ~ dgamma(0.1,0.1)
}")

# lasso
lasso.model_string = textConnection("model{
  # Likelihood
  for(i in 1:n) {
    Y[i] ~ dbern(q[i])
    logit(q[i]) = alpha+inprod(X[i,],beta[])
  }
  # Priors
  for(j in 1:p) {
    beta[j] ~ ddexp(0,taue*taub)
  }
  alpha ~ dnorm(0,0.001)
  taue ~ dgamma(0.1,0.1)
  taub ~ dgamma(0.1,0.1)
}")

# compile model(s)
model = jags.model(model_string, data=data, n.chains=n.chains, quiet=F)
ridge.model = jags.model(ridge.model_string, data=data, n.chains=n.chains, quiet=F)
lasso.model = jags.model(lasso.model_string, data=data, n.chains=n.chains, quiet=F)

# burn-in
update(model,burn)
update(ridge.model,burn)
update(lasso.model,burn)

# generate posterior samples
samples = coda.samples(model, variable.names=params, n.iter=n.iter, thin=thin)
ridge.samples = coda.samples(ridge.model, variable.names=params, n.iter=n.iter, thin=thin)
lasso.samples = coda.samples(lasso.model, variable.names=params, n.iter=n.iter, thin=thin)

# ---- convergence diagnostics ----

autocorr.plot(samples[[1]])
autocorr.plot(ridge.samples[[1]])
autocorr.plot(lasso.samples[[1]])

effectiveSize(samples[[1]])
effectiveSize(ridge.samples[[1]])
effectiveSize(lasso.samples[[1]])

geweke.diag(samples[[1]],frac1=0.2,frac2=0.2)
geweke.diag(ridge.samples[[1]],frac1=0.2,frac2=0.2)
geweke.diag(lasso.samples[[1]],frac1=0.2,frac2=0.2)

gelman.diag(samples)
gelman.diag(ridge.samples)
gelman.diag(lasso.samples)

# ---- model comparison ----

# setup
ntest = length(Y.test)
Y.pred = matrix(nrow=nrow(samples[[1]]),ncol=ntest)
ridge.Y.pred = matrix(nrow=nrow(samples[[1]]),ncol=ntest)
lasso.Y.pred = matrix(nrow=nrow(samples[[1]]),ncol=ntest)

# post pred dist
for(s in 1:nrow(samples[[1]])) {
  Y.pred[s,] = exp(samples[[1]][s,1] + X.test %*% samples[[1]][s,2:18]) / (1 + exp(samples[[1]][s,1] + X.test %*% samples[[1]][s,2:18]))
  ridge.Y.pred[s,] = exp(ridge.samples[[1]][s,1] + X.test %*% ridge.samples[[1]][s,2:18]) / (1 + exp(ridge.samples[[1]][s,1] + X.test %*% ridge.samples[[1]][s,2:18]))
  lasso.Y.pred[s,] = exp(lasso.samples[[1]][s,1] + X.test %*% lasso.samples[[1]][s,2:18]) / (1 + exp(lasso.samples[[1]][s,1] + X.test %*% lasso.samples[[1]][s,2:18]))
}

# MSE
mse = 1/ntest * sum((Y.test - apply(Y.pred,2,mean))^2)
ridge.mse = 1/ntest * sum((Y.test - apply(ridge.Y.pred,2,mean))^2)
lasso.mse = 1/ntest * sum((Y.test - apply(lasso.Y.pred,2,mean))^2)

# DIC
dic = dic.samples(model,n.iter=n.iter)
ridge.dic = dic.samples(ridge.model,n.iter=n.iter)
lasso.dic = dic.samples(lasso.model,n.iter=n.iter)

# WIDTH
width = 1/ntest * sum(apply(Y.pred,2,quantile,c(.975))-apply(Y.pred,2,quantile,c(.025)))
ridge.width = 1/ntest * sum(apply(ridge.Y.pred,2,quantile,c(.975))-apply(ridge.Y.pred,2,quantile,c(.025)))
lasso.width = 1/ntest * sum(apply(lasso.Y.pred,2,quantile,c(.975))-apply(lasso.Y.pred,2,quantile,c(.025)))

# LS
ls = 1/ntest * sum(dbinom(Y.test, 1, mean(Y.pred)))
ridge.ls = 1/ntest * sum(dbinom(Y.test, 1, mean(ridge.Y.pred)))
lasso.ls = 1/ntest * sum(dbinom(Y.test, 1, mean(lasso.Y.pred)))

# ---- posterior inference ----

par(mfrow=c(4,2))
for(j in 2:18) {
  # collect samples from chains
  s1 = c(samples[[1]][,j],samples[[2]][,j])
  s2 = c(ridge.samples[[1]][,j],ridge.samples[[2]][,j])
  s3 = c(lasso.samples[[1]][,j],lasso.samples[[2]][,j])
  
  # smooth density est
  d1 = density(s1)
  d2 = density(s2)
  d3 = density(s3)
  
  # plot density est
  mx = max(c(d1$y, d2$y, d3$y))
  
  plot(d1$x,d1$y,type="l",ylim=c(0,mx),xlab=expression(beta),ylab="Posterior density",main=names[j-1])
  lines(d2$x,d2$y,lty=2)
  lines(d3$x,d3$y,lty=3)
  abline(v=0,col="red")
}
