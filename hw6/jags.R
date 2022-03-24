library(rjags)

# specify model via model string
n=nrow( cars )
dataCars=list(dist=cars$dist ,speed=cars$speed ,n=n)
model_string = textConnection("model{
  for(i in 1:n){
    dist[i] ~ dnorm(beta0 + beta1*speed[i], tau)
  }
  tau ~ dgamma(0.1, 0.1)
  sigma2=1/tau
  beta0 ~ dnorm(0, 0.001)
  beta1 ~ dnorm(0, 0.001)
}")

# initial values
inits = list(beta0=0, beta1=1, tau=1) # set ourselves
inits = list(beta0=rnorm(1), beta1=rnorm(1),tau=rgamma(1,1)) # random

# compile model
model = jags.model(model_string, data=dataCars, inits=inits, n.chains=1)

# burn-in
update(model,1000,progress.bar="none")

# generate posterior samples
params = c("beta0", "beta1", "sigma2")
samples = coda.samples(model, variable.names=params, n.iter=10000, progress.bar="none")

# summary
summary(samples)
plot(samples)