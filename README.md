# yet-another-poisfac
pytorch implementation of poisson factorization

## Model and purposes

This python3 package build a model for a count matrix `X`.  The model family is:

```
U_{ck} ~ Gamma(rho,kappa_c)
alpha_{tk} ~ Gamma(rho,lambda_t)
Z_{ctk} ~Poisson (U_{ck} alpha_{tk} )
X_{ct} = \sum_{k} Z_{ctk}
```

This code estimates a mean field variational posterior for `U,Z`, as well as maximum likelihood estimates for `kappa`,`lambda`.  The variational family is 

```
U_{ck} ~ Gamma(sU_{ck},rU_{ck})
alpha_{tk} ~ Gamma(salpha_{ck},ralpha_{ck})
Z_{ctk} ~ Multinomial( varphi_{ctk}, X_{ct})
```

Note that the variational parameters for Z are never held in memory (too big!).  So, for example, to perform an update for `U_{ck}` we use the current parameters to calculate the optimal variational parameters for `varphi_{ctk}` for all `t`,`k` (in fact only necessary for `t`,`k` where `X_{ct}` is nonzero) and then use that to get an update for `U_{ckl}`.  

## Installation

Dependencies are:`scipy`,`sklearn`,`torch`,`tqdm`.  Just git clone this into your working directory and import it.  If you really want to install this package you're probably doing it wrong.  This code is just a singly pure-python file.  By the time you find you want to install this as a package, you should probably just copy and paste my code into whatever else you're doing.  Chances are you'll want to tweak it to your purposes anyway.

## Usage:

```
import fireyfish
Nfactors=20
model=fireyfish.PoisModel(countmatrix,Nfactors,batchsize=100000,rho=1)
model.initialize()
model.double() # (optionally) do it in double precision
model.cuda()   # (optionally) do it on the GPU
trainer=Trainer(model)
trainer.go(500) # train for 500 iterations
parameters= model.save_params(parms)
```

The learned parameters will be 

* `parameters.sU` <-- shape of posterior gamma for row loadings
* `parameters.rU` <-- rate of posterior gamma for row loadings
* `parameters.kappas` <-- inverse capacity for each row
* `parameters.salpha` <-- shape of posterior gamma for column loadings
* `parameters.ralpha` <-- rate of posterior gamma for column loadings
* `parameters.lambdas` <-- inverse capacity for each column

## Inspecting progress 

You can look at the progress of the elbo over training by looking at the value of the elbo at each step.  This is also broken down by the KL terms for the gammas and the data terms.  

```
dataterms = [x.Z for x in trainer.elbos]
kls = [(x.U_kls,x.alpha_kls) for x in trainer.elbos]
elbos = [x.total for x in trainer.elbos] # <-- elbo.total = elbo.Z - elbo.U_kls - elbo.alpha_kls

import matplotlib as plt
plt.plot(elbos)
```

Thie code seems to generally be fairly threadsafe, insofar as you can launch trainer.go in a separate thread and run `plt.plot([x.total for x in trainer.elbos])` from time to time to check progress.  A nice pattern for jupyter notebooks is to use IPython.lib.jobs.  

```
if 'jobs' not in locals():
  jobs=IPython.lib.backgroundjobs.BackgroundJobManager()
print('flushing')
if 'trainer' in locals():
    trainer.KEEPGOING=False
    while len(jobs.running)>0:
        time.sleep(.05)
    jobs.flush()
print('here we go')
jobs.new(trainer.go,500)
```

