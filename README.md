# bayesian-modeling

We are using the `scallop` dataset (referencing http://matt-wand.utsacademics.info/webspr/scallop.html) to perform bayesian modeling. 

This dataset contains 148 rows and 3 columns, concerning scallop abundance in Long Island, New York. The columns it contains are the following:

`latitude`, degrees latitude (north of the Equator);

`longitude`, degrees longitude (west of Greenwich);

`tot.catch`, size of scallop catch at location specified by `latitude` and `longitude`.

## Posterior Predictive Distribution

In this section, I perform Gaussian Process modeling on all `scallop` data. After obtaining the analytical form of the predictive distribution, 
I sample from it and obtain the predictive count of scallop in the given region (i.e. the given range of latitude and longitude).

## Parameter Fitting

In this section, I fit linear regression parameters (i.e. coefficients) using Bayesian method. After using MCMC to simulate from posterior distribution of 
parameters, I compute the predictions using the mean of simulated parameters. The model checking part comes as a justification of the linear regression model.
Finally I compute the RMSE of the linear regression.

## Predective Posterior Curve Visualizatioin

In this section, I try two sets of prior distribution over kernel parameters. By fixing one dimension (let's say latitude), I generate the predictive posterior distribution of log(tot.catch) across the range of the other dimension (let's say longitude). In this way, we could visualize how the Gaussian Process looks like.

Prior set 1:

- sigma_f ~ HalfNormal(1)

- l ~ InverseGamma(5,5)

- sigma_n ~ HalfNormal(1)


Prior set 2:

- sigma_f ~ HalfCauchyl(2)

- l ~ HalfCauchy(2)

- sigma_n ~ HalfNormal(1)
