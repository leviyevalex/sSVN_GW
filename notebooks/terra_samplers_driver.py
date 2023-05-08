#%% Load libraries and setup class
import sys, os
sys.path.append("..")
from jax.config import config
from corner import corner

config.update("jax_enable_x64", True)

import pylab as plt
import tensorflow_probability.substrates.jax as tfp
from jax import random, numpy as jnp
from jax import vmap

import jaxns
from jaxns import ExactNestedSampler
from jaxns import Model
from jaxns import PriorModelGen, Prior
from jaxns import TerminationCondition
from jaxns import analytic_log_evidence
from jaxns import resample

from models.GWFAST_heterodyne import gwfast_class
import numpy as np

# gwclass = gwfast_class(eps=0.5, chi=1, mode='TaylorF2', freeze_indicies=np.array([2, 3, 4, 5, 6, 7, 8, 9, 10]))
gwclass = gwfast_class(eps=0.5, chi=1, mode='TaylorF2', freeze_indicies=np.array([]))

#%% Likelihood wrapper
tfpd = tfp.distributions
def log_likelihood(x):
    X = x[np.newaxis, ...]
    return -1 * gwclass.heterodyne_minusLogLikelihood(X)

def prior_model() -> PriorModelGen:
    x = yield Prior(tfpd.Uniform(low=gwclass.lower_bound, high=gwclass.upper_bound), name='x')
    return x

model = Model(prior_model=prior_model, log_likelihood=log_likelihood)

ns = exact_ns = ExactNestedSampler(model=model, num_live_points=1000, num_parallel_samplers=1, max_samples=1e4)

# termination_reason, state = exact_ns(random.PRNGKey(42), term_cond=TerminationCondition(live_evidence_frac=1e-4))
# termination_reason, state = exact_ns(random.PRNGKey(42), term_cond=TerminationCondition(live_evidence_frac=1e-4))
termination_reason, state = exact_ns(random.PRNGKey(42), term_cond=TerminationCondition(max_samples=1e5))

results = exact_ns.to_results(state, termination_reason)

#%% Get results
exact_ns.summary(results)
exact_ns.plot_diagnostics(results)
exact_ns.plot_cornerplot(results)

#%% Get direct access to samples
samples = resample(random.PRNGKey(43083245), results.samples, results.log_dp_mean, S=int(results.ESS))
corner(np.array(samples['x']), labels=list(np.array(gwclass.gwfast_param_order)[gwclass.active_indicies]), truths=gwclass.true_params[gwclass.active_indicies])

# %%
