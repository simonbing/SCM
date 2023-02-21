"""
2022 Simon Bing, TU Berlin, DLR
"""
from abc import ABCMeta, abstractmethod
from collections.abc import Iterable
import copy
import inspect
import logging

import numpy as np
import numpy.random
import scipy
from scipy.stats import uniform

from . import check_mechanism_signature, make_iterable


class CausalVar(object):
    """
    Causal variable class. Used to define SCMs.
    ___

    Attributes:
        parents [list[CausalVar] or None]:
            parents of a variable
        noise_distr [scipy.stats.rv_continuous]:
            probability distribution of a variable's random noise
        mechanism_fct [function]:
            function with (n_parents + 1) arguments defining dependencies of
            variable on its parents and noise
        intervention_target [bool]:
            flag indicating whether a variable is the target of an Intervention
        noise_value [float]:
            value of noise taken during sampling
        value [float]:
            value of variable taken during sampling
        seed: int
            Random seed. Set to value other than None to obtain reproducible
            samples.
        rs: RandomState
            numpy RandomState to sample from.
    """
    def __init__(self, parents, noise_distr, mechanism_fct, seed=None):
        assert ((isinstance(parents, Iterable) and
                 all(isinstance(parent, CausalVar) for parent in parents)) or
                parents is None), "Parents of a variable must be list of CausalVar objects, or None!"
        self.parents = parents

        assert isinstance(noise_distr,
                          scipy.stats._distn_infrastructure.rv_continuous_frozen), (
            "noise_distr must be a scipy rv_continuous distribution!")
        self.noise_distr = noise_distr

        assert inspect.isfunction(mechanism_fct), (
            "mechanism_fct of a causal variable must be a function!")
        assert check_mechanism_signature(mechanism_fct, parents), (
            "mechanism_fct must take (n_parents + 1) arguments!")
        self.mechanism = mechanism_fct

        # Indicates whether variable is target of an Intervention,
        # used during sampling.
        self.intervention_target = False

        # Assigned during sampling
        self.noise_value = None
        self.value = None

        # Random seed for reproducibility
        self.seed = seed
        self.rs = numpy.random.RandomState(seed=self.seed)


class Intervention(object):
    """
    Intervention class used to intervene on SCMs.

    This class is used to represent all different types of interventions.
    Depending on the type of intervention, its arguments take different
    values. Atomic interventions, as well as interventions with multiple
    targets can be passed by specifying single arguments or lists. The
    supported types of interventions and their expected arguments are:

    Do-Intervention:
        new_mechanism: constant value
        new_parents: - (gets set to None)
        new_noise_distr: - (old noise_distr is kept)

    Noise (soft) intervention:
        new_mechanism: None (or leave empty, None is default)
        new_parents: - (old parents are kept)
        new_noise_distr: specified new distribution

    General interventions:
        new_mechanism: specificed new function
        new_parents: indices if new parents (could also be None, if mechanism
            only a function of noise!)
        new_noise_distr: None to keep old distribution, or specified new
            distribution
    ___

    Attributes:
        targets [int or list[int]]:
            indices of variables to target with intervention
        new_mechanism [int/float/function or list[int/float/function]]:
            new causal mechanisms
        new_parents [int or list[int/None] or list[list[int]/None]]:
            indices of new parents for each intervention
        new_noise_distr [scipy.stats.rv_continuous
                         or list[scipy.stats.rv_continuous]]:
            new noise distribution
    """
    def __init__(self, targets, new_mechanism=None, new_parents=None, new_noise_distr=None):
        # Expand default None values to correct lengths if we have more than one target
        if hasattr(targets, '__len__'):
            n_targets = len(targets)
            if new_mechanism is None:
                new_mechanism = [None] * n_targets
            if new_parents is None:
                new_parents = [None] * n_targets
            if new_noise_distr is None:
                new_noise_distr = [None] * n_targets

        # Make all arguments iterable (turn them to a list of len 1 if they are a single object)
        targets = make_iterable(targets)
        new_parents = make_iterable(new_parents)
        # Need to ensure that this is a list of lists/None
        if np.array(new_parents).ndim == 1:
            if len(targets) == len(new_parents):  # indices correspond to different interventions
                new_parents = [
                    make_iterable(parents) if parents is not None else parents
                    for parents in new_parents]
            else:  # indices represent new parents of a single Intervention
                new_parents = [new_parents]
        new_noise_distr = make_iterable(new_noise_distr)
        new_mechanism = make_iterable(new_mechanism)

        # Assertions to make sure the correct types are passed
        assert len(targets) == len(new_mechanism) == len(new_parents) == len(new_noise_distr), (
            "All input arguments must have the same length!")
        assert all(isinstance(target, int) for target in targets), "All targets must be ints!"
        assert all(parents is None or (all(isinstance(parent, int) for parent in parents)) for parents in new_parents), (
            "new_parents must be list of int list and/or None values!")
        assert all((isinstance(noise, scipy.stats._distn_infrastructure.rv_continuous_frozen) or noise is None)
                   for noise in new_noise_distr), (
            "new_noise_distr must either be a scipy rv_continuous distribution or None!")
        assert all(mech is None or isinstance(mech, (int, float)) or inspect.isfunction(mech) for mech in
                   new_mechanism), (
            "All new_mechanism must be either constant of type int or float, or callable functions!")

        self.targets = targets
        self.new_parents = []
        self.new_noise_distr = []
        self.new_mechanism = []

        def do_operator(val):
            # Last summand needed to return correct length
            return lambda x: val + np.zeros_like(x)

        # Outer loop over potentially multiple interventions
        for i, target in enumerate(targets):
            if new_mechanism[i] is not None:  # Either do-Intervention or mechanism Intervention
                if isinstance(new_mechanism[i],
                              (int, float)):  # do-Intervention
                    self.new_parents.append(None)
                    self.new_noise_distr.append(None)  # None just means the old noise_distr is kept
                    val = float(new_mechanism[i])
                    f_new = do_operator(val)
                    self.new_mechanism.append(f_new)
                elif inspect.isfunction(new_mechanism[i]):  # General mechanism Intervention
                    assert check_mechanism_signature(new_mechanism[i], new_parents[i]), (
                        F"new_mechanism at index {i} is not compatible with the number of new_parents!")
                    # Even if parents stay constant for new mechanism, must explicitly state them
                    self.new_parents.append(new_parents[i])
                    # If new_noise_distr is None the old noise_distr is kept
                    self.new_noise_distr.append(new_noise_distr[i])
                    self.new_mechanism.append(new_mechanism[i])
                else:
                    raise TypeError(F"new_mechanism at index {i} must either be a constant or a function!")
            else:  # Soft (noise) Intervention
                if new_noise_distr[i] is not None:
                    self.new_parents.append(-1)
                    self.new_noise_distr.append(new_noise_distr[i])
                    # If new_mechanism is None, keep the old one
                    self.new_mechanism.append(None)
                else:
                    raise TypeError(F"new_mechanism and new_noise_distr at index {i} cannot both be None!")


class SCM(object):
    """
    Abstract structural causal model (SCM) metaclass. SCMs are implemented as
    children of this base class.
    ___

    Attributes:
        variables [list[CausalVar]]:
            causal variables of the SCM
        intervention_flag [bool]:
            indicates whether or not this SCM has been intervened upon
    ___

    Methods:
        sample:
            draw a sample from the SCM's current distribution
        intervene:
            perform an intervention on the SCM
        intervent_sample:
            intervene and draw a sample from the interventional distribution
        counterfact_sample:
            intervene and draw a (paired) counterfactual sample from the
            original and intervened upon SCM
    """
    def __init__(self, variables):
        """
        Args:
            variables: list(CausalVar):
                Ordered list of the causal variables of the SCM.
        """
        self.variables = variables
        assert all(isinstance(var, CausalVar) for var in self.variables), (
            "All SCM variables must be CausalVar objects!")

        self.adj_matrix = self._get_adj_matrix()

        self.intervention_flag = False

    def _get_adj_matrix(self):
        """
        Returns:
            adj_matrix: array of size (num_vars, num_vars)
                The adjacency matrix of the SCM.
        """
        adj_matrix = np.zeros(shape=(len(self.variables), len(self.variables)),
                              dtype=np.int)

        for i, var in enumerate(self.variables):
            if var.parents is not None:
                parent_idxs = np.in1d(var.parents, self.variables).nonzero()[0]
                adj_matrix[i, parent_idxs] = 1

        return adj_matrix

    def sample(self,
               n: int,
               return_noise: bool = False,
               old_noises: list[float] = None) -> (list[float], list[float]):
        """
        Method used to draw samples from the current distribution of the SCM.
        ___

        Args:
            n [int]:
                number of samples to draw
            return_noise [bool]:
                flag to return noise values from sampling
            old_noises [list[float]]:
                noise values from previous sampling procedure, used for
                counterfactual sampling

        Returns:
            values: array:
                array of sampled values of causal variables,
                output is shape (n_samples, n_variables)
            noises array:
                array of sampled noise values, output is shape
                (n_samples, n_variables)
        """
        if self.intervention_flag:
            logging.warning(
                "The SCM from which you are sampling has been intervened upon!")

        noises = []
        values = []

        for i, var in enumerate(self.variables):
            if old_noises is not None:  # For counterfactual sampling
                if var.intervention_target:  # Resample noise if variable is an intervention target
                    noise = var.noise_distr.rvs(n, random_state=self.rs)
                else:
                    noise = old_noises[i]  # Take passed noise values for other variables
            else:  # Sample noise values
                noise = var.noise_distr.rvs(n, random_state=self.rs)

            # Get values of parents
            if var.parents is not None:
                value = var.mechanism(*[parent.value for parent in var.parents], noise)
            else:  # Variable is a root node, i.e. its mechanism only depends on noise
                value = var.mechanism(noise)

            var.noise_value = noise
            var.value = value
            noises.append(noise)
            values.append(value)

            # Reset intervention_target flag
            var.intervention_target = False

        # Transform to np arrays and reshape to (n_samples, n_variables)
        noises = np.array(noises).transpose()
        values = np.array(values).transpose()

        if not return_noise:
            return values
        else:
            return values, noises


    def intervene(self, iv: list[Intervention]):
        """
        Method to perform an intervention on the SCM.
        Only call this directly if you wish to permanently modify the structure
        of the SCM!
        ___

        Args:
            iv [list[Intervention]]:
                single or multiple interventions to perform
        """
        # Set Intervention flag
        self.intervention_flag = True

        # Loop over all interventions
        for target, new_parents, new_noise_distr, new_mechanism in zip(
                iv.targets, iv.new_parents, iv.new_noise_distr, iv.new_mechanism):
            # Set intervention_target flag for variable
            self.variables[target].intervention_target = True
            # Reset sampled values to None
            self.variables[target].noise_value = None
            self.variables[target].value = None
            # Set parents
            if new_parents == -1:  # Keep old parents
                pass
            elif new_parents is None:
                self.variables[target].parents = new_parents
            else:
                self.variables[target].parents = [self.variables[parent_idx] for
                                                  parent_idx in new_parents]
            # Set noise distribution
            if new_noise_distr is None:  # Keep old noise_distr
                pass
            else:
                self.variables[target].noise_distr = new_noise_distr
            # Set mechanism
            if new_mechanism is None:  # Keep old mechanism
                pass
            else:
                self.variables[target].mechanism = new_mechanism

    def intervent_sample(self,
                         iv: list[Intervention],
                         n: int,
                         return_noise: bool = False) -> (list[float], list[float]):
        """
        Method to perform an intervention and draw from the resulting
        distribution.
        Does not change the underlying SCM permanently!
        ___

        Args:
            iv [list[Intervention]]:
                single or multiple interventions to perform
            n [int]:
                number of samples to draw
            return_noise [bool]:
                flag to return noise values from sampling

        Returns:
            values: list[float]:
                array of sampled values of causal variables
            noises [list[float]]:
                array of sampled noise values
        """
        # Make copy of SCM, don't change internal state of original SCM
        SCM_copy = copy.deepcopy(self)
        # Intervene
        SCM_copy.intervene(iv)
        # Sample
        return SCM_copy.sample(n, return_noise)

    def counterfact_sample(self,
                           iv: list[Intervention],
                           n: int,
                           return_noise: bool = False) -> (list[float], list[float]):
        """
        Method to perform an intervention and draw a paired sample from the
        resulting counterfactual distribution.
        Does not change the underlying SCM permanently!
        ___

        Args:
            iv [list[Intervention]]:
                single or multiple interventions to perform
            n [int]:
                number of samples to draw
            return_noise [bool]:
                flag to return noise values from sampling

        Returns:
            orig_values: list[float]:
                array of sampled observational values of causal variables
            orig_noises [list[float]]:
                array of sampled noise values
            cf_values: list[float]:
                array of sampled counterfactual values of causal variables
            cf_noises [list[float]]:
                array of sampled counterfactual noise values. identical to
                orig_noises with the exception of noises of intervention targets
        """
        # Sample original values and noises
        orig_values, orig_noises = self.sample(n, return_noise=True)
        # Make copy of SCM, don't change internal state of original SCM
        SCM_copy = copy.deepcopy(self)
        # Intervene
        SCM_copy.intervene(iv)
        # Sample intervened SCM with previously sample noises
        cf_values, cf_noises = SCM_copy.sample(n, return_noise=True,
                                               old_noises=orig_noises)

        if return_noise:
            return (orig_values, orig_noises), (cf_values, cf_noises)
        else:
            return orig_values, cf_values

    # TODO: add some print summary thing. Should include causal variables in correct order, their mechanisms and noises
