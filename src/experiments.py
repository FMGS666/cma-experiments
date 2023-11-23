#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
A simple script running a benchmarking experiment for the cma.fmin2 routine on the bbob-noisy test suite

This file contains the definition of four classes:

1) `CMADataStore`: 
    Basically a data store where all the data relative to the benchmarking experiments are stored

2) `CMAExperiment`:
    The class needed to run the cma.fmin2 routine on a given problem and store the results

3) `CMAEBenchmark`:
    The class running the whole benchmarking experiment by spawning a CMAExperiment instance over each problem of a suite

4) `CMALogger`:
    The logging facilities for a benchmarking experiment
"""

from cocoex import solvers, utilities
import sys
import re
import time  # output some timings per evaluation
from collections import defaultdict
from typing import Callable, Any
import numpy as np  # for median, zeros, random, asarray

from cocoex import Suite, Observer, Problem

import cma
import pickle
try: import cocopp  # post-processing module
except: pass

class CMADataStore(object):

    """
    `CMADataStore`:
        The class needed for storing the data relative to a benchmarking experiment
    Attributes:
        timings -> defaultdictionary containing a map from the problem dimensions to the list of all the timings of the benchmarking experiment on that dimension
        evolution_strategies -> defaultdictionary containing a map from the problem indexes to the list of all the evolution strategies generated on that problem
    
    This class doesn't implement any method, is just a container for data relative to a bechmarking experiment
    """
    def __init__(self) -> None:
        self._timings = defaultdict(list)
        self._evolution_strategies = defaultdict(list)

    @property 
    def timings(self) -> defaultdict[list]:
        """
        The timings, saved by dimension
        """
        return self._timings

    @property
    def evolution_strategies(self) -> defaultdict[list]:
        """
        The evolution strategies, saved by problem
        """
        return self._evolution_strategies

class CMAExperiment(object):
    """
    `CMAExperiment`:
        The class running the cma fmin2 routine on a single problem
    
    Attributes: 
        solver               -> The solver that is being benchmarked
        suite                -> The suite the solver is benchmarked on 
        problem              -> The current problem that is being run
        observer             -> The observer used on the current problem
        data_store           -> The data store for saving the results and the evolution strategies
        budget_multiplier    -> The budget multiplier used to compute the experiments budget based on the number of dimensions
        restarts             -> The number of restarts on the current problem
    
    This function is supposed to call the underlying cma algorithm, which can be called through the following signature:
    
    from cma import fmin2 as fmin2
    fmin2(objective_function, x0, sigma0,
         options=None,
         args=(),
         gradf=None,
         restarts=0,
         restart_from_best='False',
         incpopsize=2,
         eval_initial_x=False,
         parallel_objective=None,
         noise_handler=None,
         noise_change_sigma_exponent=1,
         noise_kappa_exponent=0,
         bipop=False,
         callback=None):

    """
    def __init__(
            self, 
            solver: Callable,
            suite: Suite, 
            problem: Problem, 
            observer: Observer, 
            printer: utilities.MiniPrint, 
            data_store: CMADataStore, 
            cma_options: cma.CMAOptions,
            budget_multiplier: int,
            sigma0: float,
            verbose: int, 
            **cma_kwargs: dict[str, Any]
        ) -> None:
        """
        Initializes all the class attributes and observes the problem
        """
        self.solver = solver
        self.suite = suite
        self.problem = problem
        self.observer = observer
        self.printer = printer 
        self.data_store = data_store
        self.budget_multiplier = budget_multiplier
        self._sigma0 = sigma0
        self._restarts = -1
        self._verbose = verbose
        self._cma_options = cma_options
        self._cma_kwargs = cma_kwargs
        self.__observe_problem()
        self._ran = False

    @property
    def sigma0(self) -> float:
        """
        The starting sigma for the CMA algorithm
        """
        return self._sigma0

    @property
    def restarts(self) -> int:
        """
        The number of restarts on the current problem
        """
        return self._restarts

    @property
    def evalsleft(self) -> int:
        """
        The remaining budget to the solver
        """
        return int(self.problem.dimension * self.budget_multiplier + 1 -
                max((self.problem.evaluations, self.problem.evaluations_constraints)))

    @property
    def idx(self) -> int:
        """
        The index of the current problem on the given suite
        """
        return self.problem.index

    @property
    def verbose(self) -> int:
        """
        The index of the current problem on the given suite
        """
        return self._verbose
    
    @property
    def ran(self) -> bool:
        """
        Whether the experiments has already been ran
        """
        return self._ran

    @property
    def evolution_strategy(self) -> list[cma.CMAEvolutionStrategy] | cma.CMAEvolutionStrategy:
        """
        Returns the evolution strategies generated in the experiment
        """
        # there is no evolution strategy if we don't run the algorithm
        assert(self.ran), "You need to run the experiment before accessing the evolution strategy associated with it"
        try:
            assert(isinstance(self.data_store._evolution_strategies[self.idx], list)), "self.data._store_evolution_strategies expected to be a list"
        except AssertionError as _error:
            raise TypeError(_error)
        evolution_strategies = self.data_store.evolution_strategies[self.idx]
        # a bit of sanity check
        for evolution_strategy in evolution_strategies:
            try:
                assert(isinstance(evolution_strategy, cma.CMAEvolutionStrategy)), f"self.data._store_evolution_strategies expected to be a cma.CMAEvolutionStrategy object (given type: {type(evolution_strategy)})"
            except AssertionError as _error:
                raise TypeError(_error)
        return evolution_strategies[0] if len(evolution_strategies) == 1 else evolution_strategies

    @property
    def cma_options(self):
        """
        Returns the option argument with which to call the cma algorithm
        """
        try:
            assert(isinstance(self._cma_options, cma.CMAOptions)), "self.cma_option is supposed to be a cma.CMAOptions object"
        except AssertionError as _error:
            raise TypeError(_error)
        return self._cma_options

    @property
    def cma_kwargs(self):
        """
        Returns the option argument with which to call the cma algorithm
        """
        try:
            assert(isinstance(self._cma_kwargs, dict)), "self.cma_option is supposed to be a dictionary object"
        except AssertionError as _error:
            raise TypeError(_error)
        if not len(self._cma_kwargs):
            UserWarning("W: Passing an empty dictionary of keyword arguments to the cma optimizer. Running the cma algorithm with the default settings")
        return self._cma_kwargs

    def free(self) -> None:
        """
        Frees the memory allocated for the problem
        """
        self.problem.free()

    def __observe_problem(self) -> None:
        """
        Observes the self.problem using the self.observer object
        """
        self.problem.observe_with(self.observer)

    def __call__(self) -> None:
        """
        Runs the optimization routine on the given problem
        """
        self._ran = True
        time1 = time.time()

        self.problem(np.zeros(self.problem.dimension))  # making algorithms more comparable
        while self.evalsleft > 0 and not self.problem.final_target_hit:
            self._restarts += 1
            xopt, es = self.solver(self.problem, self.problem.initial_solution_proposal, self.sigma0, self.cma_options, **self.cma_kwargs)
            self.data_store.evolution_strategies[self.idx].append(es)

        self.data_store._timings[self.problem.dimension].append((time.time() - time1) / self.problem.evaluations
            if self.problem.evaluations else 0)
        if self.verbose:
            self.printer(self.problem, restarted = self._restarts, final=self.idx == len(self.suite) - 1)

class CMABenchmark(object):
    """
    `CMABenchmark`:
        The classed used for running a whole benchmarking session on the given suite
    
    Attributes:
        solver               -> The solver that is being benchmarked
        suite                -> The suite the solver is benchmarked on 
        observer             -> The observer used on the current problem
        data_store           -> The data store for saving the results and the evolution strategies
        budget_multiplier    -> The budget multiplier used to compute the experiments budget based on the number of dimensions
        restarts             -> The number of restarts on the current problem
    """
    def _init(
            self, 
            solver: Callable, 
            suite: Suite, 
            observer: Observer, 
            printer: utilities.MiniPrint, 
            data_store: CMADataStore, 
            cma_options: cma.CMAOptions,
            budget_multiplier: int,
            sigma0: float,
            verbose: int,
            **cma_kwargs: dict[str, Any]
        ) -> None:
        """
        Initializes classes attributes
        """
        self.solver = solver
        self.suite = suite 
        self.observer = observer 
        self.printer = printer
        self.data_store = data_store
        self.budget_multiplier = budget_multiplier
        self.verbose = verbose
        self._cma_options = cma_options
        self._sigma0 = sigma0
        self._cma_kwargs = cma_kwargs

    @staticmethod
    def __set_num_threads(
            nt: int = 1, 
            disp: int = 1
        ) -> None:
        """see https://github.com/numbbo/coco/issues/1919
        and https://twitter.com/jeremyphoward/status/1185044752753815552
        """
        try: import mkl
        except ImportError: disp and print("mkl is not installed")
        else:
            mkl.set_num_threads(nt)
        nt = str(nt)
        for name in ['OPENBLAS_NUM_THREADS',
                    'NUMEXPR_NUM_THREADS',
                    'OMP_NUM_THREADS',
                    'MKL_NUM_THREADS']:
            os.environ[name] = nt
        disp and print("setting mkl threads num to", nt)

    @property
    def cma_options(self):
        """
        Returns the option argument with which to call the cma algorithm
        """
        try:
            assert(isinstance(self._cma_options, cma.CMAOptions)), "self.cma_option is supposed to be a cma.CMAOptions object"
        except AssertionError as _error:
            raise TypeError(_error)
        return self._cma_options

    @property
    def cma_kwargs(self):
        """
        Returns the option argument with which to call the cma algorithm
        """
        try:
            assert(isinstance(self._cma_kwargs, dict)), "self.cma_option is supposed to be a dictionary object"
        except AssertionError as _error:
            raise TypeError(_error)
        if not len(self._cma_kwargs):
            UserWarning("W: Passing an empty dictionary of keyword arguments to the cma optimizer. Running the cma algorithm with the default settings")
        return self._cma_kwargs

    @property
    def sigma0(self) -> float:
        """
        The starting sigma for the CMA algorithm
        """
        return self._sigma0

    def __len__(self) -> None:
        """
        Returns the number of problems in the suite to be benchmarked
        """
        return len(self.suite)

    def __getitem__(
            self, 
            idx: int
        ) -> CMAExperiment:
        """
        Gets the idx-th problem in the suite and initializes a `CMAExperiment` object oer it
        """
        problem = self.suite[idx]
        return CMAExperiment(
            self.solver, 
            self.suite, 
            problem, 
            self.observer, 
            self.printer, 
            self.data_store, 
            self.cma_options, 
            self.budget_multiplier, 
            self.sigma0, 
            self.verbose, 
            **self.cma_kwargs
        )

    def __init__(
            self,
            solver: Callable,
            suite_name: str = "bbob-noisy",
            suite_year_option: str = "",
            suite_filter_option: str = "", 
            output_folder: str = "./",
            budget_multiplier: int = 10,
            verbose: int = 0,
            run: bool = False,
            sigma0: float = 1., 
            cma_options: cma.CMAOptions = cma.CMAOptions(), # default options
            **cma_kwargs: dict[str, Any]
        ) -> None:
        suite = Suite(suite_name, suite_year_option, suite_filter_option)
        observer = Observer(suite_name, "result_folder: " + output_folder)
        printer = utilities.MiniPrint()
        data_store = CMADataStore()
        self._init(solver, suite, observer, printer, data_store, cma_options, budget_multiplier, sigma0, verbose, **cma_kwargs)
    