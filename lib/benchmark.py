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

from __future__ import absolute_import, division, print_function, unicode_literals
from cocoex import solvers, utilities
import sys
import re
import time  # output some timings per evaluation
from collections import defaultdict
from typing import Callable
import os, webbrowser  # to show post-processed results in the browser
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
    """
    def __init__(
            self, 
            solver: Callable,
            suite: Suite, 
            problem: Problem, 
            observer: Observer, 
            printer: utilities.MiniPrint, 
            data_store: CMADataStore, 
            budget_multiplier: int,
            sigma0: float = 1.,
            verbose: int = 0
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
        self.__observe_problem()
        self._ran = False

    @property
    def sigma0(self) -> float:
        """
        The starting sigma for the CMA algorithm
        """
        return self._sigma0

    @property
    def dimension(self) -> int:
        """
        The number of dimensions of the problem
        """
        return self.problem.dimension

    @property
    def final_target_hit(self) -> bool:
        """
        Whether the final target value has been hit
        """
        return self.problem.final_target_hit

    @property
    def initial_solution(self) -> np.ndarray:
        """
        The initial solution which to start the optimization routine from
        """
        return self.problem.initial_solution_proposal

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
        return int(self.dimension * self.budget_multiplier + 1 -
                max((self.evaluations, self.problem.evaluations_constraints)))

    @property
    def evaluations(self) -> int:
        """
        The number of objective function evaluations
        """
        return self.problem.evaluations

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
    def evolution_strategy(self) -> list[cma.CMAEvolutionStrategy]| cma.CMAEvolutionStrategy:
        """
        Returns the evolution strategies generated in the experiment
        """
        assert(self.ran), "You need to run the experiment before accessing the evolution strategy associated with it"
        return self.data_store._evolution_strategies[self.idx] if len(self.data_store._evolution_strategies[self.idx]) > 1 else \
            self.data_store._evolution_strategies[self.idx][0]

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

        self.problem(np.zeros(self.dimension))  # making algorithms more comparable
        while self.evalsleft > 0 and not self.final_target_hit:
            self._restarts += 1
            xopt, es = self.solver(self.problem, self.initial_solution, self.sigma0, {'maxfevals': self.evalsleft, 'verbose':-9}, restarts = 9)
            self.data_store.evolution_strategies[self.idx].append(es)

        self.data_store.timings[self.dimension].append((time.time() - time1) / self.evaluations
            if self.evaluations else 0)
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
            budget_multiplier: int,
            verbose: int = 0,
            run: bool = False
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
        self._run = run

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
        return CMAExperiment(self.solver, self.suite, problem, self.observer, self.printer, self.data_store, self.budget_multiplier, verbose = self.verbose)

    def run(self, **kwargs) -> None:
        """
        Iterates over the suite, initializing and calling a `CMAExperiment` object at each iteration
        """
        self.__set_num_threads(**kwargs)
        for idx in range(self.__len__()):
            experiment = self.__getitem__(idx)
            experiment()
            experiment.free()
    
    def __call__(self, **kwargs) -> None:
        """
        Iterates over the suite, initializing and calling a `CMAExperiment` object at each iteration
        """
        self._run(**kwargs)

    def __init__(
            self,
            solver: Callable, 
            suite_name: str = "bbob-noisy",
            suite_year_option: str = "",
            suite_filter_option: str = "", 
            output_folder: str = "./",
            budget_multiplier: int = 2,
            show_on_browser: bool = False, 
            run: bool = False
        ) -> None:
        suite = Suite(suite_name, suite_year_option, suite_filter_option)
        observer = Observer(suite_name, "result_folder: " + output_folder)
        printer = utilities.MiniPrint()
        data_store = CMADataStore()
        self._init(solver, suite, observer, printer, data_store, budget_multiplier)
        if self._run:
            self.run()
            if show_on_browser:
                cocopp.main(observer.result_folder) 
                webbrowser.open("file://" + os.getcwd() + "/ppdata/index.html") 
        
    
    