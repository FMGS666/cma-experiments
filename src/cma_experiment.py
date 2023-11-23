from cocoex import solvers, utilities
import time
from collections import defaultdict
from typing import Callable, Any
import numpy as np

from cocoex import Suite, Observer, Problem

import cma
import pickle
import cocopp

class CMAExperiment:
    def __init__(
            self, 
            solver: Callable,
            suite: Suite, 
            problem: Problem, 
            observer: Observer, 
            printer: utilities.MiniPrint,
            cma_options: cma.CMAOptions,
            budget_multiplier: int,
            sigma0: float,
            verbose: bool, 
            **cma_kwargs: dict[str, Any]
        ) -> None:
        self.solver = solver
        self.suite = suite
        self.problem = problem
        self.observer = observer
        self.printer = printer 
        self.timings = defaultdict(list)
        self.evolution_strategies = defaultdict(list)
        self.budget_multiplier = budget_multiplier
        self.sigma0 = sigma0
        self.restarts = -1
        self.verbose = verbose
        self.cma_options = cma_options
        self.cma_kwargs = cma_kwargs
        self.problem.observe_with(self.observer)
        self.ran = False

    @property
    def evalsleft(self) -> int:
        return int(self.problem.dimension * self.budget_multiplier + 1 -
                max((self.problem.evaluations, self.problem.evaluations_constraints)))

    @property
    def idx(self) -> int:
        return self.problem.index

    @property
    def evolution_strategy(self) -> list[cma.CMAEvolutionStrategy] | cma.CMAEvolutionStrategy:
        assert(self.ran), "You need to run the experiment before accessing the evolution strategy associated with it"
        if not isinstance(self.evolution_strategies[self.idx], list):
            raise TypeError("self.data._store_evolution_strategies expected to be a list")
        evolution_strategies = self.evolution_strategies[self.idx]
        for evolution_strategy in evolution_strategies:
            if not isinstance(evolution_strategy, cma.CMAEvolutionStrategy):
                raise TypeError( f"self.data._store_evolution_strategies expected to be a cma.CMAEvolutionStrategy object (given type: {type(evolution_strategy)})")
        return evolution_strategies[0] if len(evolution_strategies) == 1 else evolution_strategies

    def free(self) -> None:
        self.problem.free()

    def run(self) -> None:
        self.ran = True
        time1 = time.time()
        self.problem(np.zeros(self.problem.dimension))
        while self.evalsleft > 0 and not self.problem.final_target_hit:
            self._restarts += 1
            xopt, es = self.solver(
                self.problem, 
                self.problem.initial_solution_proposal, 
                self.sigma0, 
                self.cma_options, 
                **self.cma_kwargs
            )
            self.evolution_strategies[self.idx].append(es)
        self.timings[self.problem.dimension].append((time.time() - time1) / self.problem.evaluations
            if self.problem.evaluations else 0)
        if self.verbose:
            self.printer(
                self.problem, 
                restarted = self.restarts, 
                final = self.idx == len(self.suite) - 1
            )
