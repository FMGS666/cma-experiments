import os
import time
import pickle
import cma
import cocopp
import numpy as np

from typing import Callable, Any
from pathlib import Path
from cocoex import Suite, Observer, Problem, solvers, utilities

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
            **cma_kwargs: dict[str, Any]
        ) -> None:
        self.solver = solver
        self.suite = suite
        self.problem = problem
        self.observer = observer
        self.printer = printer 
        self.timings = []
        self.evolution_strategies = []
        self.budget_multiplier = budget_multiplier
        self.sigma0 = sigma0
        self.restarts = -1
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

    def free(self) -> None:
        self.problem.free()

    def run(self) -> None:
        self.ran = True
        time1 = time.time()
        self.problem(np.zeros(self.problem.dimension))
        while self.evalsleft > 0 and not self.problem.final_target_hit:
            self.restarts += 1
            options = self.cma_options | {
                "maxfevals": self.evalsleft
            } 
            xopt, es = self.solver(
                self.problem, 
                self.problem.initial_solution_proposal, 
                self.sigma0, 
                options, 
                **self.cma_kwargs
            )
            result = es.result._asdict()
            result["stop"] = dict(result["stop"])
            self.evolution_strategies.append(result)
        self.timings.append((time.time() - time1) / self.problem.evaluations
            if self.problem.evaluations else 0)
        self.printer(
            self.problem, 
            restarted = self.restarts, 
            final = self.idx == len(self.suite) - 1
        )
    
    def save_history(
            self,
            dumps_folder: str | Path
        ) -> None:
        dump_file = os.path.join(dumps_folder, f"{self.problem.id}.pkl")
        history = {
            "timings": self.timings,
            "evolution_strategies": self.evolution_strategies
        }
        with open(dump_file, "wb") as _dump_buffer:
            pickle.dump(history, _dump_buffer)
