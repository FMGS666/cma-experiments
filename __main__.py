from .src.cma_benchmark import CMABenchmark
from .src.cma_argparser import CMAArgParser
from .src.utils import read_cma_settings_from_yaml

import os
import yaml
import cma
import cocopp


if __name__ == "__main__":

    argument_parser = CMAArgParser()
    arguments = argument_parser.parse_args()

    solver = cma.fmin_lq_surr2 if arguments.surrogate\
        else cma.fmin2

    budget_multipliers, cma_options, cma_kwargs = read_cma_settings_from_yaml(
        arguments.settings_file
    )

    for budget_multiplier in budget_multipliers:
        output_folder = '%s_%dx_on_%s' % (
            solver.__name__, int(budget_multiplier), arguments.suite_name)
        output_folder = os.path.join(arguments.output_folder, output_folder)
        dumps_folder = os.path.join(output_folder, "dumps")
        os.mkdir(output_folder)
        os.mkdir(dumps_folder)
        benchmark = CMABenchmark(
            solver,
            budget_multiplier,
            suite_name = arguments.suite_name,
            suite_year_option = arguments.year_options,
            suite_filter_option = arguments.filter_options,
            output_folder = output_folder,
            verbose = arguments.verbose,
            sigma0 = arguments.sigma0,
            cma_options = cma_options,
            **cma_kwargs
        )
        benchmark.set_num_threads(
            nt = arguments.number_of_threads,
            disp = arguments.disp
        )
        n_problems = len(benchmark)
        for idx in range(n_problems):
            experiment = benchmark[idx]
            experiment.run()
            experiment.save_history(dumps_folder)
            experiment.free()
        if arguments.post_processing:
            cocopp.main(benchmark.observer.result_folder)
