import mkl

from .cma_experiment import * 
from .cma_bbob_noisy_settings import CMANoisySettings, OptionType


class CMABenchmark:
    def __init__(
            self,
            solver: Callable,
            budget_multiplier: int,
            suite_name: str = "bbob-noisy",
            suite_year_option: str = "",
            suite_filter_option: str = "", 
            output_folder: str = "./",
            sigma0: float = 2.,
            cma_options: cma.CMAOptions = cma.CMAOptions(),
            noisy_settings: bool = False,
            evaluate_origin: bool = True,
            **cma_kwargs: dict[str, Any]
        ) -> None:
        self.suite_name = suite_name
        self.suite_year_option = suite_year_option
        self.suite_filter_option = suite_filter_option
        self.output_folder = output_folder
        self.solver = solver
        self.suite = Suite(
            self.suite_name, 
            self.suite_year_option, 
            self.suite_filter_option
        ) 
        self.observer = Observer(
            self.suite_name, 
            "result_folder: " + self.output_folder
        ) 
        self.printer = utilities.MiniPrint()
        self.budget_multiplier = budget_multiplier
        self.cma_options = cma_options
        self.sigma0 = sigma0
        self.cma_kwargs = cma_kwargs
        self.noisy_settings = noisy_settings
        self.evaluate_origin = evaluate_origin

    @staticmethod
    def set_num_threads(
            nt: int = 1, 
            disp: int = 1
        ) -> None:
        mkl.set_num_threads(nt)
        nt = str(nt)
        for name in ['OPENBLAS_NUM_THREADS',
                    'NUMEXPR_NUM_THREADS',
                    'OMP_NUM_THREADS',
                    'MKL_NUM_THREADS']:
            os.environ[name] = nt
        disp and print("setting mkl threads num to", nt)

    def get_noisy_settings_for_problem_dimension(
            self,
            dimension: int
        ) -> dict[str, OptionType]:
        """
        
        """
        noisy_options = CMANoisySettings(dimension)
        return noisy_options.get_opions_dictionary()

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
        print(f"Running {problem.id=}")
        extra_options = dict()
        if self.noisy_settings:
            extra_options = self.get_noisy_settings_for_problem_dimension(
                problem.dimension
            )
        options = self.cma_options | extra_options
        return CMAExperiment(
            self.solver, 
            self.suite, 
            problem, 
            self.observer, 
            self.printer, 
            options, 
            self.budget_multiplier, 
            self.sigma0, 
            **self.cma_kwargs
        )