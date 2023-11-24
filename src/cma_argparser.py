from argparse import ArgumentParser

class CMAArgParser(ArgumentParser):
    def __init__(
            self,
            prog: str = "",
            description: str = "",
            epilog: str = ""
        ) -> None:
        self.prog = prog
        self.description = description
        self.epilog = epilog
        super(CMAArgParser, self).__init__(
            prog = self.prog,
            description = self.description,
            epilog = self.epilog
        )
        self.add_argument(
            '-s', 
            '--surrogate',
            help = "Whether to use the `cma.fmin_lq_surr2` function",
            action = "store_true"
        )
        self.add_argument(
            '-n', 
            '--suite-name',
            help = "The suite which to run the benchmarking over",
            type = str,
            required = False,
            default = "bbob-noisy"   
        )
        self.add_argument(
            '-y', 
            '--year-options',
            help = "The year options which to define the suite with",
            type = str,
            required = False,
            default = ""   
        )
        self.add_argument(
            '-f', 
            '--filter-options',
            help = "The filter options which to define the suite with",
            type = str,
            required = False,
            default = ""   
        )
        self.add_argument(
            '-o', 
            '--output-folder',
            help = "The folder where to save the resulst",
            type = str,
            required = False,
            default = "./results/"   
        )
        self.add_argument(
            '-v', 
            '--verbose',
            help = "Whether to run verbose experiments or not",
            action = "store_true"
        )
        self.add_argument(
            '-S', 
            '--sigma0',
            help = "The sigma0 parameter of the CMA-ES algorithm",
            type = float,
            required = False,
            default = 1.0  
        )
        self.add_argument(
            '-sf', 
            '--settings-file',
            help = "The file from whih to read the settings for the CMA-ES algorithm",
            type = str,
            required = False,
            default = "./cma-experiments/settings.yaml"  
        )
        self.add_argument(
            '-pp', 
            '--post-processing',
            help = "Whether to perform post-processing after the experiments are run",
            action = "store_true"
        )
        self.add_argument(
            '-nt', 
            '--number-of-threads',
            help = "The number of threads",
            type = int,
            required = False,
            default = 50  
        )
        self.add_argument(
            '-d', 
            '--disp',
            help = "IDK",
            type = int,
            required = False,
            default = 0 
        )