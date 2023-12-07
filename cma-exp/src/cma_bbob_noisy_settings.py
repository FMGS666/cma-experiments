import numpy as np

from typing import Union

OptionType = Union[float, int, list[float]]

class CMANoisySettings(object):
    def __init__(
            self,
            dimension: int
        ) -> None:
        self.dimension = dimension
    
    @property
    def popsize(self) -> int:
        return int(4 + 3*np.log(self.dimension))

    @property
    def mu(self) -> int:
        return self.popsize // 2

    @property
    def mu_w(self) -> float:
        return 1 / np.sum(
            [
                self.rank_weight(idx) ** 2
                for idx in range(1, self.mu + 1)
            ]
        )

    @property 
    def c_sigma(self) -> float:
        num = self.mu_w + 2
        den = self.dimension + self.mu_w + 5
        return num / den
    
    @property
    def c_c(self) -> float:
        num = 4 + (self.mu_w / self.dimension)
        den = self.dimension + 4 + 2 * (self.mu_w / self.dimension)
        return num / den

    @property
    def c_1(self) -> float:
        den = (self.dimension + 1.3)**2 + self.mu_w
        return 2 / den
    
    @property
    def c_mu(self) -> float:
        num = self.mu_w + 2 + (1 / self.mu_w)
        den = (self.dimension + 2)**2 + self.mu_w
        c_mu = 2*(num/den)
        return np.min(
            [
                c_mu, 
                1 - self.c_1
            ]
        )

    @property
    def d_sigma(self) -> float:
        num = self.mu - 1
        den = self.dimension + 1
        addend = np.sqrt(num / den) - 1
        d_sigma = 1 + c_sigma
        if addend > 0:
            return d_sima + addend
        return d_sigma

    @property
    def recombination_weights(self) -> list[float]:
        return [
            self.rank_weight(idx)
            for idx in range(1, self.mu + 1)
        ] + [0]

    def rank_weight(
            self,
            idx: int
        ) -> float:
        assert idx <= self.mu, f"{idx=} shouldn't be larger than {self.mu=}"
        assert idx > 0, f"{idx=} should be greater than 0"
        importance = np.log(self.mu + 1) - np.log(idx)
        normalization_factor = np.sum([np.log(self.mu + 1) - np.log(j) for j in range(1, self.mu + 1)])
        return importance / normalization_factor


    def get_opions_dictionary(self) -> dict[str, OptionType]:
        return {
            "CMA_recombination_weights": self.recombination_weights,
            "CMA_rankmu": self.c_mu,
            "CMA_rankone": self.c_1,            
        }
