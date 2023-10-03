#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""

A simple script containing some simple utility functions for plotting runs of the CMA algorithm

"""

from __future__ import absolute_import

import itertools
import cma

import numpy as np  # for median, zeros, random, asarray
import matplotlib.pyplot as plt

from typing import Callable, Any
from matplotlib.animation import FuncAnimation as FuncAnimation
from functools import partial
from matplotlib.figure import Figure as Figure
from matplotlib.axes import Axes as Axes

from .parallelization import compute_mutation_probability_map
from .experiments import CMABenchmark, CMADataStore, CMAExperiment


class CMAParamHistory(object):
    """
    Generator object holding the data of each generation of a CMA run
    """
    def __init__(
            self, 
            mean_history, 
            sigma_history, 
            covariance_history, 
            phenotypes_history,
            number_of_generations,
            number_of_offsprings
        ):
        self.mean_history = mean_history
        self.sigma_history = sigma_history
        self.covariance_history = covariance_history
        self.phenotypes_history = phenotypes_history
        self.number_of_generations = number_of_generations
        self.number_of_offsprings = number_of_offsprings
        self.current_idx = 0
    
    def __len__(self):
        return self.number_of_generations * self.number_of_offsprings

    def __iter__(self):
        return self

    def __next__ (self):
        if self.current_idx < len(self):
            generation_idx = current_idx // number_of_phenotypes
            phenotypes_idx = current_idx % number_of_phenotypes
            current_mean = self.evolution_strategy.mean_history[generation_idx]
            current_sigma = self.evolution_strategy.sigma_history[generation_idx]
            current_covariance = self.evolution_strategy.covariance_history[generation_idx]
            current_phenotypes = self.evolution_strategy.samples_history[generation_idx][phenotypes_idx]
            self.current_idx += 1
            return current_mean, current_sigma, current_covariance, current_phenotypes
        else: 
            raise StopIteration


class CMAPlotter(object):
    """
    The object handling all the plotting tasks

    Attributes:
        experiment: the experiment attached to the plotter
        problem: the problem associated to the experiment attached to  the plotter 
        n_points: The number of points used for creating the grid, which will have shape of (n_points, n_points)
        evolution_strategy: The evolution strategy found by the CMA algorithm on the problem
        lbounds: The axis-wise lower bounds for the plots
        ubounds: The axis-wise upper bounds for the plots
        subplot_settings: The settings to configure the subplots with
        contour_ploy_setting: The setting used to plot the contour plot
        probability_map_setting: The setting used to plot the mutation probability map
        x_axis: The array containing the x coordinates used in the plots
        y_axis: The array containing the y coordinates used in the plots
        grid: The grid used for the plots
        z_values: The objective function values computed on the grid
        x_mesh: The mesh object for the x axis 
        y_mesh: The mesh object for the y axis 
        figure: The Figure object of the plotter
        axis: The Axes obect of the plotter
    """
    def __init__(
            self, 
            experiment: CMAExperiment,
            xoffset: float = 5.0, 
            yoffset: float = 5.0, 
            n_points: int = 2001,
            subplots_settings: dict[str, Any] = {"nrows": 1, "ncols": 2},
            axes1_plotting_options: dict[str, Any] = {"alpha": 0.5},
            axes2_plotting_options: dict[str, Any] = {"levels": 15, "alpha": 0.5},
            point_plotting_options: dict[str, Any] = {"fmt": 'ro'},
            animation_options: dict[str, Any] = {}
        ) -> None:
        """
        Arguments:
            experiment: the (already ran) experiment attached to the plotter
            xoffset: the horizontal offset from the xopt to use for plotting (The plots are centered around the best value) 
            yoffset: the vertical offset from the yopt to use for plotting (The plots are centered around the best value) 
            n_points: The number of points used for creating the grid, which will have shape of (n_points, n_points)
            subplot_settings: The settings to configure the subplots with
            contour_ploy_setting: The setting used to plot the contour plot
            probability_map_setting: The setting used to plot the mutation probability map
        
        """
        self.experiment = experiment
        assert(self.experiment.ran), "You need to run the experiment before instantiating the plotter"
        self.problem = self.experiment.problem
        if n_points % 2 == 1:
            self.n_points = n_points
        else:
            self.n_points =  n_points + 1 
            UserWarning("You need to set the argument n_points to an odd integer, increasing it by one")
        try:
            assert(isinstance(self.experiment.evolution_strategy, cma.CMAEvolutionStrategy)), "More ES are associated to the problem, currently not supported"
            self.evolution_strategy = self.experiment.evolution_strategy
        except AssertionError as _error:
            raise ValueError(_error)
        self.lbounds = self.problem.best_decision_vector - np.array([xoffset, yoffset])
        self.ubounds = self.problem.best_decision_vector + np.array([xoffset, yoffset])
        self.subplots_settings = subplots_settings
        self._axes1_plotting_options = axes1_plotting_options
        self._axes2_plotting_options = axes2_plotting_options
        self._animation_options = animation_options
        self.x_axis = np.linspace(self.lbounds[0], self.ubounds[0], self.n_points)
        self.y_axis = np.linspace(self.lbounds[1], self.ubounds[1], self.n_points)
        self.grid = self.__initialize_grid()
        self.z_values = self.__compute_z_values()
        self.x_mesh, self.y_mesh = np.meshgrid(self.x_axis, self.y_axis)
        self._point_radius_x = xoffset * 2 / self.n_points
        self._point_radius_y = yoffset * 2 / self.n_points
        self.cma_run_history = CMAParamHistory(
            self.evolution_strategy.mean_history, 
            self.evolution_strategy.sigma_history, 
            self.evolution_strategy.covariance_history, 
            self.evolution_strategy.samples_history,
            self.number_of_generations,
            self.evolution_strategy.popsize
        )
        try:
            assert(self.subplots_settings["ncols"] == 2), "The number of columns expected for the subplots is 2"
            self.figure, (self.axes1, self.axes2)  = self.__init_subplots()
        except AssertionError as _error:
            raise ValueError(_error)

    @property
    def dimension(self) -> int:
        """
        Returns the dimensionality of the decision variable, must be 2 for plotting
        """
        assert(self.problem.dimension == 2)
        return self.problem.dimension

    @property
    def axes1_plotting_options(self) -> dict["str", Any]:
        """
        Returns the dictionary containing the option settings for the first axes
        """
        try:
            assert(isinstance(self._axes1_plotting_options, dict)), "self.axes1_option is expected to be a dictionary object"
        except AssertionError as _error:
            raise TypeError(_error)
        return self._axes1_plotting_options

    @property
    def axes2_plotting_options(self) -> dict["str", Any]:
        """
        Returns the dictionary containing the option settings for the first axes
        """
        try:
            assert(isinstance(self._axes2_plotting_options, dict)), "self.axes2_option is expected to be a dictionary object"
        except AssertionError as _error:
            raise TypeError(_error)
        return self._axes2_plotting_options

    @property
    def animation_options(self) -> dict["str", Any]:
        """
        Returns the dictionary containing the option settings for the first axes
        """
        try:
            assert(isinstance(self._animation_options, dict)), "self.animation_options is expected to be a dictionary object"
        except AssertionError as _error:
            raise TypeError(_error)
        return self._animation_options

    @property
    def number_of_generations(self) -> int:
        """
        Returns the number of generations of the evolution strategy, same as `len(plotter)`
        """
        return len(self)

    def __len__(self) -> int:
        """
        Returns the number of generations of the evolution strategy, same as `len(plotter)`
        """
        assert(len(self.evolution_strategy.mean_history) == len(self.evolution_strategy.sigma_history))
        assert(len(self.evolution_strategy.mean_history) == len(self.evolution_strategy.covariance_history))
        assert(len(self.evolution_strategy.mean_history) == len(self.evolution_strategy.samples_history))
        return len(self.evolution_strategy.mean_history)

    def __initialize_grid(self) -> np.ndarray:
        """
        Initializes the grid for plottings
        """
        grid_list = list(itertools.product(self.x_axis, self.y_axis))
        return np.array(grid_list).reshape((self.n_points, self.n_points, self.dimension))

    def __compute_z_values(self) -> np.ndarray:
        """
        Evaluates the objective function on the grid
        """
        z = np.array([self.problem(point) for point in self.grid.reshape((-1, self.dimension))])
        return z.reshape(self.n_points, self.n_points)

    def __set_axes_limits(self, axes) -> None:
        """
        Sets the limits for the axes's x and y axes
        """
        try:
            assert(isinstance(axes, Axes)), "the axes arguments is expected to be a matplotlib.axes.Axes object"
        except AssertionError as _error:
            raise TypeError(_error)
        axes.set_xlim(self.lbounds[0], self.ubounds[0])
        axes.set_ylim(self.lbounds[1], self.ubounds[1])

    def __set_axes_labels(self, axes) -> None:
        """
        Set the labels for the axes's x and y axes
        """
        try:
            assert(isinstance(axes, Axes)), "the axes arguments is expected to be a matplotlib.axes.Axes object"
        except AssertionError as _error:
            raise TypeError(_error)
        axes.set_xlabel(r"$x_1$")
        axes.set_ylabel(r"$x_2$")
    
    def __init_subplots(self) -> tuple[Figure, Axes]:
        """
        Initializes the subplots of the plotter 
        """
        figure, (axes1, axes2) = plt.subplots(**self.subplots_settings)
        self.__set_axes_limits(axes1)
        self.__set_axes_limits(axes2)
        self.__set_axes_labels(axes1)
        self.__set_axes_labels(axes2)
        return figure, (axes1, axes2)

    def __compute_mutation_probability_map(self, generation_idx) -> np.ndarray:
        """
        Argument:

            generation_idx: The id of the generation of which to plot the mutation probability density

        Computes the mutation probability map on the grid for a given generation
        """
        return compute_mutation_probability_map(
            self.grid, 
            self.n_points,
            self.evolution_strategy.covariance_history[generation_idx], 
            self.evolution_strategy.sigma_history[generation_idx], 
            self.evolution_strategy.mean_history[generation_idx]
        )

    def __find_decision_vector_pixel_coordinate(self, decision_vector) -> tuple[int]:
        """
        Finds the pixels of a point in the grid
        """
        try:
            assert(decision_vector.ndim == 1 and decision_vector.shape[0] == 2), "The provided decision vector must have shape of (2)"
            assert(decision_vector[0] < self.x_axis[-1] and decision_vector[0] > self.x_axis[0]), "The x-cooridnate of the decision vector is outside the displayed grid"
            assert(decision_vector[1] < self.y_axis[-1] and decision_vector[1] > self.y_axis[0]), "The y-cooridnate of the decision vector is outside the displayed grid"
        except AssertionError as _error:
            raise ValueError(_error)
        x, y = decision_vector
        if x in self.x_axis:
            x_condition = self.x_axis == x
        else:
            x_condition = np.abs(self.x_axis - x) < self._point_radius_x
        if y in self.y_axis:
            y_condition = self.y_axis == y
        else:
            y_condition = np.abs(self.y_axis - y) < self._point_radius_y
        x_coord, = np.where(x_condition)
        y_coord, = np.where(y_condition)
        try:
            assert(x_coord.size), "No pixels were found for the x coordinate"
            assert(y_coord.size), "No pixels were found for the y coordinate"
        except AssertionError as _error:
            raise ValueError(_error)
        if x_coord.size > 1:
            x_coord = np.median(x_coord)
            x_coord = np.floor(x_coord)
        else:
            x_coord = x_coord[0]
        if y_coord.size > 1:
            y_coord = np.median(y_coord)
            y_coord = np.floor(y_coord)
        else:
            y_coord = y_coord[0]        
        return int(x_coord), int(y_coord)
  
    def __plot_point_by_coord(self, point_pixel_coordinates, axes = 0, *args, **kwargs) -> None:
        """
        Plots a poit given its pixel coordinates
        """
        if not axes:
            self.axes1.plot(*point_pixel_coordinates, *args, **kwargs)
            self.axes2.plot(*point_pixel_coordinates, *args, **kwargs)
        elif axes == 1:
            self.axes1.plot(*point_pixel_coordinates, *args, **kwargs)
        elif axes == 2:
            self.axes2.plot(*point_pixel_coordinates, *args, **kwargs)
        else:
            raise ValueError(f"The specified `axes` parameter has to be either 0, 1 or 2")
            
    def __plot_point(self, point, axes = 0, *args, **kwargs) -> None:
        """
        Plots a point
        """
        point_coordinates = self.__find_decision_vector_pixel_coordinate(point)
        self.__plot_point_by_coord(point_coordinates, axes = axes, *args, **kwargs)
    
    def __init_animation(self, *args, **kwargs) -> None:
        """
        Initialization function for the plotting routine
        """
        self.axes1.contourf(self.x_mesh, self.y_mesh, self.z_values, **self.axes1_plotting_options)
        self.__plot_point(self.problem.best_decision_vector, *args, **kwargs)
    
    def __update_animation(self, frame) -> None:
        """
        Updating function for the plotting routine
        """
        current_mean, current_sigma, current_covariance, current_phenotypes = next(frame)
    
    def __call__(self, inteval: int = 20) -> None:
        """
        Runs the plotting routine
        """
        callback_function = partial(self.__update_animation)
        animation = FuncAnimation(
            self.figure, 
            callback_function, 
            init_func = self.__init_animation, 
            frames = self.cma_run_history,
            **self.animation_options
        )
