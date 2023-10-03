#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""

A simple script containing some simple utility functions for plotting runs of the CMA algorithm

"""

from __future__ import absolute_import, division, print_function, unicode_literals
import sys
import re
import time  # output some timings per evaluation
from collections import defaultdict
import itertools
from typing import Callable, Any
import os, webbrowser  # to show post-processed results in the browser
import numpy as np  # for median, zeros, random, asarray
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from matplotlib import cm 
 
from matplotlib.figure import Figure as Figure
from matplotlib.axes import Axes as Axes
from matplotlib.ticker import MaxNLocator

from sklearn.preprocessing import MinMaxScaler as MinMaxScaler

from .parallelization import compute_mutation_probability_map

import cocoex
import cma
from .benchmark import CMABenchmark, CMADataStore, CMAExperiment

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
        x_mesh: The mesh object for the x axis (probably useless)
        y_mesh: The mesh object for the y axis (probably useless)
        fig: The Figure object of the plotter
        axis: The Axes obect of the plotter
    """
    def __init__(
            self, 
            experiment: CMAExperiment,
            xoffset: float = 5.0, 
            yoffset: float = 5.0, 
            n_points: int = 2001,
            subplots_settings: dict[str, Any] = {"nrows": 1, "ncols": 2},
            contour_plot_settings: dict[str, Any] = {"alpha": 0.5},
            probability_map_settings: dict[str, Any] = {"levels": 15, "alpha": 0.5},
            point_plotting_settings: dict[str, Any] = {"fmt": 'ro'},
            n_ticks: int = 8,
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
            assert(len(self.experiment.data_store.evolution_strategies[self.experiment.idx]) == 1), "More ES are associated to the problem, currently not supported"
            self.evolution_strategy = self.experiment.data_store.evolution_strategies[self.experiment.idx][0]
        except AssertionError as _error:
            raise ValueError(_error)
        self.lbounds = self.problem.best_decision_vector - np.array([xoffset, yoffset])
        self.ubounds = self.problem.best_decision_vector + np.array([xoffset, yoffset])
        self.subplots_settings = subplots_settings
        self.contour_plot_settings = contour_plot_settings
        self.probability_map_settings = probability_map_settings
        self.point_plotting_settings = point_plotting_settings
        self.n_ticks = n_ticks
        self.x_axis = np.linspace(self.lbounds[0], self.ubounds[0], self.n_points)
        self.y_axis = np.linspace(self.lbounds[1], self.ubounds[1], self.n_points)
        self.grid = self.__initialize_grid()
        self.z_values = self.__compute_z_values()
        self.x_mesh, self.y_mesh = np.meshgrid(self.x_axis, self.y_axis)
        self._point_radius_x = xoffset * 2 / self.n_points
        self._point_radius_y = yoffset * 2 / self.n_points
        try:
            assert(self.subplots_settings["ncols"] == 2), "The number of columns expected for the subplots is 2"
            self.fig, (self.axes1, self.axes2)  = self.__init_subplots()
            #self.__set_yticks()
            #self.__set_xticks()
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
    def number_of_generations(self) -> int:
        """
        Returns the number of generations of the evolution strategy, same as `len(plotter)`
        """
        assert(len(self.evolution_strategy.mean_history) == len(self.evolution_strategy.sigma_history))
        assert(len(self.evolution_strategy.mean_history) == len(self.evolution_strategy.covariance_history))
        assert(len(self.evolution_strategy.mean_history) == len(self.evolution_strategy.samples_history))
        return len(self.evolution_strategy.mean_history)

    def __len__(self) -> int:
        """
        Returns the number of generations of the evoution strategy, same as `plotter.number_of_generations`
        """
        return self.number_of_generations

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

    def __init_subplots(self) -> tuple[Figure, Axes]:
        """
        Initializes the subplots of the plotter 
        """
        figure, (axes1, axes2) = plt.subplots(**self.subplots_settings)
        axes1.set_xlim(self.lbounds[0], self.ubounds[0])
        axes1.set_ylim(self.lbounds[1], self.ubounds[1])
        axes2.set_xlim(self.lbounds[0], self.ubounds[0])
        axes2.set_ylim(self.lbounds[1], self.ubounds[1])
        axes1.set_xlabel(r"$x_1$")
        axes2.set_xlabel(r"$x_1$")
        axes1.set_ylabel(r"$x_2$")
        axes2.set_ylabel(r"$x_2$")
        return figure, (axes1, axes2)    
    
    def __best_decision_vector_pixel(self) -> tuple[int]:
        """
        Returns the coordinate on the grid of the best decision vector, at the center by default
        """
        assert(self.n_points % 2 == 1), "You need to set a odd number of points"
        return self.n_points//2, self.n_points//2
#"""
#    def __set_yticks(self) -> None:
#        """
#        Computes the labels of the y-axis
#        """
#        diff = np.max(self.y_axis) - np.min(self.y_axis)
#        ticks = np.arange(np.min(self.y_axis), np.max(self.y_axis), diff / self.n_ticks)
#        self.axis.set_yticks(ticks)
#        _locator= MaxNLocator(nbins=self.n_ticks)
#        self.axis.yaxis.set_major_locator(_locator)
#        self.axis.set_yticklabels(ticks)
#
#    def __set_xticks(self) -> None:
#        """
#        Computes the labels of the x-axis
#        """
#        diff = np.max(self.x_axis) - np.min(self.x_axis)
#        ticks = np.arange(np.min(self.x_axis), np.max(self.x_axis), diff / self.n_ticks)
#        print(ticks)
#        self.axis.set_xticks(ticks)
#        _locator= MaxNLocator(nbins=self.n_ticks)
#        self.axis.xaxis.set_major_locator(_locator)
#       self.axis.set_xticklabels(ticks)

    def __scale_zvalues(self) -> np.ndarray:
        """
        Min-Max scaling for plotting the z-values as an image
        """
        scaler = MinMaxScaler()
        return scaler.fit_transform(self.z_values)
    
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
        x_coord, = np.where(x_condition, )
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
  
    def __plot_point_by_coord(self, point_pixel_coordinates, *args, **kwargs) -> None:
        """
        Plots a poit given its pixel coordinates
        """
        self.axes1.plot(*point_pixel_coordinates, *args, **kwargs)
        self.axes2.plot(*point_pixel_coordinates, *args, **kwargs)

    def compute_mutation_probability_map(self, generation_idx) -> np.ndarray:
        """
        Argument:

            generation_idx: The id of the generation of which to plot the mutation probability density

        Computes the mutation probability map on the grid for a given generation
        """
        return self.__compute_mutation_probability_map(generation_idx)


    def find_decision_vector_pixel_coordinate(self, decision_vector) -> tuple[int]:
        """
        Finds the pixels of a point in the grid
        """
        return self.__find_decision_vector_pixel_coordinate(decision_vector)

    def contourplot(self) -> None:
        """
        Creates the contour plot of the z-values
        """
        if self.n_levels and "levels" not in self.contour_plot_settings.keys(): 
            interval = np.linspace(0.0, 1.0, num=self.n_levels)**2
            levels = (self.z_values.max() - self.problem.best_value) * interval + self.problem.best_value
            self.contour_plot_settings["levels"] = levels
        elif self.n_levels and "levels" in self.contour_plot_settings.keys():
            UserWarning("Both self.n_levels and the key pair 'levels':levels in self.contour_plot_settings are defined, using the latter as default")
        elif not self.n_levels and "levels" not in self.contour_plot_settings.keys():
            UserWarning("No levels set, setting the number of levels by default to 15")
            interval = np.linspace(0.0, 1.0, num=self.n_levels)**2
            levels = (self.z_values.max() - self.problem.best_value) * interval + self.problem.best_value
            self.contour_plot_settings["levels"] = levels
        self.axis.contourf(self.z_values, **self.contour_plot_settings)

    def plot_offsprings(self, generation_idx) -> None:
        offsprings = self.evolution_strategy.samples_history[generation_idx]
        for offspring in offsprings:
            _coord =self.__find_decision_vector_pixel_coordinate(offspring)
            self.__plot_point_by_coord(_coord, "ro")

    def plot_best_decision_value(self) -> None:
        """
        Plots the best decision value
        """
        best_decision_value_coordinates = self.__best_decision_vector_pixel()
        self.__plot_point_by_coord(best_decision_value_coordinates, "g*")

    def plot_point(self, point, *args, **kwargs) -> None:
        """
        Plots a point
        """
        point_coordinates = self.__find_decision_vector_pixel_coordinate(point)
        self.__plot_point_by_coord(point_coordinates, *args, **kwargs)

    def imshow(self) -> None:
        """
        Scales the z-values and shows them as an image
        """
        z = self.__scale_zvalues()
        self.axis.imshow(z)
    
    def mutation_probability_plot(self, generation_idx) -> None:
        """
        Creates the mutation probability map image
        """
        mutation_probability_map = self.__compute_mutation_probability_map(generation_idx)
        if self.n_levels and "levels" not in self.probability_map_settings.keys(): 
            levels = np.linspace(np.min(mutation_probability_map), np.max(mutation_probability_map), num=self.n_levels)**2
            self.contour_plot_settings["levels"] = levels
        elif self.n_levels and "levels" in self.probability_map_settings.keys():
            UserWarning("Both self.n_levels and the key pair 'levels':levels in self.probability_map_settings are defined, using the latter as default")
        elif not self.n_levels and "levels" not in self.probability_map_settings.keys():
            UserWarning("No levels set, setting the number of levels by default to 15")
            levels = np.linspace(np.min(mutation_probability_map), np.max(mutation_probability_map), num=self.n_levels)**2
            self.contour_plot_settings["levels"] = levels
        self.axis.contourf(mutation_probability_map, **self.probability_map_settings)
        
    def reset_subplots(self) -> None:
        """
        Resets the subplots of the plotter
        """
        self.fig, (self.axes1, self.axes2) = self.__init_subplots()
        #self.__set_xticks()
        #self.__set_yticks() 