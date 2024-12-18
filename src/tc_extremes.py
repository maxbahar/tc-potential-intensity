# Dependencies
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tropycal.tracks as tracks

from matplotlib.axes import Axes
from scipy.stats import genextreme, rankdata, kstest
from typing import Union, Any

class TCExtremes:
    """
    Class that examines storm data from HURDAT2 and IBTrACS using extreme value theory.

    Attributes:
        ocean_basin (str): The ocean basin for which storm data is fetched. Possible values are:
            - `'north_atlantic'`
            - `'east_pacific'`
            - `'both'`
            - `'west_pacific'`
            - `'north_indian'`
            - `'south_indian'`
            - `'australia'`
            - `'south_pacific'`
            - `'south_atlantic'`
            - `'all'`

            Defaults to `None` until set by the `fetch` method.

        basin_dataset (tropycal.tracks.TrackDataset): The dataset containing storm track information for the specified basin.
            Initialized by the `fetch` method.

        storm_data (pd.DataFrame): The pandas DataFrame containing fetched storm data.
            Populated by the `fetch` method.

        ev_params_ (tuple or dict): The parameters of the fitted Generalized Extreme Value (GEV) distribution.
            Set by the `fit` method.
            - If `method` is `'all'`, `ev_params_` is a tuple `(shape, loc, scale)`.
            - If `method` is `'season'`, `ev_params_` is a dictionary mapping each season to a tuple `(shape, loc, scale)`.
    """

    def __init__(self):
        """Initialize the attributes of the class instance."""
        # Initialize all attributes to None
        self.ocean_basin = None
        self.basin_dataset = None
        self.storm_data = None
        self.ev_params_ = None

        # Hidden variables for internal use
        self._available_basins = ['north_atlantic','east_pacific','both',
                                  'west_pacific','north_indian','south_indian',
                                  'australia','south_pacific','south_atlantic','all']
        self._available_vars = ['vmax','mslp']
        self._var_labels = {
            'vmax': 'Max. Wind Speed (in knots)',
            'mslp': 'Min. Central Pressure (in millibars)'
        }

    def fetch(self, start_year: int = None, end_year:int = None, ocean_basin: str = 'north_atlantic') -> pd.DataFrame:
        """
        Fetches storm data for the specified timeframe and location.

        Args:
            start_year (int, optional): The starting year (inclusive). Defaults to `None`, fetching all available storms.
            end_year (int, optional): The ending year (inclusive). Defaults to `None`, fetching all available storms.
            ocean_basin (str, optional): The ocean basin to fetch data for. Possible values are:
                - `'north_atlantic'`
                - `'east_pacific'`
                - `'both'`
                - `'west_pacific'`
                - `'north_indian'`
                - `'south_indian'`
                - `'australia'`
                - `'south_pacific'`
                - `'south_atlantic'`
                - `'all'`
                Defaults to `'north_atlantic'`.
        
        Returns:
            pd.DataFrame: Pandas DataFrame containing storm data for the specified time and location.

        Raises:
            ValueError: If the specified `ocean_basin` is not supported.
        """
        # Specify the ocean basin, check for valid values
        ocean_basin = ocean_basin.lower()
        if not ocean_basin in self._available_basins:
            raise ValueError(f"Specified ocean basin '{ocean_basin}' is not supported. Please use one of the following values: {self._available_basins}")
        source = 'hurdat' if ocean_basin in ['north_atlantic','east_pacific','both'] else 'ibtracs'

        # If the basin is the same, no need to re-initialize the dataset
        if ocean_basin != self.ocean_basin:
            self.ocean_basin = ocean_basin
            self.basin_dataset = tracks.TrackDataset(basin=ocean_basin, source=source)

        # Filter storms from the dataset according to the timeframe
        year_range = None if start_year is None or end_year is None else (start_year, end_year)
        self.storm_data = self.basin_dataset.filter_storms(year_range=year_range, return_keys=False)

        # Take unique values from each storm, maximum of vmax, minimum of mslp
        self.storm_data = self.storm_data.groupby('stormid').agg({'season':'first','type':'first','vmax':'max','mslp':'min'})
        self.storm_data['basin'] = ocean_basin
        self.storm_data['source'] = source

        return self.storm_data
    
    def fit(self, storm_data: pd.DataFrame = None, variable: str = 'vmax', fit_window: int = 0
        ) -> Union[tuple[float, float, float], dict[tuple[int,int]: tuple[float, float, float]]]:
        """
        Fits `storm_data` to the Generalized Extreme Value (GEV) distribution.

        Args:
            storm_data (pd.DataFrame, optional): 
                A pandas DataFrame containing the storm data to fit. 
                Defaults to `None`, which uses the fetched data stored in the class attribute `storm_data`.
            variable (str, optional): 
                The variable in the DataFrame to fit to. Possible values are:

                - `'vmax'`: maximum wind speeds.
                - `'mslp'`: minimum central pressure. 
                    Fitting to `mslp` requires flipping the data to accurately model the behavior of the tail.

                Defaults to `'vmax'`.
            fit_window (int, optional): 
                How many years/seasons of data to use when fitting. 
                Defaults to 0, fit to the entire dataset.

        Returns:
            tuple[float, float, float] or dict[tuple[int,int]: tuple[float, float, float]]: 
                - If `fit_window` is `0`, returns a tuple `(shape, loc, scale)`.
                - If `fit_window` is `>0`, returns a dictionary mapping each season window to a tuple `(shape, loc, scale)`.

        Raises:
            ValueError: If `variable` is not supported or `fit_window` is negative or not an int.
        """
        # If no dataframe was passed, use the class attribute
        if storm_data is None:
            storm_data = self.storm_data

        # Check if fit_window is valid
        if not isinstance(fit_window, int):
            raise ValueError(f'The fit_window {fit_window} is not supported. Please specify a positive integer.')
        elif fit_window < 0:
            raise ValueError(f'The fit_window {fit_window} is not supported. Please specify a positive integer.')

        # Check if variable is valid
        if not variable in self._available_vars:
            raise ValueError(f'The variable {variable} is not supported. Please use one of the following values {self._available_vars}')
        
        # If variable passed is minimum central pressure, transform data
        transform = -1 if variable == 'mslp' else 1

        # Warn user if missing data was detected and drop missing values        
        if storm_data[variable].isna().any():
            print(f'Warning: Missing values were detected in the variable `{variable}`. Missing values will be dropped.')

        # Fit data differently according to fit_window passed
        if fit_window == 0:
            params = genextreme.fit(storm_data[variable].dropna() * transform, 
                                    loc=storm_data[variable].dropna().mean() * transform, 
                                    scale=storm_data[variable].dropna().std())
            self.ev_params_ = (params[0], params[1] * transform, params[2])
        else:
            self.ev_params_ = dict()
            earliest_year = storm_data['season'].min()
            latest_year = storm_data['season'].max()
            for season in range(latest_year,earliest_year-1,-fit_window):
                # Filter data for the current season
                season_data = storm_data[storm_data['season'].isin(range(season-fit_window+1, season+1))][variable].dropna()
                if season_data.empty:
                    continue  # Skip if there's no data for this season
                params = genextreme.fit(season_data * transform, 
                                        loc=season_data.mean() * transform, 
                                        scale=season_data.std())
                self.ev_params_[(int(season-fit_window+1),int(season+1))] = (params[0], params[1] * transform, params[2])
        
        return self.ev_params_
    
    def evaluate(self, storm_data: pd.DataFrame = None, variable: str = 'vmax', ev_params = None) -> dict:
        """Evaluates the fit of the GEV distribution to the real data by comparing 
        GEV probabilities to the empirical data distribution.

        Args:
            storm_data (pd.DataFrame, optional): The storm data to evaluate. 
                Defaults to the class attribute `storm_data`.
            variable (str, optional): The variable to evaluate. Defaults to `'vmax'`.
            ev_params: The GEV parameters to use for scoring.
                - If array-like, should be an array like of `(shape, loc, scale)`.
                - If a dict, should be `{season: (shape, loc, scale)}`.

        Returns:
            dict: A dictionary containing the empirical and GEV probabilities for comparison.
        """
        # Use class data if no storm_data is provided
        if storm_data is None:
            storm_data = self.storm_data
        
        # Check if variable is valid
        if not variable in self._available_vars:
            raise ValueError(f'The variable {variable} is not supported. Please use one of the following values {self._available_vars}')

        # Ensure the GEV parameters are available
        if ev_params is None:
            ev_params = self.ev_params_

        results = {}

        # Check if `ev_params_` is a dictionary (seasonal fit) or a tuple (all fit)
        if isinstance(ev_params, dict):
            # Evaluate for each season
            for season, params in ev_params.items():
                shape, loc, scale = params
                # Filter data for the current season
                season_data = storm_data[storm_data['season'].isin(range(*season))][variable].dropna()
                if season_data.empty:
                    continue  # Skip if there's no data for this season
                sorted_data = np.sort(season_data)
                gev_probs = genextreme.cdf(sorted_data, shape, loc=loc, scale=scale)
                empirical_probs = rankdata(sorted_data) / len(sorted_data)
                results[season] = {'empirical_probs': empirical_probs, 'gev_probs': gev_probs}
            return results
        else:
            # Use single set of parameters for all data
            shape, loc, scale = ev_params
            sorted_data = np.sort(storm_data[variable].dropna())
            gev_probs = genextreme.cdf(sorted_data, shape, loc=loc, scale=scale)
            empirical_probs = rankdata(sorted_data) / len(sorted_data)
            return {'empirical_probs': empirical_probs, 'gev_probs': gev_probs}
        
    def score(self, storm_data: pd.DataFrame = None, variable: str = 'vmax', ev_params = None) -> Union[dict, Any]:
        """Scores the fit of the data to the GEV distribution with the Kolmogorov-Smirnov test.

        Args:
            storm_data (pd.DataFrame, optional): The storm data to score against. 
                Defaults to the class attribute `storm_data`.
            variable (str): The variable to score. Defaults to `'vmax'`.
            ev_params: The GEV parameters to use for scoring.
                - If array-like, should be an array like of `(shape, loc, scale)`.
                - If a dict, should be `{season: (shape, loc, scale)}`.

        Returns:
            KstestResult: An object with the following attributes:
                - statistic (float): KS test statistic.
                - pvalue (float): One-tailed or two-tailed p-value.
                - statistic_location (float): The distance between the empirical distribution function and hypothesized cumulative distribution function is measured at this observation.
                - statistic_sign (int):
                
        Raises:
            ValueError: If `variable` is not supported.
        """
        # Use class data if no storm_data is provided
        if storm_data is None:
            storm_data = self.storm_data
        
        # Check if variable is valid
        if not variable in self._available_vars:
            raise ValueError(f'The variable {variable} is not supported. Please use one of the following values {self._available_vars}')

        # If no extreme value parameters were passed, use the class attribute
        if ev_params is None:
            ev_params = self.ev_params_
    
        # If variable passed is minimum central pressure, transform data
        transform = -1 if variable == 'mslp' else 1

        # Warn user if missing data was detected and drop missing values        
        if storm_data[variable].isna().any():
            print(f'Warning: Missing values were detected in the variable `{variable}`. Missing values will be dropped.')

        # Check if `ev_params` is a dictionary (seasonal fit) or a tuple (all fit)
        if isinstance(ev_params, dict):
            results = {}
            for season, params in ev_params.items():
                season_data = storm_data[storm_data['season'].isin(range(*season))]
                ks_result = kstest(season_data[variable].dropna() * transform, genextreme(params[0], params[1] * transform, params[2]).cdf)
                ks_result.statistic_location = ks_result.statistic_location * transform
                results[season] = ks_result
            return results
        else:
            ks_result = kstest(storm_data[variable].dropna() * transform, genextreme(ev_params[0], ev_params[1] * transform, ev_params[2]).cdf)
            ks_result.statistic_location = ks_result.statistic_location * transform
            return ks_result
        
    def _plot_helper(self, storm_data: pd.DataFrame, variable: str, ev_params, 
                     show_plot: bool, plot_path: str) -> Axes:
        """Helper method that plots the fit of the data for a single set of extreme value distribution parameters.
        
        Args:
            storm_data (pd.DataFrame): The storm data to plot.
            variable (str): The variable to plot.
            ev_params (array-like): The GEV parameters to use for fitting, an array like of `(shape, loc, scale)`.

        Returns:
            Axes: The matplotlib Axes object with the plot.
        """
        # If variable passed is minimum central pressure, transform data
        transform = -1 if variable == 'mslp' else 1

        # Initialize plotting variables
        plot_padding = storm_data[variable].std()
        plot_range = np.array([max(0,storm_data[variable].min() - plot_padding), storm_data[variable].max() + plot_padding]) // 10 * 10
        print(plot_range)
        plot_vals = np.linspace(*plot_range, 250)
        season_start = storm_data['season'].min()
        season_end = storm_data['season'].max()
        plot_label = f'Empirical data\n({season_start}-{season_end})' if season_start != season_end else f'{self._var_labels[variable]}\n(Data from {season_start})'
        gev = genextreme(ev_params[0],ev_params[1]*transform,ev_params[2])

        basin_val = storm_data['basin'].iloc[0]
        if basin_val == 'all':
            basin_title = 'all basins'
        elif basin_val == 'both':
            basin_title = 'the North Atlantic and East Pacific basins'
        else:
            basin_title = f'the {basin_val.replace("_", " ").title()} basin'
        source_title = 'HURDAT2' if storm_data['source'].iloc[0] == 'hurdat' else 'IBTrACS'
        plot_title = f'Storms recorded in {basin_title}\nn = {len(storm_data.index)}, {source_title}'

        # PLot the data with seaborn and matplotlib
        fig, ax = plt.subplots()
        sns.histplot(data=storm_data[variable], 
                     stat='density', bins=np.arange(*plot_range,10), alpha=0.5, 
                     color=f'C{self._available_vars.index(variable)}',
                     label=plot_label)
        sns.lineplot(x=plot_vals, y=gev.pdf(plot_vals* transform), 
                     color='k', linestyle='--', 
                     label=f'Fitted GEV c = {ev_params[0]:.2f}')
        plt.xlim(*plot_range)
        plt.xlabel(f'{self._var_labels[variable]}')
        plt.title(plot_title)
        plt.grid(axis='y',alpha=0.3)

        # Show plot or save if needed
        if plot_path:
            plt.savefig(plot_path)
        if show_plot:
            plt.show()

        return ax   
    
    def plot_fit(self, storm_data: pd.DataFrame = None, variable: str = 'vmax', ev_params = None,
                 show_plot: bool = True, plot_path: str = '') -> Union[Axes, list[Axes]]:
        """
        Plots the fit of the data.
        
        Args:
            storm_data (pd.DataFrame, optional): The storm data to plot. Defaults to the class attribute `storm_data`.
            variable (str): The variable to plot. Defaults to `'vmax'`.
            ev_params: The GEV parameters to use for fitting.
                
                - If array-like, should be an array like of `(shape, loc, scale)`.
                - If a dict, should be `{season: (shape, loc, scale)}`.
                
                Defaults to `None`, which uses the fitted parameters stored in the class attribute `ev_params_`.
            show_plot (bool): Whether to display the plot(s). Defaults to `True`.
            plot_path (str): Path to save the plot(s). If plotting multiple seasons, provide a directory path.
                Defaults to `''` (does not save).

        Returns:
            Axes or list[Axes]: The matplotlib Axes object(s) with the plot(s).

        Raises:
            ValueError: If `variable` is not supported.
            TypeError: If `ev_params` is not a tuple, list, or dict.
        """
        # If no dataframe was passed, use the class attribute
        if storm_data is None:
            storm_data = self.storm_data

        # If no extreme value parameters were passed, use the class attribute
        if ev_params is None:
            ev_params = self.ev_params_

        # Check if variable is valid
        if not variable in self._available_vars:
            raise ValueError(f'The variable {variable} is not supported. Please use one of the following values {self._available_vars}')

        # Validate plot_path
        if plot_path:
            # If plotting multiple sets of params, make sure plot_path is a directory
            if isinstance(ev_params, dict): 
                if not os.path.exists(plot_path):
                    os.makedirs(plot_path, exist_ok=True)  
                elif not os.path.isdir(plot_path):
                    raise ValueError("For multiple plots, plot_path must be a valid directory.")
            # If plotting the fit for a single set of params, check if directory is valid
            else:
                directory = os.path.dirname(plot_path)
                if directory and not os.path.exists(directory):
                    os.makedirs(directory, exist_ok=True)

        # If plotting multiple sets of params, call _plot_helper function multiple times
        if isinstance(ev_params, dict):
            axes = []
            for season, season_params in ev_params.items():
                current_plot_path = f'{plot_path}/{season}.png' if plot_path else ''
                season_data = storm_data[storm_data['season'].isin(range(*season))]
                if season_data.empty:
                    continue 
                axes.append(self._plot_helper(season_data, variable, season_params, show_plot, current_plot_path))
            return axes
        else:
            # Attempt to read params as array like
            ev_params = np.asarray(ev_params, dtype=float)
            if ev_params.shape[0] != 3:
                raise TypeError('ev_params must be a tuple or list of (shape, loc, scale) or a dictionary of season: (shape, loc, scale) key, value pairs.')
            return self._plot_helper(storm_data, variable, ev_params, show_plot, plot_path)   

    def plot_agg(self, storm_data: pd.DataFrame = None, variable: str = 'vmax', window: int = 10, min_periods: int = None, 
                 show_plot: bool = True, plot_path: str = '') -> Axes:
        """Plots the moving average of the annual maximum or minimum of the specified variable over time.
        
        Args:
            storm_data (pd.DataFrame, optional): The storm data to plot. Defaults to the class attribute `storm_data`.
            variable (str): The variable to plot. Defaults to `'vmax'`.
            window (int): Number of years to consider for moving average. Defaults to `10`.
            min_periods (int): Minimum number of years required to have valid values, result is otherwise np.nan. Defaults to `None`, the size of the window.
            show_plot (bool): Whether to display the plot(s). Defaults to `True`.
            plot_path (str): Path to save the plot(s). If plotting multiple seasons, provide a directory path.
                Defaults to `''` (does not save).
        
        Returns:
            Axes: The matplotlib Axes object with the plot.

        Raises:
            ValueError: If `variable` is not supported.
        """
        # If no dataframe was passed, use the class attribute
        if storm_data is None:
            storm_data = self.storm_data

        # Check if variable is valid
        if not variable in self._available_vars:
            raise ValueError(f'The variable {variable} is not supported. Please use one of the following values {self._available_vars}')

        # Calculate maximum or minimum over entire dataset, use moving average for smoothing
        if variable == 'mslp':
            y_label_prefix = 'Season Min. of'
            agg_data = storm_data[[variable,'season']].groupby('season')[variable].min()
        else:
            y_label_prefix = 'Season Max. of'
            agg_data = storm_data[[variable,'season']].groupby('season')[variable].max()
        plot_data = agg_data.rolling(window=window, min_periods=min_periods, center=True).mean()

        # Initialize plotting variables
        basin_val = storm_data['basin'].iloc[0]
        if basin_val == 'all':
            basin_title = 'all basins'
        elif basin_val == 'both':
            basin_title = 'the North Atlantic and East Pacific basins'
        else:
            basin_title = f'the {basin_val.replace("_", " ").title()} basin'
        source_title = 'HURDAT2' if storm_data['source'].iloc[0] == 'hurdat' else 'IBTrACS'
        plot_title = f'Storms recorded in {basin_title}\nn = {len(storm_data.index)}, {source_title}'

        # Plot the data
        fig, ax = plt.subplots()
        sns.lineplot(data=agg_data, 
                     label='Raw data', 
                     color=f'C{self._available_vars.index(variable)}', 
                     alpha=0.3)
        sns.lineplot(data=plot_data, 
                     color='k',
                     label=f'{window}-year moving average')
        plt.xlabel('Year')
        plt.ylabel(f'{y_label_prefix} {self._var_labels[variable]}')
        plt.title(plot_title)

        # Show plot or save if needed
        if plot_path:
            plt.savefig(plot_path)
        if show_plot:
            plt.show()

        return ax

    def plot_dist(self, storm_data: pd.DataFrame = None, variable: str = 'vmax', window: int = 5,
                  show_plot: bool = True, plot_path: str = '') -> Axes:
        """Plots the kernel density of the data over specified time periods.

        Args:
            storm_data (pd.DataFrame, optional): The storm data to plot. Defaults to the class attribute `storm_data`.
            variable (str): The variable to plot. Defaults to `'vmax'`.
            window (int): The number of years to group for each time window. Defaults to `5`.
            show_plot (bool): Whether to display the plot(s). Defaults to `True`.
            plot_path (str): Path to save the plot(s). If plotting multiple seasons, provide a directory path.
                Defaults to `''` (does not save).
        
        Returns:
            Axes: The matplotlib Axes object with the plot.

        Raises:
            ValueError: If `variable` is not supported.
        """
        # If no dataframe was passed, use the class attribute
        if storm_data is None:
            storm_data = self.storm_data

        # Check if variable is valid
        if not variable in self._available_vars:
            raise ValueError(f'The variable {variable} is not supported. Please use one of the following values {self._available_vars}')

        # Initialize plotting variables
        basin_val = storm_data['basin'].iloc[0]
        if basin_val == 'all':
            basin_title = 'all basins'
        elif basin_val == 'both':
            basin_title = 'the North Atlantic and East Pacific basins'
        else:
            basin_title = f'the {basin_val.replace("_", " ").title()} basin'
        source_title = 'HURDAT2' if storm_data['source'].iloc[0] == 'hurdat' else 'IBTrACS'
        plot_title = f'Storms recorded in {basin_title}\nn = {len(storm_data.index):,d}, {source_title}\nData grouped into {window} year windows'

        # Group data
        grouped_data = storm_data.copy() # Avoid inplace changes

        for season in range(grouped_data['season'].max(),grouped_data['season'].min()-1,-window):
                # Filter data for the current group
                grouped_data.loc[storm_data['season'].isin(range(season-window+1, season+1)), 'bin_end'] = season+1

        # Plot the data
        fig, ax = plt.subplots()
        sns.kdeplot(data=grouped_data,x=variable,hue='bin_end',alpha=0.7,palette='viridis_r',common_norm=False,legend=False,ax=ax)

        # Definte color map
        norm = plt.Normalize(vmin=min(grouped_data['season']), vmax=max(grouped_data['season']))
        sm = plt.cm.ScalarMappable(cmap="viridis_r", norm=norm)
        sm.set_array([])

        # Add color bar with appropriate label
        cbar = plt.colorbar(sm,ax=ax)
        cbar.set_label('Year')

        plt.xlabel(f'{self._var_labels[variable]}')
        plt.title(plot_title)
        
        # Show plot or save if needed
        if plot_path:
            plt.savefig(plot_path)
        if show_plot:
            plt.show()

        return ax
    

    def plot_time(self, storm_data: pd.DataFrame = None, variable: str = 'vmax', 
                  ev_params: dict[tuple[int,int]: tuple[float, float, float]] = None,
                  show_plot: bool = True, plot_path: str = '') -> Axes:
        """Plots the empirical mean and standard deviation and the parameters of the fitted GEV distribution over time.

        Args:
            storm_data (pd.DataFrame, optional): The storm data to plot. Defaults to the class attribute `storm_data`.
            variable (str): The variable to plot. Defaults to `'vmax'`.
            ev_params (dict): The GEV parameters to use for fitting, should be `{season: (shape, loc, scale)}`.
                Defaults to `None`, which uses the fitted parameters stored in the class attribute `ev_params_`.
            show_plot (bool): Whether to display the plot(s). Defaults to `True`.
            plot_path (str): Path to save the plot(s). If plotting multiple seasons, provide a directory path.
                Defaults to `''` (does not save).
        
        Raises:
            ValueError: If `variable` is not supported.
            TypeError: If `ev_params` is not a dictionary.
        """
        # If no dataframe was passed, use the class attribute
        if storm_data is None:
            storm_data = self.storm_data

        # Ensure the GEV parameters are available
        if ev_params is None:
            ev_params = self.ev_params_

        # Check if variable is valid
        if not variable in self._available_vars:
            raise ValueError(f'The variable {variable} is not supported. Please use one of the following values {self._available_vars}')

        # Check if ev_params is a dictionary
        if not isinstance(ev_params, dict):
            raise TypeError('Parameters passed must be a dictionary for multiple time periods. Please call fit() with fit_window > 0.')

        # Initialize resutls
        res = pd.DataFrame(ev_params).transpose().rename({0:'shape',1:'loc',2:'scale'},axis=1)
        ks_all = self.score(storm_data, variable=variable, ev_params=ev_params)
        ks_stats = {season: [ks_res.statistic, ks_res.statistic_location, ks_res.pvalue, ks_res.statistic_sign] for season, ks_res in ks_all.items()}
        ks_stats = pd.DataFrame(ks_stats).transpose().rename({0:'statistic',1:'stat_loc',2:'p_value', 3:'sign'},axis=1)
        
        # Group data for plotting
        grouped_data = storm_data.copy()
        window = list(ev_params.keys())[0][1] - list(ev_params.keys())[0][0]
        for season in range(grouped_data['season'].max(),grouped_data['season'].min()-1,-window):
                # Filter data for the current group
                grouped_data.loc[storm_data['season'].isin(range(season-window+1, season+1)), 'bin_end'] = season+1
        data_mean = grouped_data.groupby('bin_end')[variable].mean()
        data_std = grouped_data.groupby('bin_end')[variable].std()

        # Initialize plotting variables
        basin_val = storm_data['basin'].iloc[0]
        if basin_val == 'all':
            basin_title = 'all basins'
        elif basin_val == 'both':
            basin_title = 'the North Atlantic and East Pacific basins'
        else:
            basin_title = f'the {basin_val.replace("_", " ").title()} basin'
        source_title = 'HURDAT2' if storm_data['source'].iloc[0] == 'hurdat' else 'IBTrACS'
        plot_title = f'{self._var_labels[variable]}\nStorms recorded in {basin_title}\nn = {len(storm_data.index):,d}, {source_title}\nData grouped into {window} year windows'

        # Plot the fitted parameters and empirical data
        fig, (ax1,ax2,ax3,ax4) = plt.subplots(nrows=4, figsize=(10,10))

        mask_res = res.index.to_series().apply(lambda x: x[0] >= 0)
        masked_res = res.loc[mask_res]
        masked_res['year'] = masked_res.index.to_series().apply(lambda x: x[1])

        mask_ks = ks_stats.index.to_series().apply(lambda x: x[0] >= 0)
        masked_ks = ks_stats.loc[mask_ks]
        masked_ks['year'] = masked_ks.index.to_series().apply(lambda x: x[1])

        sns.lineplot(data=masked_res ,x='year', y='loc', alpha=0.7, color='C2', marker='o', label='Fitted values', ax=ax1)
        sns.lineplot(data=data_mean, alpha=0.7, color='darkgreen', marker='D', label='Empirical data', ax=ax1)
        ax1.grid(alpha=0.3)
        ax1.set_ylabel('Mean', fontsize=14)
        ax1.set_xlabel('')

        sns.lineplot(data=masked_res ,x='year', y='scale', alpha=0.7, color='C3', marker='o', label='Fitted values', ax=ax2)
        sns.lineplot(data=data_std, alpha=0.7, color='darkred', marker='D', label='Empirical data', ax=ax2)
        ax2.grid(alpha=0.3)
        ax2.set_ylabel('Standard Deviation', fontsize=14)
        ax2.set_xlabel('')

        sns.lineplot(data=masked_res ,x='year', y='shape', alpha=0.7, color='C4', marker='o', ax=ax3)
        ax3.grid(alpha=0.3)
        ax3.set_ylabel('Shape Parameter', fontsize=14)
        ax3.set_xlabel('')

        sns.lineplot(data=masked_ks ,x='year', y='statistic', alpha=0.7, color='C5', marker='o', ax=ax4)
        ax4.grid(alpha=0.3)
        ax4.set_ylabel('KS Test Statistic', fontsize=14)
        ax4.set_xlabel('Year',fontsize=14)

        plt.suptitle(plot_title, fontsize=14)
        plt.tight_layout()
        
        # Show plot or save if needed
        if plot_path:
            plt.savefig(plot_path)
        if show_plot:
            plt.show()

        return [ax1,ax2,ax3,ax4]