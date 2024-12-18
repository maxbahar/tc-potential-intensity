# Dependencies
import cartopy.crs as ccrs
import geopandas as gpd
import intake
import intake_esm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rioxarray
import xarray as xr

from dask.distributed import Client, LocalCluster
from matplotlib.axes import Axes
from shapely.affinity import translate
from typing import Any, Sequence
from tcpyPI import pi

# Global constant variables
CMIP6_URL = "https://storage.googleapis.com/cmip6/pangeo-cmip6.json"

class DistributedManager:
    """
    Manages the Dask distributed client to enable parallel computing.

    This class provides an interface for initializing and shutting down a 
    Dask distributed client for parallelized computations.

    Attributes:
        cluster (dask.distributed.LocalCluster): Local Dask cluster instance.
        client (dask.distributed.Client): Dask client instance for managing distributed tasks.
    """
    def __init__(self, n_workers: int = None, threads_per_worker: int = None):
        """
        Initialize the distributed computing client.

        Args:
            n_workers (int, optional): Number of workers (default: determined by Dask).
            threads_per_worker (int, optional): Threads per worker (default: determined by Dask).
        """
        self.cluster = LocalCluster(n_workers=n_workers, threads_per_worker=threads_per_worker)
        self.client = Client(self.cluster)
        print(f"Dask Client initialized: {self.client}")

    def close(self):
        """
        Shuts down the Dask client and cluster.

        This method safely closes the Dask cluster and client resources.
        """
        self.client.close()
        self.cluster.close()
        print("Dask Client shut down.")

class CMIP6DataFetcher:
    """
    Fetches and queries climate data from the CMIP6 dataset.

    Provides tools for querying and retrieving CMIP6 datasets for
    specific institutions, experiments, and climate variables.
    """

    def __init__(self, catalog_url=CMIP6_URL):
        """
        Initializes the CMIP6 data fetcher with the specified catalog URL.

        Args:
            catalog_url (str): URL to the CMIP6 data catalog. Defaults to a globally accessible catalog.
        """
        self.catalog = intake.open_esm_datastore(catalog_url)

    def fetch_data(
        self, 
        institution_id: str = None,
        source_id: str = None, 
        experiment_id: str = None, 
        member_id: str = None,
        grid_label: str = None,
        variable_id: Sequence[str] = ["ts","psl","ta","hus"], 
        table_label: str = "Amon",
        return_options_only: bool = False,
        set_attr: bool = True,
        **data_kwargs
    ) -> dict[str, xr.Dataset] | pd.DataFrame:
        """
        Fetches monthly mean climate data from the CMIP6 dataset.

        Args:
            institution_id (str, optional): Institution responsible for the dataset.
            source_id (str, optional): Climate model identifier.
            experiment_id (str, optional): Experiment configuration (e.g., "historical", "ssp119").
            member_id (str, optional): Ensemble member identifier (e.g. "r1i1p3f1").
            grid_label (str, optional): Type of model grid (e.g. "gn", "gr"). 
            variable_id (Sequence[str], optional): List of variable IDs to retrieve. Defaults to `["ts", "psl", "ta", "hus"]`.
            table_label (str, optional): Source table for the variables. Defaults to `"Amon"`.
            return_options_only (bool, optional): If `True`, returns a DataFrame of dataset options. Defaults to `False`.
            set_attr (bool, optional): If `True`, saves the query as an attribute. Defaults to `True`.
            **data_kwargs: Additional filtering arguments for the query.

        Returns:
            dict[str, xr.Dataset] | pd.DataFrame:
                - If `return_options_only=True`: Returns a DataFrame of available dataset options.
                - If `return_options_only=False`: Returns a dictionary of xarray Datasets for the selected variables.
        """
        # Build the search query
        query_kwargs = {
            "institution_id": institution_id,
            "source_id": source_id,
            "experiment_id": experiment_id,
            "member_id": member_id,
            "grid_label": grid_label,
            "variable_id": variable_id,
            "table_id": table_label,
            **data_kwargs
        }
        # Filter out None values
        query_kwargs = {k: v for k, v in query_kwargs.items() if v is not None}
        
        # Query the catalog
        query = self.catalog.search(**query_kwargs)

        if set_attr:
            self.query = query

        if return_options_only:
            return query.df
        else:
            return query.to_dataset_dict(xarray_open_kwargs={"consolidated": True})

class PIAnalysis:
    """
    Manages the calculation, analysis, and visualization of tropical cyclone potential intensity (PI).
    """

    def __init__(self) -> None:
        """
        Initializes the analysis class with empty datasets and variables.
        """
        self.env_ds = None
        self.pi_ds = None
        self.mean_df = None

    def set_data(
        self,
        env_ds: xr.Dataset = None,
        year_mon_limits: Sequence[str] = [None, None],
        month_limits: Sequence[int] = [None, None],
        lat_limits: Sequence[float] = [None, None],
        lon_limits: Sequence[float] = [None, None],
        geo_limits: gpd.GeoDataFrame = None,
        geo_limits_touching: bool = False,
        convert_vars: bool = True
    ) -> Any: 
        """
        Configures environmental data for PI analysis by applying spatial and temporal limits.

        Args:
            env_ds (xr.Dataset, optional): Dataset containing necessary environmental variables for PI calculation.
            year_mon_limits (Sequence[str], optional): Inclusive time range limits as `[start, end]` in ISO format.
            month_limits (Sequence[int], optional): Inclusive month range limits as `[start, end]`.
            lat_limits (Sequence[float], optional): Inclusive latitude range limits as `[min, max]`.
            lon_limits (Sequence[float], optional): Inclusive longitude range limits as `[min, max]`.
            geo_limits (gpd.GeoDataFrame, optional): Geospatial constraints as a GeoDataFrame.
            geo_limits_touching (bool, optional): Whether to include geometries that touch the boundary. Defaults to `False`.
            convert_vars (bool, optional): Whether to convert variable units and formats. Defaults to `True`.

        Returns:
            Any: Reference to the class instance with updated environmental data.
        """
        
        # Set the `env_ds` attribute if `env_ds` argument is passed
        if env_ds is not None:
            self.env_ds = self.convert_variables(env_ds) if convert_vars else env_ds

        # Limit the `env_ds` attribute
        self.env_ds = self.env_ds.sel(
            time=slice(*year_mon_limits),
            lat=slice(*lat_limits),
            lon=slice(*lon_limits),
        )

        # Limit to certain months
        if month_limits and all(month_limits):
            self.env_ds = self.env_ds.sel(
                time=self.env_ds['time'].dt.month.isin(range(month_limits[0], month_limits[1]+1))
            )
        
        # Limit the `pi_ds` attribute if it exists
        if self.pi_ds is not None:
            self.pi_ds = self.pi_ds.sel(
                time=slice(*year_mon_limits),
                lat=slice(*lat_limits),
                lon=slice(*lon_limits),
            )
            if month_limits and all(month_limits):
                self.pi_ds = self.pi_ds.sel(
                    time=self.pi_ds['time'].dt.month.isin(range(month_limits[0], month_limits[1]+1))
                )

        # Use the geodataframe limits if it is passed
        if geo_limits is not None:
            # Create a copy to avoid making changes to the original dataframe
            geo_limits_copy = geo_limits.copy()
            geo_limits_copy.geometry = geo_limits_copy.geometry.apply(
                lambda geom: translate(geom, xoff=360) if geom.bounds[0] < 0 else geom
            )
            # Set the environmental variable dataset
            self.env_ds = self.env_ds.rio.write_crs(
                geo_limits_copy.crs
            ).rio.set_spatial_dims(
                x_dim="lon", 
                y_dim="lat"
            ).rio.clip(
                geo_limits_copy.geometry, 
                geo_limits_copy.crs, 
                all_touched=geo_limits_touching
            )
            # Set the PI dataset if it exists
            if self.pi_ds is not None:
                self.pi_ds = self.pi_ds.rio.write_crs(
                    geo_limits_copy.crs
                ).rio.set_spatial_dims(
                    x_dim="lon", 
                    y_dim="lat"
                ).rio.clip(
                    geo_limits_copy.geometry, 
                    geo_limits_copy.crs, 
                    all_touched=False
                )

        return self

    def analyze_pi(self, set_attr:bool = True, **pi_kwargs) -> xr.Dataset:
        """
        Calculates potential intensity (PI) values from the environmental dataset.

        Args:
            set_attr (bool, optional): If `True`, sets the calculated PI dataset as a class attribute. Defaults to `True`.
            **pi_kwargs: Additional arguments for the PI calculation algorithm.

        Returns:
            xr.Dataset: Dataset containing potential intensity calculations (`vmax`, `pmin`, `ifl`, etc.).
        """
        # Raise an error if no data is detected
        if self.env_ds is None:
            raise ValueError("Please use set_data() to pass an xarray Dataset with the environmental variables needed for potential intensity calculation.")

        # Calculate PI
        pi_ds = self.calculate_pi(self.env_ds, **pi_kwargs).compute()

        # Set class attribute if specified
        if set_attr:
            self.pi_ds = pi_ds

        return pi_ds

    def calculate_mean(
        self,
        env_vars: Sequence = ["sst", "msl", "t", "r"],
        pi_vars: Sequence = ["vmax", "pmin"],
        set_attr: bool = True
    ) -> pd.DataFrame:
        """
        Calculates the spatial mean of environmental and PI variables over time.

        Args:
            env_vars (Sequence[str], optional): List of environmental variables to include. Defaults to `["sst", "msl", "t", "r"]`.
            pi_vars (Sequence[str], optional): List of PI variables to include. Defaults to `["vmax", "pmin"]`.
            set_attr (bool, optional): If `True`, saves the results as a class attribute. Defaults to `True`.

        Returns:
            pd.DataFrame: DataFrame of mean values for the selected variables.
        """
        # Raise an error if PI has not been calculated and PI variables are being requested
        if not pi_vars and not self.pi_ds:
            raise ValueError("Potential intensity has not been calculated for this dataset, please call the method analyze_pi().")

        # Get mean values for the latitude longitude span
        mean_dfs = []
        for ds, vars in zip([self.env_ds, self.pi_ds],[env_vars, pi_vars]):

            # Skip if variables is empty
            if not vars:
                continue

            # Separate two dimensional/three dimensional variables
            two_dim_vars = [var for var in vars if "plev" not in ds[var].dims]
            three_dim_vars = [var for var in vars if "plev" in ds[var].dims]

            # Calculate mean values for two-dimensional variables if not empty
            if two_dim_vars:
                two_dim_mean_df = ds[two_dim_vars].mean(dim=["lat", "lon"]).compute().to_dataframe()
            else:
                two_dim_mean_df = pd.DataFrame()
            # Calculate mean values for three-dimensional variables if not empty
            if three_dim_vars:
                three_dim_mean_df = ds[three_dim_vars].mean(dim=["lat", "lon", "plev"]).compute().to_dataframe()
            else:
                three_dim_mean_df = pd.DataFrame()
                
            combined_df = pd.concat([two_dim_mean_df, three_dim_mean_df], axis=1)

            mean_dfs.append(combined_df)

        # Restructure output
        mean_result = pd.concat(mean_dfs, axis=1)
        mean_result = mean_result.reset_index()[["time", *env_vars, *pi_vars]].rename({"time":"cftime"}, axis=1)
        mean_result["pdtime"] = mean_result["cftime"].map(lambda x: pd.to_datetime(x.isoformat()))
        mean_result["year"] = mean_result["cftime"].apply(lambda x: x.year)
        mean_result = mean_result.sort_values("pdtime")

        if set_attr:
            self.mean_df = mean_result

        return mean_result
    
    def fit_mean_trends(
        self,
        annual_trend: bool = False,
        return_errors: bool = False,
        set_attr: bool = True
    ) -> pd.DataFrame | tuple[pd.DataFrame, pd.DataFrame]:
        """
        Fits linear trends to the mean variable time series.

        Args:
            annual_trend (bool, optional): Whether to compute trends on an annual basis. Defaults to `False` (monthly trends).
            return_errors (bool, optional): Whether to return residuals and errors. Defaults to `False`.
            set_attr (bool, optional): Whether to save the trend results as a class attribute. Defaults to `True`.

        Returns:
            pd.DataFrame | tuple[pd.DataFrame, pd.DataFrame]: 
                - Trend coefficients and errors, optionally including residuals.
        """
        # Handling of errors
        if self.mean_df is None:
            raise ValueError("Please call calculate_mean().")
        
        # Initialize dictionary to save results and dataframe to save residuals
        params_dict = {"variable":[], "intercept":[], "slope":[], "mse":[], "rmse":[]}
        errors_df = self.mean_df.copy()

        # Fit trendlines to all mean variables
        for var in self.mean_df.drop(columns=["cftime","pdtime","year"]).columns:

            # Specify index and fit to linear trendline
            idx = self.mean_df["year"] if annual_trend else self.mean_df.index
            params = np.polyfit(idx, self.mean_df[var], 1)
            
            # Save coefficients and MSE/RMSE
            params_dict["variable"].append(var)
            params_dict["intercept"].append(params[1])
            params_dict["slope"].append(params[0])

            # Calculate residuals
            trend_func = np.poly1d(params)
            errors_df[f"{var}_trend"] = trend_func(idx)
            errors_df[f"{var}_error"] = errors_df[var] - errors_df[f"{var}_trend"]

            # Calculate MSE and RMSE
            mse = np.mean(np.power(errors_df[f"{var}_error"], 2))
            params_dict["mse"].append(mse)
            params_dict["rmse"].append(np.sqrt(mse))

        params_df = pd.DataFrame(params_dict)

        # Set attributes if set_attr is True
        if set_attr:
            self.params_df = params_df
            if return_errors:
                self.errors_df = errors_df

        return (params_df, errors_df) if return_errors else params_df

    def plot_map(
        self,
        variable: str = "sst",
        time: str|Sequence[str] = None,
        cmap_limits: Sequence[float] = [None, None],
        plot_title: str = "Map of CMIP6 Data",
        ax: Axes = None,
        **plot_kwargs
    ) -> tuple[Axes, Any]:
        """
        Plots the specified variable as a map using a geographic projection.

        Args:
            variable (str, optional): The variable to plot (e.g., "sst"). Defaults to "sst".
            time (str | Sequence[str], optional): The time period to plot. Can be a single time (str)
                or a range of times (Sequence[str]). If None, the mean over all times is plotted. Defaults to None.
            cmap_limits (Sequence[float], optional): The minimum and maximum values for the color scale. Defaults to [None, None].
            plot_title (str, optional): The title of the plot. Defaults to "Map of CMIP6 Data".
            ax (Axes, optional): The matplotlib Axes on which to plot. If None, a new Axes is created. Defaults to None.
            **plot_kwargs: Additional keyword arguments to pass to the `xarray.plot` function.

        Returns:
            tuple[Axes, Any]: The matplotlib Axes and the plot object created by xarray.
        """
        # Specify PI Dataset or environmental variable Dataset according to variable passed in
        if self.pi_ds is None or variable in self.env_ds:
            ds = self.env_ds
        else:
            ds = self.pi_ds

        # Calculate mean if time is not specified
        if time is None:
            var_ds = ds[variable].mean(dim="time").compute()
            var_ds.attrs = ds[variable].attrs
        elif isinstance(time, str):
            var_ds = ds[variable].sel(time=time)
        else:
            var_ds = ds[variable].sel(time=slice(*time)).mean(dim="time").compute()
            var_ds.attrs = ds[variable].attrs

        # Plot and return the axis
        if ax is None:
            fig, ax = plt.subplots(subplot_kw=dict(projection=ccrs.PlateCarree()))
        p = var_ds.plot(ax=ax, vmin=cmap_limits[0], vmax=cmap_limits[1], **plot_kwargs)
        p.axes.coastlines()
        ax.set_title(plot_title)
        return ax, p

    def plot_mean(
        self,
        variable: str = "sst",
        yearly_window: int = None,
        trendline: bool = False,
        additional_df: pd.DataFrame = None,
        plot_title: str = "Monthly Mean CMIP6 Data",
        ax: Axes = None,
        **plot_kwargs
    ) -> Axes:
        """
        Plots the time series of the mean of a variable.

        Args:
            variable (str, optional): The variable to plot (e.g., "sst"). Defaults to "sst".
            yearly_window (int, optional): The size of the moving average window in years. If None, no smoothing is applied. Defaults to None.
            trendline (bool, optional): Whether to overlay a trendline on the plot. Defaults to False.
            additional_df (pd.DataFrame, optional): An additional DataFrame to concatenate with the mean DataFrame for extended plotting. Defaults to None.
            plot_title (str, optional): The title of the plot. Defaults to "Monthly Mean CMIP6 Data".
            ax (Axes, optional): The matplotlib Axes on which to plot. If None, a new Axes is created. Defaults to None.
            **plot_kwargs: Additional keyword arguments to pass to the `pandas.Series.plot` function.

        Returns:
            Axes: The matplotlib Axes containing the plot.
        """
        # Handling of errors
        if self.mean_df is None:
            raise ValueError("Please call calculate_mean().")
        if variable not in self.mean_df:
            raise ValueError("The variable is not in the Dataframe.")
        
        # Concatenate with additional data if needed
        # Used when using moving average
        if additional_df is None:
            plot_df = self.mean_df
            max_year = None
        else:
            plot_df = pd.concat([additional_df, self.mean_df])
            max_year = additional_df["year"].max()
        
        # Set the series to monthly average or moving average
        if yearly_window is None:
            plot_series = plot_df.set_index("pdtime")[variable]
        else:
            plot_series = plot_df.groupby("year")[variable].mean().rolling(yearly_window).mean().loc[max_year:]

        # Plot and return the axis
        if ax is None:
            fig, ax = plt.subplots()
        
        # Plot the trendline if specified
        if trendline:
            if not hasattr(self, "params_df"):
                raise ValueError("Please call fit_mean_trends().")
            trend_func = np.poly1d(self.params_df[self.params_df["variable"] == variable][["slope","intercept"]].to_numpy()[0])
            indices = np.linspace(0, self.mean_df.index.max(), 100)
            if yearly_window is None:
                plot_indices = pd.date_range(self.mean_df["pdtime"].min(), 
                                             self.mean_df["pdtime"].max(), 
                                             periods=100)
            else:
                plot_indices = np.linspace(self.mean_df["year"].min(), self.mean_df["year"].max(), 100)
            ax.plot(plot_indices, trend_func(indices), color="k", linestyle="--", label="trend")

        plot_series.plot(ax=ax, **plot_kwargs)
        ax.set_title(plot_title)

        return ax

    @staticmethod
    def convert_variables(
        cmip6_ds:xr.Dataset, 
        ts_label:str = "ts", 
        psl_label:str = "psl",
        plev_label:str = "plev", 
        ta_label:str = "ta", 
        hus_label:str = "hus"
    ) -> xr.Dataset:
        """
        Converts variables in a CMIP6 dataset to a format compatible with potential intensity calculations.

        Args:
            cmip6_ds (xr.Dataset): The input xarray Dataset containing raw CMIP6 variables.
            ts_label (str, optional): Variable name for sea surface temperature. Defaults to "ts".
            psl_label (str, optional): Variable name for sea level pressure. Defaults to "psl".
            plev_label (str, optional): Variable name for pressure levels. Defaults to "plev".
            ta_label (str, optional): Variable name for air temperature. Defaults to "ta".
            hus_label (str, optional): Variable name for specific humidity. Defaults to "hus".

        Returns:
            xr.Dataset: The xarray Dataset with variables converted to the appropriate units and names for potential intensity calculations.
        """
        converted_ds = xr.Dataset({
            "sst": (cmip6_ds[ts_label] - 273.15),                    # Convert from K to C
            "msl": (cmip6_ds[psl_label] / 100),                      # Convert from Pa to hPa
            "p": (cmip6_ds[plev_label] / 100),                       # Convert from Pa to hPa
            "t": (cmip6_ds[ta_label] - 273.15),                      # Convert from K to C
            "r": (cmip6_ds[hus_label] / (1 - cmip6_ds[hus_label]))   # Convert from specific humidity to mixing ratio
        })

        converted_ds["sst"].attrs = {"long_name" : "Sea Surface Temperature", "units": "C"}
        converted_ds["msl"].attrs = {"long_name" : "Sea Level Pressure", "units": "hPa"}
        converted_ds["p"].attrs = {"long_name" : "Air Pressure", "units": "hPa"}
        converted_ds["t"].attrs = {"long_name" : "Air Temperature", "units": "C"}
        converted_ds["r"].attrs = {"long_name" : "Mixing Ratio", "units": "g/kg"}

        return converted_ds

    @staticmethod
    def calculate_pi(
        env_ds:xr.Dataset, 
        sst_label:str = "sst", 
        msl_label:str = "msl", 
        p_label:str = "p", 
        t_label:str = "t", 
        r_label:str = "r",
        **pi_kwargs
    ) -> xr.Dataset:
        """
        Calculates potential intensity (PI) values from the given environmental dataset.

        Args:
            env_ds (xr.Dataset): The xarray Dataset containing environmental variables needed for PI calculations.
            sst_label (str, optional): Variable name for sea surface temperature. Defaults to "sst".
            msl_label (str, optional): Variable name for mean sea level pressure. Defaults to "msl".
            p_label (str, optional): Variable name for pressure levels. Defaults to "p".
            t_label (str, optional): Variable name for air temperature. Defaults to "t".
            r_label (str, optional): Variable name for mixing ratio. Defaults to "r".
            **pi_kwargs: Additional keyword arguments to pass to the PI calculation function.

        Returns:
            xr.Dataset: A dataset containing the following variables:
                - `vmax` (float): Potential intensity in meters per second.
                - `pmin` (float): Minimum central pressure in hectopascal.
                - `ifl` (int): Flag indicating calculation status.
                    - `0`: Invalid input data.
                    - `1`: Successful calculation.
                    - `2`: Convergence failure.
                    - `3`: Missing values in the temperature profile.
                - `t0` (float): Outflow temperature in Kelvin.
                - `otl` (float): Outflow temperature level in hectopascal.
        """
        vmax, pmin, ifl, t0, ot1 = xr.apply_ufunc(
            pi,
            env_ds[sst_label], 
            env_ds[msl_label], 
            env_ds[p_label], 
            env_ds[t_label], 
            env_ds[r_label],                                    # Pass in the environmental variables
            input_core_dims=[[],[],["plev"],["plev"],["plev"]], # Specify input dimensions
            output_core_dims=[[],[],[],[],[]],                  # Specify output dimensions
            vectorize=True,
            kwargs=pi_kwargs,                                   # Feed in keyword arguments
            dask="parallelized"
        )

        pi_ds = xr.Dataset({
            "vmax": vmax,
            "pmin": pmin,
            "ifl" : ifl,
            "t0" : t0,
            "ot1" : ot1
        })

        pi_ds["vmax"].attrs = {"long_name" : "Potential Intensity", "units": "m/s"}
        pi_ds["pmin"].attrs = {"long_name" : "Minimum Central Pressure", "units": "hPa"}
        pi_ds["ifl"].attrs = {"long_name" : "Algorithm Status Flag", "units": "-"}
        pi_ds["t0"].attrs = {"long_name" : "Outflow Temperature", "units": "K"}
        pi_ds["ot1"].attrs = {"long_name" : "Outflow Temperature Level", "units": "hPa"}

        return pi_ds