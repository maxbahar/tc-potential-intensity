# Dependencies
import geopandas as gpd
import intake
import intake_esm
import numpy as np
import pandas as pd
import rioxarray
import xarray as xr
from shapely.geometry import Polygon
import cartopy.crs as ccrs

import matplotlib
matplotlib.use("Agg") # Ensure non-GUI matplotlib
import matplotlib.pyplot as plt

from shapely.affinity import translate

from dask.distributed import Client, LocalCluster
from matplotlib.axes import Axes
from typing import Any, Sequence
from tcpyPI import pi

import sys
import os
import pytest

# Add the src directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.tc_potential import *

def test_distributed_manager():
    manager = DistributedManager(n_workers=2, threads_per_worker=2)
    assert manager.client.status == "running"
    manager.close()

def test_fetch_data():
    """
    Test the data fetching functionality
    """
    INSTITUTION_ID = "NASA-GISS"
    SOURCE_ID = "GISS-E2-1-G"
    GRID_LABEL = "gn"
    MEMBER_ID = "r1i1p3f1"

    # SSP pathways that appeared in the Sixth IPCC Report
    EXPERIMENT_LIST = ["historical","ssp119","ssp126","ssp245","ssp370","ssp585"]

    data_fetcher = CMIP6DataFetcher()

    # Test fetching data with return_options_only=True
    result_df = data_fetcher.fetch_data(return_options_only=True)
    assert isinstance(result_df, pd.DataFrame), "Expected a DataFrame when return_options_only=True"
    assert not result_df.empty, "Expected non-empty DataFrame when return_options_only=True"
    
    nasa_catalog = data_fetcher.fetch_data(
        institution_id=INSTITUTION_ID,
        source_id=SOURCE_ID,
        experiment_id=EXPERIMENT_LIST,
        member_id=MEMBER_ID,
        grid_label=GRID_LABEL,
        return_options_only=True           
    )

    assert isinstance(nasa_catalog, pd.DataFrame), "Expected a DataFrame when fetching with specific parameters"
    assert not nasa_catalog.empty, "Expected non-empty DataFrame when fetching with specific parameters"
    assert "ts" in nasa_catalog["variable_id"].values, "Expected 'ts' variable in the results"
    assert "NASA-GISS" in nasa_catalog["institution_id"].values, "Expected 'NASA-GISS' institution in the results"

def test_pi_analysis_initialization():
    # Test initialization
    analysis = PIAnalysis()
    assert analysis.env_ds is None, "Expected env_ds to be None upon initialization"
    assert analysis.pi_ds is None, "Expected pi_ds to be None upon initialization"
    assert analysis.mean_df is None, "Expected mean_df to be None upon initialization"

def test_set_data():
    # Create a sample xarray Dataset
    data = xr.Dataset({
        "temperature": (("time", "lat", "lon"), 20 + 5 * np.random.randn(2, 2, 2)),
        "pressure": (("time", "lat", "lon"), 1000 + 10 * np.random.randn(2, 2, 2))
    }, coords={
        "time": pd.date_range("2000-01-01", periods=2),
        "lat": [10, 20],
        "lon": [30, 40]
    })

    # Create a sample GeoDataFrame
    gdf = gpd.GeoDataFrame({
        "geometry": [Polygon([(30, 10), (40, 10), (40, 20), (30, 20), (30, 10)])]
    }, crs="EPSG:4326")

    # Initialize PIAnalysis
    analysis = PIAnalysis()

    # Test set_data with the sample Dataset and GeoDataFrame
    analysis.set_data(
        env_ds=data,
        year_mon_limits=["2000-01-01", "2000-12-31"],
        month_limits=[1, 12],
        lat_limits=[0, 30],
        lon_limits=[20, 50],
        geo_limits=gdf,
        geo_limits_touching=True,
        convert_vars=False
    )

    assert analysis.env_ds is not None, "Expected env_ds to be set"
    assert analysis.env_ds.sizes["time"] == 2, "Expected time dimension to have 2 elements"
    assert analysis.env_ds.sizes["lat"] == 2, "Expected lat dimension to have 2 elements"
    assert analysis.env_ds.sizes["lon"] == 2, "Expected lon dimension to have 2 elements"
    assert "temperature" in analysis.env_ds.data_vars, "Expected 'temperature' variable in env_ds"
    assert "pressure" in analysis.env_ds.data_vars, "Expected 'pressure' variable in env_ds"

def test_analyze_pi():
    """
    Test the potential intensity calculation functionality
    """

    # Create a sample xarray Dataset
    data = xr.Dataset({
        "temperature": (("time", "lat", "lon"), 20 + 5 * np.random.randn(2, 2, 2)),
        "pressure": (("time", "lat", "lon"), 1000 + 10 * np.random.randn(2, 2, 2))
    }, coords={
        "time": pd.date_range("2000-01-01", periods=2),
        "lat": [10, 20],
        "lon": [30, 40]
    })

    # Create a sample GeoDataFrame
    gdf = gpd.GeoDataFrame({
        "geometry": [Polygon([(30, 10), (40, 10), (40, 20), (30, 20), (30, 10)])]
    }, crs="EPSG:4326")

    # Initialize PIAnalysis
    analysis = PIAnalysis()

    # Set the data
    analysis.set_data(
        env_ds=data,
        year_mon_limits=["2000-01-01", "2000-12-31"],
        month_limits=[1, 12],
        lat_limits=[0, 30],
        lon_limits=[20, 50],
        geo_limits=gdf,
        geo_limits_touching=True,
        convert_vars=False
    )

    # Mock the calculate_pi method
    def mock_calculate_pi(env_ds):
        # Create a mock PI dataset
        pi_data = xr.Dataset({
            "pi": (("time", "lat", "lon"), np.random.rand(2, 2, 2))
        }, coords=env_ds.coords)
        return pi_data

    # Replace the calculate_pi method with the mock
    analysis.calculate_pi = mock_calculate_pi

    # Analyze PI
    pi_ds = analysis.analyze_pi()

    # Assertions
    assert pi_ds is not None, "Expected pi_ds to be set"
    assert "pi" in pi_ds.data_vars, "Expected 'pi' variable in pi_ds"
    assert pi_ds.sizes["time"] == 2, "Expected time dimension to have 2 elements"
    assert pi_ds.sizes["lat"] == 2, "Expected lat dimension to have 2 elements"
    assert pi_ds.sizes["lon"] == 2, "Expected lon dimension to have 2 elements"

def test_calculate_mean():
    """
    Test the spatial mean calculation functionality
    """
     # Create a sample xarray Dataset for environmental variables
    env_data = xr.Dataset({
        "sst": (("time", "lat", "lon"), 20 + 5 * np.random.randn(2, 2, 2)),
        "msl": (("time", "lat", "lon"), 1000 + 10 * np.random.randn(2, 2, 2)),
        "t": (("time", "lat", "lon", "plev"), 15 + 5 * np.random.randn(2, 2, 2, 3)),
        "r": (("time", "lat", "lon", "plev"), 0.5 + 0.1 * np.random.randn(2, 2, 2, 3))
    }, coords={
        "time": pd.date_range("2000-01-01", periods=2),
        "lat": [10, 20],
        "lon": [30, 40],
        "plev": [1000, 850, 500]
    })

    # Create a sample xarray Dataset for potential intensity variables
    pi_data = xr.Dataset({
        "vmax": (("time", "lat", "lon"), 50 + 10 * np.random.randn(2, 2, 2)),
        "pmin": (("time", "lat", "lon"), 950 + 5 * np.random.randn(2, 2, 2))
    }, coords=env_data.coords)

    # Initialize PIAnalysis
    analysis = PIAnalysis().set_data(env_ds=env_data, convert_vars=False)
    analysis.pi_ds = pi_data

    # Calculate mean
    mean_df = analysis.calculate_mean(env_vars=["sst", "msl", "t", "r"], pi_vars=["vmax", "pmin"])

    # Assertions
    assert mean_df is not None, "Expected mean_df to be set"
    assert isinstance(mean_df, pd.DataFrame), "Expected mean_df to be a DataFrame"
    assert "sst" in mean_df.columns, "Expected 'sst' column in mean_df"
    assert "msl" in mean_df.columns, "Expected 'msl' column in mean_df"
    assert "t" in mean_df.columns, "Expected 't' column in mean_df"
    assert "r" in mean_df.columns, "Expected 'r' column in mean_df"
    assert "vmax" in mean_df.columns, "Expected 'vmax' column in mean_df"
    assert "pmin" in mean_df.columns, "Expected 'pmin' column in mean_df"
    assert "cftime" in mean_df.columns, "Expected 'cftime' column in mean_df"
    assert "pdtime" in mean_df.columns, "Expected 'pdtime' column in mean_df"
    assert "year" in mean_df.columns, "Expected 'year' column in mean_df"

    # assert sensical values for means
    assert mean_df["sst"].mean() > 10, "Expected mean sea surface temperature to be positive and above 10 celsius"
    assert mean_df["msl"].mean() > 900, "Expected mean mean sea level to be positive and above 900 hPa"
    assert mean_df["t"].mean() > 10, "Expected mean temperature to be positive and above 10 celsius"
    assert 0 < mean_df["r"].mean() < 40, "Expected mean mixing ratio to be between 0 and 40 g/kg"
    assert 40 < mean_df["vmax"].mean() < 300, "Expected mean maximum wind speed to be positive and within reasonable threshold determined by demo.ipynb"
    assert 800 < mean_df["pmin"].mean() < 1000, "Expected mean minimum pressure to be positive and within reasonable threshold determined by demo.ipynb"

def test_convert_variables():
    # Create a sample xarray Dataset with CMIP6 variables
    cmip6_data = xr.Dataset({
        "ts": (("time", "lat", "lon"), 273.15 + 20 + 5 * np.random.randn(2, 2, 2)),  # Temperature in K
        "psl": (("time", "lat", "lon"), 100000 + 1000 * np.random.randn(2, 2, 2)),  # Pressure in Pa
        "plev": (("time", "lat", "lon", "plev"), 100000 + 1000 * np.random.randn(2, 2, 2, 3)),  # Pressure in Pa
        "ta": (("time", "lat", "lon", "plev"), 273.15 + 15 + 5 * np.random.randn(2, 2, 2, 3)),  # Temperature in K
        "hus": (("time", "lat", "lon", "plev"), 0.5 + 0.1 * np.random.randn(2, 2, 2, 3))  # Specific humidity
    }, coords={
        "time": pd.date_range("2000-01-01", periods=2),
        "lat": [10, 20],
        "lon": [30, 40]
    })

    # Convert variables using the method
    converted_ds = PIAnalysis.convert_variables(cmip6_data)

    assert "sst" in converted_ds.data_vars, "Expected 'sst' variable in converted_ds"
    assert "msl" in converted_ds.data_vars, "Expected 'msl' variable in converted_ds"
    assert "p" in converted_ds.data_vars, "Expected 'p' variable in converted_ds"
    assert "t" in converted_ds.data_vars, "Expected 't' variable in converted_ds"
    assert "r" in converted_ds.data_vars, "Expected 'r' variable in converted_ds"

    assert converted_ds["sst"].attrs["long_name"] == "Sea Surface Temperature", "Expected 'sst' long_name attribute to be 'Sea Surface Temperature'"
    assert converted_ds["sst"].attrs["units"] == "C", "Expected 'sst' units attribute to be 'C'"
    assert converted_ds["msl"].attrs["long_name"] == "Sea Level Pressure", "Expected 'msl' long_name attribute to be 'Sea Level Pressure'"
    assert converted_ds["msl"].attrs["units"] == "hPa", "Expected 'msl' units attribute to be 'hPa'"
    assert converted_ds["p"].attrs["long_name"] == "Air Pressure", "Expected 'p' long_name attribute to be 'Air Pressure'"
    assert converted_ds["p"].attrs["units"] == "hPa", "Expected 'p' units attribute to be 'hPa'"
    assert converted_ds["t"].attrs["long_name"] == "Air Temperature", "Expected 't' long_name attribute to be 'Air Temperature'"
    assert converted_ds["t"].attrs["units"] == "C", "Expected 't' units attribute to be 'C'"
    assert converted_ds["r"].attrs["long_name"] == "Mixing Ratio", "Expected 'r' long_name attribute to be 'Mixing Ratio'"
    assert converted_ds["r"].attrs["units"] == "g/kg", "Expected 'r' units attribute to be 'g/kg'"

    # test sensibility 
    assert converted_ds["sst"].mean() > 10, "Expected mean sea surface temperature to be positive and above 10 celsius"
    assert converted_ds["msl"].mean() > 900, "Expected mean mean sea level to be positive and above 900 hPa"
    assert converted_ds["p"].mean() > 900, "Expected mean pressure to be positive and above 900 hPa"
    assert converted_ds["t"].mean() > 10, "Expected mean temperature to be positive and above 10 celsius"
    assert 0 < converted_ds["r"].mean() < 40, "Expected mean mixing ratio to be between 0 and 40 g/kg"

def test_calculate_pi():

    # Create a sample xarray Dataset with environmental variables
    env_data = xr.Dataset({
        "sst": (("time", "lat", "lon"), 273.15 + 20 + 5 * np.random.randn(2, 2, 2)),  # Sea surface temperature in K
        "msl": (("time", "lat", "lon"), 100000 + 1000 * np.random.randn(2, 2, 2)),  # Mean sea level pressure in Pa
        "t": (("time", "lat", "lon", "plev"), 273.15 + 15 + 5 * np.random.randn(2, 2, 2, 3)),  # Temperature in K
        "r": (("time", "lat", "lon", "plev"), 0.5 + 0.1 * np.random.randn(2, 2, 2, 3)),  # Specific humidity in kg/kg
        "p": (("time", "lat", "lon", "plev"), 100000 + 1000 * np.random.randn(2, 2, 2, 3))  # Pressure in Pa
    }, coords={
        "time": pd.date_range("2000-01-01", periods=2),
        "lat": [10, 20],
        "lon": [30, 40],
        "plev": [1000, 850, 500]
    })

    pi_data = PIAnalysis.calculate_pi(env_data)

    # Assertions
    assert pi_data is not None, "Expected pi_data to be set"
    assert "vmax" in pi_data.data_vars, "Expected 'vmax' variable in pi_data"
    assert "pmin" in pi_data.data_vars, "Expected 'pmin' variable in pi_data"
    assert pi_data.sizes["time"] == 2, "Expected time dimension to have 2 elements"
    assert pi_data.sizes["lat"] == 2, "Expected lat dimension to have 2 elements"
    assert pi_data.sizes["lon"] == 2, "Expected lon dimension to have 2 elements"

def test_plot_map():
    env_data = xr.Dataset({
        "sst": (("time", "lat", "lon"), 273.15 + 20 + 5 * np.random.randn(2, 2, 2)),  # Sea surface temperature in K
        "msl": (("time", "lat", "lon"), 100000 + 1000 * np.random.randn(2, 2, 2)),  # Mean sea level pressure in Pa
        "t": (("time", "lat", "lon", "plev"), 273.15 + 15 + 5 * np.random.randn(2, 2, 2, 3)),  # Temperature in K
        "r": (("time", "lat", "lon", "plev"), 0.5 + 0.1 * np.random.randn(2, 2, 2, 3)),  # Specific humidity in kg/kg
        "p": (("time", "lat", "lon", "plev"), 100000 + 1000 * np.random.randn(2, 2, 2, 3))  # Pressure in Pa
    }, coords={
        "time": pd.date_range("2000-01-01", periods=2),
        "lat": [10, 20],
        "lon": [30, 40],
        "plev": [1000, 850, 500]
    })
    
    # Initialize PIAnalysis
    analysis = PIAnalysis().set_data(env_ds=env_data, convert_vars=False)

    # Calculate variables
    analysis.analyze_pi()
    analysis.calculate_mean()

    # Check various options
    ax, p = analysis.plot_map(variable="sst", time="2000-01-01")
    assert isinstance(ax, Axes)
    ax, p = analysis.plot_map(variable="vmax", cmap_limits=[0,300])
    assert isinstance(ax, Axes)
    ax, p = analysis.plot_map(variable="pmin", plot_title="Test plot title")
    assert isinstance(ax, Axes)

def test_plot_mean():
    env_data = xr.Dataset({
        "sst": (("time", "lat", "lon"), 273.15 + 20 + 5 * np.random.randn(2, 2, 2)),  # Sea surface temperature in K
        "msl": (("time", "lat", "lon"), 100000 + 1000 * np.random.randn(2, 2, 2)),  # Mean sea level pressure in Pa
        "t": (("time", "lat", "lon", "plev"), 273.15 + 15 + 5 * np.random.randn(2, 2, 2, 3)),  # Temperature in K
        "r": (("time", "lat", "lon", "plev"), 0.5 + 0.1 * np.random.randn(2, 2, 2, 3)),  # Specific humidity in kg/kg
        "p": (("time", "lat", "lon", "plev"), 100000 + 1000 * np.random.randn(2, 2, 2, 3))  # Pressure in Pa
    }, coords={
        "time": pd.date_range("2000-01-01", periods=2),
        "lat": [10, 20],
        "lon": [30, 40],
        "plev": [1000, 850, 500]
    })
    
    # Initialize PIAnalysis
    analysis = PIAnalysis().set_data(env_ds=env_data, convert_vars=False)

    # Calculate variables
    analysis.analyze_pi()
    analysis.calculate_mean()
    analysis.fit_mean_trends()

    # Check various options
    assert isinstance(analysis.plot_mean(variable="sst", trendline=True), Axes)
    assert isinstance(analysis.plot_mean(variable="sst", yearly_window=5), Axes)
    assert isinstance(analysis.plot_mean(variable="sst", plot_title="Test Plot"), Axes)
    assert isinstance(analysis.plot_mean(variable="vmax", trendline=True), Axes)

# Run the tests
if __name__ == "__main__":
    pytest.main([__file__])