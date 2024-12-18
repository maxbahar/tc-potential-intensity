# Suppress warnings originating from tropycal
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning, module="pkg_resources")
warnings.filterwarnings("ignore", category=DeprecationWarning, module="tropycal")

import pytest
import sys
import os
import pandas as pd
import numpy as np
from matplotlib.axes import Axes
from scipy.stats import kstest, genextreme


#get current directory and append root to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.tc_extremes import TCExtremes

# instantiate the class
ev = TCExtremes()  

def test_init():
    """Test the initialization of the TCExtremes class."""
    assert isinstance(ev, TCExtremes)

def test_fetch():
    """Test the fetch method of the TCExtremes class."""
    
    # Call the fetch method with one year and basin for runtime efficiency during testing
    result = ev.fetch(start_year=2020, end_year=2020, ocean_basin='north_atlantic')
    
    # Check results
    assert isinstance(result, pd.DataFrame)
    assert not result.empty
    assert 'type' in result.columns
    assert 'source' in result.columns
    assert result['basin'].iloc[0] == 'north_atlantic'
    assert result['source'].iloc[0] == 'hurdat'

def test_fit():
    """Test the fit method of the TCExtremes class."""
    
    # Mock the _available_vars attribute
    ev._available_vars = ['vmax', 'mslp']
    
    mock_vmax_params = (-0.3, 48, 18)
    mock_mslp_params = (-0.4, -1000, 12)

    # Create mock data
    data = {
        'storm_id': [i for i in range(100)],
        'season': [2014 + i//20 for i in range(100)],
        'vmax': np.round(genextreme(*mock_vmax_params).rvs(100), 0),
        'mslp': np.round(genextreme(*mock_mslp_params).rvs(100), 0) * -1
    }
    storm_data = pd.DataFrame(data)

    # Test the 'all' method with sensical values for shape, loc, scale for max wind speed
    params_all = ev.fit(storm_data=storm_data, variable='vmax', fit_window=0)
    assert isinstance(params_all, tuple)
    assert len(params_all) == 3
    assert params_all[0] < 0
    assert params_all[1] > 20
    assert params_all[2] > 0

    # Test variable 'mslp' and make sure transformation works
    params_mslp = ev.fit(storm_data=storm_data, variable='mslp', fit_window=0)
    assert isinstance(params_mslp, tuple)
    assert len(params_mslp) == 3
    assert params_mslp[0] < 0
    assert params_mslp[1] > 800 # negative of mslp
    assert params_mslp[2] > 0

    # Test the 'season' method for same values as 'all' within dict structure
    params_season = ev.fit(storm_data=storm_data, variable='vmax', fit_window=5)
    assert isinstance(params_season, dict)
    for season, params in params_season.items():
        assert isinstance(params, tuple)
        assert len(params) == 3
        assert params[0] < 0
        assert params[1] > 20
        assert params[2] > 0

    # Test invalid variable
    with pytest.raises(ValueError):
        ev.fit(storm_data=storm_data, variable='invalid_var', fit_window=0)
    
    # Test invalid method
    with pytest.raises(ValueError):
        ev.fit(storm_data=storm_data, variable='vmax', fit_window='invalid_window')

def test_evaluate():
    """Test the evaluate method of the TCExtremes class."""
    
    ev._available_vars = ['vmax', 'mslp']
    
    data = {
        'storm_id': [1, 2, 3, 4, 5, 6],
        'year': [2020, 2020, 2021, 2021, 2022, 2022],
        'season': [2020, 2020, 2021, 2021, 2022, 2022],
        'vmax': [100, 110, 120, 130, 140, 150],
        'mslp': [950, 940, 930, 920, 910, 900]
    }
    storm_data = pd.DataFrame(data)
    
    # Test the evaluate method with sensical values for max wind speed
    result = ev.evaluate(storm_data=storm_data, variable='vmax', ev_params={(2020,2023): (-7,100,10)})
    assert isinstance(result, dict)

    # Check if the returned dict is not empty
    assert bool(result) 

    # make sure dict has expected keys
    for season, results in result.items():
        assert 'empirical_probs' in results
        assert 'gev_probs' in results

    # check that probabilities are between 0 and 1 for empirical and GEV
    for season, results in result.items():
        assert results is not None
        for result in results['empirical_probs']:
            assert result >= 0
            assert result <= 1
        for result in results['gev_probs']:
            assert result >= 0
            assert result <= 1
    
        # ensure that extreme value distribution can be fit well to mock data by performing Kolmogorov-Smirnov test 
        ks_stat, p_value = kstest(results['empirical_probs'], results['gev_probs'])
        assert ks_stat < 0.6

    # Test the evaluate method with sensical values for max wind speed
    result = ev.evaluate(storm_data=storm_data, variable='mslp',ev_params={(2020,2023): (-2,1000,20)})
    assert isinstance(result, dict)

    # same procedure for empirical and GEV
    for season, results in result.items():
        assert results is not None
        for result in results['empirical_probs']:
            assert result >= 0
            assert result <= 1
        for result in results['gev_probs']:
            assert result >= 0
            assert result <= 1

    # test ValueError for invalid variable
    with pytest.raises(ValueError):
        ev.evaluate(storm_data=storm_data, variable='invalid')

def test_score():
    """Test the score method of the TCExtremes class."""

    ev._available_vars = ['vmax', 'mslp']
    
    data = {
        'storm_id': [1, 2, 3, 4, 5, 6],
        'year': [2020, 2020, 2021, 2021, 2022, 2022],
        'season': [2020, 2020, 2021, 2021, 2022, 2022],
        'vmax': [100, 110, 120, 130, 140, 150],
        'mslp': [950, 940, 930, 920, 910, 900]
    }
    storm_data = pd.DataFrame(data)
    
    # Fit the data to get the GEV params
    params_all = ev.fit(storm_data=storm_data, variable='vmax', fit_window=0)
    params_season = ev.fit(storm_data=storm_data, variable='vmax', fit_window=5)
    
    # Test the score method with 'all' method params
    score_all = ev.score(storm_data=storm_data, variable='vmax', ev_params=params_all)
    assert isinstance(score_all, object)
    assert score_all.statistic >= 0
    assert 0 <= score_all.pvalue <= 1
    
    # Test the score method with 'season' method parameters
    score_season = ev.score(storm_data=storm_data, variable='vmax', ev_params=params_season)
    assert isinstance(score_season, dict)
    for season, result in score_season.items():
        assert isinstance(result, object)
        assert result.statistic >= 0
        assert 0 <= result.pvalue <= 1
    
    # Test the score method with 'mslp' variable
    params_mslp = ev.fit(storm_data=storm_data, variable='mslp', fit_window=0)
    score_mslp = ev.score(storm_data=storm_data, variable='mslp', ev_params=params_mslp)
    assert isinstance(score_mslp, object)
    assert score_mslp.statistic >= 0
    assert 0 <= score_mslp.pvalue <= 1

    # Test ValueError for invalid variable
    with pytest.raises(ValueError):
        ev.score(storm_data=storm_data, variable='invalid', ev_params=params_all)

def test_plots():

    mock_vmax_params = (-0.3, 48, 18)
    mock_mslp_params = (-0.4, -1000, 12)

    # Create mock data
    data = {
        'storm_id': [i for i in range(100)],
        'season': [2014 + i//20 for i in range(100)],
        'source' : ['hurdat' for _ in range(100)],
        'basin' : ['north_atlantic' for _ in range(100)],
        'vmax': np.round(genextreme(*mock_vmax_params).rvs(100), 0),
        'mslp': np.round(genextreme(*mock_mslp_params).rvs(100), 0) * -1
    }
    storm_data = pd.DataFrame(data)

    # Check helper plotting function
    assert isinstance(ev._plot_helper(storm_data, 'vmax', mock_vmax_params, False, ''), Axes)
    assert isinstance(ev._plot_helper(storm_data, 'mslp', mock_mslp_params, False, ''), Axes)

    # Check fit plotting function
    assert isinstance(ev.plot_fit(storm_data,'vmax', mock_vmax_params, False, ''), Axes)
    assert isinstance(ev.plot_fit(storm_data,'mslp', mock_mslp_params, False, ''), Axes)
    assert isinstance(ev.plot_fit(storm_data,'vmax', {(2014,2019) : mock_vmax_params}, False, ''), list)
    assert isinstance(ev.plot_fit(storm_data,'mslp', {(2014,2019) : mock_mslp_params}, False, ''), list)
    with pytest.raises(ValueError):
        ev.plot_fit(storm_data, 'invalid_variable', mock_vmax_params, False, '')

    # Check distribution plotting function
    assert isinstance(ev.plot_dist(storm_data, 'vmax', 5, False, ''), Axes)
    assert isinstance(ev.plot_dist(storm_data, 'mslp', 2, False, ''), Axes)
    with pytest.raises(ValueError):
        ev.plot_dist(storm_data, 'invalid_variable', 2, False, '')

    # Check aggregate plotting function
    assert isinstance(ev.plot_agg(storm_data, 'vmax', 5, 1, False, ''), Axes)
    assert isinstance(ev.plot_agg(storm_data, 'mslp', 5, 1, False, ''), Axes)
    with pytest.raises(ValueError):
        ev.plot_agg(storm_data, 'invalid_variable', mock_vmax_params, 2, False, '')

    # Check time plotting function
    time_plot_vmax = ev.plot_time(storm_data,'vmax', {(2014,2019) : mock_vmax_params}, False, '')
    time_plot_mslp = ev.plot_time(storm_data,'mslp', {(2014,2019) : mock_mslp_params}, False, '')
    assert isinstance(time_plot_vmax, list)
    assert isinstance(time_plot_vmax[0], Axes)
    assert isinstance(time_plot_mslp, list)
    assert isinstance(time_plot_mslp[0], Axes)
    with pytest.raises(ValueError):
        ev.plot_time(storm_data,'invalid_variable', {(2014,2019) : mock_vmax_params}, False, '')
    with pytest.raises(TypeError):
        ev.plot_time(storm_data,'vmax', mock_vmax_params, False, '')


if __name__ == "__main__":
    pytest.main()