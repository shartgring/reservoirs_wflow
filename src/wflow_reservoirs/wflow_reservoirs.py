import pandas as pd
import numpy as np

from . import operation_rules

class WflowReservoir:
    def __init__(self):
        """
        The little brother of a reservoir in a WflowModel!
        It can be used to simulate reservoir performance without initializing running a whole WflowModel
        
        Each instance of a reservior class contains:
        - params: the parameters of the reservoir necessary to run simulation
        - reservoir_type: 3 types of reservoir are currently possible: 'simple', 'sqtable' and 'custom'
        - update: method for updating the reservoir storage and outflow during simulation
        - run: method for running a simulation
        """
        self.params = None
        self.reservoir_type = None

    def update(self, time, inflow, storage, timestepsecs, *args):
        """
        Update method for a reservoir without rules (storage remains constant and outflow equals inflow)

        """
        return inflow, storage
            
    def run(self, inflow: pd.Series, initial_storage: float = 0, timestepsecs: int = 86400):
        """
        Method to run a simulation of each reservoir type using the 'update' method creating
        2 properties for the WflowReservoir: reservoir outflow [m3/s] and storage [m3] timeseries

        Parameters
        ----------
        - inflow: pandas timeseries of reservoir inflow [m3/s]
        - initial_storage: simulation start with empty reservoir if no value is provided
        - timestepsecs: model time step in seconds, default: 86400 s (1 day)

        """

        time = inflow.index
        timesteps = len(time)
        storage = np.zeros(timesteps)
        outflow = np.zeros(timesteps)
        storage[0] = initial_storage
        outflow[0] = np.nan
        for i in range(timesteps-1):
            outflow[i+1], storage[i+1] = self.update(time[i], inflow[i], storage[i], timestepsecs, *self.params)
                                                        
        self.outflow = pd.Series(outflow, index=time)
        self.storage = pd.Series(storage, index=time)
        return

    def setup_simple(
        self,
        maxvolume: float,
        maxrelease: float,
        demand: float,
        targetminfrac: float|list,
        targetfullfrac: float|list,
    ):
        """
        Setting the parameters of the WflowReservoir for a simple reservoir

        It is possible to provide targetminfrac and targetfullfrac as a cyclic parameter
        for monthly cycling (length 12) or daily cycling (length 365)
        """
        if not isinstance(targetminfrac, float):
            if len(targetminfrac) != 12 and len(targetminfrac) != 365:
                raise ValueError("Cyclic parameter targetminfrac/targetfullfrac must have length 12 (monthly) or 365 (daily)")
            elif not type(targetminfrac) == type(targetfullfrac):
                raise TypeError("targetminfrac and targetfullfrac must be of the same type")
            elif len(targetminfrac) == 12:
                _targetminfrac = [targetminfrac[moy(i)-1] for i in range(365)]
                _targetfullfrac = [targetfullfrac[moy(i)-1] for i in range(365)]
            else:
                _targetminfrac = targetminfrac
                _targetfullfrac = targetfullfrac
        else:
            _targetminfrac = [targetminfrac for i in range(365)]
            _targetfullfrac = [targetfullfrac for i in range(365)]

        self.params = (maxvolume, demand, maxrelease, _targetminfrac, _targetfullfrac)
        self.update = update_simple
        self.reservoir_type = "simple"
        return

    def setup_sqtable(
        self,
        maxvolume: float,
        csv_path: str,
        delimiter: str = ","
    ):
        """
        Setting the parameters of the WflowReservoir for volume-based reservoir rules

        The csv-file is read using np.loadtxt for which the delimiter can be specified

        """    
        sq = np.loadtxt(csv_path, delimiter=delimiter)
        self.params = (maxvolume, sq[:,0], sq[:,1:-1])
        self.update = update_sqtable
        self.reservoir_type = "sqtable"
        return

    def setup_custom(self, func, *args):
        """
        Setting the parameters of the WflowReservoir for a custom function

        At least 4 arguments will be passed to the function in the following order:
        1) time [datetime format], 2) inflow[m3/s], 3) storage [m3], 4) timestepsecs [s]

        Additional parameters will be passed to self.params using '*arg'

        The function will return 1) outflow [m3/s] and 2) storage [m3]
        
        """   
        self.params = args
        self.update = func
        self.reservoir_type = "custom"
        return

    ### TO DO: setup_hydromt(self, WflowModel) to use hydromt functions to obtain parameters from a hydromt_wflow WflowModel
    ### TO DO: create example notebook comparing the reservoirs