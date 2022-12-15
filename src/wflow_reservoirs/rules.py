import numpy as np

# Some helper functions
def scurve(x, a, b, c):
    """
    Simple sigmoid function to determine environmental flow

    """
    s = 1.0 / (b + np.exp(-c * (x - a)))
    return s

def moy(doy):
    """
    Determine month of year (moy) based on day of year (doy)

    """
    for month, firstday in enumerate([0,32,60,91,121,152,182,213,244,274,305,335,366]):
        if doy < firstday:
            return month

# The reservoir operation rules
def update_simple(time, inflow, storage, timestepsecs, maxvolume, demand, maxrelease, targetminfrac, targetfullfrac):
    """
    Reservoir rules based on simple reservoir module from Wflow
 
    Parameters
    ----------
    - time: date and time of timestep i in datetime format
    - inflow: inflow to the reservoir at time i [m3/s]
    - timestepsecs: model time step in seconds
    - maxvolume: maximum volume of reservoir in [m3]
    - maxrelease: maximum discharge of reservoir through normal operations [m3/s]
    - demand: minimum (environmental) flow 
    - targetminfrac and targetfullfrac: boundaries for the reservoir volume, defined as a fraction of volume,
      fraction lies between 0 and 1 and can be provided cyclic with length 12 (monthly) or 365 (daily)

    Returns
    -------
    - outflow at time i+1 [m3/s]
    - storage at time i+1 [m3]

    """
    doy = min(time.dayofyear,365)
    # Simple reservoir module taken from Wflow.jl
    vol = max(0.0, storage + (inflow * timestepsecs))
    percfull = vol / maxvolume
    # first determine minimum (environmental) flow using a simple sigmoid curve to scale for target level
    fac = scurve(percfull, targetminfrac[doy-1], 1.0, 30.0)
    demandrelease = min(fac * demand * timestepsecs, vol)
    vol = vol - demandrelease
    wantrel = max(0.0, vol - (maxvolume * targetfullfrac[doy-1]))
    # Assume extra maximum Q if spilling
    overflow_q = max((vol - maxvolume), 0.0)
    torelease = min(wantrel, overflow_q + maxrelease* timestepsecs - demandrelease)
    storage = vol - torelease
    outflow = torelease + demandrelease
    return outflow/timestepsecs, storage

def update_sqtable(time, inflow, storage, timestepsecs, maxvolume, sq_S, sq_Q):
    """
    Volume-based reservoir rules with seasonal variations represented in a SQ-table,
    based on the lake module in Wflow. It uses interpolation to determine the discharge
    for a given day of the year and storage level

    Parameters
    ----------
    - time: date and time of timestep i in datetime format
    - inflow: inflow to the reservoir at time i [m3/s]
    - storage: inflow to the reservoir at time i [m3]
    - timestepsecs: model time step in seconds
    - maxvolume: maximum volume of reservoir in [m3]
    - sq_S: column containing N discrete storage levels
    - sq_Q: table containing discharges corresponding to the storage levels 
      for each day in year resulting in a table with size 365 x N

    Returns
    -------
    - outflow at the next timestep i+1 [m3/s]
    - storage at the next timestep i+1 [m3]
    """
    doy = min(time.dayofyear,365)
    # Interpolate from SQ table
    outflow = np.interp(storage, sq_S, sq_Q[:,doy-1])
    storage = storage + inflow * timestepsecs - outflow * timestepsecs
    # Determine new storage. If larger than maxvolume -> overflow
    if maxvolume > 0 and storage > maxvolume:
        outflow = outflow + (storage-maxvolume)/timestepsecs
        storage = maxvolume
    return outflow, storage