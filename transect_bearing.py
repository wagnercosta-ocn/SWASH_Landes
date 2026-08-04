import numpy as np

def transect_bearing(lat1, lon1, lat2, lon2):
    """
    Compute the initial bearing (azimuth) from point 1 to point 2.

    Parameters
    ----------
    lat1, lon1 : float
        Latitude and longitude of the first point (degrees)
    lat2, lon2 : float
        Latitude and longitude of the second point (degrees)

    Returns
    -------
    bearing : float
        Bearing in degrees clockwise from North (0-360°)
    """

    lat1 = np.radians(lat1)
    lat2 = np.radians(lat2)
    dlon = np.radians(lon2 - lon1)

    x = np.sin(dlon) * np.cos(lat2)
    y = (np.cos(lat1) * np.sin(lat2) -
         np.sin(lat1) * np.cos(lat2) * np.cos(dlon))

    bearing = np.degrees(np.arctan2(x, y))

    return (bearing + 360) % 360
