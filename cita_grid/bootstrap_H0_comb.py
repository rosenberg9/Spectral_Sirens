import numpy as np
from astropy.cosmology import Planck18
from astropy.cosmology import z_at_value
import astropy.units as u
from scipy.interpolate import RegularGridInterpolator
from joblib import Parallel, delayed
from tqdm import tqdm  # For progress tracking
import pickle
from joblib import dump
    

def compute_redshift_grid_parallel(H0_min, H0_max, H0_steps, D_L_min, D_L_max, D_L_steps, n_jobs=16):
    """
    Compute a grid of redshifts for given H0 and luminosity distance ranges using parallelization.

    Parameters:
        H0_min (float): Minimum Hubble constant (km/s/Mpc)
        H0_max (float): Maximum Hubble constant (km/s/Mpc)
        H0_steps (int): Number of steps in H0 grid
        D_L_min (float): Minimum luminosity distance (Mpc)
        D_L_max (float): Maximum luminosity distance (Mpc)
        D_L_steps (int): Number of steps in D_L grid
        n_jobs (int): Number of CPU cores to use in parallelization

    Returns:
        RegularGridInterpolator: Interpolant for redshift as a function of (H0, D_L)
    """
    
    H0_values = np.linspace(H0_min, H0_max, H0_steps)
    D_L_values = np.linspace((D_L_min), (D_L_max), D_L_steps)  # Log-spaced D_L
    z_grid = np.zeros((H0_steps, D_L_steps))

    def compute_redshift_for_H0(i, H0):
        cosmo = Planck18.clone(H0=H0 * u.km / (u.Mpc * u.s))
        z_values = np.zeros(D_L_steps)

        for j, D_L in enumerate(D_L_values):
            try:
                z_values[j] = z_at_value(cosmo.luminosity_distance, D_L * u.Mpc)
            except Exception:
                z_values[j] = np.nan
        
        return i, z_values

    # Parallel computation with progress tracking
    results = Parallel(n_jobs=n_jobs, backend="loky")(
        delayed(compute_redshift_for_H0)(i, H0) for i, H0 in enumerate(tqdm(H0_values, desc="Processing H0 values"))
    )

    # Collect results
    for i, z_values in results:
        z_grid[i, :] = z_values

    return RegularGridInterpolator(
        (H0_values, D_L_values), z_grid, bounds_error=False, fill_value=np.nan
    )


# Define grid ranges
H0_min, H0_max, H0_steps = 40, 100, 256  # H0 in km/s/Mpc
D_L_min, D_L_max, D_L_steps = 1, 1e6, 2000  # D_L in Mpc

redshift_interpolant = compute_redshift_grid_parallel(H0_min, H0_max, H0_steps, D_L_min, D_L_max, D_L_steps, n_jobs=16)

print("done")


# Save the interpolant to a file
with open("redshift_interpolant.pkl", "wb") as f:
    pickle.dump(redshift_interpolant, f)

print("Interpolant saved to redshift_interpolant.pkl")



# Load the interpolant from a file
with open("redshift_interpolant.pkl", "rb") as f:
    redshift_interpolant = pickle.load(f)

print("Interpolant loaded successfully!")