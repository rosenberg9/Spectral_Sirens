import bilby as bi
from astropy.cosmology import Planck18
import numpy as np
from astropy import units as u
from scipy.interpolate import RegularGridInterpolator, interp1d
from joblib import Parallel, delayed
from tqdm import tqdm

# Global variable to store the final interpolant
_z_interp = None
_z_interp_m = None

def create_interpolant(zmin=1e-4, zmax=1e2, nz=5000,
                       H0min=0.1, H0max=200, nH0=300,
                       ndl=5000):
    """
    Precomputes the interpolant (H0, DL) -> z.
    This function must be called once before using get_z().
    """
    global _z_interp

    print("Creating redshift and H0 grids...")
    z_grid_vals = np.logspace(np.log10(zmin), np.log10(zmax), nz)
    H0_grid_vals = np.linspace(H0min, H0max, nH0)

    print("Computing DL(H0, z) grid...")
    def compute_DL_row(i):
        H0 = H0_grid_vals[i]
        cosmo = Planck18.clone(H0=H0)
        return cosmo.luminosity_distance(z_grid_vals).to_value(u.Mpc)

    DL_rows = Parallel(n_jobs=-1)(
        delayed(compute_DL_row)(i) for i in tqdm(range(len(H0_grid_vals)), desc="Stage 1: DL(H0, z)")
    )
    DL_grid = np.array(DL_rows)

    print("Defining DL grid for inversion...")
    dl_min = max(1.0, np.nanmin(DL_grid))
    dl_max = np.nanmax(DL_grid)
    dl_grid_vals = np.logspace(np.log10(dl_min), np.log10(dl_max), ndl)

    print("Inverting DL(H0, z) -> z(H0, DL)...")
    def invert_DL_for_H0(i):
        DL_row = DL_grid[i]
        valid = np.isfinite(DL_row) & (np.diff(DL_row, prepend=0) > 0)
        DL_clean = DL_row[valid]
        z_clean = z_grid_vals[valid]
        if len(DL_clean) < 2:
            return np.full_like(dl_grid_vals, np.nan)
        inv_interp = interp1d(DL_clean, z_clean, bounds_error=False,
                              fill_value=(z_clean[0], z_clean[-1]))
        return inv_interp(dl_grid_vals)

    Z_rows = Parallel(n_jobs=-1)(
        delayed(invert_DL_for_H0)(i) for i in tqdm(range(len(H0_grid_vals)), desc="Stage 2: z(H0, DL)")
    )
    Z_grid = np.array(Z_rows)

    print("Creating final interpolant...")
    _z_interp = RegularGridInterpolator(
        (H0_grid_vals, dl_grid_vals), Z_grid,
        bounds_error=False, fill_value=None
    )

    print("Interpolant created successfully.")


def get_z(H0, D_L):
    """
    Returns the redshift z corresponding to input H0 and luminosity distance D_L (in Mpc).
    Requires that create_interpolant() has been called first.
    """
    global _z_interp
    if _z_interp is None:
        raise RuntimeError("You must call create_interpolant() before using get_z().")

    H0 = np.atleast_1d(H0)
    D_L = np.atleast_1d(D_L)
    points = np.column_stack((H0, D_L))
    return _z_interp(points)


def create_interpolant_m(zmin=1e-4, zmax=1e2, nz=5000,
                         H0min=0.1, H0max=200, nH0=1000,
                         Om_mmin=0.2, Om_mmax=0.4, nOm_m=1000,
                         ndl=1000):
    """
    Precomputes the interpolant (H0, Om_m, DL) -> z.
    This function must be called once before using get_z().
    """
    global _z_interp_m

    print("Creating redshift, H0, and Om_m grids...")
    z_grid_vals = np.logspace(np.log10(zmin), np.log10(zmax), nz)
    H0_grid_vals = np.linspace(H0min, H0max, nH0)
    Om_m_vals = np.linspace(Om_mmin, Om_mmax, nOm_m)

    # Preallocate full DL grid: shape (nH0, nOm_m, nz)
    DL_grid = np.empty((nH0, nOm_m, nz))

    print("Computing DL(H0, Om_m, z) grid...")
    def compute_DL(H_idx, Om_idx):
        H0 = H0_grid_vals[H_idx]
        Om0 = Om_m_vals[Om_idx]
        cosmo = Planck18.clone(H0=H0, Om0=Om0)
        try:
            DL_vals = cosmo.luminosity_distance(z_grid_vals).to_value(u.Mpc)
        except Exception:
            DL_vals = np.full_like(z_grid_vals, np.nan)
        return (H_idx, Om_idx, DL_vals)

    results = Parallel(n_jobs=-1)(
        delayed(compute_DL)(i, j)
        for i in tqdm(range(nH0), desc="H0 loop")
        for j in range(nOm_m)
    )

    for i, j, row in results:
        DL_grid[i, j, :] = row

    print("Defining DL grid for inversion...")
    dl_min = max(1.0, np.nanmin(DL_grid))
    dl_max = np.nanmax(DL_grid)
    dl_grid_vals = np.logspace(np.log10(dl_min), np.log10(dl_max), ndl)

    print("Inverting DL(H0, Om_m, z) -> z(H0, Om_m, DL)...")
    def invert_DL(H_idx, Om_idx):
        DL_row = DL_grid[H_idx, Om_idx, :]
        valid = np.isfinite(DL_row) & (np.diff(DL_row, prepend=0) > 0)
        DL_clean = DL_row[valid]
        z_clean = z_grid_vals[valid]
        if len(DL_clean) < 2:
            return np.full_like(dl_grid_vals, np.nan)
        inv_interp = interp1d(DL_clean, z_clean, bounds_error=False,
                              fill_value=(z_clean[0], z_clean[-1]))
        return (H_idx, Om_idx, inv_interp(dl_grid_vals))

    Z_grid = np.empty((nH0, nOm_m, ndl))
    inv_results = Parallel(n_jobs=-1)(
        delayed(invert_DL)(i, j)
        for i in tqdm(range(nH0), desc="Inversion H0 loop")
        for j in range(nOm_m)
    )

    for i, j, z_row in inv_results:
        Z_grid[i, j, :] = z_row

    print("Creating final interpolant...")
    _z_interp_m = RegularGridInterpolator(
        (H0_grid_vals, Om_m_vals, dl_grid_vals), Z_grid,
        bounds_error=False, fill_value=None
    )

    print("3D interpolant (H0, Om_m, DL) -> z created successfully.")

def analyze_mass_distance_relation_evol_iter_comb(luminosityDistances, log_mass_plus_log1pz, initial_guess=(100, 0, 0,0,0), d_num=100, width_fac=1.32):
    import numpy as np
    from scipy.stats import gaussian_kde
    from scipy.signal import find_peaks
    from collections import defaultdict
    import astropy.units as u

    
    def compute_d_num(mass_cut_size):
        return ((mass_cut_size) // 100)  # Ensure d_num stays within a reasonable range
    d_num_low = d_num
    d_num_high = d_num
    for _ in range(3):  # Run the process twice
        # Define separate d_num for each cut

        # Define distance bins separately for low and high mass cuts
        d_bins_low = np.percentile(luminosityDistances, np.linspace(0, 100, num=d_num_low))
        d_bins_high = np.percentile(luminosityDistances, np.linspace(0, 100, num=d_num_high))

        # Track peaks and assign masks in a single loop
        peak_tracks = defaultdict(list)
        filtered_masks = {label: np.zeros_like(log_mass_plus_log1pz, dtype=bool) for label in ['low_mass', 'high_mass']}
        peak_group_id, previous_peaks = 0, []

        for i in range(len(d_bins_low) - 1):
            bin_mask = (d_bins_low[i] <= luminosityDistances) & (luminosityDistances < d_bins_low[i + 1])
            log_mass_bin, lum_dist_bin = log_mass_plus_log1pz[bin_mask], luminosityDistances[bin_mask]

            if len(log_mass_bin) > 3:
                kde = gaussian_kde(log_mass_bin)
                x_vals = np.linspace(log_mass_bin.min(), log_mass_bin.max(), 100)
                peaks = x_vals[find_peaks(kde(x_vals))[0]]
                errors = [np.std(log_mass_bin)] * len(peaks)
                mean_lum_dist = lum_dist_bin.mean()

                current_peaks = []
                for peak_value in peaks:
                    closest = min(previous_peaks, key=lambda p: abs(peak_value - p['value']), default=None)
                    if closest and abs(peak_value - closest['value']) < 0.2:
                        peak_tracks[closest['group']].append((mean_lum_dist, peak_value, errors[0]))
                        current_peaks.append({'value': peak_value, 'group': closest['group']})
                    else:
                        peak_tracks[peak_group_id].append((mean_lum_dist, peak_value, errors[0]))
                        current_peaks.append({'value': peak_value, 'group': peak_group_id})
                        peak_group_id += 1
                previous_peaks = current_peaks

                # Assign masks immediately
                for label, group in zip(['low_mass', 'high_mass'], list(peak_tracks.keys())[:2]):
                    distances, peak_values, _ = zip(*peak_tracks[group])
                    peak_value_bin = np.interp(mean_lum_dist, distances, peak_values)
                    peak_error_bin = np.std(log_mass_bin) / width_fac

                    within_range = np.where((log_mass_bin >= peak_value_bin - peak_error_bin) & 
                                            (log_mass_bin <= peak_value_bin + peak_error_bin))[0]
                    below, above = within_range[log_mass_bin[within_range] < peak_value_bin], \
                                   within_range[log_mass_bin[within_range] > peak_value_bin]

                    num_to_keep = min(len(below), len(above))
                    bin_peak_mask = np.zeros_like(log_mass_bin, dtype=bool)
                    bin_peak_mask[np.concatenate((below[:num_to_keep], above[:num_to_keep]))] = True
                    filtered_masks[label][bin_mask] = bin_peak_mask

        # Track peaks and assign masks in a single loop
        peak_tracks = defaultdict(list)
        filtered_masks_high = {label: np.zeros_like(log_mass_plus_log1pz, dtype=bool) for label in ['low_mass', 'high_mass']}
        peak_group_id, previous_peaks = 0, []
        
        for i in range(len(d_bins_high) - 1):
            bin_mask = (d_bins_high[i] <= luminosityDistances) & (luminosityDistances < d_bins_high[i + 1])
            log_mass_bin, lum_dist_bin = log_mass_plus_log1pz[bin_mask], luminosityDistances[bin_mask]

            if len(log_mass_bin) > 3:
                kde = gaussian_kde(log_mass_bin)
                x_vals = np.linspace(log_mass_bin.min(), log_mass_bin.max(), 100)
                peaks = x_vals[find_peaks(kde(x_vals))[0]]
                errors = [np.std(log_mass_bin)] * len(peaks)
                mean_lum_dist = lum_dist_bin.mean()

                current_peaks = []
                for peak_value in peaks:
                    closest = min(previous_peaks, key=lambda p: abs(peak_value - p['value']), default=None)
                    if closest and abs(peak_value - closest['value']) < 0.2:
                        peak_tracks[closest['group']].append((mean_lum_dist, peak_value, errors[0]))
                        current_peaks.append({'value': peak_value, 'group': closest['group']})
                    else:
                        peak_tracks[peak_group_id].append((mean_lum_dist, peak_value, errors[0]))
                        current_peaks.append({'value': peak_value, 'group': peak_group_id})
                        peak_group_id += 1
                previous_peaks = current_peaks

                # Assign masks immediately
                for label, group in zip(['low_mass', 'high_mass'], list(peak_tracks.keys())[:2]):
                    distances, peak_values, _ = zip(*peak_tracks[group])
                    peak_value_bin = np.interp(mean_lum_dist, distances, peak_values)
                    peak_error_bin = np.std(log_mass_bin) / width_fac

                    within_range = np.where((log_mass_bin >= peak_value_bin - peak_error_bin) & 
                                            (log_mass_bin <= peak_value_bin + peak_error_bin))[0]
                    below, above = within_range[log_mass_bin[within_range] < peak_value_bin], \
                                   within_range[log_mass_bin[within_range] > peak_value_bin]

                    num_to_keep = min(len(below), len(above))
                    bin_peak_mask = np.zeros_like(log_mass_bin, dtype=bool)
                    bin_peak_mask[np.concatenate((below[:num_to_keep], above[:num_to_keep]))] = True
                    filtered_masks_high[label][bin_mask] = bin_peak_mask

        # Extract filtered data
        mask_low, mask_high = filtered_masks['low_mass'], filtered_masks_high['high_mass']
        lum_dist_cut1, mass_cut1 = luminosityDistances[mask_low] * u.Mpc, log_mass_plus_log1pz[mask_low]
        lum_dist_cut2, mass_cut2 = luminosityDistances[mask_high] * u.Mpc, log_mass_plus_log1pz[mask_high]

        # Sort for consistency
        sort1, sort2 = np.argsort(lum_dist_cut1), np.argsort(lum_dist_cut2)
        lum_dist_cut1, mass_cut1 = lum_dist_cut1[sort1], mass_cut1[sort1]
        lum_dist_cut2, mass_cut2 = lum_dist_cut2[sort2], mass_cut2[sort2]

        # Adjust d_num separately for each cut
        d_num_low = compute_d_num(len(mass_cut1))
        d_num_high = compute_d_num(len(mass_cut2))

    # Perform the fitting
    fit = fit_and_optimize_evol_comb([lum_dist_cut1, lum_dist_cut2], [mass_cut1,mass_cut2], initial_guess)
    
    return fit, (lum_dist_cut1, mass_cut1), (lum_dist_cut2, mass_cut2)

def fit_and_optimize_evol_comb(luminosityDistances, log_mass_plus_log1pz, initial_guess):

    """
    Fit for H0, mass offset, and mass evolution slope (k).
    """
    from scipy.optimize import minimize

    result = minimize(
        objective_evol_comb,
        initial_guess,  # Initial guess for [H0, mass_offset, k]
        args=(luminosityDistances, log_mass_plus_log1pz),
        bounds=[(40, 100), (5, 100), (-0.3, 1.4),(5, 100), (-1.4, 1.4)],  # Add bounds for k if needed
        method="L-BFGS-B"
    )
    
    if result.success:
        best_H0, best_mass_offset_low, best_k_low,best_mass_offset_high, best_k_high = result.x
    else:
        best_H0, best_mass_offset_low, best_k_low,best_mass_offset_high, best_k_high = None, None, None, None, None
    return best_H0, best_mass_offset_low, best_k_low,best_mass_offset_high, best_k_high

def objective_evol_comb(params, luminosityDistances, log_mass_plus_log1pz):
    """
    Objective function to fit H0, mass offset, and mass evolution slope k, with weights.
    """
    H0, mass_offset_low, k_low,mass_offset_high, k_high = params

    lum_dist_cut1 = luminosityDistances[0]
    lum_dist_cut2 = luminosityDistances[1]

    mass_cut1 = log_mass_plus_log1pz[0]
    mass_cut2 = log_mass_plus_log1pz[1]

    # Prepare the input for the interpolant
    #points_low = np.vstack((np.full_like(lum_dist_cut1.value, H0), lum_dist_cut1.value)).T  # Shape (N, 2)

    # Get interpolated redshifts
    #redshifts_low = interpolant(points_low)

    # Prepare the input for the interpolant
    #points_high = np.vstack((np.full_like(lum_dist_cut2.value, H0), lum_dist_cut2.value)).T  # Shape (N, 2)

    # Get interpolated redshifts
    #redshifts_high = interpolant(points_high)

    redshifts_low = get_z(np.full_like(lum_dist_cut1.value, H0),lum_dist_cut1.value)


    redshifts_high = get_z(np.full_like(lum_dist_cut2.value, H0),lum_dist_cut2.value)

    # Handle NaNs (optional): Assign a high residual if any z is NaN
    #if np.isnan(redshifts_low).any() or np.isnan(redshifts_high).any():
     #   return np.inf  # Penalize invalid z values

    # Compute the modeled log(1+z) + mass_offset with evolving mass peak
    modeled_log1pz_plus_logM_low = np.log10(mass_offset_low + k_low * redshifts_low) + np.log10(1 + redshifts_low)

    modeled_log1pz_plus_logM_high = np.log10(mass_offset_high + k_high * redshifts_high) + np.log10(1 + redshifts_high)

    # Calculate weighted residuals
    residuals_low = mass_cut1 - modeled_log1pz_plus_logM_low
    residuals_high = mass_cut2 - modeled_log1pz_plus_logM_high

    #weighted_residuals = weights * (residuals ** 2)  # Apply weights
    resid_tot = np.sum(residuals_low**2)+np.sum(residuals_high**2)

    return resid_tot  # Sum of squared residuals

def massDensity_evol(mu_1_base, sigma_1, f_p1, alpha_1, m_min, m_b, 
                     mu_2_base, sigma_2, f_p2, alpha_2, delta_m, m_max, redshifts, ev_1, ev_2):
    from astropy import units as u
    import numpy as np
    from scipy import integrate

    ev_1 = ev_1*u.Msun   
    ev_2 = ev_2*u.Msun    

    N = 1000
    m_1 = np.linspace(7, m_max.value, N) * u.Msun
    m_1_value = m_1.value  # Remove unnecessary unit operations in loops

    # Precompute redshift-dependent mu values
    mu_1_z = (mu_1_base + ev_1 * redshifts[:, None]) * u.Msun
    mu_2_z = (mu_2_base + ev_2 * redshifts[:, None]) * u.Msun

    # Precompute logical masks
    mask1 = m_1_value < m_min.value
    mask2 = (m_1_value >= m_min.value) & (m_1_value < m_b.value)
    mask3 = (m_1_value >= m_b.value) & (m_1_value < m_max.value)

    # Initialize gamma array
    gamma = np.zeros((len(redshifts), N))

    # Compute gamma in one step using broadcasting
    gamma[:, mask1] = np.exp(-((m_1_value[mask1] - m_min.value) ** 2 / (2 * delta_m.value ** 2))) \
                      * (m_1_value[mask1] / m_b.value) ** alpha_1
    gamma[:, mask2] = (m_1_value[mask2] / m_b.value) ** alpha_1
    gamma[:, mask3] = (m_1_value[mask3] / m_b.value) ** alpha_2

    # Normalize gamma only once per redshift
    integral_gamma = integrate.simpson(gamma, x=m_1_value, axis=1)[:, None]
    gamma /= integral_gamma  # Broadcasting normalization

    # Compute Gaussian distributions using broadcasting
    N_1 = (f_p1 / (sigma_1.value * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((m_1_value - mu_1_z.value) / sigma_1.value) ** 2)
    N_2 = (f_p2 / (sigma_2.value * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((m_1_value - mu_2_z.value) / sigma_2.value) ** 2)

    # Compute probability density
    p_values = (N_1 + N_2 + (1 - f_p1 - f_p2) * gamma) * u.Msun

    return p_values, m_1

# Adjust sampling function to handle redshift evolution
def inverseTransformSamplingWithEvolution(cdf_list, m_1, redshifts, num_samples):
    import numpy as np
    import astropy.units as u

    samples = []
    for i, z in enumerate(redshifts):
        uniform_samples = np.random.uniform(0, 1, num_samples // len(redshifts))  # Uniform samples
        inverse_samples = np.interp(uniform_samples, cdf_list[i], m_1.value)  # Inverse transform
        samples.append(inverse_samples)
    return np.concatenate(samples) * u.Msun

def fit_and_plot_evol(H0, mass_offset,k, luminosityDistances, log_mass_plus_log1pz, label_suffix, color_obs, color_fit):
    import numpy as np
    import matplotlib.pyplot as plt
    
    # Plot the observed data with the specified color
    

    # Generate a range of distances and corresponding redshifts for the interpolant
    distance_samples = np.linspace(
        luminosityDistances.min().value, 
        luminosityDistances.max().value, 
        1000
    )    


    # Prepare the input for the interpolant
    #points = np.vstack((np.full_like(distance_samples, H0), distance_samples)).T  # Shape (N, 2)
    
    # Get interpolated redshifts
    #redshifts = interpolant(points)

    redshifts = get_z(np.full_like(distance_samples, H0),distance_samples)



    modeled_values = []
    for i in range(len(distance_samples)):
    #redshift = distance_to_redshift_interp(dist)
        log1pz = np.log10(1 + redshifts[i])
    
        #if mass_offset + k * redshifts[i] > 0:
        modeled_values.append(log1pz + np.log10(mass_offset + k * redshifts[i]))


    plt.scatter(
        luminosityDistances, 
        log_mass_plus_log1pz, 
        s=3, 
        color=color_obs, 
        label=f"Observed Data ({label_suffix})"
    )
    
    # Plot the fitted curve with the specified color and line style
    plt.plot(
        distance_samples, 
        modeled_values, 
        linestyle="--", 
        color=color_fit, 
        label=f'Best fit H0 ({label_suffix}) = {H0:.4f}'
    )

# Define the prior for redshift with uniform comoving volume
def generate_redshift_samples(min_redshift, max_redshift, num_samples, cosmology=Planck18):
    import bilby as bi
    # Set up the uniform comoving volume prior
    redshift_prior = bi.gw.prior.UniformComovingVolume(
        minimum=min_redshift, 
        maximum=max_redshift, 
        cosmology=cosmology, 
        name="redshift"
    )

    # Sample redshifts
    redshift_samples = redshift_prior.sample(num_samples)
    return redshift_samples

# Function to compute the cumulative density function
def cumulativeDensity(p, m_1):
    import numpy as np

    cdf = np.cumsum(p)  # Compute the CDF
    cdf /= cdf[-1]      # Normalize the CDF
    return cdf

# Define the function to add observational error with constant log distance scatter
def add_observational_error(luminosityDistances, log_mass_plus_log1pz, 
                            mass_error_fraction=0.1, log_distance_scatter=0.05):
    """
    Add observational error to redshifted masses and luminosity distances.

    Parameters:
    - luminosityDistances (array): Array of luminosity distances.
    - log_mass_plus_log1pz (array): Array of log(M) + log(1+z).
    - mass_error_fraction (float): Fractional error for non-logged masses (default 10%).
    - log_distance_scatter (float): Constant scatter in log(luminosity distance) (default 0.05).

    Returns:
    - perturbed_luminosityDistances: Luminosity distances with added error.
    - perturbed_log_mass_plus_log1pz: log(M) + log(1+z) with added error.
    """
    import numpy as np
    
    num_samples = len(luminosityDistances)

    # Convert log(M) + log(1+z) back to non-logged mass * (1 + z)
    non_logged_mass_plus_1pz = 10**log_mass_plus_log1pz

    # Compute standard deviations for the mass errors
    mass_error_std = mass_error_fraction * non_logged_mass_plus_1pz

    # Apply constant scatter in log(luminosity distance)
    log_luminosity_distances = np.log10(luminosityDistances)
    perturbed_log_luminosity_distances = log_luminosity_distances + np.random.normal(
        loc=0, scale=log_distance_scatter, size=num_samples
    )
    
    perturbed_luminosityDistances = 10**perturbed_log_luminosity_distances  # Convert back to linear scale

    # Perturb the mass * (1 + z) values
    perturbed_non_logged_mass_plus_1pz = non_logged_mass_plus_1pz + np.random.normal(
        loc=0, scale=mass_error_std, size=num_samples
    )

    # Convert back to log(M) + log(1+z)
    perturbed_log_mass_plus_log1pz = np.log10(perturbed_non_logged_mass_plus_1pz)

    return perturbed_luminosityDistances, perturbed_log_mass_plus_log1pz

def fit_and_optimize_evol_comb_m(luminosityDistances, log_mass_plus_log1pz, initial_guess, redshift_interpolant):

    """
    Fit for H0, mass offset, and mass evolution slope (k).
    """
    from scipy.optimize import minimize

    result = minimize(
        objective_evol_comb_m,
        initial_guess,  # Initial guess for [H0, mass_offset, k]
        args=(luminosityDistances, log_mass_plus_log1pz, redshift_interpolant),
        bounds=[(40, 100), (5, 100), (-0.3, 1.4),(5, 100), (-1.4, 1.4),(0.25,0.35)],  # Add bounds for k if needed
        method="L-BFGS-B"
    )
    
    if result.success:
        best_H0, best_mass_offset_low, best_k_low,best_mass_offset_high, best_k_high,best_Om_m = result.x
    else:
        best_H0, best_mass_offset_low, best_k_low,best_mass_offset_high, best_k_high,best_Om_m = None, None, None, None, None,None
    return best_H0, best_mass_offset_low, best_k_low,best_mass_offset_high, best_k_high,best_Om_m

def objective_evol_comb_m(params, luminosityDistances, log_mass_plus_log1pz, interpolant):
    """
    Objective function to fit H0, mass offset, and mass evolution slope k, with weights.
    """
    H0, mass_offset_low, k_low,mass_offset_high, k_high, Om_m = params

    lum_dist_cut1 = luminosityDistances[0]
    lum_dist_cut2 = luminosityDistances[1]

    mass_cut1 = log_mass_plus_log1pz[0]
    mass_cut2 = log_mass_plus_log1pz[1]

    # Prepare the input for the interpolant
    points_low = np.vstack((
        np.full_like(lum_dist_cut1.value, H0),      # (N,) array of H0
        lum_dist_cut1.value,                         # (N,) array of distances
        np.full_like(lum_dist_cut1.value, Om_m)    # (N,) array of Om_m
    )).T 

    # Get interpolated redshifts
    redshifts_low = interpolant(points_low)
    

    # Prepare the input for the interpolant
    points_high = np.vstack((
        np.full_like(lum_dist_cut2.value, H0),      # (N,) array of H0
        lum_dist_cut2.value,                         # (N,) array of distances
        np.full_like(lum_dist_cut2.value, Om_m)    # (N,) array of Om_m
    )).T 

    # Get interpolated redshifts
    redshifts_high = interpolant(points_high)

    
    # Handle NaNs (optional): Assign a high residual if any z is NaN
    if np.isnan(redshifts_low).any() or np.isnan(redshifts_high).any():
        return np.inf  # Penalize invalid z values

    # Compute the modeled log(1+z) + mass_offset with evolving mass peak
    modeled_log1pz_plus_logM_low = np.log10(mass_offset_low + k_low * redshifts_low) + np.log10(1 + redshifts_low)

    modeled_log1pz_plus_logM_high = np.log10(mass_offset_high + k_high * redshifts_high) + np.log10(1 + redshifts_high)


    # Calculate weighted residuals
    residuals_low = mass_cut1 - modeled_log1pz_plus_logM_low
    residuals_high = mass_cut2 - modeled_log1pz_plus_logM_high

    #weighted_residuals = weights * (residuals ** 2)  # Apply weights

    # --- Optional prior on Omega_m (Gaussian) ---
   # Om_m_prior_mean = 0.30966     # or your preferred prior mean
   # Om_m_prior_sigma = 0.02   # or your preferred uncertainty

    #prior_penalty = ((Om_m - Om_m_prior_mean) ** 2) / (Om_m_prior_sigma ** 2)

    return np.sum(residuals_low**2) + np.sum(residuals_high**2)# + prior_penalty*15

def analyze_mass_distance_relation_evol_iter_comb_m(luminosityDistances, log_mass_plus_log1pz, redshift_interpolant,initial_guess=(100, 0, 0,0,0,0.3), d_num=100, width_fac=1.32):
    import numpy as np
    from scipy.stats import gaussian_kde
    from scipy.signal import find_peaks
    from collections import defaultdict
    import astropy.units as u
    
    def compute_d_num(mass_cut_size):
        return ((mass_cut_size) // 100)  # Ensure d_num stays within a reasonable range
    d_num_low = d_num
    d_num_high = d_num
    for _ in range(3):  # Run the process twice
        # Define separate d_num for each cut

        # Define distance bins separately for low and high mass cuts
        d_bins_low = np.percentile(luminosityDistances, np.linspace(0, 100, num=d_num_low))
        d_bins_high = np.percentile(luminosityDistances, np.linspace(0, 100, num=d_num_high))

        # Track peaks and assign masks in a single loop
        peak_tracks = defaultdict(list)
        filtered_masks = {label: np.zeros_like(log_mass_plus_log1pz, dtype=bool) for label in ['low_mass', 'high_mass']}
        peak_group_id, previous_peaks = 0, []

        for i in range(len(d_bins_low) - 1):
            bin_mask = (d_bins_low[i] <= luminosityDistances) & (luminosityDistances < d_bins_low[i + 1])
            log_mass_bin, lum_dist_bin = log_mass_plus_log1pz[bin_mask], luminosityDistances[bin_mask]

            if len(log_mass_bin) > 3:
                kde = gaussian_kde(log_mass_bin)
                x_vals = np.linspace(log_mass_bin.min(), log_mass_bin.max(), 100)
                peaks = x_vals[find_peaks(kde(x_vals))[0]]
                errors = [np.std(log_mass_bin)] * len(peaks)
                mean_lum_dist = lum_dist_bin.mean()

                current_peaks = []
                for peak_value in peaks:
                    closest = min(previous_peaks, key=lambda p: abs(peak_value - p['value']), default=None)
                    if closest and abs(peak_value - closest['value']) < 0.2:
                        peak_tracks[closest['group']].append((mean_lum_dist, peak_value, errors[0]))
                        current_peaks.append({'value': peak_value, 'group': closest['group']})
                    else:
                        peak_tracks[peak_group_id].append((mean_lum_dist, peak_value, errors[0]))
                        current_peaks.append({'value': peak_value, 'group': peak_group_id})
                        peak_group_id += 1
                previous_peaks = current_peaks

                # Assign masks immediately
                for label, group in zip(['low_mass', 'high_mass'], list(peak_tracks.keys())[:2]):
                    distances, peak_values, _ = zip(*peak_tracks[group])
                    peak_value_bin = np.interp(mean_lum_dist, distances, peak_values)
                    peak_error_bin = np.std(log_mass_bin) / width_fac

                    within_range = np.where((log_mass_bin >= peak_value_bin - peak_error_bin) & 
                                            (log_mass_bin <= peak_value_bin + peak_error_bin))[0]
                    below, above = within_range[log_mass_bin[within_range] < peak_value_bin], \
                                   within_range[log_mass_bin[within_range] > peak_value_bin]

                    num_to_keep = min(len(below), len(above))
                    bin_peak_mask = np.zeros_like(log_mass_bin, dtype=bool)
                    bin_peak_mask[np.concatenate((below[:num_to_keep], above[:num_to_keep]))] = True
                    filtered_masks[label][bin_mask] = bin_peak_mask

        # Track peaks and assign masks in a single loop
        peak_tracks = defaultdict(list)
        filtered_masks_high = {label: np.zeros_like(log_mass_plus_log1pz, dtype=bool) for label in ['low_mass', 'high_mass']}
        peak_group_id, previous_peaks = 0, []
        
        for i in range(len(d_bins_high) - 1):
            bin_mask = (d_bins_high[i] <= luminosityDistances) & (luminosityDistances < d_bins_high[i + 1])
            log_mass_bin, lum_dist_bin = log_mass_plus_log1pz[bin_mask], luminosityDistances[bin_mask]

            if len(log_mass_bin) > 3:
                kde = gaussian_kde(log_mass_bin)
                x_vals = np.linspace(log_mass_bin.min(), log_mass_bin.max(), 100)
                peaks = x_vals[find_peaks(kde(x_vals))[0]]
                errors = [np.std(log_mass_bin)] * len(peaks)
                mean_lum_dist = lum_dist_bin.mean()

                current_peaks = []
                for peak_value in peaks:
                    closest = min(previous_peaks, key=lambda p: abs(peak_value - p['value']), default=None)
                    if closest and abs(peak_value - closest['value']) < 0.2:
                        peak_tracks[closest['group']].append((mean_lum_dist, peak_value, errors[0]))
                        current_peaks.append({'value': peak_value, 'group': closest['group']})
                    else:
                        peak_tracks[peak_group_id].append((mean_lum_dist, peak_value, errors[0]))
                        current_peaks.append({'value': peak_value, 'group': peak_group_id})
                        peak_group_id += 1
                previous_peaks = current_peaks

                # Assign masks immediately
                for label, group in zip(['low_mass', 'high_mass'], list(peak_tracks.keys())[:2]):
                    distances, peak_values, _ = zip(*peak_tracks[group])
                    peak_value_bin = np.interp(mean_lum_dist, distances, peak_values)
                    peak_error_bin = np.std(log_mass_bin) / width_fac

                    within_range = np.where((log_mass_bin >= peak_value_bin - peak_error_bin) & 
                                            (log_mass_bin <= peak_value_bin + peak_error_bin))[0]
                    below, above = within_range[log_mass_bin[within_range] < peak_value_bin], \
                                   within_range[log_mass_bin[within_range] > peak_value_bin]

                    num_to_keep = min(len(below), len(above))
                    bin_peak_mask = np.zeros_like(log_mass_bin, dtype=bool)
                    bin_peak_mask[np.concatenate((below[:num_to_keep], above[:num_to_keep]))] = True
                    filtered_masks_high[label][bin_mask] = bin_peak_mask

        # Extract filtered data
        mask_low, mask_high = filtered_masks['low_mass'], filtered_masks_high['high_mass']
        lum_dist_cut1, mass_cut1 = luminosityDistances[mask_low] * u.Mpc, log_mass_plus_log1pz[mask_low]
        lum_dist_cut2, mass_cut2 = luminosityDistances[mask_high] * u.Mpc, log_mass_plus_log1pz[mask_high]

        # Sort for consistency
        sort1, sort2 = np.argsort(lum_dist_cut1), np.argsort(lum_dist_cut2)
        lum_dist_cut1, mass_cut1 = lum_dist_cut1[sort1], mass_cut1[sort1]
        lum_dist_cut2, mass_cut2 = lum_dist_cut2[sort2], mass_cut2[sort2]

        # Adjust d_num separately for each cut
        d_num_low = compute_d_num(len(mass_cut1))
        d_num_high = compute_d_num(len(mass_cut2))

    # Perform the fitting
    fit = fit_and_optimize_evol_comb_m([lum_dist_cut1, lum_dist_cut2], [mass_cut1,mass_cut2], initial_guess, redshift_interpolant)

    return fit, (lum_dist_cut1, mass_cut1), (lum_dist_cut2, mass_cut2)

def fit_and_plot_evol_m(H0, mass_offset,k,Om_m, luminosityDistances, log_mass_plus_log1pz,interpolant_m, label_suffix, color_obs, color_fit):
    import numpy as np
    import matplotlib.pyplot as plt
    
    # Plot the observed data with the specified color
    

    # Generate a range of distances and corresponding redshifts for the interpolant
    distance_samples = np.linspace(
        luminosityDistances.min().value, 
        luminosityDistances.max().value, 
        1000
    )    

     # Prepare the input for the interpolant
    points = np.vstack((
            np.full_like(distance_samples, H0),      # (N,) array of H0
            distance_samples,                         # (N,) array of distances
            np.full_like(distance_samples, Om_m)    # (N,) array of Om_m
        )).T 
    
    # Get interpolated redshifts
    redshifts = interpolant_m(points)

    modeled_values = []
    for i in range(len(distance_samples)):
    #redshift = distance_to_redshift_interp(dist)
        log1pz = np.log10(1 + redshifts[i])
    
        #if mass_offset + k * redshifts[i] > 0:
        modeled_values.append(log1pz + np.log10(mass_offset + k * redshifts[i]))


    plt.scatter(
        luminosityDistances, 
        log_mass_plus_log1pz, 
        s=3, 
        color=color_obs, 
        label=f"Observed Data ({label_suffix})"
    )
    
    # Plot the fitted curve with the specified color and line style
    plt.plot(
        distance_samples, 
        modeled_values, 
        linestyle="--", 
        color=color_fit, 
        label=f'Best fit H0 ({label_suffix}) = {H0:.4f}'
    )