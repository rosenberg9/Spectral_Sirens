import numpy as np
import pickle
from astropy import units as u
from astropy.cosmology import Planck18, z_at_value
from scipy.interpolate import RegularGridInterpolator
from mpi4py import MPI
import time

# Parameter grids
H0_values = np.linspace(40, 100, 256)
D_L_values = np.linspace(1, 1_000_000, 2500)
omega_m_values = np.linspace(0.25, 0.35, 50)

H0_values_clone = H0_values.copy()

# Preallocate output
z_grid_m = np.full((len(H0_values), len(D_L_values), len(omega_m_values)), 0)

def compute_z(i, j, k):
    H0 = H0_values[i]
    D_L = D_L_values[j]
    Om0 = omega_m_values[k]

    try:
        cosmo = Planck18.clone(H0=H0 * u.km / (u.Mpc * u.s), Om0=Om0)
        z = z_at_value(cosmo.luminosity_distance, D_L * u.Mpc)
        try:
            cosmo = Planck18.clone(H0=H0 * u.km / (u.Mpc * u.s), Om0=Om0)
            z = z_at_value(cosmo.luminosity_distance, D_L * u.Mpc)


            return (i, j, k, z)
        except Exception as e:
            print(f"Warning: H0={H0}, Om0={Om0}, D_L={D_L} — {e}")
            return (i, j, k, np.nan)
        return (i, j, k, z)
    except Exception as e:
        print(f"Warning: H0={H0}, Om0={Om0}, D_L={D_L} — {e}")
        return (i, j, k, np.nan)

# Split the work among MPI processes
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# Determine the workload for each process
workload = len(H0_values) * len(D_L_values) * len(omega_m_values) // size
remainder = len(H0_values) * len(D_L_values) * len(omega_m_values) % size

# Compute the start and end indices for each process
start = rank * workload
end = (rank + 1) * workload
if rank == size - 1:
    end += remainder

results = []
total_jobs = end - start
jobs_completed = 0
last_update = time.time()
report_interval = 15  # seconds

start_time = time.time()

for idx in range(start, end):
    i, j, k = np.unravel_index(idx, (len(H0_values), len(D_L_values), len(omega_m_values)))
    result = compute_z(i, j, k)
    results.append(result)
    jobs_completed += 1

    current_time = time.time()
    if current_time - last_update >= report_interval or jobs_completed == total_jobs:
        elapsed = current_time - start_time
        print(f"[Rank {rank}] {jobs_completed}/{total_jobs} jobs completed "
              f"({100 * jobs_completed / total_jobs:.2f}%) — Elapsed: {elapsed:.1f}s")
        last_update = current_time
# Gather results from all processes
all_results = comm.gather(results, root=0)

# Process 0 collects all results and updates the grid
if rank == 0:
    for process_results in all_results:
        for i, j, k, z in process_results:
            z_grid_m[i, j, k] = z.value

# Broadcast the updated grid to all processes
comm.Bcast(z_grid_m, root=0)

# Create the interpolant
redshift_interpolant_m = RegularGridInterpolator(
    (H0_values, D_L_values, omega_m_values),
    z_grid_m,
    bounds_error=False,
    fill_value=np.nan
)

# Save the interpolant to a file
if rank == 0:
    with open("redshift_interpolant_m.pkl", "wb") as f:
        pickle.dump(redshift_interpolant_m, f)

    print("Interpolant saved to redshift_interpolant_m.pkl")
