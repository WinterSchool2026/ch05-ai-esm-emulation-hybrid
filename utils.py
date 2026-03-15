import xarray as xr
import matplotlib.pyplot as plt 
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import gc

class OceanDatasetLazy(Dataset):
    def __init__(self, filepath, input_steps=1, target_steps=5, normalize=True):
        """
        filepath     : path to NetCDF file (data loaded lazily per sample)
        input_steps  : number of input timesteps (can be >1 for history)
        target_steps : number of future steps to predict (N)
        """
        self.filepath     = filepath
        self.input_steps  = input_steps
        self.target_steps = target_steps
        self.normalize    = normalize

        # open once just to get dimensions and compute stats
        ds = xr.open_dataset(filepath)

        # total valid windows
        self.n_times   = ds.dims['Time']
        self.n_samples = self.n_times - input_steps - target_steps + 1

        if normalize:
            self._compute_stats(ds)

        ds.close()

    def _compute_stats(self, ds):
        """compute mean/std for normalization, ignoring land (NaNs)"""
        self.stats = {}
        for name in ['temp', 'salt', 'u', 'v', 'ssh',
                     'surface_taux', 'surface_tauy', 'ml_qnet', 'ml_qsol']:
            arr  = ds[name].values                  # load full var once for stats
            mask = ~np.isnan(arr)
            if mask.any():
                mean = float(arr[mask].mean())
                std  = float(arr[mask].std()) + 1e-8
            else:
                mean, std = 0.0, 1.0
            self.stats[name] = (mean, std)
            del arr, mask   # explicitly delete
            gc.collect()    # force garbage collection

    def _normalize(self, tensor, name):
        mean, std = self.stats[name]
        return (tensor - mean) / std

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        t     = idx
        t_end = t + self.input_steps + self.target_steps
        t_tgt = t + self.input_steps

        # open file and read only the needed timesteps for this sample
        ds = xr.open_dataset(self.filepath)

        # --- load only t:t_end timesteps for each variable ---
        temp  = torch.tensor(ds['temp']        [t:t_end].values, dtype=torch.float32)
        salt  = torch.tensor(ds['salt']        [t:t_end].values, dtype=torch.float32)
        u     = torch.tensor(ds['u']           [t:t_end].values, dtype=torch.float32)
        v     = torch.tensor(ds['v']           [t:t_end].values, dtype=torch.float32)
        ssh   = torch.tensor(ds['ssh']         [t:t_end].values, dtype=torch.float32)
        taux  = torch.tensor(ds['surface_taux'][t:t_end].values, dtype=torch.float32)
        tauy  = torch.tensor(ds['surface_tauy'][t:t_end].values, dtype=torch.float32)
        qnet  = torch.tensor(ds['ml_qnet']     [t:t_end].values, dtype=torch.float32)
        qsol  = torch.tensor(ds['ml_qsol']     [t:t_end].values, dtype=torch.float32)

        ds.close()

        # --- normalize ---
        if self.normalize:
            temp  = self._normalize(temp,  'temp')
            salt  = self._normalize(salt,  'salt')
            u     = self._normalize(u,     'u')
            v     = self._normalize(v,     'v')
            ssh   = self._normalize(ssh,   'ssh')
            taux  = self._normalize(taux,  'surface_taux')
            tauy  = self._normalize(tauy,  'surface_tauy')
            qnet  = self._normalize(qnet,  'ml_qnet')
            qsol  = self._normalize(qsol,  'ml_qsol')

        # --- replace NaNs with 0 (land mask) ---
        temp  = torch.nan_to_num(temp)
        salt  = torch.nan_to_num(salt)
        u     = torch.nan_to_num(u)
        v     = torch.nan_to_num(v)
        ssh   = torch.nan_to_num(ssh)
        
        # --- input state: (input_steps, zt, yt, xt) per 3D var ---
        state_3d = torch.stack([
            temp [:self.input_steps],   # (input_steps, zt, yt, xt)
            salt [:self.input_steps],
            u    [:self.input_steps],
            v    [:self.input_steps],
        ], dim=1)                       # (input_steps, 4, zt, yt, xt)

        state_2d = ssh[:self.input_steps].unsqueeze(1)  # (input_steps, 1, yt, xt)

        # --- forcings over input + target window ---
        forcings = torch.stack([
            taux, tauy, qnet, qsol,
        ], dim=1)                       # (input_steps+target_steps, 4, yt, xt)

        # --- target: next N steps ---
        target_3d = torch.stack([
            temp [self.input_steps:],   # (target_steps, zt, yt, xt)
            salt [self.input_steps:],
            u    [self.input_steps:],
            v    [self.input_steps:],
        ], dim=1)                       # (target_steps, 4, zt, yt, xt)

        target_2d = ssh[self.input_steps:].unsqueeze(1)  # (target_steps, 1, yt, xt)

        return {
            'state_3d':  state_3d,   # (input_steps, 4, zt, yt, xt)
            'state_2d':  state_2d,   # (input_steps, 1, yt, xt)
            'forcings':  forcings,   # (input_steps+target_steps, 4, yt, xt)
            'target_3d': target_3d,  # (target_steps, 4, zt, yt, xt)
            'target_2d': target_2d,  # (target_steps, 1, yt, xt)
        }

class OceanDataset(Dataset):
    def __init__(self, ds, input_steps=1, target_steps=5, normalize=True):
        """
        ds           : xarray Dataset
        input_steps  : number of input timesteps (can be >1 for history)
        target_steps : number of future steps to predict (N)
        """
        
        print("initializing ocean dataset")
        
        self.input_steps = input_steps
        self.target_steps = target_steps
        self.normalize = normalize

        # --- state variables (Time, zt, yt, xt) ---
        self.temp  = torch.tensor(ds['temp'].values,  dtype=torch.float32)
        self.salt  = torch.tensor(ds['salt'].values,  dtype=torch.float32)
        self.u     = torch.tensor(ds['u'].values,     dtype=torch.float32)
        self.v     = torch.tensor(ds['v'].values,     dtype=torch.float32)

        # --- 2D state (Time, yt, xt) ---
        self.ssh   = torch.tensor(ds['ssh'].values,   dtype=torch.float32)

        # --- forcings (Time, yt, xt) ---
        self.taux  = torch.tensor(ds['surface_taux'].values,      dtype=torch.float32)
        self.tauy  = torch.tensor(ds['surface_tauy'].values,      dtype=torch.float32)
        self.qnet  = torch.tensor(ds['ml_qnet'].values,           dtype=torch.float32)
        self.qsol  = torch.tensor(ds['ml_qsol'].values,           dtype=torch.float32)
       

        if normalize:
            self._compute_stats()
            
        # replace NaNs with 0 (land mask)
        self.temp  = torch.nan_to_num(self.temp)
        self.salt  = torch.nan_to_num(self.salt)
        self.u     = torch.nan_to_num(self.u)
        self.v     = torch.nan_to_num(self.v)
        self.ssh   = torch.nan_to_num(self.ssh)
        
        # total valid windows
        self.n_times = self.temp.shape[0]
        self.n_samples = self.n_times - input_steps - target_steps + 1

    def _compute_stats(self):
        """compute mean/std for normalization, ignoring land (zeros from nan_to_num)"""
        self.stats = {}
        for name, arr in [
            ('temp', self.temp), ('salt', self.salt),
            ('u', self.u),       ('v', self.v),
            ('ssh', self.ssh),   ('taux', self.taux),
            ('tauy', self.tauy), ('qnet', self.qnet),
            ('qsol', self.qsol),
        ]:
            # compute stats only over non-nan (ocean) points
            mask = ~torch.isnan(arr)
            if mask.any():
                mean = arr[mask].mean()
                std  = arr[mask].std() + 1e-8
            else:
                mean, std = 0.0, 1.0
            self.stats[name] = (mean, std)

        # normalize in place
        self.temp  = (self.temp  - self.stats['temp'][0])  / self.stats['temp'][1]
        self.salt  = (self.salt  - self.stats['salt'][0])  / self.stats['salt'][1]
        self.u     = (self.u     - self.stats['u'][0])     / self.stats['u'][1]
        self.v     = (self.v     - self.stats['v'][0])     / self.stats['v'][1]
        self.ssh   = (self.ssh   - self.stats['ssh'][0])   / self.stats['ssh'][1]
        self.taux  = (self.taux  - self.stats['taux'][0])  / self.stats['taux'][1]
        self.tauy  = (self.tauy  - self.stats['tauy'][0])  / self.stats['tauy'][1]
        self.qnet  = (self.qnet  - self.stats['qnet'][0])  / self.stats['qnet'][1]
        self.qsol  = (self.qsol  - self.stats['qsol'][0])  / self.stats['qsol'][1]

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        t = idx  # start of input window

        # --- input state: (input_steps, zt, yt, xt) per 3D var ---
        state_3d = torch.stack([
            self.temp[t : t + self.input_steps],   # (input_steps, zt, yt, xt)
            self.salt[t : t + self.input_steps],
            self.u   [t : t + self.input_steps],
            self.v   [t : t + self.input_steps],
        ], dim=1)  # (input_steps, 4, zt, yt, xt)

        state_2d = torch.stack([
            self.ssh [t : t + self.input_steps],   # (input_steps, yt, xt)
        ], dim=1)  # (input_steps, 1, yt, xt)

        # --- forcings over input + target window ---
        t_end = t + self.input_steps + self.target_steps
        forcings = torch.stack([
            self.taux [t:t_end],
            self.tauy [t:t_end],
            self.qnet [t:t_end],
            self.qsol [t:t_end],
        ], dim=1)  # (input_steps+target_steps, 5, yt, xt)

        # --- target: next N steps ---
        t_tgt = t + self.input_steps
        target_3d = torch.stack([
            self.temp[t_tgt : t_tgt + self.target_steps],
            self.salt[t_tgt : t_tgt + self.target_steps],
            self.u   [t_tgt : t_tgt + self.target_steps],
            self.v   [t_tgt : t_tgt + self.target_steps],
        ], dim=1)  # (target_steps, 4, zt, yt, xt)

        target_2d = torch.stack([
            self.ssh[t_tgt : t_tgt + self.target_steps],
        ], dim=1)  # (target_steps, 1, yt, xt)

        return {
            'state_3d':   state_3d,    # (input_steps, 4, zt, yt, xt)
            'state_2d':   state_2d,    # (input_steps, 1, yt, xt)
            'forcings':   forcings,    # (input_steps+target_steps, 5, yt, xt)
            'target_3d':  target_3d,   # (target_steps, 4, zt, yt, xt)
            'target_2d':  target_2d,   # (target_steps, 1, yt, xt)
        }

def plot_ocean_temperature(file_path):
    print(f"Loading {file_path} for visualization...")
    ds = xr.open_dataset(file_path)

    # ---------------------------------------------------------
    # THE FIX: Convert TimeDelta (ns) to plain Floats (Days)
    # This prevents Matplotlib from crashing when it builds polygons
    # ---------------------------------------------------------
    time_days = ds['Time'].astype('float64') / (1e9 * 60 * 60 * 24)

    # Create a beautifully formatted 2x2 figure
    fig = plt.figure(figsize=(18, 12))
    plt.subplots_adjust(hspace=0.3, wspace=0.2)

    # =========================================================
    # 1. CLIMATOLOGICAL SST (Time-Averaged Surface Temperature)
    # =========================================================
    print("Calculating Climatological SST...")
    ax1 = fig.add_subplot(2, 2, 1)
    
    sst_clim = ds['temp'].isel(zt=-1).mean(dim='Time')
    
    sst_plot = sst_clim.plot(ax=ax1, cmap='RdYlBu_r', add_colorbar=True, 
                             cbar_kwargs={'label': 'Temperature (°C)'}, vmin = np.nanmin(sst_clim), vmax = np.nanmax(sst_clim))
    ax1.set_title("Climatological Sea Surface Temperature (SST)", fontsize=14, fontweight='bold')
    ax1.set_xlabel("Longitude")
    ax1.set_ylabel("Latitude")

    # =========================================================
    # 2. STRATIFICATION (Zonal Mean Temperature)
    # =========================================================
    print("Calculating Zonal Mean Stratification...")
    ax2 = fig.add_subplot(2, 2, 2)
    
    zonal_mean_temp = ds['temp'].mean(dim=['Time', 'xt'])
    
    zonal_mean_temp.plot.contourf(ax=ax2, levels=20, cmap='RdYlBu_r', 
                                  add_colorbar=True, cbar_kwargs={'label': 'Temperature (°C)'}, vmin = np.nanmin(sst_clim), vmax = np.nanmax(sst_clim))
    ax2.set_title("Global Stratification (Zonal Mean Temp)", fontsize=14, fontweight='bold')
    ax2.set_xlabel("Latitude")
    ax2.set_ylabel("Depth (m)")

    # =========================================================
    # 3. TEMPERATURE TIME SERIES AT SPECIFIC DEPTHS
    # =========================================================
    print("Extracting Global Mean Time Series at Depths...")
    ax3 = fig.add_subplot(2, 2, 3)
    
    depths_to_plot = {
        "Surface": ds['temp'].isel(zt=-1),
        "200m": ds['temp'].sel(zt=-200, method='nearest'),
        "500m": ds['temp'].sel(zt=-500, method='nearest')
    }

    for label, temp_data in depths_to_plot.items():
        global_mean_ts = temp_data.mean(dim=['xt', 'yt'])
        # Using our clean time_days array here
        ax3.plot(time_days, global_mean_ts, label=label, linewidth=2)

    ax3.set_title("Global Mean Temperature Time Series", fontsize=14, fontweight='bold')
    ax3.set_xlabel("Time (Simulation Days)")
    ax3.set_ylabel("Average Temperature (°C)")
    ax3.grid(True, linestyle='--', alpha=0.6)
    ax3.legend(loc='upper right')

    # =========================================================
    # 4. EL NIÑO INDEX (Niño 3.4 SST Anomaly)
    # =========================================================
    print("Calculating Niño 3.4 Index...")
    ax4 = fig.add_subplot(2, 2, 4)
    
    nino_sst = ds['temp'].isel(zt=-1).sel(
        yt=slice(-5, 5), 
        xt=slice(190, 240)
    ).mean(dim=['xt', 'yt'])
    
    nino_clim_mean = nino_sst.mean(dim='Time')
    nino_anomaly = nino_sst - nino_clim_mean
    
    # We use .values to guarantee Matplotlib gets raw NumPy floats/booleans, not Xarray objects
    nino_vals = nino_anomaly.values
    
    ax4.plot(time_days, nino_vals, color='black', linewidth=1)
    ax4.fill_between(time_days, nino_vals, 0, where=(nino_vals >= 0), 
                     facecolor='red', alpha=0.6, label='El Niño')
    ax4.fill_between(time_days, nino_vals, 0, where=(nino_vals < 0), 
                     facecolor='blue', alpha=0.6, label='La Niña')

    ax4.set_title("Niño 3.4 Index (SST Anomaly)", fontsize=14, fontweight='bold')
    ax4.set_xlabel("Time (Simulation Days)")
    ax4.set_ylabel("Temperature Anomaly (°C)")
    ax4.axhline(0, color='black', linewidth=1)
    ax4.grid(True, linestyle='--', alpha=0.6)
    ax4.legend(loc='upper right')

    plt.tight_layout()
    plt.show()
    
    
def plot_salinity_velocity(file_path):
    print(f"Loading {file_path} for dynamics visualization...")
    ds = xr.open_dataset(file_path)

    # Create a beautifully formatted 2x2 figure
    fig = plt.figure(figsize=(18, 12))
    plt.subplots_adjust(hspace=0.3, wspace=0.2)

    # =========================================================
    # 1. CLIMATOLOGICAL SEA SURFACE HEIGHT (SSH)
    # =========================================================
    print("Calculating Mean Sea Surface Height...")
    ax1 = fig.add_subplot(2, 2, 1)
    
    # Average SSH over time. Shows the geostrophic gyres!
    ssh_clim = ds['ssh'].mean(dim='Time')
    
    ssh_clim.plot(ax=ax1, cmap='viridis', add_colorbar=True, 
                  cbar_kwargs={'label': 'Sea Surface Height (m)'}, vmin = -0.5, vmax = 0.5)
    ax1.set_title("Mean Sea Surface Height (SSH)", fontsize=14, fontweight='bold')
    ax1.set_xlabel("Longitude")
    ax1.set_ylabel("Latitude")

    # =========================================================
    # 2. SEA SURFACE SALINITY (SSS)
    # =========================================================
    print("Calculating Mean Sea Surface Salinity...")
    ax2 = fig.add_subplot(2, 2, 2)
    
    # Extract surface layer (zt=-1) and average over Time
    sss_clim = ds['salt'].isel(zt=-1).mean(dim='Time')
    
    # 'YlGnBu_r' or 'viridis' are great for salinity
    sss_clim.plot(ax=ax2, cmap='YlGnBu_r', add_colorbar=True, 
                  cbar_kwargs={'label': 'Salinity (psu)'})
    ax2.set_title("Climatological Sea Surface Salinity (SSS)", fontsize=14, fontweight='bold')
    ax2.set_xlabel("Longitude")
    ax2.set_ylabel("Latitude")

    # =========================================================
    # 3. SURFACE CURRENT SPEED (Derived from U and V)
    # =========================================================
    print("Calculating Surface Current Speeds...")
    ax3 = fig.add_subplot(2, 2, 3)
    
    # Extract surface U and V, average over time
    u_surf = ds['u'].isel(zt=-1).mean(dim='Time')
    v_surf = ds['v'].isel(zt=-1).mean(dim='Time')
    
    # Calculate magnitude: Speed = sqrt(U^2 + V^2)
    speed = np.sqrt(u_surf**2 + v_surf**2)
    
    # 'magma' or 'YlOrRd' are perfect for visualizing kinetic energy/speed
    speed.plot(ax=ax3, cmap='magma', add_colorbar=True, vmax=0.5, # Capped at 0.5 m/s for contrast
               cbar_kwargs={'label': 'Current Speed (m/s)'})
    ax3.set_title("Mean Surface Current Speed", fontsize=14, fontweight='bold')
    ax3.set_xlabel("Longitude")
    ax3.set_ylabel("Latitude")

    # =========================================================
    # 4. EQUATORIAL UNDERCURRENT (Depth-Longitude slice of U)
    # =========================================================
    print("Slicing the Equatorial Undercurrent...")
    ax4 = fig.add_subplot(2, 2, 4)
    
    # Select the equator (yt=0 or nearest), average over time
    # We slice the top 500 meters (zt >= -500) to zoom in on the subsurface currents
    euc_slice = ds['u'].sel(yt=0, method='nearest').sel(zt=slice(-500, 0)).mean(dim='Time')
    
    # RdYlBu_r is perfect here: Red is eastward flow, Blue is westward flow
    euc_slice.plot.contourf(ax=ax4, levels=20, cmap='RdBu_r', center=0,
                            add_colorbar=True, cbar_kwargs={'label': 'Zonal Velocity U (m/s)'})
    
    ax4.set_title("Equatorial Undercurrent (Zonal Velocity at 0° Lat)", fontsize=14, fontweight='bold')
    ax4.set_xlabel("Longitude")
    ax4.set_ylabel("Depth (m)")

    plt.tight_layout()
    plt.show()
    
def plot_amoc(file_path):
    print(f"Loading {file_path}...")
    ds = xr.open_dataset(file_path)
    
    # 1. Convert Time to Days
    time_days = ds['Time'].astype('float64') / (1e9 * 60 * 60 * 24)

    # 2. Let's find the Atlantic center at 26.5N
    # We'll take a wider slice to be safe: 280 to 350 (spanning the whole Atlantic)
    v_section = ds['v'].sel(yt=26.5, method='nearest').sel(xt=slice(280, 350))
    v_section = v_section.fillna(0.0)

    # 3. Zonal Integration Math
    # dx = R * cos(lat) * d_lon
    R_earth = 6371000.0
    dx_factor = R_earth * np.cos(np.deg2rad(26.5)) * (np.pi / 180.0)
    
    # Integration over longitude (xt)
    # Use 'dim' instead of the coordinate name for clarity
    v_zonal = v_section.integrate('xt') * dx_factor

    # 4. Vertical Integration (The Overturning Streamfunction)
    # AMOC is the sum of transport from the SURFACE downwards.
    # zt is usually 0 (surface) to -5000 (bottom).
    # We sort to ensure we start at the top.
    v_zonal = v_zonal.sortby('zt', ascending=False)
    
    # Calculate dz (layer thickness) manually to avoid alignment errors
    depths = v_zonal.zt.values
    dz = np.abs(np.diff(np.append(0, depths))) # Pad with surface 0
    dz_xr = xr.DataArray(dz, coords={'zt': v_zonal.zt}, dims='zt')

    # Streamfunction Psi(z) = integral from surface to z of v dx dz
    # We convert to Sverdrups (1e6 m3/s)
    psi = (v_zonal * dz_xr).cumsum(dim='zt') / 1e6
    
    # The AMOC Index is the maximum of this streamfunction in the vertical
    # We take the max over the 'zt' dimension
    amoc_timeseries = psi.max(dim='zt')

    # Diagnostic Prints
    print(f"Max Streamfunction in first step: {psi.isel(Time=0).max().values} Sv")
    print(f"Max Streamfunction in last step: {psi.isel(Time=-1).max().values} Sv")

    # =========================================================
    # PLOTTING
    # =========================================================
    plt.figure(figsize=(10, 5))
    plt.plot(time_days, amoc_timeseries, color='blue', linewidth=2)
    plt.axhline(0, color='black', alpha=0.3)
    plt.title("Atlantic Meridional Overturning Circulation (AMOC) at 26.5°N", fontweight='bold')
    plt.xlabel("Simulation Days")
    plt.ylabel("Transport (Sv)")
    plt.grid(True, alpha=0.3)
    plt.show()
    return time_days, amoc_timeseries
