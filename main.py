import io
import requests
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import metpy.calc as mpcalc
from metpy.units import units

# 1. Ambil Data IGRA untuk stasiun Giles (ASM00094461)
url = "https://www.ncei.noaa.gov/data/integrated-global-radiosonde-archive/access/data-y2d/ASM00094461-data.txt"
response = requests.get(url)
lines = response.text.splitlines()

# Ekstrak observasi terakhir
header_indices = [i for i, line in enumerate(lines) if line.startswith('#')]
last_obs_start = header_indices[-1]
data_str = '\n'.join(lines[last_obs_start+1:])

# Parse data format fixed-width
colspecs = [(9, 15), (16, 21), (22, 27), (34, 39), (40, 45), (46, 51)]
col_names = ['pressure', 'height', 'temperature', 'dewpoint_dep', 'direction', 'speed']
df = pd.read_fwf(io.StringIO(data_str), colspecs=colspecs, names=col_names, na_values=[-9999, -8888])

# Konversi unit dasar
df['pressure'] = df['pressure'] / 100.0
df['height'] = df['height']  # Meter
df['temperature'] = df['temperature'] / 10.0
df['dewpoint_dep'] = df['dewpoint_dep'] / 10.0
df['speed'] = df['speed'] / 10.0
df['dewpoint'] = df['temperature'] - df['dewpoint_dep']

df = df.dropna(subset=['pressure', 'height', 'temperature', 'dewpoint', 'speed', 'direction']).reset_index(drop=True)

# 2. Assign MetPy Units
p = df['pressure'].values * units.hPa
z = df['height'].values * units.meters
T = df['temperature'].values * units.degC
Td = df['dewpoint'].values * units.degC
wind_speed = df['speed'].values * units('m/s')
wind_dir = df['direction'].values * units.degrees

# Komponen angin (u, v)
u, v = mpcalc.wind_components(wind_speed, wind_dir)

# 3. Kalkulasi Termodinamika & Bulk Richardson Number
# Hitung mixing ratio dari dewpoint dan tekanan
mixing_ratio = mpcalc.saturation_mixing_ratio(p, Td)

# Hitung Suhu Potensial Virtual (Theta-v)
theta_v = mpcalc.virtual_potential_temperature(p, T, mixing_ratio)

# Ambil nilai permukaan (indeks 0) sebagai referensi
z_s = z[0]
theta_v_s = theta_v[0]
u_s = u[0]
v_s = v[0]

# Hitung Bulk Richardson Number (Rib) untuk setiap level
# Formula: Rib = (g / theta_v_s) * (z - z_s) * (theta_v - theta_v_s) / ((u - u_s)**2 + (v - v_s)**2)
g = 9.81 * units('m/s^2')
delta_z = z - z_s
delta_theta_v = theta_v - theta_v_s
wind_shear_sq = (u - u_s)**2 + (v - v_s)**2

# Hindari pembagian dengan nol dengan menambahkan nilai epsilon yang sangat kecil
epsilon = 1e-6 * units('m^2/s^2')
Rib = (g / theta_v_s) * delta_z * delta_theta_v / (wind_shear_sq + epsilon)

# Jadikan array tak berdimensi agar mudah diolah
Rib_values = Rib.to('dimensionless').m
z_values = z.m

# 4. Tentukan Estimasi PBLH
# Cari indeks pertama di mana Rib melebihi nilai kritis (0.25)
critical_Rib = 0.25

# Pastikan kita hanya mencari di ketinggian logis untuk PBL (misal, di bawah 4000 meter)
valid_indices = np.where((z_values - z_values[0]) < 4000)[0]

pblh_index = None
pblh_height = None

for i in valid_indices:
    if i == 0: continue
    if Rib_values[i] >= critical_Rib:
        # Interpolasi linier sederhana untuk akurasi ketinggian
        # (z - z1) / (z2 - z1) = (Ri_crit - Ri1) / (Ri2 - Ri1)
        Ri1, Ri2 = Rib_values[i-1], Rib_values[i]
        z1, z2 = z_values[i-1], z_values[i]
        
        fraction = (critical_Rib - Ri1) / (Ri2 - Ri1)
        pblh_height = z1 + fraction * (z2 - z1)
        pblh_index = i
        break

# Jika ketinggian AGL (Above Ground Level) diinginkan:
pblh_agl = pblh_height - z_values[0] if pblh_height else np.nan

print(f"Estimasi PBLH (ASL): {pblh_height:.1f} m")
print(f"Estimasi PBLH (AGL): {pblh_agl:.1f} m")

# 5. Plot Profil Vertikal 
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 8), sharey=True)

# Batasi plot hingga 4000m AGL agar visualisasi fokus pada Boundary Layer
max_plot_height = z_values[0] + 4000
plot_mask = z_values <= max_plot_height

# Plot 1: Virtual Potential Temperature
ax1.plot(theta_v[plot_mask].m, z_values[plot_mask], marker='o', color='orange')
ax1.set_xlabel('Virtual Potential Temperature (K)')
ax1.set_ylabel('Height ASL (m)')
ax1.set_title(r'Profil $\theta_v$')
ax1.grid(True)
if pblh_height:
    ax1.axhline(pblh_height, color='red', linestyle='--', label=f'PBLH: {pblh_height:.1f} m')
    ax1.legend()

# Plot 2: Bulk Richardson Number
ax2.plot(Rib_values[plot_mask], z_values[plot_mask], marker='o', color='blue')
ax2.axvline(critical_Rib, color='green', linestyle=':', label=f'Critical Ri = {critical_Rib}')
ax2.set_xlabel('Bulk Richardson Number (Ri_b)')
ax2.set_title('Profil Bulk Richardson Number')
ax2.set_xlim(-0.5, 2.0) # Batasi sumbu X agar transisi di sekitar 0.25 terlihat jelas
ax2.grid(True)
if pblh_height:
    ax2.axhline(pblh_height, color='red', linestyle='--')
    ax2.legend()

plt.suptitle('Estimasi Planetary Boundary Layer Height (PBLH) - Stasiun Giles (WMO 94461)', fontsize=14)
plt.tight_layout()
plt.show()