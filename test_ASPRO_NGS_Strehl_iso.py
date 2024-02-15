#####################################################################
# 'test_ASPRO_NGS_Strehl_iso.py'
#
# Created: 2023.10 (yyyy.mm)
# Author: Laurent  Bourgès (JMMC - OSUG, CNRS)
# License: GPL3 (see LICENSE)
#
#####################################################################

import ASPRO_NGS.aspro as aspro
import numpy as np
import matplotlib.pyplot as plt


trace = False

# Turbulence
# average conditions: "50% (Seeing < 1.0  arcsec, t0 > 3.2 ms)"
config_turbulence = {}
config_turbulence["seeing"] = 1.0
config_turbulence["tau0"] = 3.2  # (ms)
config_turbulence["h_0"] = 4300.0  # Altitude of the turbulent layers (m) (could be a list) (for isoplanetism)
config_turbulence["Cn2"] = 1.0  # Cn2 weight (could be a list) (for collapsing h_0 and v_0 to an equivalent individual layer)

# seeing = 1 as gives r0:
# Fried's parameter @500 nm (m):
config_turbulence["r_0"] = (1.028993 * (0.5e-6 / config_turbulence["seeing"]) / np.pi * (180.0 * 3600.0))  # m

# tau0 (+r0) gives v0:
# # Wind speed of the turbulence layers (m.s-1) (could be a list) (for isoplanetism)
# unused for strehl_iso:
config_turbulence["v_0"] = (1000.0 * config_turbulence["r_0"] / config_turbulence["tau0"])  # (m.s-1)

# Derive seeing & tau0:
config_turbulence["seeing"] = (1.028993 * (0.5e-6 / config_turbulence["r_0"]) / np.pi * (180.0 * 3600.0))  # as
config_turbulence["tau0"] = (1000.0 * config_turbulence["r_0"] / config_turbulence["v_0"])  # (ms)

if trace:
    print(f"seeing: {config_turbulence['seeing']:.2f} as")
    print(f"r0:     {config_turbulence['r_0']:.6f} m")
    print(f"v0:     {config_turbulence['v_0']:.3f} m.s-1")
    print(f"tau0:   {config_turbulence['tau0']:.2f} ms")


def computeStrehl_UT_NGS_iso(flag_mode, target_ao_mag, distance_ao_as):
    ##### User parameters #####
    # Mode to simulate
    # flag_mode = 'NGS_IR' or ''NGS_VIS'
    if flag_mode[4:7] != "VIS" and flag_mode[4:7] != "IR":
        raise ValueError(flag_mode + " -> Unknown mode (*_VIS / *_IR)")

    # NGS
    config_NGS = {}
    config_NGS["magnitude"] = target_ao_mag  # Magnitude of the NGS
    config_NGS["zenith"] = 0.0  # For the airmass (deg), 0.0 for zenith

    # Target
    config_target = {}
    config_target['wavelength'] = 2.2e-06   # Wavelength of the target (science or fringe tracker) channel (m)
    config_target['theta'] = distance_ao_as # Angle between the target (science or fringe tracker) and the NGS (arcsecond)

    # AO system
    config_ao = {}
    config_ao['TelescopeDiameter'] = 8.0    # Telescope diameter (m)
    config_ao['transmission'] = 0.3         # Global transmission of the WFS channel (to compute the number of photons)
    config_ao['sig_RON'] = 0.2              # Readout noise of the camera
    config_ao['ExcessNoiseFactor'] = 2      # Excess noise factor
    config_ao['g_loop'] = 0.5               # Loop gain
    ##### User parameters #####

    ##### Mode-dependent variables #####
    # config_NGS['wavelength']  -> Wavelength of the HO NGS channel (m)
    # config_NGS['mag2flux']    -> Convertion magnitude to flux / Magnitude 0-point (ph/s/m2 for mag=0)
    # config_ao['n_mode']       -> Number of corrected modes corrected models (to compute the equivalent DM number of actuators)
    # config_ao['f_loop']       -> Loop frequency (Hz)
    # config_ao['SH_diam']      -> SH-WFS diameter (number of lenslets)
    # config_ao['pixScale']     -> pixel scale (milliarcsecond / pixel)
    # config_ao['n_pix']        -> number of pixels per lenslet

    if flag_mode[4:7] == "VIS":
        config_NGS["wavelength"] = 750e-9
        config_NGS["mag2flux"] = 2.63e10
        config_ao["n_mode"] = 800
        config_ao["f_loop"] = 1000.0
        config_ao["SH_diam"] = 40
        config_ao["pixScale"] = 420
        config_ao["n_pix"] = 6
    elif flag_mode[4:7] == "IR":
        config_NGS["wavelength"] = 2.2e-6
        config_NGS["mag2flux"] = 1.66e9
        config_ao["n_mode"] = 44
        config_ao["f_loop"] = 500.0
        config_ao["SH_diam"] = 9
        config_ao["pixScale"] = 510
        config_ao["n_pix"] = 8
    ##### Mode-dependent variables #####

    ##### Calibration of the Maréchal approximation #####
    # Values obtained with TIPTOP
    config_Strehl = {}
    if flag_mode[4:7] == "VIS":
        config_Strehl["geom"] = [0.26705087, 0.98968173]
        config_Strehl["lag"] = [8.48317135, 2.15500641]
        config_Strehl["ph"] = [11.97305155]
        config_Strehl["ron"] = [0.51996901]
        config_Strehl["iso"] = [4.33657467, 1.86425362]
    elif flag_mode[4:7] == "IR":
        config_Strehl["geom"] = [0.24405723, 0.86477159]
        config_Strehl["lag"] = [2.08400088, 2.09918214]
        config_Strehl["ph"] = [15.17856885]
        config_Strehl["ron"] = [1.65331745]
        config_Strehl["iso"] = [1.74957095, 1.97261581]
    ##### Calibration of the Maréchal approximation #####


    # Running Maréchal approximation (Anthony Berdeu, LESIA, OBSPM)
    # return aspro.compute_Marechal_NGS(config_NGS, config_target, config_ao, config_turbulence, config_Strehl)

    ##### Computing individual Strehl contributions #####

    ##### Loading configuration #####

    # Loading atmosphere
    r_0 = config_turbulence['r_0']
    Cn2 = config_turbulence['Cn2']
    h_0 = config_turbulence['h_0']
    h_0 = (np.sum(Cn2 * np.power(h_0, 5.0 / 3.0)) / np.sum(Cn2))**(3.0 / 5.0)
    v_0 = config_turbulence['v_0']
    v_0 = (np.sum(Cn2 * np.power(np.abs(v_0), 5.0 / 3.0)) / np.sum(Cn2))**(3.0 / 5.0)

    # Loading AO system
    ExcessNoiseFactor = config_ao['ExcessNoiseFactor']
    sigRON = config_ao['sig_RON']
    pixScale = config_ao['pixScale'] / 1000.0 # arcsecond
    [eqDM_pitch, eqDMn_act] = aspro.modes2eqDM(config_ao)
    f_loop = config_ao['f_loop']
    g_loop = config_ao['g_loop']
    n_pix = config_ao['n_pix']
    D_WFS = config_ao['TelescopeDiameter'] / config_ao['SH_diam']


    # Loading NGS
    wavelength_NGS = config_NGS['wavelength']
    n_ph = config_ao['transmission'] * D_WFS**2 * \
        config_NGS['mag2flux'] * 10.0**(-config_NGS['magnitude'] / 2.5) / f_loop
    zenith_angle = config_NGS['zenith']
    airmass = 1.0 / np.cos(np.radians(zenith_angle))



    # Loading target
    wavelength_target = config_target['wavelength']
    theta = config_target['theta']

    # Loading Strehl damping coefficient
    coeff_geom = config_Strehl['geom']
    coeff_lag = config_Strehl['lag']
    coeff_ph = config_Strehl['ph']
    coeff_ron = config_Strehl['ron']
    coeff_iso = config_Strehl['iso']
    ##### Loading configuration #####

    # print(f"Strehl_iso: coeff_iso={coeff_iso} airmass={airmass} theta={theta} h_0={h_0} r_0={r_0} wavelength={wavelength_target}")

    return aspro.Strehl_iso(coeff_iso, airmass, theta, h_0, r_0, wavelength_target)

    ##### Computing individual Strehl contributions #####
    # SR_geom = Strehl_geom(coeff_geom, airmass, eqDM_pitch, r_0, wavelength_target)
    # SR_lag = Strehl_lag(coeff_lag, airmass, v_0, r_0, wavelength_target, f_loop, g_loop)
    # SR_ph = Strehl_ph(coeff_ph, n_ph, wavelength_target, wavelength_NGS, g_loop, ExcessNoiseFactor)
    # SR_ron = Strehl_ron(coeff_ron, sigRON, n_ph, pixScale, n_pix, g_loop)
    # SR_iso = Strehl_iso(coeff_iso, airmass, theta, h_0, r_0, wavelength_target)
    ##### Computing individual Strehl contributions #####

    # print(f"SR_geom: {SR_geom}")
    # print(f"SR_lag:  {SR_lag}")
    # print(f"SR_ph:   {SR_ph}")
    # print(f"SR_ron:  {SR_ron}")
    # print(f"SR_iso:  {SR_iso}")

    ##### Output #####
    # SR = SR_geom * SR_lag * SR_ph * SR_ron * SR_iso


# --- main ---
if __name__ == "__main__":
    # unused args:
    ft_ao_dist = 0.0
    ao_Rmag = 5.0

    # config_NGS["zenith"] = 0.0
    # config_target['wavelength'] = 2.2e-06

    ho_values = np.array([3000.0, 3500.0, 4300.0, 6000.0, 8000.0, 10000.0])

    plt.figure(figsize=(16, 10))

    for h0 in ho_values:
        print(f"- h0 = {h0}:")

        config_turbulence["h_0"] = h0  # Altitude of the turbulent layers (m)

        dists_AO = np.arange(0.0, 60.0, (60.0 / 180.0), dtype=float)
        sr_iso = np.zeros_like(dists_AO)

        print("distance_ao_as\tstrehl_ratio")

        for i in range(len(dists_AO)):
            distance_ao_as = dists_AO[i]  # as

            sr_iso[i] = computeStrehl_UT_NGS_iso("NGS_VIS", ao_Rmag, distance_ao_as)
            print(f"{distance_ao_as:.2f}\t{sr_iso[i]:.4e}")

        plt.plot(dists_AO, sr_iso, marker='o', label=f"h0: {h0:.1f}")

    plt.xlabel('dist (as)')
    plt.ylabel('SR_iso')
    plt.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
    plt.legend()
    plt.show()

    print("That's All, folks !'")
