#####################################################################
# 'ranking.py'
#
# Created: 2023.10 (yyyy.mm)
# Author: Laurent  Bourgès (JMMC - OSUG, CNRS)
# License: GPL3 (see LICENSE)
#
#####################################################################

import ASPRO_NGS.aspro as aspro
import asproGravFT
import numpy as np


show_plot = False
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


def computeStrehl_UT_NGS(flag_mode, target_ao_mag, distance_ao_as):
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
    return aspro.compute_Marechal_NGS(config_NGS, config_target, config_ao, config_turbulence, config_Strehl)


def compute_GRAVITY_UT(target_ft_mag, SR_ft, distance_ft_sci_as):
    if trace:
        print(f"compute_GRAVITY_UT(target_ft_mag={target_ft_mag}, SR_ft={SR_ft}, distance_ft_sci_as={distance_ft_sci_as})")

    # only import GRAVITYSimulator if needed:
    import GRAVITYSimulator.gravmodel as gravmodel

    config_obs = {}
    config_obs["tel"] = "UT"
    config_obs["res"] = "MEDIUM"  # or 'MEDIUM' or 'HIGH'
    config_obs["pol"] = "COMBINED"  # or 'SPLIT'
    config_obs["mode"] = "DUAL"  # or 'DUAL'
    config_obs["grism_intervention"] = "post"  # post 2021 observations
    config_obs["laser_on"] = True
    config_obs["ut_vib_rms"] = 200.0

    # target properties:
    config = {}
    config["sci_mag"] = 100.0  # unused
    config["sci_dit"] = 0.0  # unused

    config["ft_mag"] = target_ft_mag

    config["strehl_ratio"] = SR_ft
    config["tau0"] = config_turbulence["tau0"]

    sci_wavelength = 2.2e-6
    tel_diam = 8.0  # UT

    ft_dits = [1000.0, 303.0, 100.0]

    ft_snr = np.zeros(len(ft_dits))
    sig_opd = np.zeros(len(ft_dits))

    for i, ft_freq in enumerate(ft_dits):
        # use model to get FT info:
        gv = gravmodel.GravModel(
            config["sci_mag"],
            config["ft_mag"],
            tel=config_obs["tel"],
            res=config_obs["res"],
            pol=config_obs["pol"],
            mode=config_obs["mode"],
            grism_intervention=config_obs["grism_intervention"],
            laser_on=config_obs["laser_on"],
            ut_vib_rms=config_obs["ut_vib_rms"],
            dit=config["sci_dit"],
            ft_freq=ft_freq,
            strehl_ratio=config["strehl_ratio"],
            tau0=config["tau0"],
        )

        ft_signal = gv.get_ft_signal(plot=show_plot)

        ft_phot = gv.get_ft_phot(ft_signal)

        ft_snr[i] = gv.get_ft_snr(ft_phot)

        sig_opd[i] = gv.get_opd_rms(ft_snr[i])

        # print(f"FT freq: {ft_freq} hz")
        # print(f"ft_photons: {ft_phot}")
        # print(f"ft_snr: {ft_snr[i]}")
        # print(f"sig_opd: {sig_opd[i]} nm")

    if trace:
        print(f"sig_opds: {sig_opd} nm")

    # Use best (ie min):
    best = sig_opd.argmin()

    # best_ft_freq = ft_dits[best]
    # print(f"best FT freq: {best_ft_freq} hz")

    # best_ft_dit = 1000.0 / best_ft_freq
    # print(f"best FT dit:  {best_ft_dit} ms")

    best_sig_opd = sig_opd[best]
    # print(f"best FT sigma_opd: {best_sig_opd} nm")

    ft_vis_loss = gravmodel.ft_vis_loss(best_sig_opd, sci_wavelength)
    # print(f"ft_vis_loss: {ft_vis_loss}")

    offaxis_loss = gravmodel.elhalkouj_vis_loss(
        tel_diam,
        config_turbulence["seeing"],
        config_turbulence["h_0"],
        distance_ft_sci_as,
        sci_wavelength,
    )
    # print(f"offaxis_loss: {offaxis_loss}")

    total_coherence_loss = ft_vis_loss * offaxis_loss

    return total_coherence_loss


def compute_ASPRO_GRAVITY_UT(target_ft_mag, SR_ft, distance_ft_sci_as):
    if trace:
        print(f"compute_ASPRO_GRAVITY_UT(target_ft_mag={target_ft_mag}, SR_ft={SR_ft}, distance_ft_sci_as={distance_ft_sci_as})")

    config_obs = {}
    config_obs["tel"] = "UT"
    config_obs["res"] = "MEDIUM"  # or 'MEDIUM' or 'HIGH'
    config_obs["pol"] = "COMBINED"  # or 'SPLIT'
    config_obs["mode"] = "DUAL"  # or 'DUAL'

    # target properties:
    config = {}
    config["ft_mag"] = target_ft_mag

    config["strehl_ratio"] = SR_ft
    config["tau0"] = config_turbulence["tau0"]

    sci_wavelength = 2.2e-6
    tel_diam = 8.0  # UT

    ft_dits = [1000.0, 303.0, 100.0]

    ft_snr = np.zeros(len(ft_dits))
    sig_opd = np.zeros(len(ft_dits))

    for i, ft_freq in enumerate(ft_dits):
        # use model to get FT info:
        gv = asproGravFT.AsproGravFT(
            config["ft_mag"],
            tel=config_obs["tel"],
            res=config_obs["res"],
            pol=config_obs["pol"],
            mode=config_obs["mode"],
            ft_freq=ft_freq,
            strehl_ratio=config["strehl_ratio"],
            tau0=config["tau0"]
        )

        ft_signal = gv.get_ft_signal()

        ft_phot = gv.get_ft_phot(ft_signal)

        ft_snr[i] = gv.get_ft_snr(ft_phot)

        sig_opd[i] = gv.get_opd_rms(ft_snr[i])

        # print(f"FT freq: {ft_freq} hz")
        # print(f"ft_photons: {ft_phot}")
        # print(f"ft_snr: {ft_snr[i]}")
        # print(f"sig_opd: {sig_opd[i]} nm")

    if trace:
        print(f"sig_opds: {sig_opd} nm")

    # Use best (ie min):
    best = sig_opd.argmin()

    # best_ft_freq = ft_dits[best]
    # print(f"best FT freq: {best_ft_freq} hz")

    # best_ft_dit = 1000.0 / best_ft_freq
    # print(f"best FT dit:  {best_ft_dit} ms")

    best_sig_opd = sig_opd[best]
    # print(f"best FT sigma_opd: {best_sig_opd} nm")

    ft_vis_loss = asproGravFT.ft_vis_loss(best_sig_opd, sci_wavelength)
    # print(f"ft_vis_loss: {ft_vis_loss}")

    offaxis_loss = asproGravFT.elhalkouj_vis_loss(
        tel_diam,
        config_turbulence["seeing"],
        config_turbulence["h_0"],
        distance_ft_sci_as,
        sci_wavelength,
    )
    # print(f"offaxis_loss: {offaxis_loss}")

    total_coherence_loss = ft_vis_loss * offaxis_loss

    return total_coherence_loss


# --- score ---

def ranking_GRAVITY_UT(sci_Kmag, ft_Kmag, sci_ft_dist,
                       ao_mode, ao_Rmag, sci_ao_dist, ft_ao_dist):
    # 1. Strehl ratios:
    # print("--- 1. Strehl ---")

    strehl_ft = computeStrehl_UT_NGS(ao_mode, ao_Rmag, ft_ao_dist)
    # print(f"strehl_ft:  {strehl_ft}")

    strehl_sci = computeStrehl_UT_NGS(ao_mode, ao_Rmag, sci_ao_dist)
    # print(f"strehl_sci: {strehl_sci}")

    # 2. FT SNR:
    # print("--- 2. GRAVITY FT ---")

    total_vis_loss = compute_GRAVITY_UT(ft_Kmag, strehl_ft, sci_ft_dist)
    # print(f"total_vis_loss: {total_vis_loss}")

    # 3. ranking:
    # TODO: use sci_Kmag to rank science targets too:
    return strehl_sci * total_vis_loss


def ranking_GRAVITY_UT_NGS(sci_Kmag, ft_Kmag, sci_ft_dist,
                           ao_Rmag, sci_ao_dist, ft_ao_dist):
    """  interface use by the jmmc python webservice """

    ao_mode = "NGS_VIS"
    strehl_ft = computeStrehl_UT_NGS(ao_mode, ao_Rmag, ft_ao_dist)
    strehl_sci = computeStrehl_UT_NGS(ao_mode, ao_Rmag, sci_ao_dist)

    # Use Aspro2 (derived) transmission tables:
    if True:
        total_vis_loss = compute_ASPRO_GRAVITY_UT(ft_Kmag, strehl_ft, sci_ft_dist)
    else:
        total_vis_loss = compute_GRAVITY_UT(ft_Kmag, strehl_ft, sci_ft_dist)

    # print(f"total_vis_loss:       {total_vis_loss}")
    score = strehl_sci * total_vis_loss

    if score < 1e-3:
        score = 0.0
    return score

# --- main ---
if __name__ == "__main__":
    print(f"seeing: {config_turbulence['seeing']} as")
    print(f"tau0:   {config_turbulence['tau0']} ms")

    score = ranking_GRAVITY_UT_NGS(12.0, 5.0, 3.0, 7.0, 2.0, 5.0)
    print(f"Score: {score}")

    # limit FT on UT ~ 9+3 = 12
    score = ranking_GRAVITY_UT_NGS(12.0, 12.0, 3.0, 7.0, 2.0, 5.0)
    print(f"Score: {score}")
