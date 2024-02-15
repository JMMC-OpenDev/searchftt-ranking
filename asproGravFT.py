#####################################################################
# 'asproGravFT.py'
#
# Created: 2023.10 (yyyy.mm)
# Author: Laurent  Bourgès (JMMC - OSUG, CNRS)
# License: GPL3 (see LICENSE)
#
#####################################################################

import numpy as np


# from simulator.py:
def ft_vis_loss(opd_rms, sci_wavelength):
    """
    Calculate the coherence loss due to measurement noise in fringe tracking and
    coherence time.
    """
    sig_phi = opd_rms * 1e-9 * 2 * np.pi / sci_wavelength
    ft_vis_loss = np.exp(-(sig_phi)**2 / 2.0)
    return ft_vis_loss


def elhalkouj_vis_loss(tel_diam, seeing, h_turb, distance_ft_as, sci_wavelength):
    """
    Model of visibility loss from Elhalkouj+2008 due to atmospheric turbulence and
    separation between fringe tracker and science object
    """
    lambda_500 = 500 * 1e-9

    r0 = 0.98 * lambda_500 / (seeing * 0.48 * 1e-5)
    theta0 = 0.31 * (r0 / h_turb) / (0.48 * 1e-5)
    sigma_p = 0.12 * np.pi ** (1 / 3) * lambda_500 * (tel_diam / r0) ** (-1 / 6) * (distance_ft_as / theta0)
    vis_loss = np.exp(-2.0 * (np.pi * sigma_p / sci_wavelength) ** 2)

    return vis_loss


class AsproGravFT:

    def __init__(self, ft_mag, tel='UT', res='LOW',
                 pol='COMBINED', mode='SINGLE', ft_freq=1000.0,
                 strehl_ratio=0.4, tau0=3.2):

        self.ft_mag = ft_mag

        # Set the FT DIT
        self.ft_dit = 1.0 / ft_freq

        self.tel = tel
        self.resolution = res
        self.polarization = pol

        self.strehl_ratio = strehl_ratio
        self.tau0 = tau0 / 1000.0

        if mode in ['SINGLE', 'DUAL']:
            self.mode = mode
        else:
            raise ValueError('mode can only be SINGLE or DUAL')

        if tel == 'UT':
            self.collarea = 49.29      # meter^2
            self.vib_rms = 200.0       # nm
        elif tel == 'AT':
            self.collarea = 2.53       # meter^2
            self.vib_rms = 100.0       # nm
        else:
            raise Exception('tel has to be UT or AT')

        # Detector quantum efficiency
        self.ft_qe = 0.8

        # Same zero flux as Aspro2 (Band.K):
        if True:
            self.specphotons = 4.74099226559661E15
        else:
            self.specphotons = 4.53e15   # photons/(s m^2 m)

        # ASPRO2 GRAVITY_FT LOW data table:
        # resolution = "LOW" polarization = "COMBINED" mode = "DUAL"

        # lambda	dlambda	nb_photon_thermal	trans_AT	trans_UT
        dataTable = np.array([
            [2.000790e-06, 4.250e-08, 6.000e+03, 0.014771936, 0.022978567],
            [2.071220e-06, 4.250e-08, 6.000e+03, 0.065422218, 0.101767894],
            [2.165540e-06, 4.250e-08, 6.000e+03, 0.056927240, 0.088553484],
            [2.266240e-06, 4.250e-08, 6.000e+03, 0.036612801, 0.056953246],
            [2.356060e-06, 4.250e-08, 6.000e+03, 0.039008008, 0.060679123],
            [2.383110e-06, 4.250e-08, 6.000e+03, 0.036737830, 0.057147735],
        ])

        # Atmosphere transmission (aspro2 atm profile):
        self.ft_atm_trans = np.array([0.6658563730994165, 0.8392996080223335, 0.9854283009518033, 0.950518184966206, 0.8986955388903424, 0.8991685963419173])

        # Set the effective wavelength and effective wavelength band
        self.ft_wl = dataTable[:, 0]  # 1st column
        self.ft_wl_band = dataTable[:, 1]  # 2nd column

        # Set the throughputs
        trans_idx = 4 if tel == 'UT' else 3

        # 4th or 5th column (AT/UT):
        self.ft_throughput = dataTable[:, trans_idx]

        # FT bkg (/pix = 4):
        self.ft_bkg = dataTable[:, 2] / 4.0  # 3rd column

    def get_ft_signal(self):

        # Flux Calculations
        self.ftobjflux = 10**(-0.4 * self.ft_mag) * self.specphotons * self.ft_wl_band  # per channel
        self.ftrecflux = self.ftobjflux * self.collarea  # photons/sec*nm
        self.ftrecelectrons = self.ftrecflux * self.ft_qe  # electrons/channel

        det_signal = np.zeros_like(self.ft_wl)

        for i in range(len(self.ft_wl)):
            det_signal[i] = self.ftrecelectrons[i] * self.ft_throughput[i] * self.ft_atm_trans[i]

        # print(f"det_signal: {det_signal}")

        det_signal_final = det_signal * self.strehl_ratio * self.ft_dit

        # print(f"det_signal_final: {det_signal_final}")

        if self.mode == 'SINGLE':
            det_signal_final /= 2

        if self.polarization == 'SPLIT':
            det_signal_final /= 2

        self.ft_signal = det_signal_final

        # print(f"det_signal_final: {det_signal_final}")
        return det_signal_final

    def get_ft_phot(self, ft_signal):
        """
        Convert FT signal to FT photons (transmission depends)
        """
        # LBO: use N_FT = N / 4:
        if True:
            ft_phot = ft_signal / 4.0
        else:
            # from simulator.py:
            if self.polarization == 'SPLIT':
                ft_phot = ft_signal / 4.0  # 4 interferometric channels, 2 polarizations, 2 telescopes
            else:
                # polarization == 'COMBINED'
                ft_phot = ft_signal / 2.0  # 4 interferometric channels, 2 telescopes
        return ft_phot

    def get_ft_bkg_phot(self):
        return self.ft_bkg

    def getVis2SNR(self, vis2, num_samples, photons, bkg, readout):
        """
        SNR calculation for V2
        following ten Brummelaar 1997, Equ A13
        """

        nom = np.sqrt(num_samples) * vis2 * photons**2
        nn = photons + bkg
        denom = np.sqrt(2 * nn**3 * vis2 + nn**2 +
                        readout**2 * (2 * nn**2 * vis2 + 2 * nn + 1.0 / 4) +
                        readout**4)
        return nom / denom

    def get_ft_snr(self, ft_photons):
        ft_rn = 0.57
        ft_bkg = self.get_ft_bkg_phot() * self.ft_qe * self.ft_dit

        # print(f"ft_phot: {ft_photons}")
        # print(f"ft_bkg : {ft_bkg}")

        ftvis2snr = self.getVis2SNR(vis2=1.0, num_samples=1, photons=ft_photons, bkg=ft_bkg,
                                    readout=ft_rn)

        # print(f"ftvis2snr : {ftvis2snr}")

        if True:
            ftvis_snr = 2.0 * ftvis2snr
            # print(f"ftvis_snr : {ftvis_snr}")

            sigma_phi = 1.0 / ftvis_snr
            weight = ft_photons ** 2.0

            # print(f"sigma_phi : {sigma_phi}")
            # print(f"weight    : {weight}")

            sum_weight_mean = np.sum((sigma_phi * weight) ** 2.0)
            sum_weight = np.sum((1.0 * weight) ** 2.0)

            snrFT = np.sqrt(sum_weight / sum_weight_mean)

            # print(f"snrFT(aspro): {snrFT}")

            # LBO: use fixed formula (23.12.14):
            return snrFT

        # former formula (not really a weighted mean):
        varphi = ft_photons ** 4 / 4 / ftvis2snr ** 2
        ft_snr = np.sum(ft_photons ** 2) / np.sqrt(np.sum(varphi))

        # print(f"snrFT(taro) : {ft_snr}")
        return ft_snr

    def get_opd_rms(self, ft_snr):
        # LBO: fixed formula (23.12.14):
        opd_rms_snr = (4.0 * 2200.0) / (2.0 * np.pi * ft_snr)

        opd_rms_tau = np.power(self.ft_dit / (2.6 * self.tau0), 5.0 / 6.0) * (2200.0 / (2.0 * np.pi) )

        return np.sqrt(opd_rms_snr**2 + opd_rms_tau**2 + self.vib_rms**2)  # nm
