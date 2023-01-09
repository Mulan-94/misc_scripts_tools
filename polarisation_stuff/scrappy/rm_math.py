"""
Some useful equations from Brentjens 2005
Useful for some of the usual suspect calculations
"""
import numpy as np
from astropy.constants import c

# Get the value from this quantity
c = c.value

def chan_width_wavelengths(center_freq: float, chan_width: float):
    """Equation 35
    Channel width in wavelengths
    """
    cww = ((2 * c**2 * chan_width)/center_freq**2) \
        (1 + 0.5 * (chan_width/center_freq)**2)
    return cww

def freq_sqrt(lamsq):
    #speed of light in a vacuum
    global c

    lamsq = np.sqrt(lamsq)
    # frequency to wavelength
    freq_ghz = (c/lamsq)
    return freq_ghz

def lambda_sq(center_freq: float, chan_width: float):
    """Equation 34
    Calculate the lambda squared value
    """
    lsq = (c**2 / center_freq**2) * \
        (1 + ( 0.75 * (chan_width/center_freq)**2))
    return lsq

def lambda_sq_err(n_chans: int, lambda_sq_0: float, lambda_sq: tuple):
    """Equation 35
    Error in the lambda squared
    """
    lam_sums = 0
    for n in range(nchans):
        lam_sums += lambda_sq[n]**4 - lambda_sq_0**4
    err_lambda = np.sqrt(lam_sums/(n_chans - 1))
    return err_lambda

def linear_polzn(q, u):
    """Amplitude of the total polarised intensity"""
    linear = q + 1j*u
    return np.abs(linear)

def linear_polzn_error(q, u, q_err, u_err):
    """Appendix A.7
    Error in total linear polarised intensity
    """
    lpol = linear_polzn(q, u)
    err_sq = (q**2 * q_err**2)/ lpol**2 + \
             (u**2 * u_err**2)/lpol**2
    return np.sqrt(err_sq)

def polzn_angle(q, u):
    """Angle of polarisation"""
    angle = 0.5 * np.arctan2(u,q)
    return np.rad2deg(angle)

def polzn_angle_error(q, u, q_err, u_err):
    """Equation 54 and 55"""
    lpol = linear_polzn(q, u)
    err_sq = ((u**2 * q_err**2) + (q**2 * u_err**2)) \
            / (4 * lpol**4)
    return np.sqrt(err_sq)

def rm_error(pangle_err: float, lam_sq_err: float, n_chans: int):
    """Equation 52 and 53"""
    err_rm = pangle_err \
            / (lam_sq_err * np.sqrt(n_chans - 2))
    return err_rm

def frac_polzn(i, q, u):
    # this gets the amplitude of linear polarisation
    frac_pol = linear_polzn(q, u) / i
    return frac_pol

def frac_polzn_error(i, q, u, i_err, q_err, u_err):
    fpol = frac_polzn(i, q, u)
    p = linear_polzn(q, u)
    p_err = linear_polzn_error(q, u, q_err, u_err)
    res = np.abs(fpol) * np.sqrt(np.square((p_err/p)) + np.square((i_err/i)))
    return res


######################################
# some checks here
######################################


def polarised_snr(q: float, u: float, q_err: float, u_err: float):
    """
    From Feain 2012: Section 5
    Signal-to-noise ratio of the polarised signal
    Taken to be the ratio between the:
    1. Total polarised intensity
    2. Standard error of (1)
    """
    lpol = linear_polzn(q, u)
    lpol_err = linear_polzn_error(q, u, q_err, u_err)
    return lpol/lpol_err



######################################
# RM SYNTHESIS some values
######################################


def max_faraday_depth(center_freq: float, chan_width: float):
    """
    Equation 63
    Approximate max faraday depth at which one has more than 50% sensitivity
    """
    cw_wave = chan_width_wavelengths(center_freq, chan_width)
    mfd = 3**0.5 / cw_wave
    return mfd

def max_faraday_scale(lambda_sq_min: float):
    """
    Equation 62
    Scale in depth space where sensitivty is at 50%
    """
    mscale = np.pi / lambda_sq_min
    return mscale

def rmtf_resolution(lambda_sq_min: float, lambda_sq_max: float):
    """
    Equation 61
    Approximate FWHM of the RMSF
    """
    msens = (2 * 3**0.5)/(lambda_sq_max - lambda_sq_min)
    return msens

