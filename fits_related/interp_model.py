import numpy as np

"""
infreqs 
 - should contain the input frequencies of the model cube so you populate that with the CRVAL3 values you get from the headers
outfreqs 
 - should contain the frequencies where you want to evaluate the new higher resolution cube. So let's say you pass in a cube with 8 imaging bands and you want to evaluate it at 32 bands, then you need to divide the frequency range by 32 to get your output frequencies. Remember to also change the CRVAL3 and CDELT3's of the output fits files when you save them

And yes, ref_freq is the ref_freq from the spw subtable
Is that clear?


I think you should be able to use the subtract-model option. A good test would be to clean with a low frequency resolution and let wsclean write out residuals for you. Then you interpolate with your script and rerun using the subtract-model option and with no cleaning (i.e. niters=0). You can then compare the low resolution residuals with the high resolution residuals to see if they have improved. If the interpolation is working as expected you should get an improvement, especially for bright sources with steep spectra

"""

# concatenate the model images into a model cube
# concatenate their wsums
# get infreqs from crval3
# 


def interp_cube(model, wsums, infreqs, outfreqs, ref_freq, spectral_poly_order):
    """
    model: array
        model array containing model image's data for each stokes parameter
    wsums: list or array
        concatenated wsums for each of stokes parameters
    infreqs: list or array
        a list of input frequencies for the images?
    outfreqs: int
        Number of output frequencies i.e how many frequencies you want out
    ref_freq: float
        The reference frequency. A frequency representative of this spectral window, usually the sky frequency corresponding to the DC edge of the baseband. Used by the calibration system if a fixed scaling frequency is required or **in algorithms to identify the observing band**. 
        see https://casa.nrao.edu/casadocs/casa-5.1.1/reference-material/measurement-set
    spectral_poly_order: int
        the order of the spectral polynomial
    """

    # is this hdu0.data.shape?
    nband, nx, ny = model.shape
    

    #is this mask = np.any(hdu.data, axis=0) ?
    mask = np.any(model, axis=0)

    # components excluding zeros
    beta = model[:, mask]
    if spectral_poly_order > infreqs.size:
        raise ValueError("spectral-poly-order can't be larger than nband")

    # we are given frequencies at bin centers, convert to bin edges

    #delta_freq is the same as CDELt value in the image header
    delta_freq = infreqs[1] - infreqs[0] 

    wlow = (infreqs - delta_freq/2.0)/ref_freq
    whigh = (infreqs + delta_freq/2.0)/ref_freq
    wdiff = whigh - wlow

    # set design matrix for each component
    # look at Offringa and Smirnov 1706.06786
    Xfit = np.zeros([nband, spectral_poly_order])
    for i in range(1, spectral_poly_order+1):
        Xfit[:, i-1] = (whigh**i - wlow**i)/(i*wdiff)

    # we want to fit a function modeli = Xfit comps
    # where Xfit is the design matrix corresponding to an integrated
    # polynomial model. The normal equations tells us
    # comps = (Xfit.T wsums Xfit)**{-1} Xfit.T wsums modeli
    # (Xfit.T wsums Xfit) == hesscomps
    # Xfit.T wsums modeli == dirty_comps

    dirty_comps = Xfit.T.dot(wsums*beta)

    hess_comps = Xfit.T.dot(wsums*Xfit)

    comps = np.linalg.solve(hess_comps, dirty_comps)

    # now we want to evaluate the unintegrated polynomial coefficients
    # the corresponding design matrix is constructed for a polynomial of
    # the form
    # modeli = comps[0]*1 + comps[1] * w + comps[2] w**2 + ...
    # where w = outfreqs/ref_freq 
    w = outfreqs/ref_freq
    # nchan = outfreqs
    # Xeval = np.zeros((nchan, spectral_poly_order))
    # for c in range(spectral_poly_order):
    #     #Xeval[c,:] = w**c added by me
    #     Xeval[:, c] = w**c
    Xeval = np.tile(w, spectral_poly_order)**np.arange(spectral_poly_order)

    betaout = Xeval.dot(comps)

    modelout = np.zeros((nchan, nx, ny))
    modelout[:, mask] = betaout

    return modelout