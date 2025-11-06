import numpy as np

hann = lambda win: 0.5*(1-np.cos(2*np.pi*np.arange(win)/win))

def get_cepstrum(x):
    """
    Compute the cepstrum of an entire chunk of audio

    Parameters
    ----------
    x: ndarray(N)
        Audio samples
    
    Returns
    -------
    ndarray(N)
        Cepstrum
    """
    x = x*hann(x.size)
    F = np.abs(np.fft.rfft(x))
    F = np.fft.irfft(np.log(F+1e-8))
    return F

def get_z_score(c, delta, buff=0, start_buff=0):
    """
    Compute a z-score for the a correlation vector or cepstrum
    at a particular offset
    The mean/std are computed ignoring the offset location, and
    there is the option to ignore locations from the beginning or
    slightly to the left / slightly to the right of the location

    Parameters
    ----------
    c: ndarray(N)
        Correlation vector/cepstrum
    delta: int
        Delay at which to check for the pseudorandom sequence
    buff: int
        Buffer on either side of delta to ignore when computing mu/std
        for z-score
    start_buff: int
        Ignore this many from the start when computing mu/std 
        for z-score
    """
    cmu = np.array(c)
    if start_buff > 0:
        cmu[0:start_buff] = np.nan
    cmu[delta-buff:delta+buff+1] = np.nan
    mu = np.nanmean(cmu)
    std = np.nanstd(cmu)
    return (c[delta]-mu)/std


def get_local_peak_z(c, delta, search_width=3, buff=0, start_buff=0):
    """
    pass
    """
    start = max(delta - search_width, 0)
    end = min(delta + search_width + 1, len(c))
    
    local_idx = np.argmax(c[start:end]) + start
    local_val = c[local_idx]
    
    cmu = np.array(c)
    if start_buff > 0:
        cmu[0:start_buff] = np.nan
    cmu[local_idx-buff:local_idx+buff+1] = np.nan
    mu = np.nanmean(cmu)
    std = np.nanstd(cmu)
    z = (local_val - mu) / std
    return z
    
def moving_average(x, window_size=5):
    return np.convolve(x, np.ones(window_size)/window_size, mode='same')
    
    
def get_median_z_score(c, delta, buff=0, start_buff=0):
    """
    aaa
    """
    cmu = np.array(c)
    if start_buff > 0:
        cmu[0:start_buff] = np.nan
    cmu[delta-buff:delta+buff+1] = np.nan
    mu = np.nanmedian(cmu)
    mad = np.nanmedian(np.absolute(cmu - np.nanmedian(cmu)))
    return (c[delta]-mu)/mad    

