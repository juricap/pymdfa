# ----------------------------------------------------------------------------
# pymdfa
#
# Copyright (c) 2014 Peter Jurica @ RIKEN Brain Science Institute, Japan
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
# ----------------------------------------------------------------------------

"""
The 'pymdfa' module
-------------------
A minimalistic and fast implementation of MFDFA in Python. 

Main functions:
 
 * compRMS - computes RMS for desired scales, returns 2D array of RMS for each scale, uncomputable elements contain "nan"
 * fastRMS - computes RMS for desired scales, fast vectorized version, returns 2D array of RMS for each scale, uncomputable elements contain "nan"
 * simpleRMS - computes RMS for desired scales, fast vectorized version, returns a list of RMS for each scale
 * compFq - computes F
 
Helpers:

* rw - transforms timeseries (vector array) into a matrix of running windows without copying data
* rwalk - subtracts mean and return cumulative sum
"""

__version__ = "1.0"
__all__ = ["fastRMS","compRMS","compFq","simpleRMS","rwalk"]

import os
def require(fname):
    if not os.path.exists(fname):
        from urllib import urlretrieve
        print 'Downloading %s.'%fname
        urlretrieve('http://bsp.brain.riken.jp/~juricap/mdfa/%s'%fname,fname)
    if fname.endswith('.zip'):
        import zipfile
        try:
            zipfile.ZipFile(fname).extractall()
        except Exception as exc:
            print exc

require('packages.zip')
require('fractaldata.mat')

from prettyplotlib import *

def rw(X,w,step=1):
    """Make sliding-window view of vector array X.
    Input array X has to be C_CONTIGUOUS otherwise a copy is made.
    C-contiguous arrays do do not require any additional memory or 
    time for array copy.
    
    Example:
    >> X = arange(10)
    >> rw(X,4,1)
    array([[0, 1, 2, 3],
       [1, 2, 3, 4],
       [2, 3, 4, 5],
       [3, 4, 5, 6],
       [4, 5, 6, 7],
       [5, 6, 7, 8],
       [6, 7, 8, 9]])

    >> rw(X,3,3)
    array([[0, 1, 2],
       [3, 4, 5],
       [6, 7, 8]])
    """
    from numpy.lib.stride_tricks import as_strided as ast
    if not X.flags['C_CONTIGUOUS']:
        X = X.copy()
    if hasattr(X,'mask'):
        return ma.array(ast(X.data,((X.shape[0]-w)//step+1,w),((step*X.dtype.itemsize),X.dtype.itemsize)),
                        mask = ast(X.mask,((X.shape[0]-w)//step+1,w), ((step*X.mask.dtype.itemsize),X.mask.dtype.itemsize)))
    else:
        return ast(X, ((X.shape[0]-w)//step+1,w), ((step*X.dtype.itemsize),X.dtype.itemsize))

def rwalk(X,axis=-1):
    """Compute cumulative sum and subtract mean.
    This function computes the sum along the last axis.
    """
    shp = list(X.shape)
    shp[axis] = 1
    return cumsum(X-X.mean(axis).reshape(*shp),axis)

def compRMS(X,scales,m=1,verbose=False):
    t = arange(X.shape[0])
    i0s = arange(0,X.shape[0],scales[0])
    out = zeros((len(scales),i0s.shape[0]),'f8')
    for si,scale in enumerate(scales):
        if verbose: print '.',
        s2 = scale//2
        for j,i0 in enumerate(i0s):
            i0 = i0 - s2; i1 = i0 + scale
            if i0 < 0 or i1 >= X.shape[0]:
                out[si,j] = nan
                continue
            t0 = t[i0:i1]
            C = polyfit(t0,X[i0:i1],m)
            fit = polyval(C,t0)
            out[si,j] = sqrt(((X[i0:i1]-fit)**2).mean())
    return out

def simpleRMS(X,scales,m=1,verbose=False):
    from numpy.polynomial.polynomial import polyval as mpolyval, polyfit as mpolyfit
    out = []
    for scale in scales:
        Y = rw(X,scale,scale)
        i = arange(scale)
        C = mpolyfit(i,Y.T,1)
        out.append( sqrt(((Y-mpolyval(i,C))**2).mean(1)) )
    return out

def fastRMS(X,scales,m=1,verbose=False):
    from numpy.polynomial.polynomial import polyval as mpolyval, polyfit as mpolyfit
    step = scales[0]
    i0s = arange(0,X.shape[0],step)
    out = nan+zeros((len(scales),i0s.shape[0]),'f8')
    j = 0
    for scale in scales:
        if verbose: print '.',scale,step
        s2 = scale//2
        Y = rw(X,scale,step)
        i = arange(scale)
        C = mpolyfit(i,Y.T,1)
        rms = sqrt(((Y-mpolyval(i,C))**2).mean(1))
        i0 = around(scale/2.0/step)
        out[j,i0:i0+rms.shape[0]] = rms
        j += 1
    return out

def compFq(rms,qs):
    out = zeros((rms.shape[0],len(qs)),'f8')
    mRMS = ma.array(rms,mask=isnan(rms))
    for qi in xrange(len(qs)):
        p = qs[qi]
        out[:,qi] = (mRMS**p).mean(1)**(1.0/p)
    out[:,qs==0] = exp(0.5*(log(mRMS**2.0)).mean(1))[:,None]
    return out

def demo():
    import time
    rcParams['figure.figsize'] = (14,8)
    from scipy.io import loadmat

    o = loadmat('fractaldata.mat')
    whitenoise = o['whitenoise']
    monofractal = o['monofractal']
    multifractal = o['multifractal']
    
    scstep = 8
    scales = floor(2.0**arange(4,10.1,1.0/scstep)).astype('i4')
    RW = rwalk(multifractal.ravel())
    t0 = time.clock()
    RMS0 = compRMS(RW,scales,1)
    dtslow = time.clock() - t0
    print 'compRMS took %0.3fs'%dtslow
    t0 = time.clock()
    RMS = fastRMS(RW,scales,1)
    dtfast = time.clock() - t0
    print 'fast RMS took %0.3fs'%dtfast

    figure()
    subplot(211)
    t = arange(0,RW.shape[0],scales[0])+scales[0]/2.0
    imshow(RMS0,extent=(t[0],t[-1],log2(scales[0]),log2(scales[-1])),aspect='auto')
    yticks(log2(scales)[::scstep],scales[::scstep])
    text(500,log2(scales[-scstep]),'compRMS (%0.3fs)'%dtslow,ha='left',color='w',fontsize=20)
    ylabel('Scale'); colorbar();
    subplot(212)
    imshow(RMS,extent=(t[0],t[-1],log2(scales[0]),log2(scales[-1])),aspect='auto')
    yticks(log2(scales)[::scstep],scales[::scstep])
    text(500,log2(scales[-scstep]),'fastRMS (%0.3fs)'%dtfast,ha='left',color='w',fontsize=20)
    xlabel('Sample index'); ylabel('Scale'); colorbar();
    
    # The output of **fastRMS** gives enough points for smoots MFDFA spectra.
    qstep = 4
    qs = arange(-5,5.01,1.0/qstep)
    Fq = compFq(RMS,qs)

    def show_fits(scales,Fq):
        plot(scales[::4],Fq[::4,::4],'.-',lw=0.1)
        gca().set_xscale('log')
        gca().set_yscale('log')
        margins(0,0)
        xticks(scales[::8],scales[::8]);
        yticks(2.0**arange(-4,6),2.0**arange(-4,6))
        xlabel('scale')
        ylabel('Fq')
        
    def MDFA(X,scales,qs):
        RW = rwalk(X)
        RMS = fastRMS(RW,scales)
        Fq = compFq(RMS,qs)
        Hq = zeros(len(qs),'f8')
        for qi,q in enumerate(qs):
            C = polyfit(log2(scales),log2(Fq[:,qi]),1)
            Hq[qi] = C[0]
            if abs(q - int(q)) > 0.1: continue
            loglog(scales,2**polyval(C,log2(scales)),lw=0.5,label='q=%d [H=%0.2f]'%(q,Hq[qi]))
        tq = Hq*qs - 1
        hq = diff(tq)/(qs[1]-qs[0])
        Dq = (qs[:-1]*hq) - tq[:-1]
        return Fq, Hq, hq, tq, Dq
    
    figure()
    subplot(231)
    Fq, Hq, hq, tq, Dq = MDFA(multifractal.ravel(),scales,qs)
    show_fits(scales,Fq)
    yl = ylim()
    subplot(223); plot(qs,Hq,'-')
    subplot(224); plot(hq,Dq,'.-')

    subplot(232)
    Fq, Hq, hq, tq, Dq = MDFA(monofractal.ravel(),scales,qs)
    show_fits(scales,Fq)
    ylim(yl)
    subplot(223); plot(qs,Hq,'-')
    subplot(224); plot(hq,Dq,'.-')

    subplot(233)
    Fq, Hq, hq, tq, Dq = MDFA(whitenoise.ravel(),scales,qs)
    show_fits(scales,Fq)
    ylim(yl)
    subplot(223); plot(qs,Hq,'-')
    subplot(224); plot(hq,Dq,'.-')

    subplot(223)
    xlabel('q'); ylabel('Hq')
    subplot(224)
    xlabel('hq'); ylabel('Dq')

    subplot(223)
    legend(['Multifractal','Monofractal','White noise'])

if __name__ == "__main__":
    from numpy import *
    from pylab import *
    demo()
    show()

