##
## Graham Kerr
## graham.s.kerr@nasa.gov; kerrg@cua.edu
##
## SPICE_anal.py
## 
## A set of useful routines to analyse SPICE data.
## THese are primarily designed to interact with objects created
## using sunraster, and also utilise routines in the sospice 
## libraries:
##
## https://docs.sunpy.org/projects/sunraster/en/latest/index.html
##
## https://sospice.readthedocs.io/en/stable/index.html
##
## There are probably (definitely!) more efficient or clever 
## ways to dal with SPICE data within the framework of sunraster 
## and sospice but for now I figured these separate routines
## can be useful for specifics whereas those libraries/tools can be 
## more general.
##
####################################################################
####################################################################

import numpy as np
import copy 

## These are because I haven't yet installed sospice 'properly'. You should 
## either replace the path to sospice, or if you installed it properly then 
## "import sospice" etc., should work
import sys
sys.path.insert(0,'/Users/gskerr1/Documents/Research/Python_Programs/sospice/sospice/')
from sospice import Release
from sospice import Catalog
from sospice import FileMetadata
from sospice import spice_error

import astropy.units as u
from astropy.io import fits
from astropy.coordinates import SkyCoord
from astropy.wcs.utils import wcs_to_celestial_frame
from astropy.nddata import NDUncertainty

from sunraster.instr.spice import read_spice_l2_fits

####################################################################
####################################################################
    
def print_windows(raster, verbose = True):
    '''
    Graham Kerr
    NASA/GSFC & CUA
    10th Jan 2024
   
    NAME:               print_windows

    PURPOSE:            To extract the wavelength windows from a SPICE L2
                        observion

    INPUTS:             raster -- A sunraster SPICE object  
                        
    OPTIONAL
    INPUTS:             verbose -- Prints the windows to screen

    OUTPUTS:            A list of strings 

    NOTES:               
 
    '''
    keys = list(raster.keys())
    if verbose == True:
        print(keys)
    return keys

####################################################################
####################################################################
    
def grab_wavel(raster, winid=None, verbose = True, nounit = False): 
    '''
    Graham Kerr
    NASA/GSFC & CUA
    10th Jan 2024
   
    NAME:               grab_wavel

    PURPOSE:            To extract the wavelength values from a SPICE L2
                        observion.

    INPUTS:             raster -- A sunraster SPICE object
                        winid -- A string noting which wavelength window 
                                 to use. Default is to use the first index.  
                        
    OPTIONAL
    INPUTS:             verbose -- Prints the windows to screen
                        nounits -- Removes the astropy units

    OUTPUTS:            The cell-centered wavelengths, in angstroms. 

    NOTES:              By default, the output is a list with a defined
                        astropy unit
 
    '''
    if winid == None:
        keys = print_windows(raster,verbose=False)
        winid = keys[0]
        print('No window ID set, using',keys[0])
    window = raster[winid]
    if verbose == True:
        print((window.axis_world_coords_values('em.wl')[0]).to(u.angstrom))
    if nounit == True:
        return (window.axis_world_coords_values('em.wl')[0]).to(u.angstrom).value
    else:
        return (window.axis_world_coords_values('em.wl')[0]).to(u.angstrom)

####################################################################
####################################################################
    
def grab_time(raster, winid=None, verbose = True, nounit = False): 
    '''
    Graham Kerr
    NASA/GSFC & CUA
    10th Jan 2024
   
    NAME:               grab_time

    PURPOSE:            To extract the time values from a SPICE L2
                        observion, using the WCS header info.

    INPUTS:             raster -- A sunraster SPICE object
                        winid -- A string noting which wavelength window 
                                 to use. Default is to use the first index.  
                        
    OPTIONAL
    INPUTS:             verbose -- Prints the windows to screen
                        nounits -- Removes the astropy units

    OUTPUTS:            The cell-centered time since the start of the raster, 
                        in seconds. 

    NOTES:              By default, the output is a list with a defined
                        astropy unit
 
    '''
    if winid == None:
        keys = print_windows(raster,verbose=False)
        winid = keys[0]
        print('No window ID set, using',keys[0])
    window = raster[winid]
    if verbose == True:
        print((window.axis_world_coords_values('time')[0][0]))
    if nounit == True:
        return (window.axis_world_coords_values('time')[0][0]).value
    else:
        return (window.axis_world_coords_values('time')[0][0]) 


####################################################################
####################################################################
    
def wavel2pix(ras_window, wavels, frame=None, Tx=0, Ty=0, 
              verbose=False, outputall=False): 
    '''
    Graham Kerr
    NASA/GSFC & CUA
    11th Jan 2024
   
    NAME:               wavel2pix

    PURPOSE:            To return pixel # given wavelengths 

    INPUTS:             ras_window -- A sunraster SPICE object that is 'window-ed', 
                                      which is that the specific window extracted
                                      from the raster object, 
                                      e.g. window = raster['Ly Beta 1025 - LH']

                                      This should have been 'sliced', such that
                                      the two dimensions are spatial (pixel lat 
                                      and long), and the third is spectral, 
                                      That is, time has been sliced out.
                                      e.g ndslice = window[0,:,:,:]
                        wavels -- float, or list of floats, representing 
                                  requested wavelengths in angstrom
                        
    OPTIONAL            Tx, Ty -- the longitude and latitude in arcsec
    INPUTS:             verbose -- Print some steps
                        outputall -- Return spatial pixels also 


    OUTPUTS:            Flt array containing the pixels corresponding  
                        to the input wavelengths


    NOTES:              While intended to be general, this might be 
                        rather specific to certain observing modes/SPICE
                        fits files.

                        Note that dummy vars are used for the latitude 
                        and longitude, unless speficied otherwhise. 
                        It Tx and Ty are input also, and the outputall 
                        keyword switched on then the function instead 
                        returns [lon, lat, wavelength] pixels
    '''

    ## Set the celestial frame, to be used when creating the SkyCoords
    if frame == None:
        frame = ras_window.celestial
        # frame=wcs_to_celestial_frame(window.wcs))

    ## Extract the various pixels
    pix = ras_window.wcs.world_to_pixel(SkyCoord(Tx=Tx*u.arcsec, 
                                                Ty=Ty*u.arcsec, 
                                                frame=frame),
                          wavels*u.angstrom)

    if outputall == True:
        if verbose==True:
            print('>>> Outputting spatial (dim 0 & 1), and spectral (dim 2) pixel #s')
        return pix
    else:
        if verbose==True:
            print('>>> Outputting only spectral pixel #s')
        return pix[-1]

####################################################################
####################################################################

def wintegrate_rebin(ras_window, w1=None, w2=None, wavels=[],
                     noconvert=False): 
    '''
    Graham Kerr
    NASA/GSFC & CUA
    15th Jan 2024
   
    NAME:               wintegrate_rebin

    PURPOSE:            To integrate over wavelength, using ndslice's in-built
                        rebing method, given which pixels to integrate over.

                        By default, it is assumed that the intensity units are
                        in per W/m^2/sr/nm, and the sum is multiplied by the spectral 
                        bin size to obtain W/m^2/sr. This can be suppressed by the 
                        keyword noconvert = True. 

    INPUTS:             ras_window -- A sunraster SPICE object that is 'window-ed', 
                                      which is that the specific window extracted
                                      from the raster object, 
                                      e.g. window = raster['Ly Beta 1025 - LH']

                                      This should have been 'sliced', such that
                                      the two dimensions are spatial (pixel lat 
                                      and long), and the third is spectral, 
                                      That is, time has been sliced out.
                                      e.g ndslice = window[0,:,:,:]
                        
    OPTIONAL            
    INPUTS:             wavels -- float, or list of floats
                                  Wavelengths of the observing window in angstrom. 
                                  Default is wavels = None, and they are 
                                  read from the ras_slice object.
                        w1, w2 -- The wavelengths to integrate over, in angstrom.
                                  Default is that w1 = w2 = None, and the range 
                                  is taken to be the start and end of the wavels
                                  array.
                        noconvert -- bool, default = False
                                     If False then the integrated wavelengths are 
                                     multiplied by the pixel spacing in nm, since 
                                     the L2 SPICE data are in W/m^2/sr/nm. If other 
                                     units, or DN, are used then set to True and
                                     manually deal with pixel scale outside the 
                                     function.
                        
    OUTPUTS:            An NDCUBE object containing the integrated intensities of the 
                        data. The WCS coords of that object match the input, and
                        know that the rebinning has taken place. 
                        

    NOTES:              While intended to be general, this might be 
                        rather specific to certain observing modes/SPICE
                        fits files.

                        There is probably a way to drop the dummy spectral axis
                        from the output object... should investigate that.

                        
    '''

    ## The wavelength axis
    if len(wavels) == 0:
        wavels = (ras_window.axis_world_coords_values('em.wl')[0]).to(u.angstrom).value


    ## If required, set the wavelength integration limits
    if w1 == None:
        w1 = wavels[0].value
        print(w1)
    if w2 == None:
        w2 = wavels[-1].value

    ## Extract a slice so that we can grab wavelength ranges in pixels
    ind = 0
    ndslice = ras_window[0,:,:,ind]
    wlimpix = wavel2pix(ndslice, [w1,w2], outputall=False, verbose=False)
    wind1 = int(np.round(wlimpix)[0])
    wind2 = int(np.round(wlimpix)[1])

    ## Number of pixels
    nw = (wind2-wind1)+1

    ## Extract a slice that corresponds to the wavelength range, then rebin using 
    ## numpy's nansum function. 
    ndslice_wrange = ras_window[0,wind1:wind2+1,:,:]
    ndslice_wrange_integ = ndslice_wrange.rebin((nw,1,1), operation=np.nansum)

    ## Multiply by the pixel spacing
    if noconvert == False:
        ndslice_wrange_integ*=ndslice.meta['CDELT3']

    return ndslice_wrange_integ
    
####################################################################
####################################################################
    
def wintegrate_trapz(ras_window, w1=None, w2=None, wavels=[],
                     noconvert=False): 
    '''
    Graham Kerr
    NASA/GSFC & CUA
    15th Jan 2024
   
    NAME:               wintegrate_trapz

    PURPOSE:            To integrate over wavelength, using the trapezoidal 
                        rule. This will be rather clunky, by in effect 
                        going through the rebin process in order to grab 
                        the correct WCS slice, but will replace the data array
                        at the end.

                        By default, it is assumed that the intensity units are
                        in per W/m^2/sr/nm, and the sum is multiplied by the spectral 
                        bin size to obtain W/m^2/sr. This can be suppressed by the 
                        keyword noconvert = True. 

    INPUTS:             ras_window -- A sunraster SPICE object that is 'window-ed', 
                                      which is that the specific window extracted
                                      from the raster object, 
                                      e.g. window = raster['Ly Beta 1025 - LH']

                                      This should have been 'sliced', such that
                                      the two dimensions are spatial (pixel lat 
                                      and long), and the third is spectral, 
                                      That is, time has been sliced out.
                                      e.g ndslice = window[0,:,:,:]
                        
                                               
    OPTIONAL            
    INPUTS:             wavels -- float, or list of floats
                                  Wavelengths of the observing window in angstrom. 
                                  Default is wavels = None, and they are 
                                  read from the ras_slice object.
                        w1, w2 -- The wavelengths to integrate over, in angstrom.
                                  Default is that w1 = w2 = None, and the range 
                                  is taken to be the start and end of the wavels
                                  array.
                        noconvert -- bool, default = False
                                     If False then the integrated wavelengths are 
                                     multiplied by the pixel spacing in nm, since 
                                     the L2 SPICE data are in W/m^2/sr/nm. If other 
                                     units, or DN, are used then set to True and
                                     manually deal with pixel scale outside the 
                                     function.
                        
    OUTPUTS:            An NDCUBE object containing the integrated intensities of the 
                        data. The WCS coords of that object match the input, and
                        know that the rebinning has taken place. 
                        

    NOTES:              While intended to be general, this might be 
                        rather specific to certain observing modes/SPICE
                        fits files.

                        You *could* in theory not pass the wavelength array to 
                        np.trapz, and instead multiply the result by the pixel 
                        spacing... that doesn't quite give the same result but 
                        it's pretty darn close. 

                        
    '''
    ## The wavelength axis
    if len(wavels) == 0:
        wavels = (ras_window.axis_world_coords_values('em.wl')[0]).to(u.angstrom).value


    ## If required, set the wavelength integration limits
    if w1 == None:
        w1 = wavels[0].value
        print(w1)
    if w2 == None:
        w2 = wavels[-1].value

    ## Extract a slice so that we can grab wavelength ranges in pixels
    ind = 0
    ndslice = ras_window[0,:,:,ind]
    wlimpix = wavel2pix(ndslice, [w1,w2], outputall=False, verbose=False)
    wind1 = int(np.round(wlimpix)[0])
    wind2 = int(np.round(wlimpix)[1])

    ## Number of pixels
    nw = (wind2-wind1)+1

    ## Extract a slice that corresponds to the wavelength range, then rebin using 
    ## numpy's nansum function... this is just to create the wcs object (definitely 
    ## a better way exists to do this)
    ndslice_wrange = ras_window[0,wind1:wind2+1,:,:]
    ndslice_wrange_integ = ndslice_wrange.rebin((nw,1,1), operation=np.nansum)

    ## Grab the data array to be integrated by np.trapz
    data_tmp = ndslice_wrange.data
    ## Integrate over wavelength... assumes intensity in W/m^2/nm!
    if noconvert == False:
        data_tmp_integ = np.trapz(data_tmp, x = wavels[wind1:wind2+1]/10, axis = 0)
    else:
        data_tmp_integ = np.trapz(data_tmp, axis = 0)

    ndslice_wrange_integ.data[0,:,:] = data_tmp_integ

    return ndslice_wrange_integ

####################################################################
####################################################################

def uncertaintyL2(ras_window, verbose = False): 
    '''
    Graham Kerr
    NASA/GSFC & CUA
    16th Jan 2024
   
    NAME:               uncertaintyL2.py

    PURPOSE:            Calculates the error on L2 SPICE data, following
                        Huang,Z.,etal. A&A 673, A82 (2023).

                        This is largely just a wrapper for sospice.spice_error()
                        but places that output into the uncertainty field
                        of a sunraster object, keeping all the WCS info together.

                        The input array assumes data in W/m^2/sr/nm, and the output
                        errors are in those same units. However, note that the 
                        units field of the uncertainty object is, by requirement, the
                        same as the sunraster SPICE L2 object, which is ADU.

    INPUTS:             ras_window -- A sunraster SPICE object that is 'window-ed', 
                                      which is that the specific window extracted
                                      from the raster object, 
                                      e.g. window = raster['Ly Beta 1025 - LH']

                                      This should have been 'sliced', such that
                                      the two dimensions are spatial (pixel lat 
                                      and long), and the third is spectral, 
                                      That is, time has been sliced out.
                                      e.g ndslice = window[0,:,:,:]
                        
    OPTIONAL
    INPUTS:             verbose -- Prints info from sospice to screen.
                        

    OUTPUTS:            ras_window is returned, but this time the ras_window.uncertainty
                        field has been populated with the astropy NDUncertainty object.

    NOTES:              There are ways to deal with error propagation in sunraster, 
                        but I haven't put thought in there yet... this should be done.
                        There are other fields of the astropy NDUncertainty object that 
                        I have not yet populated.

    '''

    ## Use sospice to measure the errors on the data
    av_noise_contribution, sigma = spice_error(data = ras_window.data, 
                                               header = ras_window.meta,
                                               verbose = verbose)

    
    # uncL2.unit = u.watt/u.nanometer/u.steradian/u.meter/u.meter
  
    uncL2 = spiceL2_Unc(sigma['Total'], unit=ras_window.unit, copy=True)

    ras_window.uncertainty = uncL2

    return ras_window

### A class to hold the SPICE L2 Uncertainties... it is a work-in-progress
class spiceL2_Unc(NDUncertainty):
    
    @property
    def uncertainty_type(self):
        """``"abs"`` : `spiceL2_Unc` returns the absolute uncertainty."""
        return "abs"
    
    def _propagate_add(self, other_uncert, result_data, correlation):
        result_uncertainty = 0.0
        return result_uncertainty

    def _propagate_subtract(self, other_nddata, result_data):
        result_uncertainty = 0.0
        return result_uncertainty

    def _propagate_multiply(self, other_uncert, result_data, correlation):
        result_uncertainty = 0.0
        return result_uncertainty

    def _propagate_divide(self, other_uncert, result_data, correlation):
        result_uncertainty = 0.0
        return result_uncertainty
    
    def _data_unit_to_uncertainty_unit(self, value):
        return value

####################################################################
####################################################################


# def grab_celestial(raster, winid=None, verbose = True, nounit = False): 
#     '''
#     Graham Kerr
#     NASA/GSFC & CUA
#     10th Jan 2024
   
#     NAME:               grab_celestial

#     PURPOSE:            To extract the helioprojective latitude and longitude 
#                         from a SPICE L2 observion, using the WCS header info.

#     INPUTS:             raster -- A sunraster SPICE object
#                         winid -- A string noting which wavelength window 
#                                  to use. Default is to use the first index.  
                        
#     OPTIONAL
#     INPUTS:             verbose -- Prints the windows to screen
#                         nounits -- Removes the astropy units

#     OUTPUTS:            

#     NOTES:              By default, the output is a list with a defined
#                         astropy unit
 
#     '''
#     if winid == None:
#         keys = print_windows(raster,verbose=False)
#         winid = keys[0]
#         print('No window ID set, using',keys[0])
#     window = raster[winid]
#     if verbose == True:
#         print()

####################################################################
####################################################################
    
# def grab_hplong(raster, winid=None, verbose = True, nounit = False): 
#     '''
#     Graham Kerr
#     NASA/GSFC & CUA
#     10th Jan 2024
   
#     NAME:               grab_hplong

#     PURPOSE:            To extract the helioprojective longitude from a SPICE L2
#                         observion, using the WCS header info.

#     INPUTS:             raster -- A sunraster SPICE object
#                         winid -- A string noting which wavelength window 
#                                  to use. Default is to use the first index.  
                        
#     OPTIONAL
#     INPUTS:             verbose -- Prints the windows to screen
#                         nounits -- Removes the astropy units

#     OUTPUTS:            The cell-centered helioprojective longitude of each 
#                         pixel, in arcsec. 

#     NOTES:              By default, the output is a list with a defined
#                         astropy unit
 
#     '''
#     if winid == None:
#         keys = print_windows(raster,verbose=False)
#         winid = keys[0]
#         print('No window ID set, using',keys[0])
#     window = raster[winid]
#     if verbose == True:
#         print((window.axis_world_coords_values('custom:pos.helioprojective.lon')[0][0]).to(u.arcsec))
#     if nounit == True:
#         return (window.axis_world_coords_values('custom:pos.helioprojective.lon')[0][0]).to(u.arcsec).value
#     else: 
#         return (window.axis_world_coords_values('custom:pos.helioprojective.lon')[0][0]).to(u.arcsec)