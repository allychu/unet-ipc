"""
Class for applying IPC transforms during U-Net training
"""

import phasepack
import numpy as np

class IPCTransform:
    def __init__(self, nscale=3, k=2.0):
        """
        nscale: Number of wavelet scales, try values 3-6
        k: No. of standard deviations of the noise energy
                            beyond the mean at which we set the noise threshold
                            point. You may want to vary this up to a value of
                            10 or 20 for noisy images
        """
        self.nscale = nscale
        self.k = k

    def __call__(self, input_image):
        """Apply the IPC transform,
        Using fast fourier transforms and
        monogenic filters for greater speed"""
        in_image = np.array(input_image)
        M, _, ft, _ = phasepack.phasecongmono(in_image,
                                                nscale=self.nscale,
                                                k=self.k)     
        """
        Add 2 channels:
        M: Maximum moment of phase congruency covariance, which can be used
            as a measure of edge strength
        ft: Local weighted mean phase angle at every point in the image. A
            value of pi/2 corresponds to a bright line, 0 to a step and -pi/2
            to a dark line
        """
        if len(in_image.shape) == 3:
            # if it's RGB input
            img_with_ipc = np.stack([in_image[:,:,0], 
                                     in_image[:,:,1], 
                                     in_image[:,:,2], 
                                     M, ft],
                                     axis=2)
        else:
            # if it's greyscale input
            img_with_ipc = np.stack([in_image,
                                     M, ft],
                                     axis=2)
        return img_with_ipc
    
    