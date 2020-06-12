import numpy as np
from math import sqrt,exp,floor,ceil
from moviepy.editor import vfx
import cv2
from random import shuffle
from random import getrandbits
from moviepy.editor import VideoFileClip


def fade(clip): 
    """
    fade in or out to a white or black screen
    """
    white = bool(getrandbits(1))
    revert = bool(getrandbits(1))
    step = np.random.uniform(0.06,0.1)
    def pipe(frame,frame_no):
        if white:
            panel = 255*np.ones(frame.shape, dtype=np.uint8)  
        else:
            panel = np.ones(frame.shape, dtype=np.uint8)  

        if revert:
            fadein = max(1-step-frame_no*step,0)
        else:
            fadein =  min(frame_no*step,1-step)
        dst = cv2.addWeighted( frame, 1-fadein, panel, fadein, 0)
        return(dst)
    return(clip.fl(lambda gf,t : pipe(gf(t),int(t*clip.fps))))

def shift_channel(clip):
    """
    This function returns a video with a randomized colour channels,
     to compensate background changes
    """
    channel=[0,1,2]
    shuffle(channel)
    def pipe(image):
        return image[:,:,channel]
    return(clip.fl_image(pipe))

def shift_hue(clip,h_max=10):
    """
    This function returns a video with random hue changes
    """
    if h_max==None:
        return (clip)
    def pipe(x,h_max=10):
        h_shift=int(np.random.normal(-h_max,h_max))
        x=x.astype(np.uint8)
        hsv=cv2.cvtColor(x,cv2.COLOR_BGR2HSV)
        h=hsv[:,:,0]
        h_shift=h+h_shift
        h_rot=h_shift+180
        h=h_rot%180
        hsv[:,:,0]=h
        x_shift=cv2.cvtColor(hsv,cv2.COLOR_HSV2RGB)
        return(x_shift)
    return(clip.fl_image(pipe))

def bw(clip,chance=.1):
    ''''
    Returns black and white once in 10 times, can change value with randomness.
    '''
    if np.random.rand()<chance:
        return(clip.fx(vfx.blackwhite))
    else:
        return(clip)

def blur(clip):
    ''''
    Returns temporal blured videos that imitates out of focus cases in videos.
    '''
    def sort(array, num_peaks=2, start_ascending=True):
        if num_peaks is None:
            num_peaks = len(array) // 6
        sorted_ar = sorted(array)
        subarrays = [sorted_ar[i::num_peaks] for i in range(num_peaks)]
        for i, subarray in enumerate(subarrays, start=int(not start_ascending)):
            if i % 2:
                # subarrays are in ascending order already!
                subarray.reverse()
        return sum(subarrays, [])

    rand=np.random.rand(1,np.random.randint(int(.2*clip.duration*clip.fps),int(.5*clip.duration*clip.fps)))
    rand=np.sort(rand[0]*10)
    start=int(np.random.uniform(0,0.5*clip.duration)*clip.fps)
    randx=[i+i%2+1 for i in np.array(rand).astype(np.uint8)]
    array=np.ones(int(ceil(clip.fps*clip.duration))).astype(np.uint8)
    array[start:start+len(randx)]=randx

    def pipe(image,frame_no):
        return(cv2.GaussianBlur(image,(array[frame_no],array[frame_no]),0))
    return(clip.fl(lambda gf,t : pipe(gf(t),int(t*clip.fps))))


def artifical_flash(clip):
    ''''
    Returns artificial flash scenerios in the video.
    '''
    def gamma(image,frame_no):
        gamma=array[frame_no]
        image=image.astype(np.uint8)
        invGamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** invGamma) * 255
        for i in np.arange(0, 256)]).astype("uint8")
        image_gamma=cv2.LUT(image, table).astype(np.uint8)
        return (image_gamma)

    def sort(array, num_peaks, start_ascending=True):
        if num_peaks is None:
            num_peaks = len(array) // 6
        sorted_ar = sorted(array)
        subarrays = [sorted_ar[i::num_peaks] for i in range(num_peaks)]
        for i, subarray in enumerate(subarrays, start=int(not start_ascending)):
            if i % 2:
                # subarrays are in ascending order already!
                subarray.reverse()
        return sum(subarrays, [])

    rand_i=np.random.randint(0,clip.fps/2)
    samples=np.random.rand(1,2+int(rand_i*clip.duration))

    start=int(np.random.uniform(0.1*clip.duration,0.6*clip.duration)*clip.fps)

    flash_intensity=int(np.random.uniform(5,7))

    samples=sort(array=samples[0]*flash_intensity,num_peaks=np.random.randint(2,10))

    array=np.ones(int(ceil(clip.fps*clip.duration)))
    array[start:start+len(samples)]=samples

    return(clip.fl(lambda gf,t : gamma(gf(t),int(t*clip.fps))))



