import os
import librosa
import numpy as np
import math
import json
# from numba import njit

'''
There are plenty of ways to structure the preprocessing the data the way the
directory is structured. I chose to use a preprocessing class to keep things
organized in one object.
'''

class Config:
    # When initializing, we set up necessary constants needed for preprocessing.
    def __init__(self, target,  dsd_path, data_path, sr = 44100, n_fft = 2048, n_mfcc = 13, hop_length = 512, num_chan = 2):
        '''
        Initalizing necessary information to preprocess DSD100 dataset.

        Parameters:
        -----------
        - target: (str)
        Target source of extraction. Can be either 'vocals', 'bass', 'drums', or 'other'.

        -dsd_path: (str)
        Path of the dataset.

        -data_path: (str)
        Path of where processed data is going to be stored.

        - sr: (int, optional)
        Samplerate of audio in question. Default set to 44100.

        - n_fft: (int, optional)
        Number of Fast Fourier Transformations to perform on audio. Default set to 2048.

        -n_mfcc: (int, optional)
        Number of MFCCs to find. Default set to 13.

        -hop_length: (int, optional)
        Number of points to slide over to slice audio. Best to be around a quarter of self.n_fft. Default set to 512.

        -num_chan: (int,optional)
        Number of channels in the audio. Default set to 2.
        '''
        self.data = { # Dictionary initalized to store MFCCs of Mixture and Target.
            'Mixture MFCC': [],
            'Target MFCC': []
        }        
        self.target = target # Target source extraction
        self.sr = sr # Samplerate
        self.frame_length = int(self. sr / 4)
        self.n_fft = n_fft # Number of Fast Fourier Transforms we are performing
        self.n_mfcc = n_mfcc # Number of MFCCs we are incorperating
        self.hop_length = hop_length # How many points we're sliding over
        self.num_chan = num_chan # Number of channels of audio in question
        self.dsd_path = dsd_path # Path of DSD100
        self.data_path = data_path # Path of where processed data will be stored

def multi_mfcc(audio, Config):
    '''
    The main objective of multi_mfcc is to take the MFCC of a multi-channel
    audio source by taking the MFCC of each individual channel. While this
    function can be generalized for any number of channels, I am keeping it to 
    stereophonic audio for the time being.

    This function is mainly used to improve readability of the save_mfcc function
    defined below.

    Parameters:
    -----------
    - audio: (array)
    The audio we want to find the MFCCs of.

    - Config (class)
    Configuration of needed information.

    Returns:
    --------
    - mfcc: (array)
    The MFCC of each channel converted to a 3D array with shape of 
    (Elen, self.n_mfcc, self.num_chan)

    '''
    # Take MFCC of first channel and transpose for continuitity.
    mfcc1 = librosa.feature.mfcc(audio[0, :], sr = Config.sr,
                                n_fft = Config.n_fft, n_mfcc = Config.n_mfcc, 
                                hop_length = Config.hop_length)
    # Take MFCC of second channel and transpose again.
    mfcc2 = librosa.feature.mfcc(audio[1, :], sr = Config.sr,
                                n_fft = Config.n_fft, n_mfcc = Config.n_mfcc, 
                                hop_length = Config.hop_length)
    # Vertically Stack stack the 2 MFCCs and reshape to a 3D array.
    # (26,22) -> (2,22,13); we can change this later if needed.
    mfcc = np.vstack((mfcc1, mfcc2)).reshape(Config.num_chan, mfcc1.shape[1], mfcc1.shape[0])
    return mfcc
# @njit
def save_mfcc(mix, target, Config):
    '''
    This function segments the mix and targets and takes its MFCC.

    Parameters:
    -----------
    - mix: (array)
    The numpy array that contains the mixed song.

    - target: (array)
    The numpy array that contains the target in question.

    - Config (class)
    Configuration of needed information.

    Returns:
    ---------
    MFCCs as lists in Config.data
    '''
    # Segment audio using sliding window with a quarter second frame length.
    mix = librosa.util.frame(mix, Config.frame_length, Config.hop_length)
    target = librosa.util.frame(target, Config.frame_length, Config.hop_length)
    for i in range(mix.shape[2]):
        # MFCCs of vocals
        mix_mfcc = multi_mfcc(mix[:,:,i], Config)
        # MFCCs of Vocals:
        tgt_mfcc = multi_mfcc(target[:,:,i], Config)
        # Append to data
        Config.data['Mixture MFCC'].append(mix_mfcc.tolist())
        Config.data['Target MFCC'].append(tgt_mfcc.tolist())
    print('Song has been segmented and appended.')
# @njit
def preprocess(dev_test, Config):
    '''
    This function goes to the path 

    Parameters:
    -----------
    - dev_test: (str)
    Accepts either "Dev" or "Test" as arguments. This will decide whether we are
    processing the training or test set.

    - Config (class)
    Configuration of needed information.

    Returns:
    --------
    Processed data in data/test.json or data/train.json
    '''
    mixtures = 'Mixtures'
    source = 'Sources'
    mix_path = os.path.join(Config.dsd_path, mixtures, dev_test)
    target_path = os.path.join(Config.dsd_path, source, dev_test)
    for items in sorted(os.walk(mix_path)):
        # Ensure we're not at the root level
        # Items[0] is the directory path
        if items[0] is not mix_path:
            # Get song name
            song = items[0].split('\\')[-1]
            # Paths for files
            mix_file = os.path.join(mix_path, song, mixtures[:-1].lower() + '.wav')
            target_file = os.path.join(target_path, song, Config.target + '.wav')
            # Loading in the data and performing preprocessing operations
            mix, _ = librosa.load(mix_file, sr = Config.sr, mono = False)
            target, _ = librosa.load(target_file, sr = Config.sr, mono = False)
            save_mfcc(mix, target, Config)
            print('{} is complete'.format(song))
    # Determine whether or not we are saving the test or train data
    if dev_test == 'Test':
        json_path = os.path.join(Config.data_path, (dev_test.lower() + '.json'))
    else:
        json_path = os.path.join(Config.data_path, 'train.json')
    # Save data as json file
    with open(json_path, 'w') as json_file:
        json.dump(Config.data, json_file, indent = 4)
        print('Writing in the json for {} has been complete.'.format(dev_test))

# Main function
if __name__ == "__main__":
    target = 'vocals'
    dsd_path = 'C:\\Users\\19512\\Downloads\\DSD100\\DSD100'
    data_path = 'C:\\Users\\19512\\OneDrive\\GitHub\\Audio-Source-Separation\\Audio-Source-Separation--Undergraduate-Thesis-\\data'
    c = Config(target, dsd_path, data_path)
    for dev_test in ['Dev', 'Test']:
        preprocess(dev_test, c)