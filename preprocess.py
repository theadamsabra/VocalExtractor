import os
import librosa
import numpy as np
import math
import json

'''
There are plenty of ways to structure the preprocessing the data the way the
directory is structured. I chose to use a preprocessing class to keep things
organized in one object.
'''

class Config:
    # When initializing, we set up necessary constants needed for preprocessing.
    def __init__(self, target,  dsd_path, data_path, sr = 44100, n_fft = 2048, n_mfcc = 13, 
    hop_length = 512, frame_length = 4096, block_length = 4098, num_chan = 2):
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

        -frame_length: (int, optional)
        Length of frame window when streaming in our audio. Default set to 2048.

        -num_chan: (int,optional)
        Number of channels in the audio. Default set to 2.
        '''
        self.data = { # Dictionary initalized to store MFCCs of Mixture and Target.
            'Mixture': [],
            'Target': []
        }        
        self.target = target 
        self.dsd_path = dsd_path 
        self.data_path = data_path         
        self.sr = sr
        self.n_fft = n_fft 
        self.n_mfcc = n_mfcc 
        self.hop_length = hop_length
        self.frame_length = frame_length
        self.block_length = block_length
        self.num_chan = num_chan 

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
            # Loading in the mixtures as a generator function and taking MFCCs
            mix_block = librosa.stream(mix_file, block_length = Config.block_length,
            frame_length = Config.frame_length, hop_length = Config.hop_length)
            for m_blocks in mix_block:
                mfcc_block = librosa.feature.melspectrogram(m_blocks, sr=Config.sr, n_fft=Config.n_fft, 
                hop_length=Config.hop_length, center=False)
                if mfcc_block.shape[1] == 4102:
                    Config.data['Mixture'].append(mfcc_block.tolist())
            target_block = librosa.stream(target_file, block_length = Config.block_length,
            frame_length = Config.frame_length, hop_length = Config.hop_length)
            for t_blocks in target_block:
                tmfcc_block = librosa.feature.melspectrogram(t_blocks, sr=Config.sr, n_fft=Config.n_fft, 
                hop_length=Config.hop_length, center=False)
                if tmfcc_block.shape[1] == 4102:
                    Config.data['Target'].append(mfcc_block.tolist())
            print('{} is complete. \n Length of data lists are currently {} for mixtures and {} for target.'.format(song, 
            len(Config.data['Mixture']), len(Config.data['Target'])))
    # Determine whether or not we are saving the test or train data
    if dev_test == 'Test':
        json_path = os.path.join(Config.data_path, (dev_test.lower() + '.json'))
    else:
        json_path = os.path.join(Config.data_path, 'train.json')
    # Save data as json file
    with open(json_path, 'w') as json_file:
        # Dump data to JSON
        json.dump(Config.data, json_file, indent = 4)
        # Empty Config for next round of processing
        Config.data['Mixture'] = []
        Config.data['Target'] = []
    return 'Writing in the json for {} has been complete.'.format(dev_test)

# Main function
if __name__ == "__main__":
    target = 'vocals'
    dsd_path = 'C:\\Users\\19512\\Downloads\\DSD100\\DSD100'
    data_path = 'C:\\Users\\19512\\OneDrive\\GitHub\\Audio-Source-Separation\\Audio-Source-Separation--Undergraduate-Thesis-\\data'
    c = Config(target, dsd_path, data_path)
    for dev_test in ['Dev', 'Test']:
        preprocess(dev_test, c)