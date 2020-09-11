import os
import librosa
import numpy as np
import math
import h5py

'''
There are plenty of ways to structure the preprocessing the data the way the
directory is structured. I chose to use a preprocessing class to keep things
organize the necessary constants.
'''

class Config:
    def __init__(self, target,  dsd_path, data_path, sr = 44100, n_fft = 2048,
    n_mfcc = 128, hop_length = 256, frame_length = 512, block_length = 2048,
    num_chan = 2):
        '''
        Initalizing necessary information to preprocess DSD100 dataset.

        Parameters:
        -----------
        - target: (str)
        Target source of extraction. Can be either 'vocals', 'bass', 'drums',
        or 'other'.

        -dsd_path: (str)
        Path of the DSD 100 dataset.

        -data_path: (str)
        Path of where processed data is going to be stored.

        - sr: (int, optional)
        Samplerate of audio in question. Default set to 44100.

        - n_fft: (int, optional)
        Number of Fast Fourier Transformations to perform on audio. Default set
        to 2048.

        -n_mfcc: (int, optional)
        Number of MFCCs to find. Default set to 128.

        -hop_length: (int, optional)
        Number of points to slide over to slice audio. Default set to 256.

        -frame_length: (int, optional)
        Length of frame window when streaming in our audio. Default set to 512.

        -block_length: (int, optional)
        Length of block for librosa.stream. Default set to 2048.

        -num_chan: (int,optional)
        Number of channels in the audio. Default set to 2.
        '''
        # Dictionary initalized to store MFCCs of Mixture and Target.
        self.data = {
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

def streaming(block, source, Config):
    '''
    Simplify librosa's streaming process for readability.

    Parameters:
    -----------
    - block: Generator function
    Iterable generator with all blocks of song.

    - source: (str)
    String of either 'Mixture' or 'Target'.

    - Config: (class)
    Configuration of needed information.
    '''
    for blocks in block:
        mfcc_block = librosa.feature.mfcc(blocks, sr=Config.sr,
        n_fft=Config.n_fft, n_mfcc = Config.n_mfcc,
        hop_length=Config.hop_length, center = False)
        if mfcc_block.shape[1] == 2042:
            Config.data[source].append(mfcc_block.tolist())

def preprocess(dev_test, Config):
    '''
    Preprocess all data.

    Parameters:
    -----------
    - dev_test: (str)
    Accepts either "Dev" or "Test" as arguments. This will decide whether we are
    processing the training or test set.

    - Config (class)
    Configuration of needed information.

    Returns:
    --------
    Processed data in data/Dev.hdf5 or data/Train.hdf5
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
            song = items[0].split('/')[-1]
            # Paths for files
            mix_file = os.path.join(mix_path, song,
            mixtures[:-1].lower() + '.wav')
            target_file = os.path.join(target_path, song,
            Config.target + '.wav')
            # Loading in the mixtures as a generator function and taking MFCCs
            mix_block = librosa.stream(mix_file,
            block_length = Config.block_length,
            frame_length = Config.frame_length, hop_length = Config.hop_length)
            streaming(mix_block, 'Mixture', Config)
            # Loading in the targets as a generator function and taking MFCCs
            target_block = librosa.stream(target_file,
            block_length = Config.block_length,
            frame_length = Config.frame_length, hop_length = Config.hop_length)
            streaming(target_block, 'Target', Config)
            print(f'{song} is complete')
    # Convert lists to numpy arrays
    # mix_array is automatically saved as dtype = 'float64'
    mix_array = np.array(Config.data['Mixture'])
    target_array = np.array(Config.data['Target'], dtype='float64')
    file_path = os.path.join(Config.data_path, dev_test + '.hdf5')
    # Save arrays as specific keys in hdf5 file
    with h5py.File(file_path, 'w') as f:
        f.create_dataset('mixture', data = mix_array)
        f.create_dataset('target', data = target_array)
        # Empty Config for next round of processing
        Config.data['Mixture'] = []
        Config.data['Target'] = []
        print('Writing in the file for {} has been complete.'.format(dev_test))

# Main function
if __name__ == "__main__":
    target = 'vocals'
    # Have the DSD100 dataset in your current directory called DSD100
    dsd_path = os.path.abspath('DSD100')
    # Have a folder called data to store processed data
    data_path = os.path.abspath('data')
    c = Config(target, dsd_path, data_path)
    for dev_test in ['Dev', 'Test']:
        preprocess(dev_test, c)
