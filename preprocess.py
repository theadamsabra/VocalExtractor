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
# Preprocessing class
class Preprocess:
    # When initializing, we set up necessary constants needed for preprocessing.
    def __init__(self, target = 'vocals', sr = 44100, n_fft = 2048, n_mfcc = 13, hop_length = 512, num_chan = 2, dsd_path = 'C:\\Users\\19512\\Downloads\\DSD100\\DSD100',
    input_shape = None, data_path = 'C:\\Users\\19512\\OneDrive\\GitHub\\Audio-Source-Separation\\Audio-Source-Separation--Undergraduate-Thesis-\\data'):
        self.sr = sr # Samplerate
        self.n_fft = n_fft # Number of Fast Fourier Transforms we are performing
        self.n_mfcc = n_mfcc # Number of MFCCs we are incorperating
        self.hop_length = hop_length # How many points we're sliding over
        self.num_chan = num_chan # Number of channels of audio in question # Number of segments we want to split all audio into
        self.dsd_path = dsd_path # Path of DSD100
        self.input_shape = input_shape # None for now, will be tuple of shape
        self.data_path = data_path
        self.target = target
        self.data = {
            'Mixture MFCC': [],
            'Target MFCC': []
        }
    def multi_mfcc(self, audio, start, end):
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

        - start: (int)
        An integer designating the starting point where we want to slice the audio.

        - end: (int)
        An integer designating the end point where we want to slice the audio

        Returns:
        --------
        - mfcc: (array)
        The MFCC of each channel converted to a 3D array with shape of 
        (Elen, self.n_mfcc, self.num_chan)

        '''
        # Take MFCC of first channel and transpose for continuitity.
        mfcc1 = librosa.feature.mfcc(audio[start:end, 0], sr = self.sr,
                                    n_fft = self.n_fft, n_mfcc = self.n_mfcc, 
                                    hop_length = self.hop_length)
        # Take MFCC of second channel and transpose again.
        mfcc2 = librosa.feature.mfcc(audio[start:end, 1], sr = self.sr,
                                    n_fft = self.n_fft, n_mfcc = self.n_mfcc, 
                                    hop_length = self.hop_length)
        # Horizontally stack the 2 MFCCs and reshape to a 3D array.
        mfcc = np.hstack((mfcc1, mfcc2)).reshape(mfcc1.shape[1], mfcc1.shape[0], 
                                                self.num_chan)
        return mfcc
    def save_mfcc(self, mix, target):
        '''
        This function segments the mix and targets and takes its MFCC.

        Parameters:
        -----------
        mix: (array)
        The numpy array that contains the mixed song.

        target: (array)
        The numpy array that contains the target in question.
        '''
        # Number of samples per segment. A quarter of the sample rate is a quarter of a second
        samp_per_segment = int(self.sr / 4)
        # Expected value of length of segments. We'll ignore if it's shorter
        Elen = math.ceil(samp_per_segment / self.hop_length)
        # Expected value of segments in song.
        Eseg = math.ceil(mix.shape[1] / Elen)
        # Iterating through all segments of the song.
        print(Eseg, mix.shape[1], Elen)
        # for segment in range(Eseg):
        #     # Start and end of slice of song
        #     start = samp_per_segment * segment
        #     end = start + samp_per_segment
        #     # MFCCs of Mix:
        #     mix_mfcc = self.multi_mfcc(mix, start, end)
        #     # MFCCs of Vocals:
        #     tgt_mfcc = self.save_mfcc(target, start, end)
        #     # If the mfccs are the desired length, we use it.
        #     if (len(mix_mfcc) and len(tgt_mfcc)) == Elen:
        #         # Append to list for storage.
        #         self.data['Mixture MFCC'].append(mix_mfcc.tolist())
        #         self.data['Target MFCC'].append(tgt_mfcc.tolist())


    # Preprocess data.
    def preprocess(self, dev_test):
        '''
        This function goes to the path 

        Parameters:
        -----------
        - dev_test: (str)
        Accepts either "Dev" or "Test" as arguments. This will decide whether we are
        processing the training or test set.

        Returns:
        --------
        Processed data in data/test or data/train
        '''
        mixtures = 'Mixtures'
        source = 'Sources'
        mix_path = os.path.join(self.dsd_path, mixtures, dev_test)
        target_path = os.path.join(self.dsd_path, source, dev_test)
        for dirpath, dirnames, filenames in sorted(os.walk(mix_path)):
            # Ensure we're not at the root level
            if dirpath is not mix_path:
                # Get song name
                song = dirpath.split('\\')[-1]
                # Paths for files
                mix_file = os.path.join(mix_path, song, mixtures[:-1].lower() + '.wav')
                target_file = os.path.join(target_path, song, self.target + '.wav')
                # Loading in the data and performing preprocessing operations
                mix, samplerate = librosa.load(mix_file, sr = self.sr, mono = False)
                target, samplerate = librosa.load(target_file, sr = self.sr, mono = False)
                self.save_mfcc(mix, target)
                print('{} completed.'.format(song))
        # Determine whether or not we are saving the test or train data
        if dev_test == 'Test':
            json_path = os.path.join(self.data_path, (dev_test.lower() + '.json'))
        else:
            json_path = os.path.join(self.data_path, 'train.json')
        # Save data as json file
        with open(json_path, 'w') as json_file:
            json.dump(self.data, json_file, indent = 4)
            print('Writing in the json for {} has been complete.'.format(dev_test))

# Main function
if __name__ == "__main__":
    pre = Preprocess()
    for dev_test in ['Test', 'Dev']:
        pre.preprocess(dev_test)
