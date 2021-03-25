import os
import librosa
import numpy as np
import h5py

'''
Create a preprocessing class that enables audio to be sliced into segments of
any number of seconds. Default is set to 1 second at a samplerate/frequency of 8192 Hz.
'''

class Preprocess:
    def __init__(self, target,  dsd_path, data_path, segment_length = 1, sr = 8192):
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

        -segment_length: (int)
        Length of segemented audio in seconds. Default set to 1 second.

        - sr: (int, optional)
        Samplerate of audio in question. Default set to 8192 to downsample.
        '''
        # Dictionary initalized to store MFCCs of Mixture and Target.
        self.data = {
            'Mixture': [],
            'Target': []
        }
        self.target = target
        self.dsd_path = dsd_path
        self.data_path = data_path
        self.segment_length = segment_length
        self.sr = sr
    
    def preprocess(self, dev_test):
        '''
        Preprocess all data.

        Parameters:
        -----------
        - dev_test: (str)
        Accepts either "Dev" or "Test" as arguments. This will decide whether we are
        processing the training or test set. To process both, create a simple for loop:

        for dev_test in ['Dev', 'Test']:
            Preprocess.preprocess(dev_test)

        Returns:
        --------
        Processed data in data/Dev.hdf5 or data/Train.hdf5 or both.
        '''
        mixtures = 'Mixtures'
        source = 'Sources'
        mix_path = os.path.join(self.dsd_path, mixtures, dev_test)
        target_path = os.path.join(self.dsd_path, source, dev_test)
        for items in sorted(os.walk(mix_path)):
            # Ensure we're not at the root level
            # Items[0] is the directory path
            if items[0] is not mix_path:
                # Get song name
                song = items[0].split('/')[-1]
                # Paths for files
                # Mix:
                mix_file = os.path.join(mix_path, song,
                mixtures[:-1].lower() + '.wav')
                # Target:
                target_file = os.path.join(target_path, song,
                self.target + '.wav')
                print(f'Now slicing {song}')
                file, _ = librosa.load(mix_file, sr=self.sr)
                # Get length of the file
                # Take segments of however many seconds
                num_segments = int(file.shape[0] / (self.segment_length * self.sr))
                for s in range(num_segments):
                    start = s * (self.segment_length*self.sr)
                    end = start + (self.segment_length*self.sr)
                    # Verify shape for consistency
                    # Add sliced audio to list
                    self.data['Mixture'].append(mix_file[start:end])
                    self.data['Target'].append(target_file[start:end])
        # Convert lists to numpy arrays
        # mix_array is automatically saved as dtype = 'float64'
        mix_array = np.array(self.data['Mixture'], dtype = 'float64')
        target_array = np.array(self.data['Target'], dtype='float64')
        file_path = os.path.join(self.data_path, dev_test + '.hdf5')
        # Save arrays as specific keys in hdf5 file
        with h5py.File(file_path, 'w') as f:
            f.create_dataset('mixture', data = mix_array)
            f.create_dataset('target', data = target_array)
            # Empty self for next round of processing
            self.data['Mixture'] = []
            self.data['Target'] = []
            print('Writing in the file for {} has been complete.'.format(dev_test))