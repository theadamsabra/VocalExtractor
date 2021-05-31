import os
from re import L
import librosa
import numpy as np
import h5py

'''
Create a preprocessing class that enables audio to be sliced into segments of
any number of seconds.
'''

class Preprocess:
  '''
  Preprocess class to be inherited by various other classes to inherit the necessary preprocessing parameters.
  '''
  def __init__(self, spec_type = 'spec', n_fft = 1024, hop_length = 256, segment_length = 0.5, sr = 22050):
        '''
        Initialize parameters for processing audio.

        Parameters:
        ------------
        - spec_type (str)
        Type of spectrogram used for preprocessing. Currently only allowing for spec and mel_spec as inputs.

        -n_fft: (int)
        Number of points in each Fast Fourier Transformation. Keep to a power of 2. Default set to 1024.

        -hop_length (int)
        Number of points to hop from window to window. Default set to 256.

        -segment_length: (numeric)
        Length of segemented audio in seconds. Default set to 0.5 seconds.

        - sr: (int, optional)
        Samplerate of audio in question. Default set to 22050 to downsample.
        '''
        self.spec_type = spec_type
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.segment_length = segment_length
        self.sr = sr

class DSDPreprocess(Preprocess):
    def __init__(self, target,  dsd_path, data_path):
      '''
        Parameters:
        -----------
        - target: (str)
        Target source of extraction. Can be either 'vocals', 'bass', 'drums',
        or 'other'.

        -dsd_path: (str)
        Path of the DSD 100 dataset.

        -data_path: (str)
        Path of where processed data is going to be stored.
      '''
      super().__init__()
      # Dictionary initalized to store MFCCs of Mixture and Target.
      self.data = {
          'Mixture': [],
          'Target': []
      }
      self.target = target
      # If target is vocals, target file name is 'vocals.wav', and so on.
      self._target_file_name = self.target + '.wav'
      self.dsd_path = dsd_path
      self.data_path = data_path
      # Doesn't change in DSD100. Will be used only in self.preprocessing method.
      self._mixture_file_name = 'mixture.wav'

    def spec(self, sliced_array):
      '''
      Find Spectrogram of sliced audio.

      Parameters:
      -----------
      - sliced_array: (np.ndarray)
      Sliced segment of audio. Will be of length (segment_length * sr, 1)

      Returns:
      ---------
      Spectrogram of sliced array.
      '''
      
      if self.spec_type == 'spec':
        # Magnitude spectrogram
        spec = np.abs(librosa.core.stft(sliced_array, n_fft = self.n_fft, 
                                        hop_length=self.hop_length))
      elif self.spec_type == 'mel_spec':
        spec = librosa.feature.melspectrogram(sliced_array, sr=self.sr, n_fft=self.n_fft, hop_length=self.hop_length)
      else:
        raise Exception("Type of spectrogram not specified or wrong type. The only valid options are 'spec' and 'mel_spec'.")

      return spec 

    def preprocess(self, dev_test):
        '''
        Preprocess all data in DSD100 for the training of our model.

        Parameters:
        -----------
        - dev_test: (str)
        Accepts either "Dev" or "Test" as arguments. This will decide whether we are
        processing the training or test set. To process both, create a simple for loop:

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
                mix_file_path = os.path.join(mix_path, song, self._mixture_file_name)
                # Target:
                target_file_path = os.path.join(target_path, song, self._target_file_name)
                print(f'Now slicing {song}')
                print(f'\t Length of appended mixture array: {len(self.data["Mixture"])}')
                print(f'\t Length of appended target array: {len(self.data["Target"])}')
                fileM, _ = librosa.load(mix_file_path, sr=self.sr)
                fileT, _ = librosa.load(target_file_path, sr=self.sr)
                # Get length of the file
                # Take segments of however many seconds and find spectrograms
                num_segments = int(fileM.shape[0] / (self.segment_length * self.sr))
                for s in range(num_segments):
                    start = s * int(self.segment_length*self.sr)
                    end = start + int(self.segment_length*self.sr)
                    self.data['Mixture'].append(self.spec(fileM[start:end], self.spec_type))
                    self.data['Target'].append(self.spec(fileT[start:end], self.spec_type))
        file_path = os.path.join(self.data_path, dev_test + '.hdf5')
        # Save arrays as specific keys in hdf5 file
        with h5py.File(file_path, 'w') as f:
            f.create_dataset('mixture', data = self.data['Mixture'])
            f.create_dataset('target', data = self.data['Target'])
            # Empty self for next round of processing
            self.data['Mixture'] = []
            self.data['Target'] = []
            print(f'Writing in the file for {dev_test} has been complete.')

################################################################################
# # Set target
# target = 'vocals'
# # Set Spectrogram type
# spec_type = 'spec'
# # Have the DSD100 dataset in your current directory called DSD100
# dsd_path = 'path/to/DSD100'
# # Have a folder called data to store processed data
# data_path = 'path/to/stored/data/folder'
# p = DSDPreprocess(target, dsd_path, data_path)
# for dev_test in ['Dev', 'Test']:
#   p.preprocess(dev_test)
################################################################################