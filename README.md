# Vocal Extractor
## Implementation of U-Net Architecture for [Vocal Extraction](https://pdfs.semanticscholar.org/83ea/11b45cba0fc7ee5d60f608edae9c1443861d.pdf)

# About:

The paper by Jansson et al. implements a U-Net Convolutional Neural Network to attempt to extract the singing voice. The code within this repository has been tested and trains with no errors. However, I am currently building a "transfer" from processed audio back to something interpretable by humans. Once this is complete, a simple web app will be built for user interface. In that, users will be able to upload songs and be played back the extracted vocals.

In the meantime, if you would like to utilize the trained model do as such:
```
import tensorflow as tf

unet = tf.keras.models.load_model('my_model')
# View details of model:
model.summary()
```
