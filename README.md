# Vocal Extractor
## Implementation of U-Net Architecture for Vocal Extraction

# About:
The paper by Jansson et al. [1] implements a U-Net Convolutional Neural Network to attempt to extract the singing voice. This repository aims to rebuild the code from scratch to better understand the tools necessary to approach a task that is at the intersection of Deep Learning and Music.

## Why extract Vocals?
The pursuit of splitting songs into their individual components democratizes creativity, specifically in rap. What do I mean by this? Sampling is a technique in beat-making that sections off portions of songs to use as the backdrop for rappers to perform over. In essence, by allowing the separation of vocals to be used at a high-level, vocals can now be looked at as an instrument to enhance creative pursuits in production.

# Information about Data:
The dataset used to train the model is DSD100 [2]. In it, contains 100 songs (50/50 train/test) with the mixed songs as input and the four stems of the song: vocals, bass, drums, and other. For training and testing, only the mixed songs and its respective vocals were needed.

## Exploratory Data Analysis:
To get a taste of what the dataset entails, reading through the [EDA notebook](https://github.com/theadamsabra/VocalExtractor/blob/master/notebooks/EDA.ipynb) should get you caught up rather quickly.

Furthermore, understanding how audio is processed for various ML-based audio applications in foundational. For this, refer to the [Spectrograms notebook]()


# References:
[1] Jansson, A. et al. “Singing Voice Separation with Deep U-Net Convolutional Networks.” ISMIR (2017). https://pdfs.semanticscholar.org/83ea/11b45cba0fc7ee5d60f608edae9c1443861d.pdf

[2]  Antoine  Liutkus,   Fabian-Robert  Stoter,   Zafar  Rafii,   Daichi  Kitamura,Bertrand  Rivet,  Nobutaka  Ito,  Nobutaka  Ono,  and  Julie  Fontecave.   The 2016  signal  separation  evaluation  campaign.   In  Petr  Tichavsky,  MassoudBabaie-Zadeh, Olivier J.J. Michel, and Nadege Thirion-Moreau, editors,La-tent Variable Analysis and Signal Separation - 12th International Confer-ence, LVA/ICA 2015, Liberec, Czech Republic, August 25-28, 2015, Pro-ceedings, pages 323–332, Cham, 2017. Springer International Publishing. https://sigsep.github.io/datasets/dsd100.html
