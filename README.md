## Description
This is a implementation of Face Aligment with simple cnn in Keras, which is the second step of my **FaceID system**. You can find another two repositories  as follows:
1. [Face-detection-with-mobilenet-ssd](https://github.com/bruceyang2012/Face-detection-with-mobilenet-ssd)
2. [Face-Alignment-with-simple-cnn](https://github.com/bruceyang2012/Face-Alignment-with-simple-cnn)
3. [Face-identification-with-cnn-triplet-loss](https://github.com/bruceyang2012/Face-identification-with-cnn-triplet-loss) (To do)

## Some Details
Today there are lots of excellent face alignment algorithms, but they are somehow too complex to implement, and most of methods based on deep learning don't meet the requirement of real-time, here I introduce an efficient method based on simple convolution neural network, which can realize real-time face feature points detection.

It costs me about only 10 minutes with cpu to train a model on a training set containing 7049 images. It's really fast, and the testing time is about 60ms per face. You can easily improve the accuray using different methods, such as make the convnet structure deeper or make a data augmentation and so on.

Here are some testing results.

![image](https://github.com/bruceyang2012/Face-Alignment-with-simple-cnn/raw/master/predicted.png)
