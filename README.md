# AutoRetinalImageSegmentation

  This code is used for joint optic disc and cup segmentation from retinal fundus images. The basic idea includes two steps:
  (a) use a pretrained model to localize the optic disc and then crop out the ROI including optic disc
  (b) use your model to do segmentation based on the ROI

  From coarse to fine,  this idea can make your model focus on the most interested regions, thus leading to a better result. 

  The original idea can be found from "Joint Optic Disc and Cup Segmentation Based on Multi-label Deep Network and Polar Transformation"
  The pretianed models can be found from https://github.com/HzFu/MNet_DeepCDR/tree/master/deep_model
  
# Code environment
  Python 3.6
  Keras 2.2.4
  Tensorflow 1.9.0
  
  
 

