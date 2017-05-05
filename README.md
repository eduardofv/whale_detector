# whale_detector
Whale Detector for Kaggle's Right Whale Recognition Challenge

<a href="http://kaggle.com" target="_blank">Kaggle's</a> <a href="https://www.kaggle.com/c/noaa-right-whale-recognition" target="_blank">NOAA Right Whale Recognition Challenge</a> aims to develop an algorithm to identify individuals of Right Whales, which are critically endangered. It is a great chance to study machine learning and digital image processing although looks to me as a really hard challenge. Anyway I've developed this method to detect the whale in the photograph and I'm releasing it in a hope that it may help others.

It takes advantage of the fact that most pictures are pretty plain, with almost all of the area covered by water, and have a smaller region of interest which corresponds to the whale, so the <a href="http://docs.opencv.org/3.0.0/d6/dc7/group__imgproc__hist.html" target="_blank">histogram</a> for most of the image will be similar except on the region of  interest. The algorithm looks recursively to subimages that have an HSV histogram not similar to the original image's histogram, marking those regions in white and else on black. Then searches for the biggest continuous region using contours and places a bounding box around it, assuming it's the whale. The image is called "extract" and is saved along the black & white mask. 
Uses Python 2.7 and <a href="http://opencv.org/opencv-3-0.html" target="_blank">OpenCV 3.0.</a> 

Original Image:
<img src="https://raw.githubusercontent.com/eduardofv/whale_detector/master/w_7489.jpg" alt="whale" style="max-width:500px" />

Whale found:
<img src="https://raw.githubusercontent.com/eduardofv/whale_detector/master/w_7489.jpg.areas.jpg" alt="whale" style="max-width:500px" />

Areas found mask:
<img src="https://raw.githubusercontent.com/eduardofv/whale_detector/master/w_7489.jpg.mask-areas.jpg" alt="whale" style="max-width:500px" />

ROI Mask:
<img src="https://raw.githubusercontent.com/eduardofv/whale_detector/master/w_7489.jpg.mask.jpg" alt="whale" style="max-width:500px" />

ROI Extract:
<img src="https://raw.githubusercontent.com/eduardofv/whale_detector/master/w_7489.jpg.extract.jpg" alt="whale" style="max-width:500px" />

## Running with docker, Python

The jupyter version (hist_zones.ipynb) works well with a docker image that contains OpenCV 3 and Python 3  as described <a href="http://to.predict.ch/hacking/2017/02/12/opencv-with-docker.html">here</a>. Just modify the last line of the Dockerfile to "CMD /usr/local/bin/jupyter-notebook --ip=0.0.0.0 --allow-root" to for root access.

