# import the necessary packages
from skimage.segmentation import slic
from skimage.segmentation import mark_boundaries
from skimage.util import img_as_float
from skimage import io
import numpy as np
import matplotlib.pyplot as plt
import argparse
# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required = True, help = "Path to the image")
# ap.add_argument("-d","--destination", required = True, help = "Path to destination for SLIC superpixels")
args = vars(ap.parse_args())
# load the image and convert it to a floating point data type
image = img_as_float(io.imread(args["image"]))
# loop over the number of segments
# print(args["destination"][16:]+'\n')
for numSegments in (200,):
	# apply SLIC and extract (approximately) the supplied number
	# of segments
	segments = slic(image, n_segments = numSegments, sigma = 5, compactness=10)
	print(segments)
	# np.savetxt('./SLIC200/'+args["destination"][16:]+".npy", segments)
	# show the output of SLIC
	# fig = plt.figure("Superpixels -- %d segments" % (numSegments))
	# ax = fig.add_subplot(1, 1, 1)
	# ax.imshow(mark_boundaries(image, segments))
	# plt.axis("off")
# show the plots
# plt.show()
# plt.savefig('~/Desktop'+args["destination"][16:]+'_slic4000.pdf')
