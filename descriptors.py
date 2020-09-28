#################################################################################################################
# A Modularized implementation for
# Image enhancement, extracting descriptors, clustering, evaluation and visualization
# Integrating all the algorithms stated above in one framework
# CONFIGURATION - LOAD_DATA - IMAGE_ENHANCEMENTS - IMAGE_DESCRIPTORS - CLUSTERING - EVALUATION_VISUALIZATION
# Implementation of the IMAGE_DESCRIPTOR class
# Author: Aya Saad
# Date created: 24 September 2019
#
#################################################################################################################

import cv2
from skimage import feature
#import scipy.spatial.distance as ssd
# convert the redundant n*n square matrix form into a condensed nC2 array
#distArray = ssd.squareform(distMatrix) # distArray[{n choose 2}-{n-i choose 2} + (j-i-1)] is the distance between points i and j

class Descriptor:

    def __init__(self, name, image):
        '''
        Class Descriptor constructor
        :param name:    Name of the Algorithm/model
        :param image:   Name of the image
        '''
        self.name = name
        self.image = image

    def algorithm(self, alg_name):
        '''
        Choose the feature extractor algorithm
        :param alg_name: Name of the algorithm
        '''
        if alg_name == 'HOG':
            return self._alg_hog()
        if alg_name == 'ORB':
            return self._alg_orb()
        elif alg_name == 'SIFT':
            return self._alg_sift()
        elif alg_name == 'SURF':
            return self._alg_surf()
        elif alg_name == 'FAST':
            return self._alg_fast()
        elif alg_name == 'BRIEF':
            return self._alg_brief()

    def _alg_hog(self):
        print('Extracting HOG decriptors for image ...')
        # extract Histogram of Oriented Gradients from the logo
        print(self.image.shape)
        fd, hog = feature.hog(self.image, orientations=9, pixels_per_cell=(3, 3),
                                          cells_per_block=(2, 2), transform_sqrt=True, block_norm="L1",
                                          visualize=True)
        return self.image, hog, fd

    def _alg_orb(self):
        print('Extracting ORB decriptors for image ...')

        # orb = cv.ORB_create(WTA_K=cv.NORM_HAMMING2)  # default nfeatures=500
        orb = cv2.ORB_create()
        # find the keypoints and descriptors with ORB
        kp, desc = orb.detectAndCompute(self.image, None)

        img = cv2.drawKeypoints(self.image, kp, None, (0, 0, 255), 4)

        if desc is None:
            print('None Object descriptor!!')
        else:
            print(len(desc), '  ', len(kp))
            # for d in desc:
            #  print(d)
        return img, kp, desc

    def _alg_sift(self):
        print('Extracting SIFT decriptors for image ...')
        sift = cv2.xfeatures2d.SIFT_create()

        #convert image to uint8
        #cvuint8 = cv2.convertScaleAbs(self.image)
        cv8 = (self.image * 255).astype("uint8")
        #from utils import visualize_input
        #visualize_input(cv8)

        kp, desc = sift.detectAndCompute(cv8, None)

        img = cv2.drawKeypoints(cv8, kp, None, (0, 0, 255), flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

        if desc is None:
            print('None Object descriptor!!')
        else:
            print(len(desc), '  ', len(kp))
            # for d in desc:
            #  print(d)
        return img, kp, desc

    def _alg_surf(self):
        print('Extracting SURF decriptors for image ...')
        surf = cv2.xfeatures2d.SURF_create()
        # find the keypoints and descriptors with SURF
        kp, desc = surf.detectAndCompute(self.image, None)

        img = cv2.drawKeypoints(self.image, kp, None, (0, 0, 255), 4)

        if desc is None:
            print('None Object descriptor!!')
        else:
            print(len(desc), '  ', len(kp))
            # for d in desc:
            #  print(d)
        return img, kp, desc

    def _alg_fast(self):
        print('Extracting FAST decriptors for image ...')
        fast = cv2.FastFeatureDetector_create()

        #orb = cv.ORB_create(nfeatures=100000, scoreType=cv.ORB_FAST_SCORE)
        #orb = cv.ORB_create(edgeThreshold=5, patchSize=31, nlevels=8, fastThreshold=10, scaleFactor=1.2, WTA_K=2,
        #                     scoreType=cv.ORB_HARRIS_SCORE, firstLevel=0, nfeatures=150000000) # cv.ORB_HARRIS_SCORE
        #orb = cv.ORB_create(edgeThreshold=2, patchSize=16, nlevels=8, fastThreshold=fast.getThreshold(), scaleFactor=1.2, WTA_K=2,
        #                    scoreType=cv.ORB_FAST_SCORE, firstLevel=0, nfeatures=15000) # cv.ORB_HARRIS_SCORE
        orb = cv2.ORB_create()
        # find and draw the keypoints
        kp = fast.detect(self.image, None)
        # Print all default params
        print("Threshold: {}".format(fast.getThreshold()))
        print("nonmaxSuppression:{}".format(fast.getNonmaxSuppression()))
        print("neighborhood: {}".format(fast.getType()))
        print("Total Keypoints with nonmaxSuppression: {}".format(len(kp)))
        kp2 , desc = orb.detectAndCompute(self.image, cv2.ORB_FAST_SCORE)
        img = cv2.drawKeypoints(self.image, kp, None, (0, 0, 255), flags=cv2.DrawMatchesFlags_DEFAULT)   # 4

        if kp is None:
          print('None Object descriptor!!')
        else:
          print(len(kp))
          # for d in desc:
          #  print(d)
        return img, kp, desc

    def _alg_brief(self, img):
        print('Extracting BRIEF decriptors for image ...')

        # Initiate FAST detector
        #star = cv.xfeatures2d.StarDetector_create()
        fast = cv2.FastFeatureDetector_create()
        # Initiate BRIEF extractor
        brief = cv2.xfeatures2d.BriefDescriptorExtractor_create()

        # find the keypoints with STAR
        #kp = star.detect(img, None)
        kp = fast.detect(img, None)
        # compute the descriptors with BRIEF
        kp, desc = brief.compute(img, kp)

        # find the keypoints with STAR
        img = cv2.drawKeypoints(img, kp, None, (0, 0, 255), 4)

        if desc is None:
          print('None Object descriptor!!')
        else:
          print(len(desc),'  ', len(kp))
          #for d in desc:
          #  print(d)
        return img, kp, desc


class Matcher:
    def match_images(self):
        pass
    pass

class BFMatcher(Matcher):
    #def match_images(self, img_a, img_b, kp_a, kp_b, desc_a, desc_b):
    def match_images(self, desc_a, desc_b):
        # initialize the bruteforce matcher
        # bf = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)
        bf = cv2.BFMatcher(cv2.NORM_L1, crossCheck=True)

        # match.distance is a float between {0:100} - lower means more similar
        matches = bf.match(desc_a, desc_b)
        matches = sorted(matches, key=lambda x: x.distance)
        #matching_result = cv2.drawMatches(img_a, kp_a, img_b, kp_b, matches[:50], None, flags=2)
        #cv2.imshow('Matching results ', matching_result)
        similar_regions = [i for i in matches if i.distance < 70]
        if len(matches) == 0:
            return -1
        regions = []
        for m in matches:
            regions.append(m.distance)
        return sum(regions)/len(matches) #len(similar_regions) / len(matches)


class FlannMatcher(Matcher):
    def match_images(self, img_a, img_b, kp_a, kp_b, desc_a, desc_b):
        # FLANN parameters
        FLANN_INDEX_KDTREE = 0
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=50)  # or pass empty dictionary

        flann = cv2.FlannBasedMatcher(index_params, search_params)

        matches = flann.knnMatch(desc_a, desc_b, k=2)

        # Need to draw only good matches, so create a mask
        matchesMask = [[0, 0] for i in range(len(matches))]

        # ratio test as per Lowe's paper
        for i, (m, n) in enumerate(matches):
            if m.distance < 0.7 * n.distance:
                matchesMask[i] = [1, 0]

        draw_params = dict(matchColor=(0, 255, 0),
                           singlePointColor=(255, 0, 0),
                           matchesMask=matchesMask,
                           flags=0)
        similar_regions = [1 for i, (m, n) in enumerate(matches) if m.distance < 0.7 * n.distance]
        for i, (m, n) in enumerate(matches):
            print('m.distance: ', m.distance)
            print('0.7 * n.distance: ', 0.7 * n.distance)
            print('n.distance: ', n.distance)
        #matching_result = cv2.drawMatchesKnn(img_a, kp_a, img_b, kp_b, matches, None, **draw_params)
        print('similar_regions: ', similar_regions)
        return (len(similar_regions) / len(matches))
