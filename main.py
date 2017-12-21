# main.py

import sys

import numpy as np
import cv2
import kinect as kn

import matplotlib.pyplot as plt

import view
import gabor

def main():

    cap1 = cv2.VideoCapture('data/highway_raw.AVI')
    cap2 = cv2.VideoCapture('data/rgb.avi')

    svr = kn.Saver('record.dat')


    frames = {}
    i = 0

    while(cap1.isOpened() and cap2.isOpened()):
        ret, frame1 = cap1.read()
        ret, frame2 = cap2.read()
        frames['depth'] = frame1
        frames['color'] = frame2

        svr.write(frames)

        key = cv2.waitKey(delay=1)
        if key == ord('q'):
            break
        if i > 20:
            break


def main2():
    src = kn.FakeKinect('recordings/wall1.rec')

    while (src.hasNextFrame()):
        frames = src.get_frames()
        c = frames['color']
        l,a,b = cv2.split(cv2.cvtColor(c, cv2.COLOR_BGR2LAB))
        h,s,v = cv2.split(cv2.cvtColor(c, cv2.COLOR_BGR2HSV))
        cv2.imshow('color', c)
        cv2.imshow('a', a)
        cv2.imshow('b', b)
        cv2.imshow('h', h)
        cv2.imshow('s', s)
        cv2.imshow('v', v)

        gabor.apply(c)

        key = cv2.waitKey(delay=0)
        if key == ord('q'):
            break


def main3():
    k = kn.FakeKinect('recordings/wall1.rec')
    # k = kn.Kinect()
    # svr = kn.Saver('recordings/wallmedian.rec')
    v = view.PointCloudViewer()
    # v2 = view.PointCloudViewer()

    # colors = MedianFilter()
    # undistorteds = FrameHistory()
    # depths = MedianFilter()
    # registereds = MedianFilter()


    # fgbg = cv2.createBackgroundSubtractorMOG2()
    # fgbg2 = cv2.createBackgroundSubtractorMOG2()

    while (k.hasNextFrame()):
        frames = k.get_frames()

        pts, colorspts = k.makePointCloud(frames['undistorted'], frames['registered'])
        v.show(pts, colorspts)

        view.show_frames(frames)

        if cv2.waitKey(0) == ord('q'):
            import sys
            sys.exit()



def absdiff(img1, img2):
    a = img1-img2
    b = np.uint8(img1<img2) * 254 + 1
    return a * b

def hist(img):
    plt.ion()
    plt.hist(img.ravel(), bins='auto')  # plt.hist passes it's arguments to np.histogram
    plt.title("Histogram with 'auto' bins")
    plt.show()
    plt.pause(.001)
    input('hi')

class FrameHistory():

    def __init__(self, history=50, min_samples=10):
        self._frames = []
        self._depth = False
        self.history = history
        self.min_samples = min_samples

    def add(self, frame):
        if frame.dtype == np.float32:
            self._depth = True
            # mask out invalid data (anything closer than 1mm or further away than 10m)
            masked = np.ma.masked_outside(frame, 1, 10*1000)
            self._frames.append(masked)
        else:
            self._frames.append(frame)
        if len(self._frames) > self.history:
            self._frames.pop(0)

    def median(self):
        if self._depth:
            stacked = np.ma.array(self._frames)
            median = np.ma.median(stacked, axis=0).astype(np.float32)
            insignificant = np.where(stacked.count(axis=0) < self.min_samples)
            # print('insignificant: ', insignificant)
            # print(stacked)
            median[insignificant] = 0
            return median
        else:
            return np.median(self._frames, axis=0).astype(np.uint8)

    def repairedRecent(self, history=2):
        if self._depth:
            result = self._frames[-1].copy()
            # count -2, -3, -4, ..., -history+1, -history
            history = min(history, len(self._frames))
            for i in range(-2, -history-1, -1):
                # fill in any errors with data from previous frame
                errors = np.where(result==0)
                result[errors] = (self._frames[i])[errors]
            return result.astype(np.float32)
        else:
            return self._frames[-1].astype(np.uint8)



if __name__ == '__main__':
    main3()





