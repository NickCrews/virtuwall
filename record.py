#record.py
import sys
import numpy as np
import cv2
import kinect as kn
import view

def main(fname=None):
    k = kn.Kinect()
    v = view.PointCloudViewer()
    if fname is not None:
        svr = kn.Saver(fname)

    while (k.hasNextFrame()):
        frames = k.get_frames()
        pts, colors = k.makePointCloud(frames['undistorted'], frames['registered'])

        v.show(pts, colors)
        view.show_frames(frames)


        if fname is not None:
            svr.write(frames)

def usage():
    print('usage: python record.py <outfile> [color] [depth] [ir]')
    sys.exit()
             


if __name__ == '__main__':
    fname = sys.argv[1] if len(sys.argv) > 1 else None
    main(fname)