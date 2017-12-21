#view.py

import sys
import numpy as np
import cv2

from pyqtgraph.Qt import QtCore, QtGui
import pyqtgraph.opengl as gl

from multiprocessing import Process, Queue

class PointCloudViewer():

    WHITE3 = (1.0, 1.0, 1.0)
    WHITE4 =(1.0, 1.0, 1.0, 1.0)


    def __init__(self):
        self.q = Queue()

        self.p = Process(target=self.run, args = (self.q,))
        self.p.daemon = True
        self.p.start()

    def show(self, pts, colors):
        self.q.put((pts, colors))

    def run(self, q):
        #QT app
        app = QtGui.QApplication([])
        gl_widget = gl.GLViewWidget()
        gl_widget.show()
        gl_grid = gl.GLGridItem()
        gl_widget.addItem(gl_grid)

        #initialize some points data
        pts = np.zeros((1,3))
        colors = self.WHITE4

        sp = gl.GLScatterPlotItem(pos=pts)
        sp.setGLOptions('opaque') # Ensures not to allow vertexes located behinde other vertexes to be seen.

        gl_widget.addItem(sp)

        def draw():
            '''Convert the depth frame to a point cloud and show it.
            undistorted- the undistorted depth frame, after registration has been applied
            registered- the registered color frame. If supplied, the point cloud will be colored'''
            nonlocal pts, colors
            if not q.empty():
                pts, colors = q.get()
                if colors is None:
                    colors = self.WHITE4
                else:
                    # wherever replace all block points with white points so we can see them
                    colors[np.where((colors[:]==[0,0,0]).all(1))] = self.WHITE3

            # # Calculate a dynamic vertex size based on window dimensions and camera's position - To become the "size" input for the scatterplot's setData() function.
            v_rate = 8.0 # Rate that vertex sizes will increase as zoom level increases (adjust this to any desired value).
            v_scale = np.float32(v_rate) / gl_widget.opts['distance'] # Vertex size increases as the camera is "zoomed" towards center of view.
            v_offset = (gl_widget.geometry().width() / 1000)**2 # Vertex size is offset based on actual width of the viewport.
            v_size = v_scale + v_offset

            sp.setData(pos=pts, color=colors, size=v_size)
                # print(pts)
            # end draw()

        timer = QtCore.QTimer()
        timer.timeout.connect(lambda: draw())
        timer.start(50)

        QtGui.QApplication.instance().exec_()



def show_frames(frames):
    assert isinstance(frames,dict)
    for name in frames:
        cv2.imshow(name, frames[name])
    if cv2.waitKey(1) == ord('q'):
        import sys
        sys.exit()

def show(*imgs, **kargs):
    '''Convenience method to show a sequence of images'''
    if len(imgs) > 0:
        plt.figure(kargs.get('name','show'))
        result = imgs[0]
        for img in imgs[1:]:
            result = sideBySide(result, img)

        if len(result.shape) == 2:
            plt.imshow(result, cmap='gray')
        else:
             plt.imshow(color.r2b(result))
    doShow = kargs.get('imm', True)
    if doShow==True or doShow=='True':
        plt.tight_layout()
        plt.show()















