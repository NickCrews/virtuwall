# kinect.py
#
# a wrapper for the pylibfreenect2 interface, as well as some stuff so that we can emulate a kinect but actually be reading from recorded files

from pylibfreenect2 import Freenect2, SyncMultiFrameListener
from pylibfreenect2 import FrameType, Registration, Frame, FrameMap
from pylibfreenect2 import createConsoleLogger, setGlobalLogger
from pylibfreenect2 import LoggerLevel

import numpy as np
import cv2
import time

import atexit
import pickle

# Create and set logger
logger = createConsoleLogger(LoggerLevel.Debug)
setGlobalLogger(logger)

# set up our pipeline.
pipeline = None
try:
    from pylibfreenect2 import OpenGLPacketPipeline
    pipeline = OpenGLPacketPipeline()
except:
    try:
        from pylibfreenect2 import OpenCLPacketPipeline
        pipeline = OpenCLPacketPipeline()
    except:
        from pylibfreenect2 import CpuPacketPipeline
        pipeline = CpuPacketPipeline()

# set up our context
fn = Freenect2()

class Kinect:

    # Kinects's intrinsic parameters based on v2 hardware (estimated).
    CameraParams = {
      "cx":254.878,
      "cy":205.395,
      "fx":365.456,
      "fy":365.456,
      "k1":0.0905474,
      "k2":-0.26819,
      "k3":0.0950862,
      "p1":0.0,
      "p2":0.0,
    }

    # Kinect's physical orientation in the real world.
    CameraPosition = {
        "x": -4, # distance of kinect sensor from wall.
        "y": 0, # actual position in meters of kinect sensor relative to the viewport's center.
        "z": 0, # height in meters of actual kinect sensor from the floor.
        "roll": 0, # angle in degrees of sensor's roll (used for INU input - trig function for this is commented out by default).
        "azimuth": 0, # sensor's yaw angle in degrees.
        "elevation": 0, # sensor's pitch angle in degrees.
    }

    DEPTH_SHAPE = (424, 512)
    N_DEPTH_PIXELS = DEPTH_SHAPE[0] * DEPTH_SHAPE[1]
    COLOR_SHAPE = (360, 640, 3)

    def __init__(self, frametypes=FrameType.Color|FrameType.Depth, registration=True):

        # open the device and start the listener
        num_devices = fn.enumerateDevices()
        assert num_devices>0, "Couldn't find any devices"
        serial = fn.getDeviceSerialNumber(0)
        self.device = fn.openDevice(serial, pipeline=pipeline)
        self.listener = SyncMultiFrameListener(frametypes)
        self.device.setColorFrameListener(self.listener)
        self.device.setIrAndDepthFrameListener(self.listener)
        self.device.start()

        # NOTE: must be called after device.start()
        if registration:
            ir = self.device.getIrCameraParams()
            color = self.device.getColorCameraParams()
            self.registration = Registration(ir, color)

            # create these here so we don't have to every time we call update()
            self.undistorted = Frame(512, 424, 4)
            self.registered = Frame(512, 424, 4)

        # which kinds of frames are we going to be reading?
        # will be true or false
        self.need_color = frametypes & FrameType.Color
        self.need_depth = frametypes & FrameType.Depth
        self.need_ir = frametypes & FrameType.Ir

        # initialize our frames
        self.frames = {}


        # ensure that we shut down smoothly
        atexit.register(self.close)


    def hasNextFrame(self):
        return True

    def close(self):
        self.device.stop()
        self.device.close()

    def get_frames(self):
        self._update()
        return self.frames

    def _update(self):
        if self.listener.hasNewFrame() or len(self.frames)==0:
            _frames = self.listener.waitForNewFrame()

            if self.need_color:
                # get the data, then throw out the 4th channel
                color = _frames['color'].asarray()[:,:,:3]
                h, w, d = self.COLOR_SHAPE
                color = cv2.resize(color, (w,h) )
                self.frames['color'] = color

                # print('color:' +str(_frames['color'].timestamp))
            if self.need_depth:
                depth = _frames['depth'].asarray() / 4500
                self.frames['depth'] = depth

                # print('depth:' + str(_frames['depth'].timestamp))
            if self.need_ir:
                ir = _frames['ir'].asarray() / 65535.
                self.frames['ir'] = ir

                # print('ir:' + str(_frames['ir'].timestamp))

            # register the franes if we need to
            if self.registration and self.need_color and self.need_color:
                c, d, u, r = _frames['color'], _frames['depth'], self.undistorted, self.registered
                self.registration.apply(c, d, u, r)
                self.frames['undistorted'] = self.undistorted.asarray(np.float32)
                self.frames['registered'] = self.registered.asarray(np.uint8)

            # we NEED to explicitly deallocate the frames from the listener
            self.listener.release(_frames)

    def makePointCloud(self, undistorted, registered=None):

        # convert depth frame to point cloud
        pts = self.depthMatrixToPointCloudPos(undistorted)
        # # Kinect sensor real-world orientation compensation.
        pts = self.applyCameraMatrixOrientation(pts)

        # if supplied with registered color image, figure out color of each point
        if registered is not None:
            colors = self.registered2points(registered)
        else:
            # else default to white RBGA
            colors = (1.0, 1.0, 1.0, 1.0)
        pts = pts.astype(np.float32)
        return (pts, colors)

    def depthMatrixToPointCloudPos(self, z, scale=1000):
        # calculate the real-world xyz vertex coordinates from the raw depth data matrix.
        C, R = np.indices(z.shape)

        R = np.subtract(R, self.CameraParams['cx'])
        R = np.multiply(R, z)
        R = np.divide(R, self.CameraParams['fx'] * scale)

        C = np.subtract(C, self.CameraParams['cy'])
        C = np.multiply(C, z)
        C = np.divide(C, self.CameraParams['fy'] * scale)

        return np.column_stack((z.ravel() / scale, R.ravel(), -C.ravel()))

    def applyCameraMatrixOrientation(self, pt):
        # Kinect Sensor Orientation Compensation
        # bacically this is a vectorized version of applyCameraOrientation()
        # uses same trig to rotate a vertex around a gimbal.
        def rotatePoints(ax1, ax2, deg):
            # math to rotate vertexes around a center point on a plane.
            hyp = np.sqrt(pt[:, ax1] ** 2 + pt[:, ax2] ** 2) # Get the length of the hypotenuse of the real-world coordinate from center of rotation, this is the radius!
            d_tan = np.arctan2(pt[:, ax2], pt[:, ax1]) # Calculate the vertexes current angle (returns radians that go from -180 to 180)

            cur_angle = np.degrees(d_tan) % 360 # Convert radians to degrees and use modulo to adjust range from 0 to 360.
            new_angle = np.radians((cur_angle + deg) % 360) # The new angle (in radians) of the vertexes after being rotated by the value of deg.

            pt[:, ax1] = hyp * np.cos(new_angle) # Calculate the rotated coordinate for this axis.
            pt[:, ax2] = hyp * np.sin(new_angle) # Calculate the rotated coordinate for this axis.

        #rotatePoints(1, 2, CameraPosition['roll']) #rotate on the Y&Z plane # Disabled because most tripods don't roll. If an Inertial Nav Unit is available this could be used)
        if self.CameraPosition['elevation'] != 0:
            rotatePoints(0, 2, self.CameraPosition['elevation']) #rotate on the X&Z
        if self.CameraPosition['azimuth'] != 0:
            rotatePoints(0, 1, self.CameraPosition['azimuth']) #rotate on the X&Y

        # Apply offsets for height and linear position of the sensor (from viewport's center)
        if any([self.CameraPosition['x'], self.CameraPosition['y'], self.CameraPosition['z']]):
            pt[:] += np.float_([self.CameraPosition['x'], self.CameraPosition['y'], self.CameraPosition['z']])

        return pt

    def registered2points(self, registered):
        # Format the color registration map from an 2D image of BGRA-pixels to a 1D list of RGB pixels
        colors = np.divide(registered, 255) # values must be between 0.0 - 1.0
        colors = colors.reshape(colors.shape[0] * colors.shape[1], 4 ) # From: Rows X Cols X RGB -to- [[r,g,b],[r,g,b]...]
        colors = colors[:, :3:]  # remove alpha (fourth index) from BGRA to BGR
        colors = colors[...,::-1] #BGR to RGB

        return colors

class FakeKinect(Kinect):

    def __init__(self, fname):
        self.fname = fname
        self._file = open(fname, 'rb')
        self._frames = None

    def hasNextFrame(self):
        # actually try to load the next frame and catch the error, otherwise memo it
        try:
            self._frames = pickle.load(self._file)
            return True
        except EOFError as e:
            # we must have hit the end of the file
            return False

    def get_frames(self):
        # have we already loaded the next frame?
        if self._frames is not None:
            return self._frames
        # otherwise, try to load it, and raise an error if we can't
        else:
            if self.hasNextFrame():
                return self._frames
            else:
                raise Exception('No more frames in the file')

    def forward(self):
        self.hasNextFrame()

    def back(self):
        pass

    def close(self):
        self._file.close()

class Saver():

    def __init__(self, fname):
        self.fname = fname
        self._file = None
        self._is_closed = False

        atexit.register(self.close)

    def write(self, frames):
        assert isinstance(frames, dict)
        if self._is_closed:
            raise Exception('Tried to write to a closed Saver')
        if self._file is None:
            self._file = open(self.fname, 'wb')
        pickle.dump(frames, self._file)


    def close(self):
        if self._file is not None:
            self._file.flush()
            self._file.close()
            del self._file
            self._is_closed = True






