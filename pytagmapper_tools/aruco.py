# aruco shim for new vs old aruco apis from opencv

import cv2

class ArucoDetector:
    def __init__(self):
        try:
            # old aruco api
            self.aruco_dict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_ARUCO_ORIGINAL)
            self.aruco_params = cv2.aruco.DetectorParameters_create()
            self.old_api = True
        except AttributeError:
            # new aruco api
            self.aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_ARUCO_ORIGINAL)
            self.aruco_params =  cv2.aruco.DetectorParameters()
            self.aruco_detector = cv2.aruco.ArucoDetector(self.aruco_dict, self.aruco_params)
            self.old_api = False

    def detectMarkers(self, image):
        if self.old_api:
            return cv2.aruco.detectMarkers(
                image,
                self.aruco_dict,
                parameters=self.aruco_params)
        else:
            return self.aruco_detector.detectMarkers(image)
