import json
import numpy as np


class Config:
    def __init__(self, config_file):
        with open(config_file) as json_file:
            config = json.load(json_file)

        # camera setting
        self.mh = config["camera"]["mh"]
        self.mw = config["camera"]["mw"]
        self.height = config["camera"]["h"]
        self.width = config["camera"]["w"]
        self.H = self.height - 2 * self.mh
        self.W = self.width - 2 * self.mw
        self.fx = config["camera"]["fx"]
        self.fy = config["camera"]["fy"]
        self.cx = config["camera"]["cx"]
        self.cy = config["camera"]["cy"]

        if "k1" in config["camera"]:
            k1 = config["camera"]["k1"]
            k2 = config["camera"]["k2"]
            k3 = config["camera"]["k3"]
            k4 = config["camera"]["k4"]
            k5 = config["camera"]["k5"]
            k6 = config["camera"]["k6"]
            p1 = config["camera"]["p1"]
            p2 = config["camera"]["p2"]
            self.distortion_array = np.array([k1, k2, p1, p2, k3, k4, k5, k6])

        else:
            self.distortion_array = None
