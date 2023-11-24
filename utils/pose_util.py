"""
<!-- ******************************************
*  Author : Levin Jian  
*  Created On : Fri Nov 24 2023
*  File : pose_util.py
******************************************* -->

"""
import numpy as np


def get_Tgl2cv(inv = False):
        Tgl2cv = np.array([[1,  0,  0, 0,],
                  [0, -1,  0, 0],
                  [0,  0, -1, 0],
                  [0,  0,  0, 1]])
        if inv:
            return np.linalg.inv(Tgl2cv) 
        return Tgl2cv
class PoseUtil(object):
    def __init__(self):
        return
    def run(self):
        return 

if __name__ == "__main__":   
    obj= PoseUtil()
    obj.run()
