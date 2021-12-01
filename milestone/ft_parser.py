import numpy as np

FT_FRAME_NUM_IDX = 0
FT_X_IDX = 14
FT_Y_IDX = 15
FT_THETA_IDX = 16
FT_TIMESTAMP_IDX = 21

class FtDataReader():
    def __init__(self, fn):
        ft_data_handler = open(fn, 'r')

        ft_frame = []
        ft_theta = []
        ft_x = []
        ft_y = []
        ft_timestamps = []

        ft_line = ft_data_handler.readline()
        while ft_line!="":
            ft_toks = ft_line.split(", ")
            curr_time = float(ft_toks[FT_TIMESTAMP_IDX])/1e3
            ft_frame.append(int(ft_toks[FT_FRAME_NUM_IDX]))
            ft_theta.append(float(ft_toks[FT_THETA_IDX]))
            ft_x.append(float(ft_toks[FT_X_IDX]))
            ft_y.append(float(ft_toks[FT_Y_IDX]))
            ft_timestamps.append(float(ft_toks[FT_TIMESTAMP_IDX]))
            ft_line = ft_data_handler.readline()

        self.fram_num = np.asarray(ft_frame)
        self.theta = np.asarray(ft_theta)
        self.x = np.asarray(ft_x)
        self.y = np.asarray(ft_x)
        self.t = np.asarray(ft_timestamps)
