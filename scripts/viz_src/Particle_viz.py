from viz_src.dump import dump
from viz_src.xyz import xyz
from viz_src.svg import svg
import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation
import pandas as pd
import sys
from IPython.display import HTML
sys.path.append('../scripts/')

class Particle_viz:
    def __init__(self, movie_file_name = None):
    
        self.data = dump(movie_file_name)
        self.data.map(1,"id")
        self.time_frames = self.data.time()
        #selecting last index from sorted time frames
        time_value = self.time_frames[-1]
        time_index = self.data.findtime(time_value)
        last_snap_shot = self.data.viz(time_index)

        self.box_x_lim = last_snap_shot[1][0::3]
        self.box_y_lim = last_snap_shot[1][1::3]
        self.box_z_lim = last_snap_shot[1][2::3]

        self.all_data_frames = {}
        for t_ in self.time_frames:
            time_index = self.data.findtime(t_)
            snap_shot = np.array(self.data.viz(time_index)[2])
            self.all_data_frames[t_] = snap_shot[:,2:5]


        #%matplotlib inline
        #figsize=(8, 8)
        self.fig = plt.figure(num='MD simulation output', figsize=(7, 6))
        self.ax = self.fig.add_subplot(111, projection='3d')
        #self.fig.suptitle('3D Test', fontsize=16)
        self.title = self.ax.set_title('Frame=-1')

        # Setting the axes properties
        self.ax.set_xlim3d(self.box_x_lim)
        self.ax.set_xlabel('X')

        self.ax.set_ylim3d(self.box_y_lim)
        self.ax.set_ylabel('Y')

        self.ax.set_zlim3d(self.box_z_lim)
        self.ax.set_zlabel('Z')

        #self.ax.set_xticklabels([])
        #self.ax.set_xticks([])
        #self.ax.set_yticklabels([])
        #self.ax.set_yticks([])
        #self.ax.set_zticklabels([])
        #self.ax.set_zticks([])


    def get_figure(self):
        return self.fig, self.ax

    def update_graph(self, num):
        data_frame_=self.all_data_frames[self.time_frames[num]]
        self.graph.set_data (data_frame_[:,0],data_frame_[:,1])
        self.graph.set_3d_properties(data_frame_[:,2])
        self.title.set_text('Frame={}'.format(self.time_frames[num]))
        return self.title, self.graph, 
    
    def start(self):
        self.graph, = self.ax.plot(self.all_data_frames[self.time_frames[0]][:,1], self.all_data_frames[self.time_frames[0]][:,2], self.all_data_frames[self.time_frames[0]][:,0], 
                 c='r', marker='o',  linestyle="", markersize=10)
        
        # Creating the Animation object
        self.animation = matplotlib.animation.FuncAnimation(fig=self.fig, func=self.update_graph, frames=len(self.time_frames),
                              interval=50, blit=True)
        #self.ax.plt.show()
        #animation_.save('matplot003.gif', writer='imagemagick')
        #animation_.save('myvideo.mp4', codec='h264')
        #HTML(animation_.to_html5_video())