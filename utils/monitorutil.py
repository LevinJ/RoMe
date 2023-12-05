"""
<!-- ******************************************
*  Author : Levin Jian  
*  Created On : Wed Nov 29 2023
*  File : monitorutil.py
******************************************* -->

"""
import matplotlib.pyplot as plt
import numpy as np
import torch


class MonitorUtil(object):
    def __init__(self):
        self.depth_range = ""
        return
    def vis_bev_depth(self, depth):
        depth = depth.detach().cpu().numpy().copy().squeeze()
        return self.get_heapmap(depth)
        
    def vis_depth(self, depth):
        depth = depth[0, :, :, 0].detach().cpu().numpy().copy()
        mask = depth == -1
        if len(depth[~mask]) > 0:
            self.depth_range = f"min-max-mean: {depth[~mask].min():.3f}---{depth[~mask].max():.3f}---{depth[~mask].mean():.3f}"
        return self.get_heapmap(depth, bev_depth = False)
    def vis_mesh(self, grid, visualizer):
        with torch.no_grad():
            mesh = grid(batch_size=1)
        self.depth_range = f"min-max-mean: {grid.vertices_z.min():.3f}---{grid.vertices_z.max():.3f}---{grid.vertices_z.mean():.3f}"
        print(self.depth_range)
        bev_features, bev_depth = visualizer(mesh[0])
        return self.vis_bev_depth(bev_depth)

    def get_heapmap(self, depth, bev_depth = True):
        plt.switch_backend('agg')
        fig = plt.figure()
        mask = depth == -1
        # if bev_depth:
        #     depth = -depth + 1
        depth[mask] = np.nan
        if self.depth_range is not None:
            plt.title(self.depth_range)
        plt.imshow(depth, cmap='viridis')
        plt.colorbar()
        # plt.show()
        output_name = 'outputs/depth.png'
        if bev_depth:
            # print(f"bev depth range, min-max-mean: {depth[~mask].min():.3f}---{depth[~mask].max():.3f}---{depth[~mask].mean():.3f}")
            output_name = 'outputs/bev_depth.png'
        plt.savefig(output_name)
        img = self.get_imagefrom_fig(fig)
        plt.close()
        return img
        
    def get_imagefrom_fig(self, fig):
        fig.canvas.draw()
        image_from_plot = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        image_from_plot = image_from_plot.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        return image_from_plot
    def run(self):
        depth = np.full((200, 200), -1.0)
        depth[:, 90:111] = 0.0
        img = self.get_heapmap(depth, bev_depth=True)
        return 

if __name__ == "__main__":   
    obj= MonitorUtil()
    obj.run()
