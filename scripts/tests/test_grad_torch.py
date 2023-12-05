"""
<!-- ******************************************
*  Author : Levin Jian  
*  Created On : Tue Dec 05 2023
*  File : grad_torch.py
******************************************* -->

"""
import torch
from torch import nn
from torchviz import make_dot


class App(object):
    def __init__(self):
        return
    def run(self):
        x=torch.ones(2, requires_grad=True)
        with torch.no_grad():
            y = x * 2
        print(y.requires_grad)
        make_dot(y).render("rnn_torchviz", format="png")
        return 

if __name__ == "__main__":   
    obj= App()
    obj.run()
