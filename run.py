import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from numpngw import write_apng
from IPython.display import Image
from tqdm.notebook import tqdm
from cartpole_env import CartpoleEnv

env = CartpoleEnv()
env.reset(state = np.array([0.0, 0.5, 0.5, 0.0, 0.0, 0.0]))

frames=[] #frames to create animated png
frames.append(env.render())
for i in tqdm(range(100)):
    action = env.action_space.sample()
    s = env.step(action)
    img = env.render()
    frames.append(img)

write_apng("cartpole_example.gif", frames, delay=10)
Image(filename="cartpole_example.gif")