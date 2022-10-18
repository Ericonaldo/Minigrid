import gymnasium as gym
from minigrid.wrappers import RGBImgObsWrapper, ImgObsWrapper
import cv2

env = gym.make('MiniGrid-BoxPushing-16x16-v0')
env = RGBImgObsWrapper(env, highlight=False) # Get pixel observations
env = ImgObsWrapper(env) # Get rid of the 'mission' field
obs, _ = env.reset() # This now produces an RGB tensor only

print(obs.shape)

cv2.imwrite("image.jpg", obs)
cv2.imshow("image", obs)
cv2.waitKey(0)
cv2.destroyAllWindows()