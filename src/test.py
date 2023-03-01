
from multiprocessing import Pool, cpu_count
from multiprocessing.pool import ApplyResult
import time



from simulation.environments.retina.retina_environment import HardRetinaEnvironment, VisualObject, Side

env = HardRetinaEnvironment()
#print(env)

print(env.visual_objects[14])
print(env.visual_objects[15])
print()

print(env.visual_objects[32])
print(env.visual_objects[33])
print()