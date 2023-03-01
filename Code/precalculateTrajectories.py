import model 
import pickle
import json

u= 2           # This is our mean wind speed in the horizontal (m/s) [Original value: 2]

v_sigma = 1    # This is the lateral dispersion parameter (m/s) [Original value: 1]
w_sigma = 0.25 # This is the vertical dispersion (m/s) [Original value: 0.25]

start_x = 0    # This is the starting coordinate in x-direction in m [Original Value: 0]
start_y = 0    # This is the starting coordinate in y-direction (crosswind) in m [Original Value: 0]
start_z = 100  # This is the starting coordinate in z-direction (vertical) in m [Original Value: 100]


s_Ws =[0.1, 0.25, 0.5]
s_Vs =[0.5, 1, 2]

n_trajectories = 1000  # This is the total number of trajectories 
DTrajectories = {}
for v_sigma in s_Vs:
    DTrajectories[str(v_sigma)]={}
    for w_sigma in s_Ws:
        x,y,z =model.calculate_trajectories(start_y,start_z,u,v_sigma,w_sigma,n_trajectories)
        DTrajectories[str(v_sigma)][str(w_sigma)] = (x,y,z)


with open('PreCalculatedTrajectories.pickle', 'wb') as handle:
    pickle.dump(DTrajectories, handle, protocol=pickle.HIGHEST_PROTOCOL)


import pickle
with open('PreCalculatedTrajectories.pickle', 'wb') as handle:
    pickle.dump(DTrajectories, handle, protocol=pickle.HIGHEST_PROTOCOL)


with open('PreCalculatedTrajectories.pickle', 'rb') as handle:
    b = pickle.load(handle)


paramDict = {
    "sigmaW" : s_Ws,
    "sigmaV" : s_Vs,
    "n_trajectories" : n_trajectories
}

with open('TrajectoryParams.txt', 'w') as json_file:
  json.dump(paramDict, json_file)




