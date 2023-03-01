import model 
import pickle
import json

def loadSigmas():
    with open('TrajectoryParams.txt', 'r') as json_file:
        paramDict = json.load(json_file)
        sigmaWs = [str(x)  for x in  paramDict['sigmaW']]
        sigmaVs = [str(x)  for x in  paramDict['sigmaV']]

    return sigmaVs, sigmaWs 
    
def loadTrajectories(sV,SW):
    with open('PreCalculatedTrajectories.pickle', 'rb') as handle:
        Trajectories = pickle.load(handle)
        
    return Trajectories[sV][sW]
        
#Trajectories


#maxNTrajectories = paramDict['n_trajectories']