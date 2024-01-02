# Opinion formation model with BC and noisy expression of opinions
# 09 May 2022
# Peter Steiglechner
# peter.steiglechner@gmail.com

# IMPORTS
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import scipy.stats as stats
import time
import xarray as xr
import sys
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

class Model:
    def __init__(self, params, track_times):
        """
        Create model by initialising an array of agents with opinions 
        and defining how these are updated in an update step.  
        """
        self.params = params
        seed = params["seed"]
        np.random.seed(seed)

        # Define Model parameters
        self.n_agents = params["n_agents"]          # Number of agents
        self.mu = params["mu"]                # convergence rate
        self.nu_max = params["nu_max"]                      # maximum noise level 
        self.nu_min = params["nu_min"]                      # minimum noise level 
        self.bc = params["bc"]                      # bounded confidence radius (bias)
        self.initial_dist = params["initial_dist"]  # Either "uniform" or a list of floats for a given distribution (e.g. six Americas) or, if track_times[0]!=0 the opinions of all agents.   
      
        # Create Agents
        self.schedule = []
        self.ids = np.arange(self.n_agents, dtype=int)
        if track_times[0]==0 :
            # initialise agents at t=0
            if self.initial_dist== "uniform":
                self.x = stats.uniform(0,1).rvs(size=self.n_agents)  
            else:
                segments = np.linspace(0, 1, num=len(self.initial_dist)+1)[:len(self.initial_dist)]
                self.x = np.random.choice(segments, size=self.n_agents, replace=True, p=self.initial_dist) + 1/len(segments) * np.random.random(size=self.n_agents)
        else:
            # adopt opinions given in self.initial_dist 
            if not len(self.initial_dist)==self.n_agents: 
                print("Error: if track_times does not start with 0, the initial_dist needs to represent the current opinions of the n agents")
                return 0
            self.x = self.initial_dist
        

        # for quadratic calc_nu adaptation:  # NEW
        self.rho_opt = 0.75
        self.nu_max_conformity = self.nu_max
        self.nu_max_diversity = self.nu_max

        self.time = track_times[0]
        self.track_times = list(track_times)
        self.running=1
        self.all_x = np.empty([len(self.track_times), self.n_agents])#.astype("f4")
        self.all_x[0,:] = self.x.astype("f4") 
        self.all_nu = np.empty([len(self.track_times), self.n_agents])#.astype("f4")
        self.all_nu[0,:] = self.x.astype("f4") 
        return 

    def calc_nu(self, ag_x):  # NEW
        # Quadratic
        rho = (sum(abs(self.x - ag_x)<=self.bc)-1)/self.n_agents
        if rho<self.rho_opt: # conformity pressure increases noise
            return  self.nu_min + (self.nu_max - self.nu_min) * (rho - self.rho_opt)**2 / self.rho_opt**2
        else: # diversity pressure increaes noise
            return  self.nu_min + (self.nu_max - self.nu_min) * (rho - self.rho_opt)**2 / (1-self.rho_opt)**2
        # Linear
        #self.calc_nu = lambda ag_x: self.nu_max - (sum(abs(self.x - ag_x)<self.bc)-1)/self.n_agents * (self.nu_max-self.nu_min) 
        

    def run(self, t_simulation, verbose=False):
        """
        update all agents for t_simulation times, store values if in track_times
        """
        #if verbose: startTime = time.time()
        lower = 0.0  # bound of opinion space
        upper = 1.0  # bound of opinion space

        # create nr of random updates upfront
        #randList = np.random.random(size=(2*t_simulation+2))
        # create random noises from Gaussian dist upfront (only for unbounded model)
        #randList_addNoise = np.random.normal(loc=0.0, scale=self.nu, size=(2*t_simulation+2))
        
        # Create random update order upfront, correct for those agent pairs with the same index
        orderList = np.random.choice(self.ids, size=(int(t_simulation)+1, 2), replace=True)
        n=0
        while True:  # remove all monologues (ag1 talking to ag1)
            n+=1
            monologs = np.where((orderList[:,0] == orderList[:,1]))
            if len(monologs[0])==0:
                break
            if n==100:
                print("problem")
                break
            orderList[monologs, :] = np.random.choice(self.ids, size=(len(monologs[0]), 2))

        # run through simulation
        for t in range(1,t_simulation+1):
            # 1. select agents 
            (ag1, ag2) = orderList[t,:] 
            (x1,x2) = (self.x[ag1], self.x[ag2])
            #(f1, f2) = [sum(abs(self.x - ag_x)<self.bc)/self.n_agents for ag_x in [x1,x2]]  # NEW: adaptive
            (nu1, nu2) = [self.calc_nu(ag_x) for ag_x in [x1,x2]]  # NEW

            # 2. Create messages (and check if they are within the bounds)
            while True:
                testMessage1 = np.random.normal(loc=x1, scale=nu1)   # NEW
                if testMessage1>=lower and testMessage1<=upper:
                    message1=testMessage1
                    break
            while True:
                testMessage2 = np.random.normal(loc=x2, scale=nu2)   # NEW
                if testMessage2 >= lower and testMessage2 <= upper:
                    message2=testMessage2
                    break         

            # 3. Accept/reject message and update if accepted
            if abs(message2-x1)<=self.bc: #randlist[2*t] < p_m_ag1:
                self.x[ag1] = x1 + self.mu * (message2 - x1)
            if abs(message1-x2)<= self.bc: #randlist[2*t+1] < p_m_ag2:
                self.x[ag2] = x2 + self.mu * (message1 - x2)
            
            # 4. Store at the time steps given in track_times.
            if t in self.track_times:
                ind = self.track_times.index(t)
                self.all_x[ind, :] = self.x.astype("f4")
                self.all_nu[ind, :] = np.array([ self.calc_nu(ag_x) for ag_x in self.x]).astype("f4")  

        #if verbose: print(int(time.time()-starttime), "sec")
        return self.running   

def main_adapt(expName, adaptType, n, icName, mu_arr, seeds, track_times, verbose=False):
    eps_arr = np.arange(0.025,0.35, step=0.025)
    #eps_arr = np.insert(eps_arr, 0, 1e-3)
    nu_min = 0.01
    nu_max_arr = [0.01, 0.05,0.1,0.15,0.2,0.25]
    sixAm = np.array([0.07,0.11,0.12,0.18,0.33,0.19])
    icDict = {"uniform": "uniform", "sixAmerica": sixAm, "sixAmericaExtreme2":sixAm**2/np.sum(sixAm**2)}
    
    params = dict(
        n_agents = n, 
        seed = None,
        mu = None,
        nu_min = nu_min,
        nu_max = None,
        bc = None,
        rho_opt = 0.75,
        initial_dist = icDict[icName] if type(icName)==str else icName # stats.uniform(0,1) 
    )
    t_simulation = track_times[-1] - track_times[0]

    if type(seeds)== int:
        seeds = np.array([seeds])
    
    result = np.empty([len(mu_arr), len(eps_arr), len(nu_max_arr), len(seeds), len(track_times), params["n_agents"]])
    result_nu = np.empty([len(mu_arr), len(eps_arr), len(nu_max_arr), len(seeds), len(track_times), params["n_agents"]])
    if verbose: print(result.shape)
    for n1, mu in enumerate(mu_arr):
        params["mu"] = mu
        for n2, eps in enumerate(eps_arr):
            params["bc"] = eps
            for n3, nu_max in enumerate(nu_max_arr):
                params["nu_max"] = nu_max
                for n4, s in enumerate(seeds):
                    params["seed"] = int(s) 
                    m = Model(params, track_times)
                    running = m.run(t_simulation, verbose=False)
                    result[n1,n2,n3,n4, :, :] = m.all_x 
                    result_nu[n1,n2,n3,n4, :, :] = m.all_nu 
            if verbose: print("{}".format(eps), end=", ")
        if verbose: print("..{}".format(mu))
    
    d = xr.Dataset(
        data_vars = dict(
            x = (["mu", "bc", "nu_max", "seed", "t", "id"], result),
            nu = (["mu", "bc", "nu_max", "seed", "t", "id"], result_nu),
        ),
        coords={
                "mu":mu_arr, 
                "bc":eps_arr,
                "nu_max":nu_max_arr,
                "seed": seeds,
                "t": track_times,
                "id": np.arange(params["n_agents"])
            }, 
    )
    d = d.astype("f4")
    d.attrs["nu_min"] = params["nu_min"]
    d.attrs["initial_dist"] = params["initial_dist"] if "sixAmerica" in icName or icName=="uniform" else "continue"
    # STORE
    adaptName = "linear_" if adaptType=="linear" else "quadr{}-NuMaxConf=NuMaxDiv_"
    if len(seeds)==1: 
        filename = "data/model_"+expName+adaptName+"{}Initial_seeds{}.ncdf".format(icName, seeds[0])
    else:
        filename = "data/model_"+expName+adaptName+"{}Initial_seeds{}-{}.ncdf".format(icName, seeds[0], seeds[-1])
    print("saved: ", filename)
    d.to_netcdf(filename)
    if verbose: print("finished")
    return 



    
if __name__=="__main__":


    s0 = time.time()
    
    print(sys.argv)
    
    track_times = [0, 1000, 2000,3000,4000,5000,6000,7000,8000,9000, 10000, 50000, 100000] 

    mu_arr = [0.5]

    n = 100
    
    seeds = np.arange(int(sys.argv[1]), int(sys.argv[2]))
    
    # resolution = "low"

    icName = "uniform"
    #old_data = xr.open_dataset("model-v8_data/model-v8_long_n100_mu0.5_T_05-001_uniformInitial_seeds0-99.ncdf", engine="netcdf4")

    # linear:
    # expName ="adaptive_"
    # quadratic:
    expName = "adaptive_"

    main_adapt(expName, n, icName, mu_arr, seeds, track_times, verbose=False)

    s1 = time.time()
    print("{} min {} sec".format(int((s1-s0)/60), int(s1-s0)%60 ))
    
