# Opinion formation model with BC and different types of noise
# 2023
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
import warnings
warnings.filterwarnings('ignore')

# MODEL 

class Model:
    def __init__(self, params, track_times, noise_type, **kwargs):
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
        self.nu = params["nu"]                      # noise level 
        self.bc = params["bc"]                      # bounded confidence radius (bias)
        self.initial_dist = params["initial_dist"]  
        self.noise_type = noise_type

        # Create Agents
        self.ids = np.arange(self.n_agents,dtype=int)
        if not "opinions" in kwargs.keys():
            assert track_times[0]==0
            self.x = init_opinions(self.initial_dist, self.n_agents)
        else:
            # in this case, we can start the model by fixing all agent opinions at a certain time, e.g. x_i=0.6 for all agents i.
            assert len(kwargs["opinions"]) == self.n_agents
            self.x = np.copy(kwargs["opinions"])

        self.track_times = list(track_times)
        self.all_x = np.empty([len(self.track_times), self.n_agents])#.astype("f4")
        self.all_x[0,:] = self.x.astype("f4") 
        return 

    def run(self, t_simulation):
        """
        update all agents for t_simulation times, store values if time is in track_times
        """
        _abs = abs  # for performance

        lower = 0.0  # edge of opinion space
        upper = 1.0  # edge of opinion space
        
        interaction_schedule = np.random.randint(0, self.n_agents, size=(t_simulation + 2, 2))
        
        # remove monologues (agents talking to themselves)
        monologues = np.where(interaction_schedule[:,0]==interaction_schedule[:,1])
        if len(monologues)>0:
            interaction_schedule[monologues, 1] = (interaction_schedule[monologues, 1] + np.random.randint(0,self.n_agents, size=(len(monologues))))%self.n_agents

        # Store noisy deviations for later use. Minimum 2 noisy deviations per interaction.
        noises = np.random.normal(loc=0,scale=self.nu, size=2*int(t_simulation))
        counter = -1
        if self.noise_type == "exogenousNoise":
            # random values to determine when exogenous noise perturbs the opinion. 
            randlist = np.random.random(size=2*int(t_simulation)+10) # for interaction probability
            counter2 = -1

        # run the simulation
        for t in range(1,t_simulation+1):
            # 1. select agents 
            (ag1, ag2) = interaction_schedule[t,:] 
            (x1, x2) = (self.x[ag1], self.x[ag2])

            if self.noise_type=="adaptationNoise":
                # 2. Create messages
                m1 = x1
                m2 = x2
                # 3. Accept/reject message and update if accepted
                for m, x, ag in zip([m2, m1],[x1,x2], [ag1, ag2]): # == for both agents
                    if _abs( m - x ) <= self.bc:
                         # 4. Update + noise, but only add noise when this noise keeps the opinion in the opinion space
                        while True:   
                            counter += 1
                            newx = x + self.mu * (m - x) + noises[counter]
                            if newx >= lower and newx <= upper:
                                self.x[ag] = newx
                                break
                        
            elif self.noise_type == "exogenousNoise": 
                # 2. Create messages 
                m1 = x1
                m2 = x2
                # 3. Accept/reject message and update if accepted
                for m, x, ag in zip([m2, m1],[x1,x2], [ag1, ag2]):
                    if _abs( m - x ) <= self.bc:
                        self.x[ag] += self.mu * (m-x) # Social influence
                    counter2 +=1
                    # 4. Perturb opinion through exogenous noise (with probability nu)
                    if randlist[counter2] <= self.nu: 
                        # perturbation, such that opinion remains within the bounds
                        while True:   
                            counter += 1
                            newx = x + noises[counter]
                            if newx>=lower and newx<=upper:
                                self.x[ag] = newx
                                break

            elif self.noise_type=="selectionNoise":
                # 2. Create messages (and check if they are within the bounds)
                m1 = x1
                m2 = x2
                # 3. Accept/reject message and update if accepted (with noise)
                for m, x, ag in zip([m2, m1],[x1,x2], [ag1, ag2]):
                    counter+=1
                    eps = max(self.bc + noises[counter], 0)
                    if _abs( m - x ) <= eps:
                        self.x[ag] = x + self.mu * (m-x) 
                        

            elif self.noise_type == "ambiguityNoise":
                # 2. Create messages (and check if they are within the bounds)
                while True:
                    counter+=1
                    testm1 = x1 + noises[counter]  # == np.random.normal(loc=x1, scale=self.nu)
                    if testm1>=lower and testm1<=upper:
                        m1 = testm1
                        break
                while True:
                    counter+=1
                    testm2 = x2 + noises[counter]  # == np.random.normal(loc=x2, scale=self.nu)
                    if testm2 >= lower and testm2 <= upper:
                        m2 = testm2
                        break         
                # 3. Accept/reject message and update if accepted
                for m, x, ag in zip([m2, m1], [x1,x2], [ag1, ag2]):
                    if _abs( m - x ) <= self.bc: 
                        self.x[ag] += self.mu * ( m - x )
            
            # create new noisy variations if too many have been used.
            if counter >= 0.95*len(noises):
                    noises = np.random.normal(loc=0, scale=self.nu, size=int(t_simulation))
                    counter=0

            # 5. Store opinions if time step is in track_times.
            if t in self.track_times:
                ind = self.track_times.index(t)
                self.all_x[ind, :] = self.x.astype("f4")
        return 



# INITIAL CONDITIONS


# def init6G(n, sig, sixAm):
#     """ 
#     create a distribution of initial opinions from a superposition of 6 gaussian functions a0*N(1/12,sig)+a1*N(3/12,sig)+...
#     where sixAm = (a0, ...a6)
#     """
#     gauss = lambda x, mu, sig: 1/(sig*np.sqrt(2*np.pi)) * np.exp(-0.5 * (x-mu)**2/(sig)**2) 
#     multigauss = lambda x, amp, mus, sig: np.sum( [a*gauss(x, mu, sig) for a,mu in zip(amp, mus)], axis=0) 
#     xgrid = np.linspace(0,1,1000)
#     segments = np.linspace(0, 1, num=len(sixAm)+1)[:len(sixAm)]
#     pdf = list(multigauss(xgrid, amp= sixAm, mus = segments+np.diff(segments)[0]/2, sig=sig))
#     cdf = np.cumsum(pdf) 
#     cdf /= cdf[-1]
#     initX = []
#     for i in range(n):
#         rand = np.random.random()
#         densityChoice = min(cdf[cdf>=rand])
#         initX.append(xgrid[list(cdf).index(densityChoice)])
#     initX = np.array(initX)
#     return initX


def init2G(n, a0=0.07432029, mu0=0.34018943, sig0=0.25329727, a1=0.10308521, mu1=0.77908659, sig1=0.13464983):
    """ 
    create a distribution of initial opinions from a superposition of 2 gaussian functions a0*N(mu0,sig0)+a1*N(mu1,sig1)
    """
    gauss = lambda x, mu, sig: 1/(sig*np.sqrt(2*np.pi)) * np.exp(-0.5 * (x-mu)**2/(sig)**2) 
    twogauss = lambda x, a0, mu0, sig0, a1, mu1, sig1: a0 * gauss(x, mu0, sig0) + a1 * gauss(x, mu1, sig1) 
    xgrid = np.linspace(0,1,1000)
    pdf = list(twogauss(xgrid, a0=a0, mu0=mu0, sig0=sig0, a1=a1, mu1=mu1, sig1=sig1))
    cdf = np.cumsum(pdf) 
    cdf /= cdf[-1]
    initX = []
    for i in range(n):
        rand = np.random.random()
        densityChoice = min(cdf[cdf>=rand])
        initX.append(xgrid[list(cdf).index(densityChoice)])
    initX = np.array(initX)
    return initX

def init_opinions(initial_dist, n_agents):
    # initialise agents at t=0
    if initial_dist== "uniform":
        initX = stats.uniform(0,1).rvs(size=n_agents)  
    #elif initial_dist=="6AM":
    #    sixAm = np.array([0.07,0.11,0.12,0.18,0.33,0.19])  # data from Leiserowitz 2021 and Maibach et al 2011
    #    segments = np.linspace(0, 1, num=len(sixAm)+1)[:len(sixAm)]
    #    initX = np.random.choice(segments, size=n_agents, replace=True, p=initial_dist) + 1/len(segments) * np.random.random(size=n_agents)
    elif "2G" in initial_dist:
        if initial_dist == "2G-6AM":
            a0, mu0, sig0, a1, mu1, sig1 = (0.07432029, 0.34018943, 0.25329727, 0.10308521, 0.77908659, 0.13464983)  # fitted parameters
        #elif initial_dist=="2G-Poles":
        #    a0, mu0, sig0, a1, mu1, sig1 = (0.5,0.2,0.1,0.5,0.8,0.1)
        initX = init2G(n_agents, a0, mu0, sig0, a1, mu1, sig1)
    #elif "6G-6AM" in initial_dist:
    #    assert initial_dist[-4:-2]=="0."
    #    sixAm = np.array([0.07,0.11,0.12,0.18,0.33,0.19])  # data from Leiserowitz 2021 and Maibach et al 2011
    #    initX = init6G(n_agents, float(initial_dist[-4:]), sixAm)
    #elif initial_dist == "beta":
    #    # ESS8 data in Germany: variable "wrclmch" (how much do you worry about climate change?) 
    #    # --> fit is a beta distribution.
    #    initX = stats.beta(4.026928199803696, 3.028342634840655, 0, 1).rvs(n_agents)
    else:
        print("error in initial dist")
        return 0
    return initX



def main(noise_type, expname, n, ic, mu_arr, seeds, track_times, resolution, verbose=False, **kwargs):
    '''
    run a model simulation for (various) n, mu, seeds and for given IC and track_times and a resoultion of bc and nu values
    '''    
    eps_arr = epsarrs[resolution]
    nu_arr = nuarrs[resolution]
    t_simulation = track_times[-1] - track_times[0]
    if type(seeds)== int:
        seeds = [seeds]

    params = dict(
        n_agents = n, 
        seed = None,
        mu = None,
        nu = None,
        bc = None,
        initial_dist = ic 
        )
    if verbose: print(len(eps_arr), len(mu_arr), len(nu_arr), len(seeds), len(track_times), n)
    
    for _, eps in enumerate(eps_arr):
        params["bc"] = eps
        result = np.empty([1, len(mu_arr), len(nu_arr), len(seeds), len(track_times), params["n_agents"]])
        for n2, mu in enumerate(mu_arr):
           params["mu"] = mu
           for n3, nu in enumerate(nu_arr):
                params["nu"] = nu
                for n4, s in enumerate(seeds):
                    params["seed"] = int(s) 
                    m = Model(params, track_times, noise_type, **kwargs)
                    m.run(t_simulation)
                    result[:,n2,n3,n4, :, :] = m.all_x.astype("f4")
        d = xr.DataArray(
                data= result,
                name="x",
                dims=["bc","mu", "nu", "seed", "t", "id"],
                coords={
                    "bc": [eps],  # eps_arr,      
                    "mu": mu_arr, 
                    "nu": nu_arr,
                    "seed": seeds,
                    "t": track_times,
                    "id": list(range(0,params["n_agents"]))
                }      
            )
        d = d.astype("f4")
        d.attrs["initial_dist"] = params["initial_dist"]
        # STORE
        if len(seeds)==1: 
            fname = f"data/model-{noise_type}_"+expname+"{}Initial_eps{:.3f}_seeds{}.ncdf".format(ic,  eps, seeds[0])
        else:
            fname = f"data/model-{noise_type}_"+expname+"{}Initial_eps{:.3f}_seeds{}-{}.ncdf".format(ic,  eps, seeds[0], seeds[-1])
        print("saved: ", fname)
        d.to_netcdf(fname)
    if verbose: print("finished")

    if len(seeds)==1:
        ncdfs = [xr.open_dataset(f"data/model-{noise_type}_{expname}{ic}Initial_eps{eps:.3f}_seeds{s}.ncdf", engine="netcdf4") for eps in eps_arr]
        ncdfs_merged = xr.merge(ncdfs)
        ncdfs_merged.attrs["initial_dist"] = params["initial_dist"]
        ncdfs_merged.to_netcdf(f"data/model-{noise_type}_{expname}{ic}Initial_seeds{s}.ncdf")
    return 

    
if __name__=="__main__":

    epsarrs = {
        "high": np.insert(np.arange(0.025,0.41, step=0.025), 0, 1e-3),   # high resolution of the phase space
        "low": np.insert(np.arange(0.05,0.41, step=0.05), 0, 1e-3)       # low resolution of the phase space
    }
    nuarrs = {
        "high": np.concatenate([np.array([1e-10,1e-3, 2e-3, 5e-3]), np.arange(0.01,0.301, 0.01)]),
        "low": np.concatenate([np.array([1e-10,1e-3]), np.arange(0.03,0.301, 0.03)])
    }
 

    import cProfile
    import pstats
    with cProfile.Profile() as pr:
        s0 = time.time()
        track_times = np.arange(0, int(1e4)+1, step=10000) #  np.arange(0, int(1e5)+1, step=10000) 
        mu_arr = [0.5]
        n = 100
        seeds = list(range(10))    # list(range(1000))
        resolution = "low"
        ic = "2G-6AM"
        noise_type = "ambiguityNoise"
        assert noise_type in ["ambiguityNoise", "selectionNoise", "exogenousNoise", "adaptationNoise"]

        main(noise_type, resolution+"Res_" , n, ic, mu_arr, seeds, track_times, resolution=resolution, verbose=False)

        #main(noise_type, expname, n, ic, mu_arr, seeds, track_times, resolution=resolution, verbose=True)
        s1 = time.time()
        print("{} min {} sec".format(int((s1-s0)/60), int(s1-s0)%60 ))
    stats = pstats.Stats(pr)
    stats.sort_stats(pstats.SortKey.TIME)
    # Now you have two options, either print the data or save it as a file
    #stats.print_stats() # Print The Stats
    #stats.dump_stats("cprofile.prof")
    
    



