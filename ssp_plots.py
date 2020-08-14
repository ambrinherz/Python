import arlpy.uwapm as pm
import arlpy.plot as plt
import numpy as np

bathy = [
    [0, 30],    
    [300, 20],  
    [1000, 350]  
]

ssp = [
    [0, 1430],  
    [10, 1431],  
    [20, 1431],  
    [25, 1431.32],  
    [30, 1431.39],
    [35.3, 1431.67],
    [200, 1442.5],
    [350, 1454.3]
]



env = pm.create_env2d(
    depth=bathy,
    soundspeed=ssp,
    bottom_soundspeed=1450,
    bottom_density=1200,
    bottom_absorption=1.0,
    tx_depth=15,
    max_angle=80,
    min_angle=-80
)


print(env)
pm.check_env2d(env)

rays = pm.compute_rays(env)
pm.plot_rays(rays, width=1000)

env['rx_range'] = np.linspace(0, 1000, 1001)
env['rx_depth'] = np.linspace(0, 350, 351)

# tloss = pm.compute_transmission_loss(env)
# tloss.to_csv(r'transmission_loss.csv')
# pm.plot_transmission_loss(tloss, env=env, clim=[-60,-30], width=900)

tloss = pm.compute_transmission_loss(env, mode='incoherent')
pm.plot_transmission_loss(tloss, env=env, clim=[-60,-30], width=900)
