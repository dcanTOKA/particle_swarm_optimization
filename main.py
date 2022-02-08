from particle import *
from initialize import *

###################### CHANGEABLE PARAMETERS ######################

'''
Note 1: Change objective function in utils.py
Note 2: Tested for quadratic equation with one unknown, i.e., x^2 + 6x + 8
'''

# indicate the type of problem
# "max" for maximization problem
# "min" for minimization problem
mode = "min"

w = 0.40  # inertia effect
c1 = 0.50  # acceleration coefficient c1
c2 = 2.0  # acceleration coefficient c2
n = 8  # number of swarm particles
threshold = 1e-5  # threshold value for stopping process
bounds = np.array([-5, 5])  # bounds that the positions lie on
iteration = 500  # Maximum iteration
####################################################################

# initialize first metrics of swarm
initial_swarm = InitialSwarm(n, bounds)
x = initial_swarm.initialize_position()
v = initial_swarm.initialize_velocity()

# Create particles
particles = Particles(n, bounds, w, c1, c2, x, v, threshold, mode)

exit_criteria = False

e = 1  # iteration -- starts from zero

while e < iteration and not exit_criteria:
    print(f"Iteration : {e}")
    particles.evaluate_fitness_function(x)
    particles.particle_best()
    particles.global_best()
    particles.visualize_movements_of_swarm(e)
    particles.update_velocity()
    particles.update_position()
    exit_criteria = particles.exit_criteria(e)
    e += 1
