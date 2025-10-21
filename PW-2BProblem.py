import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib.animation import FuncAnimation
from modules.ode import runge_kutta_4th
plt.style.use("dark_background")
#We define the equations of motion for the 2 body problem
M = 100
c = 300000 #km/seg
G = 1
m=1
dt=0.05 #Time step
N = 100 #Timespan length
r_s= (2*G*M)/c**2
r0=10.0
def eq_motion(u,softening=0.1):
    r,phi,v_r = u
    r_eff= max(r,r_s+softening)  
    drdt = v_r
    dphidt = l/(m*r_eff**2)
    dv_rdt = l**2/(m**2*r_eff**3) - (G*M) / (m*(r_eff-r_s)**2)
    return np.array([drdt,dphidt,dv_rdt])

#We now set the initial conditions for the orbit
u_0 =[r0,0.0,0.0] #r_0,phi_0,v_r_0
l_circular = np.sqrt((G * M * r0**3) / (r0 - r_s)**2)
l = l_circular*0.5
t=np.linspace(0,100,int(N/dt))#We must define the time

#Store the results into an array
results = runge_kutta_4th(lambda u, t: eq_motion(u),u_0,t,dt)

#Now we convert the polar coordinates into cartesian coordinates
x = results[:,0] * np.cos(results[:,1])
y = results[:,0] * np.sin(results[:,1])

#Create the empty plot
fig,ax= plt.subplots(figsize=(8,8))

#Plot the orbit
ax.plot(x,y,label="Particle orbit")

#Add the Schwarzschild radius representacion
schwarzschild_circle = Circle((0, 0), r_s, fill=True, color='red', alpha=0.5, label='Schwarzschild Radius')
ax.add_patch(schwarzschild_circle)


#Set the plot properies
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_title("Orbit arround a Schwarzschild BH")
ax.legend()
ax.grid(True)
ax.set_aspect("equal")

#To set apropiate limits for the plot
max_r=max(np.max(np.abs(x)),np.max(np.abs(y)),r_s*1.5)
ax.set_xlim(-max_r,max_r)
ax.set_ylim(-max_r,max_r)


plt.show()

#We check for the conservation of energy
# Calculate energy at each time step
kinetic = 0.5 * m * (results[:, 2]**2 + (results[:, 0] * results[:, 1])**2)
potential = -G * M * m / (results[:, 0] - r_s)
total_energy = kinetic + potential

plt.figure()
plt.plot(t[:len(total_energy)], total_energy)
plt.xlabel('Time')
plt.ylabel('Total Energy')
plt.title('Energy Conservation')
plt.show()
#Animation of the orbit



