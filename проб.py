from main import *

g_g = 100
g_l = 150
d = 60
t = 30
theta = 90


result = solve_ivp(gradient, t_span=[0, 2000],
                   y0=np.array([150]), args=(g_g, g_l, d, theta, t))
plt.plot(result.y0)
print(result)
