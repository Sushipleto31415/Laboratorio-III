import numpy as np
import matplotlib.pyplot as plt
from math import sqrt, sin, cos

# --- Parámetros físicos y numéricos (modificables) ---
mu = 0.5                    # masa relativa de los agujeros (igual masa implica mu=0.5)
x1, y1 = -0.5, 0.0          # posición de BH1
x2, y2 =  0.5, 0.0          # posición de BH2
r_s = 0.01                  # radio de Schwarzschild (p.ej., 0.01)
C = 2.0                  # constante de Jacobi (energia)
x_min, x_max = -2.0, 2.0    # rango en x
y_min, y_max = -2.0, 2.0    # rango en y
N = 256                      # resolución de la grilla reducida para prueba (original: 64)
t_max = 500.0                # tiempo máximo de integración reducido para prueba (original: 500.0)
dt = 0.01                   # paso temporal (simétrico simplecítico)


# Potencial efectivo (Ω) y su gradiente (Paczynski-Wiita + centrífugo)
def U(x, y):
    R1 = sqrt((x - x1)**2 + (y - y1)**2)
    R2 = sqrt((x - x2)**2 + (y - y2)**2)
    return 0.5/(R1 - r_s) + 0.5/(R2 - r_s) + 0.5*(x**2 + y**2)

def grad_U(x, y):
    R1 = sqrt((x - x1)**2 + (y - y1)**2)
    R2 = sqrt((x - x2)**2 + (y - y2)**2)
    dU_dx = x - 0.5*(x - x1)/(R1*(R1 - r_s)**2) - 0.5*(x - x2)/(R2*(R2 - r_s)**2)
    dU_dy = y - 0.5*(y - y1)/(R1*(R1 - r_s)**2) - 0.5*(y - y2)/(R2*(R2 - r_s)**2)
    return dU_dx, dU_dy

# Rotación de velocidades para integrador simplecítico
def rotate(vx, vy, angle):
    ca, sa = cos(angle), sin(angle)
    return vx*ca + vy*sa, -vx*sa + vy*ca

# Configurar grilla de condiciones iniciales
xs = np.linspace(x_min, x_max, N)
ys = np.linspace(y_min, y_max, N)
X0, Y0 = np.meshgrid(xs, ys)
orbit_type = np.full((N, N), -1, dtype=int)
times = np.zeros((N, N))

print(f"Processing {N}x{N} grid ({N*N} orbits total)...")

# Iteración sobre cada punto inicial en la grilla
for i in range(N):
    print(f"Row {i+1}/{N}")
    for j in range(N):
        x, y = X0[i,j], Y0[i,j]
        # Velocidades iniciales retrógradas (φ̇ < 0) según g = sqrt(2U-C)
        r0 = sqrt(x*x + y*y)
        if r0 == 0.0:
            vx = vy = 0.0
        else:
            val = 2*U(x,y) - C
            if val < 0:
                # Región prohibida (energía insuficiente)
                orbit_type[i,j] = -1
                continue
            g = sqrt(val)
            vx =  (y/r0)*g
            vy = -(x/r0)*g
        
        # Vectores iniciales para SALI
        w1 = np.array([1.0, 0.0, 0.0, 0.0])
        w2 = np.array([0.0, 1.0, 0.0, 0.0])
        w1 /= np.linalg.norm(w1)
        w2 -= np.dot(w2, w1)*w1; w2 /= np.linalg.norm(w2)
        
        t = 0.0
        escaped = collided = False
        
        # Integrar hasta t_max o hasta escape/colisión
        while t < t_max and not escaped and not collided:
            # Integrador simplecítico (rotación + kicks)
            vx, vy = rotate(vx, vy, dt)
            dU_dx, dU_dy = grad_U(x, y)
            vx += dU_dx*(dt/2); vy += dU_dy*(dt/2)
            x += vx*dt; y += vy*dt
            dU_dx, dU_dy = grad_U(x, y)
            vx += dU_dx*(dt/2); vy += dU_dy*(dt/2)
            vx, vy = rotate(vx, vy, dt)
            
            # Ecuaciones variacionales (para SALI) usando derivadas segundas aproximadas
            eps = 1e-8  # Reduced epsilon for better stability
            dU_dx, dU_dy = grad_U(x, y)
            
            # Improved second derivative calculations
            d2U_xx = (grad_U(x+eps, y)[0] - grad_U(x-eps, y)[0]) / (2*eps)
            d2U_yy = (grad_U(x, y+eps)[1] - grad_U(x, y-eps)[1]) / (2*eps)
            d2U_xy = (grad_U(x, y+eps)[0] - grad_U(x, y-eps)[0]) / (2*eps)
            
            dx1, dy1, dvx1, dvy1 = w1
            dx2, dy2, dvx2, dvy2 = w2
            
            # Fixed variational equations with correct signs
            dvx1, dvy1 = rotate(dvx1, dvy1, dt)
            dvx2, dvy2 = rotate(dvx2, dvy2, dt)
            
            dvx1 += (2*dvy1 - d2U_xx*dx1 - d2U_xy*dy1) * (dt/2)  # Fixed signs
            dvy1 += (-2*dvx1 - d2U_xy*dx1 - d2U_yy*dy1) * (dt/2) # Fixed signs
            dvx2 += (2*dvy2 - d2U_xx*dx2 - d2U_xy*dy2) * (dt/2)  # Fixed signs
            dvy2 += (-2*dvx2 - d2U_xy*dx2 - d2U_yy*dy2) * (dt/2) # Fixed signs
            
            dx1 += dvx1 * dt; dy1 += dvy1 * dt
            dx2 += dvx2 * dt; dy2 += dvy2 * dt
            
            dvx1, dvy1 = rotate(dvx1, dvy1, dt)
            dvx2, dvy2 = rotate(dvx2, dvy2, dt)
            
            w1 = np.array([dx1, dy1, dvx1, dvy1])
            w2 = np.array([dx2, dy2, dvx2, dvy2])
            w1 /= np.linalg.norm(w1)
            w2 -= np.dot(w2, w1)*w1; w2 /= np.linalg.norm(w2)
            
            t += dt
            
            # Chequear escape
            R = sqrt(x*x + y*y)
            if R >= 10.0 and (vx*x + vy*y) > 0:
                escaped = True
                times[i,j] = t
                orbit_type[i,j] = 5  # escape
                break
            # Chequear colisiones con primarios
            if sqrt((x-x1)**2 + (y-y1)**2) <= r_s:
                collided = True
                times[i,j] = t
                orbit_type[i,j] = 3  # colisión con BH1
                break
            if sqrt((x-x2)**2 + (y-y2)**2) <= r_s:
                collided = True
                times[i,j] = t
                orbit_type[i,j] = 4  # colisión con BH2
                break
        
        # Si no escapó ni colisionó, clasificar órbita ligada con SALI
        if not escaped and not collided:
            v1 = w1/np.linalg.norm(w1); v2 = w2/np.linalg.norm(w2)
            sali = min(np.linalg.norm(v1+v2), np.linalg.norm(v1-v2))
            if sali > 1e-4:
                orbit_type[i,j] = 0  # regular
            elif sali < 1e-8:
                orbit_type[i,j] = 2  # caótica
            else:
                orbit_type[i,j] = 1  # pegajosa
            times[i,j] = 0.0

# Verificar clasificaciones obtenidas
tipos_unicos = np.unique(orbit_type)
print("Valores únicos en orbit_type:", tipos_unicos)
print("Classification complete. Generating plots...")

# Graficar mapa de clasificación de órbitas
colores = np.array([[0,1,0],[1,0,1],[1,1,0],[0,0,1],[1,0,0],[0,1,1]])
cmap = plt.matplotlib.colors.ListedColormap(colores)
fig1, ax1 = plt.subplots(figsize=(6.5,6))
im1 = ax1.imshow(orbit_type, origin='lower',
                 extent=[x_min,x_max,y_min,y_max],
                 cmap=cmap, vmin=0, vmax=5)
ax1.plot([x1,x2],[y1,y2],'ko', markersize=6)
ax1.set_xlabel('x'); ax1.set_ylabel('y')
ax1.set_title(f'Clasificación de órbitas (C={C}, N={N})')
# Leyenda de categorías
import matplotlib.patches as mpatch
etiquetas = ['Regular','Pegajosa','Caótica','Colisión 1','Colisión 2','Escape']
parches = [mpatch.Patch(color=colores[i], label=etiquetas[i]) for i in range(6)]
ax1.legend(handles=parches, fontsize=8, loc='upper right')
fig1.tight_layout(); fig1.savefig('mapa_orbitas.png', dpi=300)

# Graficar tiempos de escape/colisión
fig2, ax2 = plt.subplots(figsize=(6.5,6))
esc_map = np.where(orbit_type==5, times, np.nan)
col_map = np.where((orbit_type==3)|(orbit_type==4), times, np.nan)
im_esc = ax2.imshow(esc_map, origin='lower',
                    extent=[x_min,x_max,y_min,y_max],
                    cmap='viridis', vmax=np.nanmax(esc_map))
im_col = ax2.imshow(col_map, origin='lower',
                    extent=[x_min,x_max,y_min,y_max],
                    cmap='autumn', alpha=0.7, vmax=np.nanmax(col_map))
ax2.set_xlabel('x'); ax2.set_ylabel('y')
ax2.set_title('Tiempos de escape (azul-verde) y colisión (naranja)')
cbar1 = fig2.colorbar(im_esc, ax=ax2, shrink=0.8); cbar1.set_label('Tiempo escape')
cbar2 = fig2.colorbar(im_col, ax=ax2, shrink=0.8); cbar2.set_label('Tiempo colisión')
fig2.tight_layout(); fig2.savefig('mapa_tiempos.png', dpi=300)
plt.show()

print("Done! Check mapa_orbitas.png and mapa_tiempos.png")
