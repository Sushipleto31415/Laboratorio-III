"""
Código corregido según Zotos et al. 2018 - Incluye constante de Jacobi
"""
import numpy as np
import matplotlib.pyplot as plt
from math import sqrt, sin, cos

# Parámetros del sistema (igual que paper)
mu = 0.5
x1, y1 = -0.5, 0.0
x2, y2 =  0.5, 0.0
r_s = 0.01
x_min, x_max = -2.0, 2.0
y_min, y_max = -2.0, 2.0
N = 64
t_max = 500.0
dt = 0.01

# CONSTANTE DE JACOBI (ENERGÍA) - CLAVE SEGÚN EL PAPER
C = 3.6  # Valor del Energy Case I (Figura 4 del paper)

def U(x, y, r_s):
    """Potencial efectivo completo Ω(x,y) - Ecuación (2) del paper"""
    R1 = sqrt((x - x1)**2 + (y - y1)**2)
    R2 = sqrt((x - x2)**2 + (y - y2)**2)
    # Términos Paczyński-Wiita + centrífugo
    potential = (0.5/(R1 - r_s) + 0.5/(R2 - r_s) + 0.5*(x**2 + y**2))
    return potential

def grad_Omega(x, y, r_s):
    """Gradiente del potencial (ya lo tenías correcto)"""
    R1 = sqrt((x - x1)**2 + (y - y1)**2)
    R2 = sqrt((x - x2)**2 + (y - y2)**2)
    dU_dx = x - 0.5*(x - x1)/(R1 * (R1 - r_s)**2) - 0.5*(x - x2)/(R2 * (R2 - r_s)**2)
    dU_dy = y - 0.5*(y - y1)/(R1 * (R1 - r_s)**2) - 0.5*(y - y2)/(R2 * (R2 - r_s)**2)
    return dU_dx, dU_dy

def rotate_velocity(vx, vy, angle):
    cos_a = cos(angle)
    sin_a = sin(angle)
    vx_new =  vx * cos_a + vy * sin_a
    vy_new = -vx * sin_a + vy * cos_a
    return vx_new, vy_new

# Configuración inicial
xs = np.linspace(x_min, x_max, N)
ys = np.linspace(y_min, y_max, N)
X0, Y0 = np.meshgrid(xs, ys)

orbit_type = np.full((N, N), -1, dtype=int)
times = np.zeros((N, N))

print(f"Calculando órbitas con C={C}, r_s={r_s}...")

for i in range(N):
    if i % 10 == 0:
        print(f"Procesando fila {i}/{N}")
    
    for j in range(N):
        x, y = X0[i,j], Y0[i,j]
        
        # VELOCIDADES INICIALES SEGÚN PAPER - ECUACIÓN (12)
        r0 = sqrt(x**2 + y**2)
        if r0 == 0:
            vx, vy = 0.0, 0.0
        else:
            # g(x,y) = sqrt(2U(x,y) - C) como en el paper
            g_val = sqrt(2 * U(x, y, r_s) - C)
            # Velocidades iniciales para órbitas retrógradas (˙φ < 0)
            vx = (y / r0) * g_val
            vy = (-x / r0) * g_val
        
        # Verificar que la energía sea físicamente posible
        if np.isnan(vx) or np.isnan(vy):
            orbit_type[i,j] = -1  # Región prohibida
            continue
            
        # Vectores para SALI
        w1 = np.array([1.0, 0.0, 0.0, 0.0])
        w2 = np.array([0.0, 1.0, 0.0, 0.0])
        w1 = w1 / np.linalg.norm(w1)
        w2 = w2 - np.dot(w2, w1) * w1
        w2 = w2 / np.linalg.norm(w2)

        t = 0.0
        escaped = collided = False

        while t < t_max and not escaped and not collided:
            # Paso simplecítico (igual que antes)
            vx, vy = rotate_velocity(vx, vy, dt)
            dU_dx, dU_dy = grad_Omega(x, y, r_s)
            vx += dU_dx * (dt/2.0)
            vy += dU_dy * (dt/2.0)
            x += vx * dt
            y += vy * dt
            dU_dx, dU_dy = grad_Omega(x, y, r_s)
            vx += dU_dx * (dt/2.0)
            vy += dU_dy * (dt/2.0)
            vx, vy = rotate_velocity(vx, vy, dt)

            # Sistema variacional (SALI) - mantener igual
            eps = 1e-6
            d2U_dx2 = (grad_Omega(x+eps, y, r_s)[0] - 2*dU_dx + grad_Omega(x-eps, y, r_s)[0]) / eps**2
            d2U_dy2 = (grad_Omega(x, y+eps, r_s)[1] - 2*dU_dy + grad_Omega(x, y-eps, r_s)[1]) / eps**2
            d2U_dxdy = ((grad_Omega(x+eps, y+eps, r_s)[0] - grad_Omega(x+eps, y-eps, r_s)[0])
                       - (grad_Omega(x-eps, y+eps, r_s)[0] - grad_Omega(x-eps, y-eps, r_s)[0])) / (4*eps*eps)
            
            dx1, dy1, dvx1, dvy1 = w1
            dx2, dy2, dvx2, dvy2 = w2

            dvx1, dvy1 = rotate_velocity(dvx1, dvy1, dt)
            dvx2, dvy2 = rotate_velocity(dvx2, dvy2, dt)
            dvx1 += (2*dvy1 + d2U_dx2*dx1 + d2U_dxdy*dy1) * (dt/2.0)
            dvy1 += (-2*dvx1 + d2U_dxdy*dx1 + d2U_dy2*dy1) * (dt/2.0)
            dvx2 += (2*dvy2 + d2U_dx2*dx2 + d2U_dxdy*dy2) * (dt/2.0)
            dvy2 += (-2*dvx2 + d2U_dxdy*dx2 + d2U_dy2*dy2) * (dt/2.0)
            dx1 += dvx1 * dt
            dy1 += dvy1 * dt
            dx2 += dvx2 * dt
            dy2 += dvy2 * dt
            dvx1 += (2*dvy1 + d2U_dx2*dx1 + d2U_dxdy*dy1) * (dt/2.0)
            dvy1 += (-2*dvx1 + d2U_dxdy*dx1 + d2U_dy2*dy1) * (dt/2.0)
            dvx2 += (2*dvy2 + d2U_dx2*dx2 + d2U_dxdy*dy2) * (dt/2.0)
            dvy2 += (-2*dvx2 + d2U_dxdy*dx2 + d2U_dy2*dy2) * (dt/2.0)
            dvx1, dvy1 = rotate_velocity(dvx1, dvy1, dt)
            dvx2, dvy2 = rotate_velocity(dvx2, dvy2, dt)

            w1 = np.array([dx1, dy1, dvx1, dvy1])
            w2 = np.array([dx2, dy2, dvx2, dvy2])
            w1 /= np.linalg.norm(w1)
            w2 = w2 - np.dot(w2, w1) * w1
            w2 /= np.linalg.norm(w2)

            t += dt

            # Chequear escape o colisión
            R = sqrt(x*x + y*y)
            if R >= 10.0 and (vx*x + vy*y) > 0:
                escaped = True
                times[i,j] = t
                orbit_type[i,j] = 5  # escape
                break
            
            dist1 = sqrt((x - x1)**2 + (y - y1)**2)
            dist2 = sqrt((x - x2)**2 + (y - y2)**2)
            if dist1 <= r_s:
                collided = True
                times[i,j] = t
                orbit_type[i,j] = 3  # colisión 1
                break
            if dist2 <= r_s:
                collided = True
                times[i,j] = t
                orbit_type[i,j] = 4  # colisión 2
                break

        # Clasificación final para órbitas ligadas
        if not escaped and not collided:
            v1 = w1 / np.linalg.norm(w1)
            v2 = w2 / np.linalg.norm(w2)
            sali = min(np.linalg.norm(v1 + v2), np.linalg.norm(v1 - v2))
            if sali > 1e-4:
                orbit_type[i,j] = 0  # regular
            elif sali < 1e-8:
                orbit_type[i,j] = 2  # caótica
            else:
                orbit_type[i,j] = 1  # pegajosa
            times[i,j] = 0.0

# ... (el resto del código de gráficos igual)# ---- GRÁFICOS CORREGIDOS ----

# 1. Mapa de clasificación de órbitas (CORREGIDO)
colores = np.array([
    [0.0, 1.0, 0.0],    # 0: verde (regular)
    [1.0, 0.0, 1.0],    # 1: magenta (pegajosa) 
    [1.0, 1.0, 0.0],    # 2: amarillo (caótica)
    [0.0, 0.0, 1.0],    # 3: azul (colisión 1)
    [1.0, 0.0, 0.0],    # 4: rojo (colisión 2)
    [0.0, 1.0, 1.0]     # 5: cyan (escape)
])
cmap = plt.matplotlib.colors.ListedColormap(colores)

fig1, ax1 = plt.subplots(figsize=(8, 6))
# IMPORTANTE: Especificar vmin y vmax para que los colores coincidan con los valores
im1 = ax1.imshow(orbit_type, origin='lower', extent=[x_min, x_max, y_min, y_max], 
                cmap=cmap, vmin=0, vmax=5, interpolation='nearest')
ax1.set_xlabel('x')
ax1.set_ylabel('y')
ax1.set_title('Mapa de clasificación de órbitas')

# Agregar posición de los agujeros negros
ax1.plot([x1, x2], [y1, y2], 'ko', markersize=8, label='Agujeros negros')

# Leyenda mejorada
leyenda = [
    plt.matplotlib.patches.Patch(color=colores[0], label='Regular'),
    plt.matplotlib.patches.Patch(color=colores[1], label='Pegajosa'),
    plt.matplotlib.patches.Patch(color=colores[2], label='Caótica'),
    plt.matplotlib.patches.Patch(color=colores[3], label='Colisión 1'),
    plt.matplotlib.patches.Patch(color=colores[4], label='Colisión 2'),
    plt.matplotlib.patches.Patch(color=colores[5], label='Escape')
]
ax1.legend(handles=leyenda, loc='upper right', fontsize=8)
fig1.tight_layout()
fig1.savefig('mapa_orbitas_corregido.png', dpi=300, bbox_inches='tight')

# 2. Mapa de tiempos (ya funcionaba bien)
fig2, ax2 = plt.subplots(figsize=(8, 6))
esc_map = np.where(orbit_type == 5, times, np.nan)
col_map = np.where((orbit_type == 3) | (orbit_type == 4), times, np.nan)

im_esc = ax2.imshow(esc_map, origin='lower', extent=[x_min, x_max, y_min, y_max], 
                   cmap='viridis', vmax=np.nanmax(esc_map) if np.any(~np.isnan(esc_map)) else 1)
im_col = ax2.imshow(col_map, origin='lower', extent=[x_min, x_max, y_min, y_max],
                   cmap='autumn', alpha=0.7, 
                   vmax=np.nanmax(col_map) if np.any(~np.isnan(col_map)) else 1)

ax2.set_xlabel('x')
ax2.set_ylabel('y')
ax2.set_title('Tiempos de escape (azul-verde) y colisión (rojo-naranja)')

# Barras de color
if np.any(~np.isnan(esc_map)):
    cbar1 = fig2.colorbar(im_esc, ax=ax2, shrink=0.8)
    cbar1.set_label('Tiempo de escape')
if np.any(~np.isnan(col_map)):
    cbar2 = fig2.colorbar(im_col, ax=ax2, shrink=0.8)
    cbar2.set_label('Tiempo de colisión')

fig2.tight_layout()
fig2.savefig('mapa_tiempos_corregido.png', dpi=300, bbox_inches='tight')

print("Gráficos guardados como 'mapa_orbitas_corregido.png' y 'mapa_tiempos_corregido.png'")

# Mostrar estadísticas
print("\n--- ESTADÍSTICAS ---")
labels = ['Regular', 'Pegajosa', 'Caótica', 'Colisión 1', 'Colisión 2', 'Escape']
for i, label in enumerate(labels):
    count = np.sum(orbit_type == i)
    percentage = (count / (N*N)) * 100
    print(f"{label}: {count} puntos ({percentage:.1f}%)")

# Opcional: mostrar gráficos (puedes comentar esta línea si causa problemas)
plt.show()
