import numpy as np
import matplotlib.pyplot as plt
from numba import njit, prange
import math

# --- Parámetros físicos y numéricos (modificables) ---
mu = 0.5                    # masa relativa de los agujeros (igual masa implica mu=0.5)
x1, y1 = -0.5, 0.0          # posición de BH1
x2, y2 =  0.5, 0.0          # posición de BH2
r_s = 0.01                  # radio de Schwarzschild (p.ej., 0.01)
C = 3.6                     # constante de Jacobi (energia)
x_min, x_max = -2.0, 2.0    # rango en x
y_min, y_max = -2.0, 2.0    # rango en y
N = 128                     # resolución de la grilla
t_max = 10000.0             # tiempo máximo de integración
dt = 0.01                   # paso temporal (simétrico simplecítico)

# RADIOS CRÍTICOS COMO EN EL PAPER
escape_radius = 10.0         # Radio de escape
collision_radius = 1e-5      # Para caso Newtoniano (r_s = 0)
close_encounter_additional_radius = 1e-5  # Radio adicional para close encounter

# Funciones compiladas con Numba
@njit
def U(x, y, x1, y1, x2, y2, r_s):
    R1 = math.sqrt((x - x1)**2 + (y - y1)**2)
    R2 = math.sqrt((x - x2)**2 + (y - y2)**2)
    
    # Evitar singularidades numéricas
    term1 = 0.5 / max(R1 - r_s, 1e-12)
    term2 = 0.5 / max(R2 - r_s, 1e-12)
    
    return term1 + term2 + 0.5 * (x**2 + y**2)

@njit
def grad_U(x, y, x1, y1, x2, y2, r_s):
    R1 = math.sqrt((x - x1)**2 + (y - y1)**2)
    R2 = math.sqrt((x - x2)**2 + (y - y2)**2)
    
    # Términos con protección numérica
    denom1 = R1 * max((R1 - r_s), 1e-12)**2
    denom2 = R2 * max((R2 - r_s), 1e-12)**2
    
    dU_dx = x - 0.5 * (x - x1) / max(denom1, 1e-12) - 0.5 * (x - x2) / max(denom2, 1e-12)
    dU_dy = y - 0.5 * (y - y1) / max(denom1, 1e-12) - 0.5 * (y - y2) / max(denom2, 1e-12)
    
    return dU_dx, dU_dy

@njit
def rotate(vx, vy, angle):
    ca, sa = math.cos(angle), math.sin(angle)
    return vx*ca + vy*sa, -vx*sa + vy*ca

@njit
def norm_4d(vec):
    return math.sqrt(vec[0]**2 + vec[1]**2 + vec[2]**2 + vec[3]**2)

@njit
def normalize_4d(vec):
    n = norm_4d(vec)
    return vec[0]/n, vec[1]/n, vec[2]/n, vec[3]/n

@njit
def dot_4d(v1, v2):
    return v1[0]*v2[0] + v1[1]*v2[1] + v1[2]*v2[2] + v1[3]*v2[3]

# Función principal que integra una órbita
@njit
def integrate_single_orbit(x, y, x1, y1, x2, y2, r_s, C, t_max, dt):
    # Velocidades iniciales retrógradas
    r0 = math.sqrt(x*x + y*y)
    if r0 == 0.0:
        vx = vy = 0.0
    else:
        val = 2 * U(x, y, x1, y1, x2, y2, r_s) - C
        if val < 0:
            return -1, 0.0  # Región prohibida
        g = math.sqrt(val)
        vx = (y/r0) * g
        vy = -(x/r0) * g
    
    # Vectores iniciales para SALI
    w1 = np.array([1.0, 0.0, 0.0, 0.0])
    w2 = np.array([0.0, 1.0, 0.0, 0.0])
    w1 = w1 / norm_4d(w1)
    # Ortogonalización de Gram-Schmidt
    dot_val = dot_4d(w2, w1)
    w2 = w2 - dot_val * w1
    w2 = w2 / norm_4d(w2)
    
    t = 0.0
    escaped = False
    collided = False

    # Determinar radio de colisión/close encounter según paper
    if r_s == 0.0:
        collision_threshold = collision_radius  # Newtonian collision
    else:
        collision_threshold = r_s + close_encounter_additional_radius  # Pseudo-Newtonian close encounter

    # Integrar hasta t_max o hasta escape/colisión
    while t < t_max and not escaped and not collided:
        # Guardar posición anterior para verificar cruce de límites
        x_prev, y_prev = x, y
        
        # Integrador simplecítico (rotación + kicks)
        vx, vy = rotate(vx, vy, dt)
        dU_dx, dU_dy = grad_U(x, y, x1, y1, x2, y2, r_s)
        vx += dU_dx * (dt/2)
        vy += dU_dy * (dt/2)
        x += vx * dt
        y += vy * dt
        dU_dx, dU_dy = grad_U(x, y, x1, y1, x2, y2, r_s)
        vx += dU_dx * (dt/2)
        vy += dU_dy * (dt/2)
        vx, vy = rotate(vx, vy, dt)
        
        # Ecuaciones variacionales para SALI
        eps = 1e-8
        dU_dx, dU_dy = grad_U(x, y, x1, y1, x2, y2, r_s)
        
        # Derivadas segundas
        d2U_xx = (grad_U(x+eps, y, x1, y1, x2, y2, r_s)[0] - grad_U(x-eps, y, x1, y1, x2, y2, r_s)[0]) / (2*eps)
        d2U_yy = (grad_U(x, y+eps, x1, y1, x2, y2, r_s)[1] - grad_U(x, y-eps, x1, y1, x2, y2, r_s)[1]) / (2*eps)
        d2U_xy = (grad_U(x, y+eps, x1, y1, x2, y2, r_s)[0] - grad_U(x, y-eps, x1, y1, x2, y2, r_s)[0]) / (2*eps)
        
        dx1, dy1, dvx1, dvy1 = w1
        dx2, dy2, dvx2, dvy2 = w2
        
        # Variational equations (fixed signs)
        dvx1, dvy1 = rotate(dvx1, dvy1, dt)
        dvx2, dvy2 = rotate(dvx2, dvy2, dt)
        
        dvx1 += (2*dvy1 - d2U_xx*dx1 - d2U_xy*dy1) * (dt/2)
        dvy1 += (-2*dvx1 - d2U_xy*dx1 - d2U_yy*dy1) * (dt/2)
        dvx2 += (2*dvy2 - d2U_xx*dx2 - d2U_xy*dy2) * (dt/2)
        dvy2 += (-2*dvx2 - d2U_xy*dx2 - d2U_yy*dy2) * (dt/2)
        
        dx1 += dvx1 * dt
        dy1 += dvy1 * dt
        dx2 += dvx2 * dt
        dy2 += dvy2 * dt
        
        dvx1, dvy1 = rotate(dvx1, dvy1, dt)
        dvx2, dvy2 = rotate(dvx2, dvy2, dt)
        
        w1 = np.array([dx1, dy1, dvx1, dvy1])
        w2 = np.array([dx2, dy2, dvx2, dvy2])
        
        # Renormalización
        w1 = w1 / norm_4d(w1)
        dot_val = dot_4d(w2, w1)
        w2 = w2 - dot_val * w1
        w2 = w2 / norm_4d(w2)
        
        # CONDICIONES DE TERMINACIÓN (CORREGIDAS)

        # Radio actual y anterior
        R = math.sqrt(x*x + y*y)
        R_prev = math.sqrt(x_prev*x_prev + y_prev*y_prev)
        
        # Chequear escape: más estricto como en el paper
        # Debe cruzar el límite de R=10 desde adentro hacia afuera
        if R_prev < escape_radius and R >= escape_radius:
            # Velocidad radial positiva (hacia afuera)
            vr = (vx*x + vy*y) / max(R, 1e-12)
            if vr > 0:
                escaped = True
                orbit_type = 5  # escape
                break
        
        # Chequear colisión/close encounter
        R1_current = math.sqrt((x-x1)**2 + (y-y1)**2)
        R2_current = math.sqrt((x-x2)**2 + (y-y2)**2)
        
        # Verificar si está entrando a la región prohibida
        if R1_current <= collision_threshold:
            # Solo contar si se está acercando (no alejando)
            R1_prev = math.sqrt((x_prev-x1)**2 + (y_prev-y1)**2)
            if R1_current < R1_prev:
                collided = True
                orbit_type = 3  # BH1
                break
        
        if R2_current <= collision_threshold:
            R2_prev = math.sqrt((x_prev-x2)**2 + (y_prev-y2)**2)
            if R2_current < R2_prev:
                collided = True
                orbit_type = 4  # BH2
                break
        
        t += dt

    # Clasificar órbita ligada con SALI
    if not escaped and not collided:
        v1 = w1 / norm_4d(w1)
        v2 = w2 / norm_4d(w2)
        sali1 = norm_4d(v1 + v2)
        sali2 = norm_4d(v1 - v2)
        sali = min(sali1, sali2)
        if sali > 1e-4:
            orbit_type = 0  # regular
        elif sali < 1e-8:
            orbit_type = 2  # caótica
        else:
            orbit_type = 1  # pegajosa
        t = 0.0
    
    return orbit_type, t

# Función principal que procesa toda la grilla (paralelizable)
@njit(parallel=True)
def process_grid(X0, Y0, x1, y1, x2, y2, r_s, C, t_max, dt):
    N = X0.shape[0]
    orbit_type = np.full((N, N), -1, dtype=np.int32)
    times = np.zeros((N, N))
    
    for i in prange(N):
        for j in prange(N):
            x, y = X0[i,j], Y0[i,j]
            orbit_type_ij, time_ij = integrate_single_orbit(x, y, x1, y1, x2, y2, r_s, C, t_max, dt)
            orbit_type[i,j] = orbit_type_ij
            times[i,j] = time_ij
    
    return orbit_type, times

# Configurar grilla de condiciones iniciales
print(f"Processing {N}x{N} grid ({N*N} orbits total)...")
xs = np.linspace(x_min, x_max, N)
ys = np.linspace(y_min, y_max, N)
X0, Y0 = np.meshgrid(xs, ys)

# Procesar la grilla completa
orbit_type, times = process_grid(X0, Y0, x1, y1, x2, y2, r_s, C, t_max, dt)

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
ax1.set_title(f'Clasificación de órbitas (C={C}, N={N}, $r_s$={r_s})')
# Leyenda de categorías
import matplotlib.patches as mpatch
etiquetas = ['Regular','Pegajosa','Caótica','Colisión 1','Colisión 2','Escape']
parches = [mpatch.Patch(color=colores[i], label=etiquetas[i]) for i in range(6)]
ax1.legend(handles=parches, fontsize=8, loc='upper right')
fig1.tight_layout(); fig1.savefig('mapa_orbitas_numba.png', dpi=300)

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
ax2.set_title(f'Tiempos de escape (azul-verde) y colisión (naranja) (C={C}, N={N}, $r_s$={r_s})')
cbar1 = fig2.colorbar(im_esc, ax=ax2, shrink=0.8); cbar1.set_label('Tiempo escape')
cbar2 = fig2.colorbar(im_col, ax=ax2, shrink=0.8); cbar2.set_label('Tiempo colisión')
fig2.tight_layout(); fig2.savefig('mapa_tiempos_numba.png', dpi=300)
plt.show()

print("Done! Check mapa_orbitas_numba.png and mapa_tiempos_numba.png")
