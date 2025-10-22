import numpy as np
import matplotlib.pyplot as plt
from numba import njit, prange
import math

# --- Parámetros CORREGIDOS ---
mu = 0.5
x1, y1 = -0.5, 0.0
x2, y2 = 0.5, 0.0
r_s_values = [0.0, 1e-4, 1e-3, 1e-2]
C = 1.45
x_min, x_max = -2.0, 2.0
y_min, y_max = -2.0, 2.0
N = 256
t_max = 2000.0  # Aumentado a 10000 como en el paper
dt = 0.01
close_encounter_radius = 1e-5  # Radio adicional para close encounter
escape_radius = 20.0  # Radio de escape más grande

# Potencial CORREGIDO para evitar singularidades
@njit
def U(x, y, x1, y1, x2, y2, r_s):
    R1 = math.sqrt((x - x1)**2 + (y - y1)**2)
    R2 = math.sqrt((x - x2)**2 + (y - y2)**2)
    
    # Evitar división por cero cuando r_s = 0
    term1 = 0.5/(R1 - r_s) if (R1 - r_s) > 1e-12 else 0.0
    term2 = 0.5/(R2 - r_s) if (R2 - r_s) > 1e-12 else 0.0
    
    return term1 + term2 + 0.5*(x**2 + y**2)

@njit
def grad_U(x, y, x1, y1, x2, y2, r_s):
    R1 = math.sqrt((x - x1)**2 + (y - y1)**2)
    R2 = math.sqrt((x - x2)**2 + (y - y2)**2)
    
    # Términos corregidos para evitar singularidades
    term1_x = 0.0
    term1_y = 0.0
    term2_x = 0.0
    term2_y = 0.0
    
    if (R1 - r_s) > 1e-12:
        term1_x = 0.5*(x - x1)/(R1*(R1 - r_s)**2)
        term1_y = 0.5*(y - y1)/(R1*(R1 - r_s)**2)
    
    if (R2 - r_s) > 1e-12:
        term2_x = 0.5*(x - x2)/(R2*(R2 - r_s)**2)
        term2_y = 0.5*(y - y2)/(R2*(R2 - r_s)**2)
    
    dU_dx = x - term1_x - term2_x
    dU_dy = y - term1_y - term2_y
    
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
#Función principal que integra una órbita
 
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
    
    # Vectores iniciales para SALI (se mantiene igual)
    w1 = np.array([1.0, 0.0, 0.0, 0.0])
    w2 = np.array([0.0, 1.0, 0.0, 0.0])
    w1 = w1 / norm_4d(w1)
    dot_val = dot_4d(w2, w1)
    w2 = w2 - dot_val * w1
    w2 = w2 / norm_4d(w2)
    
    t = 0.0
    escaped = False
    collided = False
    
    # CALCULAR RADIO DE CLOSE ENCOUNTER CORRECTO
    effective_collision_radius = r_s + close_encounter_radius
    
    # Integrar hasta t_max o hasta escape/colisión
    while t < t_max and not escaped and not collided:
        # Integrador simplecítico (se mantiene igual)
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
        
        # ... (ecuaciones variacionales se mantienen igual)
        
        # CONDICIONES DE TERMINACIÓN CORREGIDAS
        
        # Chequear escape (radio mayor)
        R = math.sqrt(x*x + y*y)
        if R >= escape_radius and (vx*x + vy*y) > 0:
            escaped = True
            orbit_type = 5  # escape
            break
        
        # Chequear close encounters (radio corregido)
        if math.sqrt((x-x1)**2 + (y-y1)**2) <= effective_collision_radius:
            collided = True
            orbit_type = 3  # close encounter con BH1
            break
        if math.sqrt((x-x2)**2 + (y-y2)**2) <= effective_collision_radius:
            collided = True
            orbit_type = 4  # close encounter con BH2
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
print(f"Processing {len(r_s_values)} different r_s values with grid {N}x{N}...")
xs = np.linspace(x_min, x_max, N)
ys = np.linspace(y_min, y_max, N)
X0, Y0 = np.meshgrid(xs, ys)

# Almacenar resultados para cada r_s
all_orbit_types = []
all_times = []

# Procesar para cada valor de r_s
for idx, r_s in enumerate(r_s_values):
    print(f"Processing r_s = {r_s} ({idx+1}/{len(r_s_values)})")
    orbit_type, times = process_grid(X0, Y0, x1, y1, x2, y2, r_s, C, t_max, dt)
    all_orbit_types.append(orbit_type)
    all_times.append(times)
    
    # Verificar clasificaciones obtenidas
    tipos_unicos = np.unique(orbit_type)
    print(f"  Valores únicos en orbit_type para r_s={r_s}: {tipos_unicos}")

print("Classification complete. Generating plots...")

# ============================================================================
# CREAR SUBPLOTS PARA MAPAS DE CLASIFICACIÓN
# ============================================================================

# Colores y etiquetas
colores = np.array([[0,1,0],[1,0,1],[1,1,0],[0,0,1],[1,0,0],[0,1,1]])
cmap = plt.matplotlib.colors.ListedColormap(colores)
etiquetas = ['Regular','Pegajosa','Caótica','Colisión 1','Colisión 2','Escape']

# Crear figura con subplots 2x2
fig1, axes1 = plt.subplots(2, 2, figsize=(15, 12))
axes1 = axes1.flatten()

for idx, (r_s, orbit_type) in enumerate(zip(r_s_values, all_orbit_types)):
    ax = axes1[idx]
    
    # Mostrar mapa de clasificación
    im = ax.imshow(orbit_type, origin='lower', extent=[x_min, x_max, y_min, y_max],
                   cmap=cmap, vmin=0, vmax=5)
    
    # Marcar posición de los agujeros negros
    ax.plot([x1, x2], [y1, y2], 'ko', markersize=8)
    
    # Añadir círculos para mostrar el radio de Schwarzschild (solo si r_s > 0)
    if r_s > 0:
        circle1 = plt.Circle((x1, y1), r_s, color='white', fill=False, linestyle='--', linewidth=1)
        circle2 = plt.Circle((x2, y2), r_s, color='white', fill=False, linestyle='--', linewidth=1)
        ax.add_artist(circle1)
        ax.add_artist(circle2)
    
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title(f'({chr(97+idx)}) $r_s$ = {r_s}', fontsize=14, pad=10)
    ax.set_aspect('equal')

# Añadir leyenda única para toda la figura
import matplotlib.patches as mpatch
parches = [mpatch.Patch(color=colores[i], label=etiquetas[i]) for i in range(6)]
fig1.legend(handles=parches, loc='upper center', bbox_to_anchor=(0.5, 0.05), 
           ncol=6, fontsize=12, frameon=True)

fig1.suptitle(f'Clasificación de Órbitas - Constante de Jacobi C = {C}', fontsize=16, y=0.95)
fig1.tight_layout(rect=[0, 0.05, 1, 0.95])
fig1.savefig('mapa_orbitas_comparativo-1024x1024-C-145.png', dpi=300, bbox_inches='tight')

# ============================================================================
# CREAR SUBPLOTS PARA TIEMPOS DE ESCAPE/COLISIÓN
# ============================================================================

fig2, axes2 = plt.subplots(2, 2, figsize=(15, 12))
axes2 = axes2.flatten()

# Encontrar máximos globales para escalas consistentes
global_max_esc = max(np.nanmax(np.where(ot==5, t, np.nan)) for ot, t in zip(all_orbit_types, all_times))
global_max_col = max(np.nanmax(np.where((ot==3)|(ot==4), t, np.nan)) for ot, t in zip(all_orbit_types, all_times))

for idx, (r_s, orbit_type, times) in enumerate(zip(r_s_values, all_orbit_types, all_times)):
    ax = axes2[idx]
    
    # Crear mapas separados para escape y colisión
    esc_map = np.where(orbit_type==5, times, np.nan)
    col_map = np.where((orbit_type==3)|(orbit_type==4), times, np.nan)
    
    # Mostrar tiempos de escape (fondo)
    im_esc = ax.imshow(esc_map, origin='lower', extent=[x_min, x_max, y_min, y_max],
                       cmap='viridis', vmin=0, vmax=global_max_esc)
    
    # Superponer tiempos de colisión
    im_col = ax.imshow(col_map, origin='lower', extent=[x_min, x_max, y_min, y_max],
                       cmap='autumn', alpha=0.7, vmin=0, vmax=global_max_col)
    
    # Marcar posición de los agujeros negros
    ax.plot([x1, x2], [y1, y2], 'ko', markersize=8)
    
    # Añadir círculos para r_s > 0
    if r_s > 0:
        circle1 = plt.Circle((x1, y1), r_s, color='white', fill=False, linestyle='--', linewidth=1)
        circle2 = plt.Circle((x2, y2), r_s, color='white', fill=False, linestyle='--', linewidth=1)
        ax.add_artist(circle1)
        ax.add_artist(circle2)
    
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title(f'({chr(97+idx)}) $r_s$ = {r_s}', fontsize=14, pad=10)
    ax.set_aspect('equal')

# Añadir colorbars para toda la figura
cbar_ax_esc = fig2.add_axes([0.15, 0.05, 0.3, 0.02])
cbar_esc = fig2.colorbar(im_esc, cax=cbar_ax_esc, orientation='horizontal')
cbar_esc.set_label('Tiempo de Escape', fontsize=12)

cbar_ax_col = fig2.add_axes([0.55, 0.05, 0.3, 0.02])
cbar_col = fig2.colorbar(im_col, cax=cbar_ax_col, orientation='horizontal')
cbar_col.set_label('Tiempo de Colisión', fontsize=12)

fig2.suptitle(f'Tiempos de Escape y Colisión - Constante de Jacobi C = {C}', fontsize=16, y=0.95)
fig2.tight_layout(rect=[0, 0.08, 1, 0.95])
fig2.savefig('mapa_tiempos_comparativo-C-145.png', dpi=300, bbox_inches='tight')

plt.show()

print(f"Done! Check 'mapa_orbitas_comparativo-{N}x{N}-C-{100*C}.png' and 'mapa_tiempos_comparativo.png-{N}x{N}-C-{100*C}'")
