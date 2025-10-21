

"""
Código para clasificar órbitas en un sistema binario de agujeros negros (masa igual, no girantes),
utilizando el potencial pseudo-Newtoniano de Paczyński-Wiita :contentReference[oaicite:0]{index=0}.
Empleamos un marco de referencia rotante (sinódico) e integramos las ecuaciones de
movimiento usando un método simplecítico (leapfrog), siguiendo la formulación de Zotos et al. (2018):contentReference[oaicite:1]{index=1}.
También incluimos el sistema variacional para calcular el indicador SALI y distinguir entre
órbitas regulares, caóticas o pegajosas:contentReference[oaicite:2]{index=2}, así como órbitas de escape o colisión (encuentro cercano).
El resultado son dos gráficos: un mapa de clasificación de órbitas (colores según tipo) y un mapa de tiempos de escape/colisión, análogos a las Figuras 4 y 5 de Zotos et al.:contentReference[oaicite:3]{index=3}:contentReference[oaicite:4]{index=4}.

Referencias clave: Zotos et al. 2018 (órbitas en binario de agujeros negros):contentReference[oaicite:5]{index=5}:contentReference[oaicite:6]{index=6}:contentReference[oaicite:7]{index=7};
Skokos 2001 (SALI):contentReference[oaicite:8]{index=8}.
"""
import numpy as np
import matplotlib.pyplot as plt
from math import sqrt, sin, cos

# Parámetros del sistema:
mu = 0.5            # masas iguales, sistema normalizado
# Posición de primarios en coordenadas rotantes:
x1, y1 = -0.5, 0.0  # primario 1 en (-0.5,0)
x2, y2 =  0.5, 0.0  # primario 2 en (0.5,0)
r_s = 0.01          # radio de Schwarzschild (puede variarse como parámetro)
# Límites de la grilla de condiciones iniciales:
x_min, x_max = -2.0, 2.0
y_min, y_max = -2.0, 2.0
N = 64          # tamaño de la grilla (e.g. 512x512) según indicación
# Tiempo de integración:
t_max = 500.0   # tiempo máximo (unidades adimensionales):contentReference[oaicite:9]{index=9}
dt = 0.01         # paso temporal (elegir pequeño para precisión)

def grad_Omega(x, y, r_s):
    """
    Calcula el gradiente del potencial efectivo Ω en el marco sinódico.
    Incluye el término centrífugo (x,y) y el potencial de Paczyński-Wiita de ambos agujeros.
    Ecuaciones basadas en Zotos et al. 2018:contentReference[oaicite:10]{index=10}.
    """
    R1 = sqrt((x - x1)**2 + (y - y1)**2)
    R2 = sqrt((x - x2)**2 + (y - y2)**2)
    # Derivada del término centrífugo 0.5*(x^2+y^2) es (x, y).
    # Derivadas del potencial Paczyński-Wiita: -0.5*(coord - pos_i)/(R_i*(R_i - r_s)^2).
    dU_dx = x \
            - 0.5*(x - x1) / ( R1 * (R1 - r_s)**2 ) \
            - 0.5*(x - x2) / ( R2 * (R2 - r_s)**2 )
    dU_dy = y \
            - 0.5*(y - y1) / ( R1 * (R1 - r_s)**2 ) \
            - 0.5*(y - y2) / ( R2 * (R2 - r_s)**2 )
    return dU_dx, dU_dy

# Condiciones iniciales en la grilla:
xs = np.linspace(x_min, x_max, N)
ys = np.linspace(y_min, y_max, N)
X0, Y0 = np.meshgrid(xs, ys)

# Matrices para almacenar resultado:
orbit_type = np.full((N, N), -1, dtype=int)  # código de tipo de órbita
times = np.zeros((N, N))                     # tiempo de escape/colisión

# Códigos de tipos:
# 0=regular, 1=pegajosa, 2=caótica, 3=colisión con primario1, 4=colisión con primario2, 5=escape

def rotate_velocity(vx, vy, angle):
    """
    Rota el vector velocidad (vx, vy) en el plano por el ángulo dado.
    La rotación es en el sentido horario por convención (ecuaciones de movimiento).
    """
    cos_a = cos(angle)
    sin_a = sin(angle)
    vx_new =  vx * cos_a + vy * sin_a
    vy_new = -vx * sin_a + vy * cos_a
    return vx_new, vy_new

# Iterar sobre cada condición inicial:
for i in range(N):
    for j in range(N):
        x = X0[i,j]
        y = Y0[i,j]
        vx = 0.0   # velocidades iniciales fijas (problema de Copenhague)
        vy = 0.0
        # Vectores de desviación iniciales ortonormales (para SALI):
        w1 = np.array([1.0, 0.0, 0.0, 0.0])
        w2 = np.array([0.0, 1.0, 0.0, 0.0])
        w1 = w1 / np.linalg.norm(w1)
        w2 = w2 - np.dot(w2, w1) * w1
        w2 = w2 / np.linalg.norm(w2)

        t = 0.0
        escaped = False
        collided = False

        while t < t_max:
            # Paso simplecítico Strang: rotación-Coriolis / kick / drift / kick / rotación

            # 1) Rotación (Coriolis) - medio paso:
            vx, vy = rotate_velocity(vx, vy, dt)

            # 2) Primera "kick" - medio paso:
            dU_dx, dU_dy = grad_Omega(x, y, r_s)
            vx += dU_dx * (dt/2.0)
            vy += dU_dy * (dt/2.0)

            # 3) Drift de posiciones:
            x += vx * dt
            y += vy * dt

            # 4) Segunda "kick" - medio paso:
            dU_dx, dU_dy = grad_Omega(x, y, r_s)
            vx += dU_dx * (dt/2.0)
            vy += dU_dy * (dt/2.0)

            # 5) Rotación final - medio paso:
            vx, vy = rotate_velocity(vx, vy, dt)

            # Integración del sistema variacional (derivadas lineales)
            eps = 1e-6
            # Aprox. numérica de segundas derivadas de Ω:
            d2U_dx2 = (grad_Omega(x+eps, y, r_s)[0] - 2*dU_dx + grad_Omega(x-eps, y, r_s)[0]) / eps**2
            d2U_dy2 = (grad_Omega(x, y+eps, r_s)[1] - 2*dU_dy + grad_Omega(x, y-eps, r_s)[1]) / eps**2
            d2U_dxdy = ((grad_Omega(x+eps, y+eps, r_s)[0] - grad_Omega(x+eps, y-eps, r_s)[0])
                       - (grad_Omega(x-eps, y+eps, r_s)[0] - grad_Omega(x-eps, y-eps, r_s)[0])) / (4*eps*eps)
            dx1, dy1, dvx1, dvy1 = w1
            dx2, dy2, dvx2, dvy2 = w2

            # Rotación de δv (parte Coriolis):
            dvx1, dvy1 = rotate_velocity(dvx1, dvy1, dt)
            dvx2, dvy2 = rotate_velocity(dvx2, dvy2, dt)
            # Kick de δv - medio paso:
            dvx1 += (2*dvy1 + d2U_dx2*dx1 + d2U_dxdy*dy1) * (dt/2.0)
            dvy1 += (-2*dvx1 + d2U_dxdy*dx1 + d2U_dy2*dy1) * (dt/2.0)
            dvx2 += (2*dvy2 + d2U_dx2*dx2 + d2U_dxdy*dy2) * (dt/2.0)
            dvy2 += (-2*dvx2 + d2U_dxdy*dx2 + d2U_dy2*dy2) * (dt/2.0)
            # Drift de δx, δy:
            dx1 += dvx1 * dt
            dy1 += dvy1 * dt
            dx2 += dvx2 * dt
            dy2 += dvy2 * dt
            # Kick final de δv - medio paso:
            dvx1 += (2*dvy1 + d2U_dx2*dx1 + d2U_dxdy*dy1) * (dt/2.0)
            dvy1 += (-2*dvx1 + d2U_dxdy*dx1 + d2U_dy2*dy1) * (dt/2.0)
            dvx2 += (2*dvy2 + d2U_dx2*dx2 + d2U_dxdy*dy2) * (dt/2.0)
            dvy2 += (-2*dvx2 + d2U_dxdy*dx2 + d2U_dy2*dy2) * (dt/2.0)
            # Rotación final δv:
            dvx1, dvy1 = rotate_velocity(dvx1, dvy1, dt)
            dvx2, dvy2 = rotate_velocity(dvx2, dvy2, dt)

            w1 = np.array([dx1, dy1, dvx1, dvy1])
            w2 = np.array([dx2, dy2, dvx2, dvy2])
            w1 /= np.linalg.norm(w1)
            # Hacer w2 ortogonal a w1 y normalizar:
            w2 = w2 - np.dot(w2, w1) * w1
            w2 /= np.linalg.norm(w2)

            t += dt

            # Chequear escape o colisión:
            R = sqrt(x*x + y*y)
            R_lim = 10.0  # círculo límite (Zotos et al. usan 10):contentReference[oaicite:11]{index=11}
            if R >= R_lim and (vx*x + vy*y) > 0:
                escaped = True
                times[i,j] = t
                orbit_type[i,j] = 5  # escape
                break
            dist1 = sqrt((x - x1)**2 + (y - y1)**2)
            dist2 = sqrt((x - x2)**2 + (y - y2)**2)
            if dist1 <= r_s:
                collided = True
                times[i,j] = t
                orbit_type[i,j] = 3  # colisión primario 1
                break
            if dist2 <= r_s:
                collided = True
                times[i,j] = t
                orbit_type[i,j] = 4  # colisión primario 2
                break

        # Clasificación final para órbitas ligadas (no escaparon/colisionaron):
        if not escaped and not collided:
            v1 = w1 / np.linalg.norm(w1)
            v2 = w2 / np.linalg.norm(w2)
            sali = min(np.linalg.norm(v1 + v2), np.linalg.norm(v1 - v2))
            # Umbrales de SALI (Skokos 2001):contentReference[oaicite:12]{index=12}:
            if sali > 1e-4:
                orbit_type[i,j] = 0  # órbita regular
            elif sali < 1e-8:
                orbit_type[i,j] = 2  # órbita caótica
            else:
                orbit_type[i,j] = 1  # órbita pegajosa
            times[i,j] = 0.0

# ---- Generar gráficos ----


"""
# Mapa de clasificación (colores según tipo de órbita):
colores = np.array([
    [0.0, 1.0, 0.0],    # verde (regular)
    [1.0, 0.0, 1.0],    # magenta (pegajosa)
    [1.0, 1.0, 0.0],    # amarillo (caótica)
    [0.0, 0.0, 1.0],    # azul (colisión 1)
    [1.0, 0.0, 0.0],    # rojo (colisión 2)
    [0.0, 1.0, 1.0]     # cyan (escape)
])
cmap = plt.matplotlib.colors.ListedColormap(colores)
plt.figure(figsize=(6,6))
plt.imshow(orbit_type, origin='lower', extent=[x_min,x_max,y_min,y_max], cmap=cmap)
plt.xlabel('x')
plt.ylabel('y')
plt.title('Mapa de clasificación de órbitas')
import matplotlib.patches as mpatches
leyenda = [
    mpatches.Patch(color=colores[0], label='Regular'),
    mpatches.Patch(color=colores[1], label='Pegajosa'),
    mpatches.Patch(color=colores[2], label='Caótica'),
    mpatches.Patch(color=colores[3], label='Colisión 1'),
    mpatches.Patch(color=colores[4], label='Colisión 2'),
    mpatches.Patch(color=colores[5], label='Escape')
]
plt.legend(handles=leyenda, loc='upper right', fontsize='small')
plt.tight_layout()
plt.savefig('mapa_orbitas.png', dpi=300)

# Mapa de tiempos (escape vs colisión):
plt.figure(figsize=(6,6))
esc_map = np.where(orbit_type==5, times, np.nan)
col_map = np.where((orbit_type==3)|(orbit_type==4), times, np.nan)
plt.imshow(esc_map, origin='lower', extent=[x_min,x_max,y_min,y_max], 
           cmap='viridis', vmax=np.nanmax(esc_map))
plt.imshow(col_map, origin='lower', extent=[x_min,x_max,y_min,y_max],
           cmap='autumn', alpha=0.6, vmax=np.nanmax(col_map))
plt.xlabel('x')
plt.ylabel('y')
plt.title('Tiempos de escape (verde) y colisión (rojo)')
# Barras de color separadas:
cbar1 = plt.colorbar(shrink=0.8)
cbar1.set_label('t_escape')
m = plt.cm.ScalarMappable(cmap='autumn', norm=plt.Normalize(vmin=np.nanmin(col_map), vmax=np.nanmax(col_map)))
m.set_array([])
cbar2 = plt.colorbar(m, shrink=0.8, orientation='vertical')
cbar2.set_label('t_colision')
plt.tight_layout()
plt.savefig('mapa_tiempos.png', dpi=300)
"""
#------ New ------

# Mapa de clasificación (colores según tipo de órbita):
colores = np.array([
    [0.0, 1.0, 0.0],    # verde (regular)
    [1.0, 0.0, 1.0],    # magenta (pegajosa)
    [1.0, 1.0, 0.0],    # amarillo (caótica)
    [0.0, 0.0, 1.0],    # azul (colisión 1)
    [1.0, 0.0, 0.0],    # rojo (colisión 2)
    [0.0, 1.0, 1.0]     # cyan (escape)
])
cmap = plt.matplotlib.colors.ListedColormap(colores)

fig1, ax1 = plt.subplots(figsize=(6,6))
im1 = ax1.imshow(orbit_type, origin='lower', extent=[x_min,x_max,y_min,y_max], cmap=cmap)
ax1.set_xlabel('x')
ax1.set_ylabel('y')
ax1.set_title('Mapa de clasificación de órbitas')
import matplotlib.patches as mpatches
leyenda = [
    mpatches.Patch(color=colores[0], label='Regular'),
    mpatches.Patch(color=colores[1], label='Pegajosa'),
    mpatches.Patch(color=colores[2], label='Caótica'),
    mpatches.Patch(color=colores[3], label='Colisión 1'),
    mpatches.Patch(color=colores[4], label='Colisión 2'),
    mpatches.Patch(color=colores[5], label='Escape')
]
ax1.legend(handles=leyenda, loc='upper right', fontsize='small')
fig1.tight_layout()
fig1.savefig('mapa_orbitas.png', dpi=300)

# Mapa de tiempos (escape vs colisión):
fig2, ax2 = plt.subplots(figsize=(6,6))
esc_map = np.where(orbit_type==5, times, np.nan)
col_map = np.where((orbit_type==3)|(orbit_type==4), times, np.nan)

im_esc = ax2.imshow(esc_map, origin='lower', extent=[x_min,x_max,y_min,y_max], 
                    cmap='viridis', vmax=np.nanmax(esc_map))
im_col = ax2.imshow(col_map, origin='lower', extent=[x_min,x_max,y_min,y_max],
                    cmap='autumn', alpha=0.6, vmax=np.nanmax(col_map))
ax2.set_xlabel('x')
ax2.set_ylabel('y')
ax2.set_title('Tiempos de escape (verde) y colisión (rojo)')

# Barras de color correctamente asociadas al mismo ax:
cbar1 = fig2.colorbar(im_esc, ax=ax2, shrink=0.8)
cbar1.set_label('t_escape')

cbar2 = fig2.colorbar(im_col, ax=ax2, shrink=0.8)
cbar2.set_label('t_colision')

fig2.tight_layout()
fig2.savefig('mapa_tiempos.png', dpi=300)




plt.show()
