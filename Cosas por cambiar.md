### Evolución de la Estructura de Datos (Arquitectura V2.0)

El código base actual es purista y utiliza una matriz booleana para resolver el entorno rápidamente. Para soportar termodinámica, materiales y campos de presión, la estructura booleana queda obsoleta. Debemos expandir los tensores de NumPy:

- **1. Revolución de Materiales (El Fin del Booleano):**
    
    La matriz `self.barriers` de tipo `bool` se elimina. Se reemplaza por `self.materials`, un tensor de `np.int8`:
    
    - 0 = Aire Libre.
        
    - 1 = Muro Sólido (Rebote perfecto).
        
    - 2 = Follaje / Permeable (Resta 1 nivel de intensidad vectorial, pero permite el paso).
        
    - 3 = Sumidero/Absorbedor (Aniquila el viento y la presión al contacto).
        
- **2. Presión Barométrica Activa:**
    
    Añadimos un tensor `self.static_pressure` (int8). La presión genera un gradiente. Matemáticamente, el viento debe acelerar en dirección opuesta al gradiente de presión (de alta a baja):
    
    $$\mathbf{V}_{t+1} = \mathbf{V}_t - c \nabla P$$
    
    El motor debe evaluar las celdas vecinas y generar deltas de viento si la diferencia de presión $\Delta P \ge 2$.
    
- **3. Termodinámica (Temperatura):**
    
    El viento no _es_ temperatura; la transporta (Advección). Requerimos una matriz escalar paralela `self.temperature` $(N, M)$. Si un vector de viento se mueve, arrastra su calor. La ecuación discreta que el backend debe vectorizar es:
    
    $$T_{t+1} = T_t - \nabla \cdot (\mathbf{V} T_t) + \alpha \nabla^2 T_t$$
    
- **4. Entidades Móviles (Partículas Trazadoras):**
    
    Añadimos un tensor paralelo `self.tracers` de tipo `np.int8` (donde 1 indica presencia de partícula, 0 vacío). En la fase de streaming, estas partículas consultan el vector de intensidad máxima de viento en su coordenada actual $(x, y)$ y se desplazan 1 celda en esa dirección mediante la operación de desplazamiento espacial `np.roll`.
    
- **5. Visualización (Renderizado Complejo):**
    
    Las flechas de viento se colorean según `self.temperature` (Azul a Rojo). Los trazadores (`self.tracers`) se dibujan como puntos blancos o amarillos brillantes. Los materiales usan texturas sólidas o tramadas (verde para follaje).
    

### Delegación del Escuadrón (6 Desarrolladores)

Prohibido que trabajen en el mismo archivo simultáneamente. Asigna estas ramas de Git de inmediato.

|**Escuadrón**|**Rama de Git**|**Asignación Estratégica (NumPy / Pygame)**|
|---|---|---|
|**Backend 1 y 2**|`feature/materials-pressure`|**Entorno y Presión:** Cambiar `barriers` por `materials` (int8). Reescribir `_phase_b_boundaries` usando máscaras lógicas para follaje (tipo 2) y sumideros (tipo 3). Implementar la fase matemática que inyecta vectores de viento basándose en el gradiente de `self.static_pressure`.|
|**Backend 3 y 4**|`feature/thermo-tracers`|**Advección y Cinemática:** Crear tensores `temperature` y `tracers`. Programar la propagación térmica desplazando los valores de `temperature` usando los canales de viento. Programar el movimiento de los `tracers` aplicando `np.roll` condicionado al vector dominante de viento de su misma coordenada.|
|**Frontend 1**|`feature/renderer-engine`|**Motor Gráfico:** Modificar el bucle de dibujo de Pygame. Renderizar el follaje con transparencia/trama. Leer `self.temperature` para aplicar gradientes RGB a las líneas de viento. Dibujar píxeles resaltados donde `self.tracers == 1`.|
|**Frontend 2**|`feature/gui-tools`|**UI / Herramientas:** Sustituir los controles de teclado ocultos por un HUD lateral. Crear un selector de "Pinceles" para el mouse: Pincel de Temperatura (calentar/enfriar celdas), Pincel de Follaje, Pincel de Trazadores y Pincel de Presión estática.|
