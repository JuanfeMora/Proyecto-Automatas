El código base actual es purista. Usa una matriz booleana (`self.barriers`) para resolver el entorno rápidamente.

Si queremos distintos materiales y temperatura, esa estructura booleana queda obsoleta. Debemos evolucionar el diseño de datos:

1. **Revolución de Materiales (El Fin del Booleano):**
    
    La matriz `self.barriers` de tipo `bool` debe morir. Debe ser reemplazada por `self.materials`, un tensor de `np.int8` donde:
    
    - 0 = Aire Libre.
        
    - 1 = Muro Sólido (Rebote perfecto, lo que ya hace).
        
    - 2 = Follaje / Permeable (Resta intensidad pero permite el paso).
        
    - 3 = Absorbedor (Elimina el viento sin rebotar).
        
2. **Termodinámica (Temperatura):**
    
    El viento no _es_ temperatura, el viento _transporta_ temperatura (Advección). Requerimos una matriz escalar paralela `self.temperature` de tamaño $(N, M)$. Si un vector de viento se mueve de $(x, y)$ a $(x+1, y)$, debe arrastrar un porcentaje del valor térmico de su celda de origen a la nueva celda.
    
    La ecuación discreta de advección que tu backend deberá vectorizar se verá aproximadamente así:
    
    $$T_{t+1} = T_t - \nabla \cdot (V \times T_t) + \alpha \nabla^2 T_t$$
    
3. **Visualización (Color del Aire):**
    
    Darle "color al aire" arbitrariamente es inútil. El color debe representar un dato físico. Te propongo que el color del viento en el renderizado dependa de su **Temperatura** (Rojo = Caliente, Azul = Frío) o de su **Intensidad** (Cian = Débil, Blanco = Fuerte).
    

### Delegación del Escuadrón (6 Desarrolladores)

Tus compañeros ya no pueden trabajar en el mismo archivo. Debes exigirles que trabajen en ramas de Git separadas (`feature/materials`, `feature/temperature`, `feature/gui`).

|**Escuadrón**|**Asignación Estratégica**|**Tareas Específicas a Implementar en NumPy / Pygame**|
|---|---|---|
|**Backend 1 y 2 (Física de Materiales)**|`_phase_b_boundaries` extendida|Cambiar `self.barriers` a `self.materials` de tipo entero. Implementar máscaras booleanas para cada material. Programar la atenuación del viento al pasar por material tipo 2 (follaje), restando intensidad en lugar de rebotarlo.|
|**Backend 3 y 4 (Termodinámica)**|Nueva `_phase_e_temperature`|Crear matriz escalar `self.temperature`. Programar la transferencia de calor: el viento mueve valores de la matriz de temperatura en la dirección de los canales activos. Añadir un decaimiento térmico (enfriamiento ambiental).|
|**Frontend 1 (Motor de Renderizado)**|Integración de Color Vectorial|Modificar el bucle de dibujo de Pygame. Leer la matriz `self.temperature` del backend y mapear los valores térmicos a un gradiente RGB para colorear las líneas de viento (azul a rojo).|
|**Frontend 2 (UI / UX)**|Interfaz Gráfica de Usuario Extendida|Sustituir los controles de teclado ocultos por un menú lateral en Pygame (botones clicables). Crear "Pinceles" visibles para Temperatura, Muros, Hojas y Viento.|
