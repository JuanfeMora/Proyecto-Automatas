### Grupo 1 
### Materia: Autómatas Y Lenguajes Formales C2 
### Líder: Juan Felipe Mora
### Integrantes: 
Andrés Felipe Martínez, María Paula Herrera, Maia Catalina García, Johan Felipe Prado, Wilgton Sanchez, Juan José Monsalve, Frank Nicolás Chavez, Eduardo Zambrano y Yonier Gamboa Picón

# Simulador Aerodinámico y Termodinámico

Este proyecto implementa un simulador bidimensional de flujos de aire basado en un **Autómata de Gas Reticular (LGA)**, utilizando tensores matemáticos para resolver la advección, colisión y termodinámica de corrientes de aire en una cuadrícula discreta.

## Fundamentos Matemáticos y Teóricos

A diferencia de los enfoques tradicionales de Dinámica de Fluidos Computacional (CFD) que resuelven las ecuaciones de Navier-Stokes directamente de forma continua, este simulador utiliza un modelo discreto LGA inspirado en el **modelo HPP (Hardy, Pazzis, Pomeau)** especializado en la simulación de gases.

### 1. Cuadrícula y Tensores de Estado

El espacio se divide en una cuadrícula 2D (Grid) definida por la vecindad de Von Neumann (4 direcciones cardinales). El estado del sistema se mantiene mediante un conjunto de tensores (`numpy.ndarray`) de dimensiones $(W, H)$:

- $M_{x,y}$: Matriz de Materiales (Aire, Muro, Follaje, Sumidero).
    
- $P_{x,y}$: Presión estática (escalar, tipo `int16`).
    
- $T_{x,y}$: Temperatura (escalar continuo).
    
- $V_{d,x,y}$: Tensor tridimensional de viento, donde $d \in \{0, 1, 2, 3\}$ representa los canales direccionales (Arriba, Abajo, Izquierda, Derecha).
    

### 2. Fase de Propagación (Advección)

En cada paso de tiempo discreto, las "partículas" de aire (unidades de momento) se desplazan a la celda adyacente en la dirección de su vector de velocidad. Matemáticamente, esto se resuelve mediante operaciones de desplazamiento matricial continuo (`np.roll`):

$$V'_{d}(\vec{r}) = V_d(\vec{r} - \vec{c}_d \Delta t)$$

Donde $\vec{c}_d$ es el vector de dirección unitario para el canal $d$. Los muros ($M_{x,y} = 1$) aplican una condición de frontera de rebote estricto (Bounce-back), invirtiendo la dirección de las partículas de aire entrantes para asegurar la conservación de la masa del gas.

### 3. Fase de Colisión (Modelo HPP modificado)

Las colisiones ocurren estrictamente dentro de las celdas de aire. Para satisfacer la conservación del momento lineal y la masa, se implementa una regla booleana/entera: si dos corrientes de aire colisionan frontalmente (ej. Arriba y Abajo), su estado se aniquila en ese eje y se transfiere al eje ortogonal (Izquierda y Derecha), siempre que haya "espacio" disponible en esos canales. El delta de rotación se calcula como:

$$\Delta = \min(V_{left}, V_{right}) - \min(V_{up}, V_{down})$$

### 4. Termodinámica: Ecuación de Calor y Convección

La difusión térmica se aproxima calculando el **Laplaciano discreto** sobre la matriz de temperaturas, restringido únicamente a las celdas vecinas que contienen aire:

$$\nabla^2 T \approx T_{i+1,j} + T_{i-1,j} + T_{i,j+1} + T_{i,j-1} - 4T_{i,j}$$

La convección se simula alterando probabilísticamente el momento de las celdas en función del gradiente térmico: el calor extremo transfiere momento del canal descendente al ascendente (simulando la flotabilidad natural del aire caliente), conservando la energía cinética total del sistema.

## Detalles de Implementación Técnica

- **Motor de Tensores:** Se utiliza **NumPy** para vectorizar completamente el cálculo del autómata. No existen bucles `for` que iteren sobre las celdas individualmente (lo cual colapsaría el rendimiento en Python); en su lugar, se utilizan máscaras booleanas y operadores matemáticos matriciales.
    
- **Corrección de Desbordamiento (Overflow):** Para manejar la acumulación infinita de presión (ondas de choque), los tensores críticos se han promovido de `int8` a `int16`, previniendo la corrupción matemática por desbordamiento binario.
    
- **Renderizado Optimizado:** Se emplea `pygame.surfarray` para inyectar directamente la matriz RGB pre-calculada por NumPy en la memoria de la tarjeta gráfica, evitando el cuello de botella del dibujo píxel por píxel.
    

## Interfaz e Interactividad (HUD)

El simulador cuenta con una interfaz gráfica (GUI) construida sobre Pygame que expone las herramientas del modelo tensorial:

- **Pinceles de Material:** Muros impenetrables, follaje (que aplica decaimiento lineal al momento del viento) y sumideros (agujeros negros de presión y temperatura).
    
- **Inyección de Energía:** Pinceles de presión estática, temperatura extrema y fuerza vectorial dirigida.
    
- **Trazadores de Partículas:** Partículas pasivas (color amarillo) que siguen el campo de velocidades sin alterar el flujo de aire, implementando algoritmos de rebote elástico total en colisiones con sólidos.
    

## Requisitos e Instalación

El proyecto requiere Python 3.8+ y las siguientes dependencias:

```
# Crear un entorno virtual (opcional pero recomendado)
python -m venv venv
source venv/bin/activate  # En Windows: venv\Scripts\activate

# Instalar dependencias requeridas
pip install numpy pygame
```

Para ejecutar la simulación:

```
python main_v2.6.py
```

## Limitaciones Conocidas (Análisis Crítico)

Al utilizar una cuadrícula cuadrada (Von Neumann) y un conjunto de colisiones basado en el modelo HPP, la aerodinámica macroscópica sufre de **anisotropía**. Esto significa que la simulación no es completamente invariante rotacionalmente (el aire se comporta ligeramente diferente si se mueve en diagonal frente a si se mueve paralelo a los ejes ortogonales).

Para aplicaciones de ingeniería aeronáutica rigurosas, este modelo debería actualizarse a una red FHP (hexagonal) o, idealmente, a un método de Lattice Boltzmann (LBM) completo con función de equilibrio BGK, sacrificando parte del rendimiento por una exactitud física total en las ecuaciones de Navier-Stokes para gases.
