Paula Herrera
# Documento Técnico: Arquitectura y Mejoras del Simulador (V2.0)

Este documento detalla las optimizaciones algorítmicas, físicas y de interfaz implementadas en la versión del simulador basada en tensores y Pygame. El enfoque principal del rediseño fue maximizar la eficiencia computacional y la estabilidad matemática del autómata.

---

### 1. Arquitectura Orientada a Datos (Data-Oriented Design)

El cambio más significativo es el abandono del paradigma Orientado a Objetos (donde cada celda es un objeto) en favor de un enfoque de **Data-Oriented Design (DOD)**.

* **Estructuras Tensoriales:** El estado global del sistema ya no se almacena en una lista de listas, sino en matrices contiguas de memoria gestionadas por NumPy (`np.int8`, `np.float32`). Esto permite que la CPU procese los datos secuencialmente utilizando memoria caché de manera ultraeficiente.
* **Eliminación de Bucles Anidados (Vectorización):** En la iteración de la física (`update_step`), se eliminaron por completo los bucles `for` en Python. Las operaciones de propagación se realizan mediante la función `np.roll()`, desplazando matrices enteras en un solo paso.
* **Complejidad y Rendimiento:** Aunque la complejidad teórica sigue siendo $O(N \cdot M)$ para una cuadrícula de tamaño $N \times M$, las operaciones lógicas y aritméticas ahora se ejecutan directamente en lenguaje C (bajo el capó de NumPy), erradicando el *overhead* del intérprete de Python.

### 2. Rigor Matemático y Físico

El comportamiento de los fluidos se ha estabilizado utilizando principios discretos que emulan la dinámica de fluidos real.

* **Cálculo del Gradiente de Presión ($\nabla P$):** Se implementó un sistema para inyectar viento automáticamente en celdas adyacentes si existe una diferencia de presión estática. Esto se calcula mediante diferencias finitas espaciales usando desplazamientos de matriz (e.g., `self.static_pressure - p_up >= 2`).
* **Resolución de Colisiones Paralelizada:** Las interacciones del fluido con los muros (rebote de 180°), el follaje (fricción) y los sumideros (aniquilación) se resuelven utilizando **máscaras booleanas**. Por ejemplo, el cálculo de rebote aisla las celdas de muro (`walls = self.materials == MAT_WALL`) y revierte los vectores de viento instantáneamente en toda la grilla.
* **Advección Cinemática:** Los trazadores (partículas visibles) calculan el vector dominante en su posición usando `np.argmax(self.wind, axis=0)` y son arrastrados por la corriente mediante operaciones de conjuntos (`np.logical_or`), lo que asegura que la masa no se duplique ni se pierda por errores de redondeo.

### 3. Interfaz Gráfica de Usuario (GUI) y Control

La interfaz interactiva se construyó desde cero para no depender de librerías de UI externas, garantizando compatibilidad y un control total sobre el ciclo de dibujado.

| Mejora Implementada | Descripción de la Solución |
| --- | --- |
| **Interpolación de Trazos** | Se integró el **Algoritmo de Bresenham** para trazar líneas continuas. Esto corrige el problema clásico donde mover el ratón muy rápido deja "huecos" en los muros dibujados debido a la baja tasa de muestreo del hardware. |
| **Desacoplamiento Motor/UI** | El renderizado (`model.render`) y la actualización del estado (`model.update_step`) operan de manera independiente. Esto evita que la interfaz se congele si la carga matemática de la física aumenta temporalmente. |
| **Atajos de Teclado (Hotkeys)** | Se vincularon variables de estado a interacciones de teclado no bloqueantes (Flechas direccionales para vectores, Q/W para magnitud de escalares, Espaciador para pausa/reproducción). |
| **Renderizado Reactivo** | El fondo del entorno de simulación reacciona visualmente calculando la presión dinámica global (`np.sum(self.wind, axis=0)`) y escalando un mapa de color RGB en tiempo real para visualizar la energía cinética del sistema. |

---

> **Nota de Rendimiento:** La elección de mantener los datos en formato entero de 8 bits (`np.int8`) para el viento y los materiales fue deliberada. Al restringir los valores numéricos a espacios reducidos, se minimiza la huella de memoria RAM y se maximiza la cantidad de celdas que pueden procesarse simultáneamente en las instrucciones SIMD (Single Instruction, Multiple Data) del procesador.
