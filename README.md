## 1. OBJETIVO DEL SISTEMA

Desarrollar un simulador no interactivo de corrientes de aire y obstáculos basado en un Autómata Celular discreto estricto. El sistema calcula la propagación vectorial, colisión contra fronteras y presión emergente del fluido en una cuadrícula 2D utilizando actualizaciones sincrónicas estáticas.

## 2. ARQUITECTURA DEL SISTEMA (MVC DE ACOPLAMIENTO DÉBIL)

El proyecto utiliza un patrón Modelo-Vista-Controlador estricto, implementando Data-Oriented Design para garantizar un rendimiento computacional de $O(N \cdot M)$. Se omite el uso de objetos por celda para maximizar la localidad de caché.

Modelo (M): Tensores tridimensionales paralelos basados en numpy (int8).

Vista (V): Motor gráfico (Pygame / PyQt) que lee pasivamente el modelo matricial.

Controlador (C): Bucle de reloj principal que invoca las fases de propagación y resolución de conflictos.

## 3. ALCANCE DEL PROYECTO (SCOPE)

### 3.1. LO QUE SE HARÁ (In-Scope)

Matriz de 4 Canales: Cada coordenada $(x,y)$ posee 4 canales vectoriales (Arriba, Abajo, Izquierda, Derecha) que almacenan magnitudes de intensidad (0 a 3).

Obstáculos Inmutables: Celdas lógicas marcadas en una máscara booleana (True) que actúan como barreras de rebote absoluto.

Límites Sólidos: Los bordes de la cuadrícula actúan sistemáticamente como barreras cerradas.

Colisiones Deterministas: * Choques frontales de igual intensidad se dispersan perpendicularmente.

Choques asimétricos resultan en la sustracción del vector menor y propagación del excedente para mantener la estabilidad matemática del tensor.

Presión Emergente: La presión escalar de una celda se calcula dinámicamente en tiempo de renderizado como la suma de sus 4 canales en el instante $t$.

Configuración Inicial: Interfaz estática previa al inicio de simulación que permite definir el estado $t=0$ (corrientes y obstáculos).

### 3.2. LO QUE NO SE HARÁ (Out-of-Scope)

Movimiento Diagonal: Descartado. Topología restringida estrictamente a la vecindad de Von Neumann (4 direcciones ortogonales).

Atracción Escalar por Presión: La presión se define como una consecuencia de la acumulación de masa vectorial, no como una causa directriz del movimiento.

Decaimiento Temporal Lineal: Se descarta el atributo de "edad" en memoria. La pérdida de energía/intensidad ocurre exclusivamente mediante reglas de dispersión y colisión térmica simulada.

Interactividad en Tiempo de Ejecución: El autómata es un sistema cerrado. Iniciada la iteración $t > 0$, el estado no acepta inputs externos del usuario.
