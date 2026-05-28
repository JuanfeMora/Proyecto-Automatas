### Fallos en el Motor Físico (Backend)

Tenemos un fallo crítico en la matemática de colisiones basado en las reglas originales de tu documento.

- **La Paradoja de la Colisión de 4 Vías (Intersección Perfecta):**
    
    En tu documento base se estipulaba explícitamente: _"si se chocan 4 corrientes de la misma intensidad estas rebotarán en sentido contrario con la misma intensidad"_.
    
    Si ejecutamos mentalmente el código actual (`_phase_c_collisions`) para una celda donde entran simultáneamente 4 vectores de intensidad $I = 3$, ocurre esto:
    
    1. `eq_y` es `True`. `eq_x` es `True`.
        
    2. Por el eje Y: $\Delta_{up} = -3$, $\Delta_{left} = +3$, $\Delta_{right} = +3$.
        
    3. Por el eje X: $\Delta_{left} = -3$, $\Delta_{up} = +3$, $\Delta_{down} = +3$.
        
    4. Suma total en el tensor de Deltas para Arriba: $\Delta_{up(total)} = -3 + 3 = 0$.
        
        El resultado matemático actual es que los vectores de delta se anulan entre sí. En lugar de rebotar, las 4 corrientes **atraviesan** la intersección intactas (efecto fantasma). Necesitamos aislar este caso con una máscara lógica exclusiva en NumPy.
        
- **Ausencia de Decaimiento por Fricción (Opcional pero recomendado):**
    
    Descartamos el decaimiento por tiempo para evitar el atributo de "edad" en memoria. Sin embargo, en un autómata discreto puro, la energía térmica se mantiene infinita si no hay disipación. Sugiero implementar una probabilidad de caída de intensidad $I = I - 1$ usando `np.random.random()` vectorizado para simular la disipación del viento a grandes distancias.
    
- **Limpieza de Estado (Reset):**
    
    Una vez que el usuario detiene la simulación (presionando `ESPACIO`), el tablero queda en un estado caótico. El autómata necesita un método `model.reset()` que devuelva el tensor `self.grid` a una matriz de ceros absolutos sin destruir la matriz de obstáculos `self.barriers`.
    

### 2. Deficiencias en el Renderizado y la UI (Frontend)

El bucle de tu interfaz actual es rústico y sufre de "ceguera" temporal.

- **Ceguera de Vectores durante la Simulación:**
    
    Cuando el sistema está en `simulating == False`, puedes ver las flechas cian apuntando en sus direcciones. Pero en cuanto presionas `ESPACIO` (`simulating == True`), el código solo renderiza los cuadros azules de presión. El usuario no tiene forma de ver en qué dirección se está moviendo el fluido. Es imperativo que el bucle de renderizado de vectores se ejecute también durante la simulación.
    
- **Telemetría en Consola vs. Pantalla:**
    
    Actualizar el título de la ventana con `pygame.display.set_caption()` para mostrar la herramienta actual es un _hack_ inaceptable para un proyecto universitario. Debes utilizar `pygame.font.SysFont` para renderizar un panel superpuesto en la esquina de la pantalla (HUD) que muestre:
    
    - Modo actual (Edición / Simulación).
        
    - Ciclos calculados (Variable $t$).
        
    - Herramienta de pincel y Fotogramas por Segundo (FPS).
        
- **Avance Discreto por Tick (Debugging Step-by-Step):**
    
    Para probar que la física de NumPy funciona, el usuario debe poder presionar una tecla (ej. `N`) mientras está pausado para avanzar exactamente un estado temporal ($t \to t+1$). Actualmente, solo puedes darle _Play_ y ver cómo todo ocurre a velocidad de 15 FPS, haciendo imposible auditar colisiones específicas a simple vista.
