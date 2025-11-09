# Robot Seguidor de Línea con Control PID

Sistema de control PID para seguimiento de línea usando visión por computadora en robot e-puck (Webots).

## Requisitos

- Webots R2023b o superior
- Robot e-puck con cámara (128x64px)
- Pista: línea negra (ancho 3-5cm) sobre fondo claro

## Instalación

1. Copiar `magia.cpp` en directorio de controladores
2. Configurar robot e-puck:
   - Motores: "left wheel motor", "right wheel motor"
   - Cámara: "camera" (posición: 0, 0.05, 0.08, rotación X: -0.3)
3. Desactivar sombras: `DirectionalLight { castShadows FALSE }`

## Parámetros PID

```cpp
KP = 0.015      // Proporcional
KI = 0.00005    // Integral
KD = 0.08       // Derivativa
BASE_SPEED = 3.5 rad/s
```

**Ajustes rápidos:**
- Oscila mucho: KP=0.01, KD=0.05
- Curvas cerradas: BASE_SPEED=2.5, KP=0.02
- Mayor velocidad: BASE_SPEED=4.5, KP=0.02, KD=0.1

## Funcionamiento

**Detección:** Escanea imagen al 80% altura, detecta píxeles negros (brillo<60), valida ancho (5-60px), calcula posición.

**Control:** `error = centro - posicion_linea` → PID → velocidades diferenciales

**Recuperación:** Al perder línea, reduce velocidad y gira hacia último lado conocido en 3 niveles progresivos.

## Configuración de Detección

```cpp
BRIGHTNESS_THRESHOLD 60  // Umbral línea negra
MIN_LINE_WIDTH 5         // Ancho mínimo válido
MAX_LINE_WIDTH 60        // Ancho máximo (filtrar sombras)
```

## Solución Problemas

| Problema | Solución |
|----------|----------|
| Oscilaciones | Reducir KP a 0.01 |
| No detecta línea | BRIGHTNESS_THRESHOLD=50-80 |
| Detecta sombras | MAX_LINE_WIDTH=40, castShadows=FALSE |
| Pierde curvas | BASE_SPEED=2.5, SEARCH_SPEED=2.0 |
| No se mueve | Verificar nombres motores y TIME_STEP=64 |

## Resultados

- Rectas: 100% éxito, ±5px oscilación
- Curvas suaves: 100% éxito
- Curvas cerradas: 95% éxito
- Recuperación: 10-15 frames (640-960ms)

## Estructura Código

```
main()
  ├─ Inicialización (robot, motores, cámara, variables PID)
  ├─ Bucle (64ms)
  │   ├─ Captura imagen
  │   ├─ Detección (píxeles→segmentos→validación→posición)
  │   ├─ Control PID o búsqueda activa
  │   └─ Aplicar velocidades
  └─ Limpieza
```

## Ecuaciones

**Error:** `e = x_centro - x_linea`

**PID:** `u = Kp*e + Ki*∫e + Kd*de/dt`

**Velocidades:** 
```
v_izq = v_base*f_curva - u
v_der = v_base*f_curva + u
```

**Factor curva:** `f = 0.7 si |e|>30, 1.0 caso contrario`

## Búsqueda Activa

| Frames | Acción | Velocidad |
|--------|--------|-----------|
| 0-20 | Giro suave | 0.3:1.0 |
| 20-40 | Giro agresivo | 0.5:1.8 |
| >40 | Giro en lugar | 0.2:2.25 |

## Limitaciones

- Requiere alto contraste
- No maneja intersecciones
- Líneas discontinuas reducen rendimiento
- Sensible a iluminación variable