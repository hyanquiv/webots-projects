"""
Q-Learning Controller para Navegación en Grid 4x8 (A → B)
Navega desde Start (amarillo) hasta Target (verde) evitando obstáculos (rojo)
"""

from controller import Robot, Motor, DistanceSensor, GPS
import numpy as np
import pickle
import os

class GridQLearningRobot:
    def __init__(self):
        # Inicializar robot Webots
        self.robot = Robot()
        self.timestep = int(self.robot.getBasicTimeStep())
        
        # Definir el entorno (Grid 4x8)
        self.grid_width = 4
        self.grid_height = 8
        self.cell_size = 0.5  # Tamaño de cada celda en metros (ajustar según tu mundo)
        
        # Posiciones en el grid
        self.start_pos = (2, 3)  # Posición inicial (columna 3, fila 2)
        self.target_pos = (3, 8)  # Posición meta (columna 4, fila 8)
        self.obstacles = [(3, 6)]  # Obstáculos (ajustar según tu mundo)
        
        # Parámetros Q-Learning
        self.alpha = 0.1        # Tasa de aprendizaje
        self.gamma = 0.95       # Factor de descuento
        self.epsilon = 1.0      # Exploración inicial
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.01
        
        # Q-table: diccionario {(x, y): [Q(up), Q(down), Q(left), Q(right)]}
        self.q_table = {}
        
        # Acciones: 0=Arriba, 1=Abajo, 2=Izquierda, 3=Derecha
        self.actions = {
            0: (0, 1),   # Arriba (Up)
            1: (0, -1),  # Abajo (Down)
            2: (-1, 0),  # Izquierda (Left)
            3: (1, 0)    # Derecha (Right)
        }
        self.action_names = {0: 'Up', 1: 'Down', 2: 'Left', 3: 'Right'}
        
        # Inicializar motores
        self.left_motor = self.robot.getDevice('left wheel motor')
        self.right_motor = self.robot.getDevice('right wheel motor')
        self.left_motor.setPosition(float('inf'))
        self.right_motor.setPosition(float('inf'))
        self.left_motor.setVelocity(0.0)
        self.right_motor.setVelocity(0.0)
        
        # Inicializar GPS (para conocer posición)
        self.gps = self.robot.getDevice('gps')
        if self.gps:
            self.gps.enable(self.timestep)
        
        # Inicializar sensores de distancia
        self.distance_sensors = []
        sensor_names = ['ps0', 'ps1', 'ps2', 'ps3', 'ps4', 'ps5', 'ps6', 'ps7']
        for name in sensor_names:
            sensor = self.robot.getDevice(name)
            if sensor:
                sensor.enable(self.timestep)
                self.distance_sensors.append(sensor)
        
        # Variables de control
        self.current_grid_pos = self.start_pos
        self.episode = 0
        self.steps_in_episode = 0
        self.total_steps = 0
        self.successes = 0
        self.max_steps_per_episode = 200
        
    def world_to_grid(self, x, z):
        """Convierte coordenadas del mundo a posición en el grid"""
        grid_x = int(round(x / self.cell_size)) + self.grid_width // 2
        grid_y = int(round(z / self.cell_size)) + self.grid_height // 2
        # Limitar a los bordes del grid
        grid_x = max(1, min(self.grid_width, grid_x))
        grid_y = max(1, min(self.grid_height, grid_y))
        return (grid_x, grid_y)
    
    def get_state(self):
        """Obtiene la posición actual en el grid"""
        if self.gps:
            pos = self.gps.getValues()
            return self.world_to_grid(pos[0], pos[2])
        return self.current_grid_pos
    
    def initialize_q_values(self, state):
        """Inicializa valores Q para un estado nuevo"""
        if state not in self.q_table:
            self.q_table[state] = [0.0, 0.0, 0.0, 0.0]
    
    def get_q_value(self, state, action):
        """Obtiene el valor Q para un par estado-acción"""
        self.initialize_q_values(state)
        return self.q_table[state][action]
    
    def choose_action(self, state):
        """Selecciona acción usando epsilon-greedy"""
        if np.random.random() < self.epsilon:
            # Exploración: acción aleatoria
            return np.random.randint(4)
        else:
            # Explotación: mejor acción
            self.initialize_q_values(state)
            q_values = self.q_table[state]
            max_q = max(q_values)
            best_actions = [i for i, q in enumerate(q_values) if q == max_q]
            return np.random.choice(best_actions)
    
    def is_valid_position(self, pos):
        """Verifica si una posición es válida (dentro del grid y sin obstáculos)"""
        x, y = pos
        if x < 1 or x > self.grid_width or y < 1 or y > self.grid_height:
            return False
        if pos in self.obstacles:
            return False
        return True
    
    def get_next_state(self, state, action):
        """Calcula el siguiente estado dada una acción"""
        dx, dy = self.actions[action]
        next_state = (state[0] + dx, state[1] + dy)
        
        # Si el siguiente estado no es válido, quedarse en el estado actual
        if not self.is_valid_position(next_state):
            return state
        return next_state
    
    def calculate_reward(self, state, action, next_state):
        """Calcula la recompensa para una transición"""
        # Llegar al objetivo
        if next_state == self.target_pos:
            return 100.0
        
        # Chocar con obstáculo o pared
        if not self.is_valid_position(next_state):
            return -10.0
        
        # Acercarse al objetivo (recompensa menor)
        current_distance = abs(state[0] - self.target_pos[0]) + abs(state[1] - self.target_pos[1])
        next_distance = abs(next_state[0] - self.target_pos[0]) + abs(next_state[1] - self.target_pos[1])
        
        if next_distance < current_distance:
            return 1.0  # Se acercó
        else:
            return -0.5  # Se alejó o no cambió
    
    def update_q_table(self, state, action, reward, next_state):
        """Actualiza Q-table usando la ecuación de Q-Learning"""
        self.initialize_q_values(state)
        self.initialize_q_values(next_state)
        
        current_q = self.q_table[state][action]
        max_future_q = max(self.q_table[next_state])
        
        # Q(s,a) = Q(s,a) + α[r + γ·max(Q(s',a')) - Q(s,a)]
        new_q = current_q + self.alpha * (reward + self.gamma * max_future_q - current_q)
        self.q_table[state][action] = new_q
    
    def move_to_grid_position(self, target_grid_pos):
        """Mueve el robot hacia una posición del grid"""
        # Implementación simple: mover en línea recta
        # En tu implementación real, deberías usar control de motores más sofisticado
        current_state = self.get_state()
        
        if current_state == target_grid_pos:
            self.stop_motors()
            return True
        
        # Calcular dirección
        dx = target_grid_pos[0] - current_state[0]
        dy = target_grid_pos[1] - current_state[1]
        
        # Movimiento simple (ajustar velocidades según tu robot)
        base_speed = 3.0
        
        if abs(dx) > abs(dy):
            if dx > 0:  # Derecha
                self.left_motor.setVelocity(base_speed)
                self.right_motor.setVelocity(base_speed)
            else:  # Izquierda
                self.left_motor.setVelocity(-base_speed)
                self.right_motor.setVelocity(-base_speed)
        else:
            if dy > 0:  # Arriba
                self.left_motor.setVelocity(base_speed)
                self.right_motor.setVelocity(base_speed)
            else:  # Abajo
                self.left_motor.setVelocity(-base_speed)
                self.right_motor.setVelocity(-base_speed)
        
        return False
    
    def stop_motors(self):
        """Detiene los motores"""
        self.left_motor.setVelocity(0.0)
        self.right_motor.setVelocity(0.0)
    
    def reset_position(self):
        """Reinicia el robot a la posición inicial"""
        self.current_grid_pos = self.start_pos
        self.steps_in_episode = 0
        # En Webots, podrías usar supervisorSetField para reiniciar posición
    
    def save_q_table(self, filename='q_table_grid.pkl'):
        """Guarda la Q-table"""
        with open(filename, 'wb') as f:
            pickle.dump(self.q_table, f)
        print(f"✓ Q-table guardada: {len(self.q_table)} estados")
    
    def load_q_table(self, filename='q_table_grid.pkl'):
        """Carga la Q-table"""
        if os.path.exists(filename):
            with open(filename, 'rb') as f:
                self.q_table = pickle.load(f)
            print(f"✓ Q-table cargada: {len(self.q_table)} estados")
            return True
        return False
    
    def print_policy(self):
        """Imprime la política aprendida"""
        print("\n=== POLÍTICA APRENDIDA ===")
        for y in range(self.grid_height, 0, -1):
            row = ""
            for x in range(1, self.grid_width + 1):
                pos = (x, y)
                if pos == self.start_pos:
                    row += " S "
                elif pos == self.target_pos:
                    row += " T "
                elif pos in self.obstacles:
                    row += " X "
                elif pos in self.q_table:
                    best_action = np.argmax(self.q_table[pos])
                    arrows = ['↑', '↓', '←', '→']
                    row += f" {arrows[best_action]} "
                else:
                    row += " . "
            print(f"Fila {y}: {row}")
        print("=" * 40)
    
    def run(self):
        """Bucle principal de entrenamiento"""
        print("=" * 60)
        print("Q-LEARNING ROBOT - NAVEGACIÓN A → B (Grid 4x8)")
        print("=" * 60)
        print(f"Start: {self.start_pos}, Target: {self.target_pos}")
        print(f"Obstáculos: {self.obstacles}")
        print(f"Parámetros: α={self.alpha}, γ={self.gamma}, ε={self.epsilon}")
        print("=" * 60)
        
        # Cargar Q-table si existe
        self.load_q_table()
        
        while self.robot.step(self.timestep) != -1:
            # Obtener estado actual
            state = self.get_state()
            
            # Verificar si llegó al objetivo
            if state == self.target_pos:
                print(f"🎯 ¡ÉXITO! Episodio {self.episode} completado en {self.steps_in_episode} pasos")
                self.successes += 1
                self.episode += 1
                self.reset_position()
                self.save_q_table()
                continue
            
            # Verificar límite de pasos
            if self.steps_in_episode >= self.max_steps_per_episode:
                print(f"⏱ Episodio {self.episode} - Tiempo límite alcanzado")
                self.episode += 1
                self.reset_position()
                continue
            
            # Seleccionar acción
            action = self.choose_action(state)
            next_state = self.get_next_state(state, action)
            
            # Calcular recompensa
            reward = self.calculate_reward(state, action, next_state)
            
            # Actualizar Q-table
            self.update_q_table(state, action, reward, next_state)
            
            # Ejecutar acción (mover robot)
            self.move_to_grid_position(next_state)
            self.current_grid_pos = next_state
            
            # Incrementar contadores
            self.steps_in_episode += 1
            self.total_steps += 1
            
            # Mostrar progreso cada 100 pasos
            if self.total_steps % 100 == 0:
                success_rate = (self.successes / max(1, self.episode)) * 100 if self.episode > 0 else 0
                print(f"📊 Paso {self.total_steps} | Episodio {self.episode} | "
                      f"Éxitos: {self.successes} ({success_rate:.1f}%) | "
                      f"ε: {self.epsilon:.3f} | Estados: {len(self.q_table)}")
            
            # Guardar cada 500 pasos
            if self.total_steps % 500 == 0:
                self.save_q_table()
                self.print_policy()
            
            # Reducir epsilon
            if self.epsilon > self.epsilon_min:
                self.epsilon *= self.epsilon_decay


# Ejecutar el robot
if __name__ == "__main__":
    robot = GridQLearningRobot()
    robot.run()