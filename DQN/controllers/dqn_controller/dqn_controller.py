"""
Controlador DQN actualizado para Webots - Versión Corregida
Compatible con Webots R2023b y versiones posteriores
"""

from controller import Supervisor
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random
import sys

# Hiperparámetros
STATE_SIZE = 10  # 8 sensores + distancia + ángulo
ACTION_SIZE = 3
GAMMA = 0.95
LR = 1e-2
BATCH_SIZE = 32
MEMORY_SIZE = 2000
EPSILON_DECAY = 0.85
MIN_EPSILON = 0.1
TARGET_UPDATE = 10

class DQN(nn.Module):
    """Red neuronal para DQN"""
    def __init__(self, state_size, action_size):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_size, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, action_size)
        
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        return self.fc4(x)

class ReplayMemory:
    """Memoria de experiencias para replay"""
    def __init__(self, capacity):
        self.memory = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)
    
    def __len__(self):
        return len(self.memory)

class DQNAgent:
    """Agente DQN"""
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.epsilon = 1.0
        
        # Redes policy y target
        self.policy_net = DQN(state_size, action_size)
        self.target_net = DQN(state_size, action_size)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=LR)
        self.memory = ReplayMemory(MEMORY_SIZE)
        self.steps = 0
        
    def act(self, state):
        """Selecciona acción usando epsilon-greedy"""
        if random.random() < self.epsilon:
            return random.randint(0, self.action_size - 1)
        
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            q_values = self.policy_net(state_tensor)
            return q_values.argmax().item()
    
    def remember(self, state, action, reward, next_state, done):
        """Almacena experiencia en memoria"""
        self.memory.push(state, action, reward, next_state, done)
    
    def replay(self):
        """Entrena la red con un batch de experiencias"""
        if len(self.memory) < BATCH_SIZE:
            return
        
        batch = self.memory.sample(BATCH_SIZE)
        states, actions, rewards, next_states, dones = zip(*batch)
        
        states = torch.FloatTensor(np.array(states))
        actions = torch.LongTensor(actions)
        rewards = torch.FloatTensor(rewards)
        next_states = torch.FloatTensor(np.array(next_states))
        dones = torch.FloatTensor(dones)
        
        # Q-values actuales
        current_q = self.policy_net(states).gather(1, actions.unsqueeze(1))
        
        # Q-values máximos del siguiente estado
        with torch.no_grad():
            max_next_q = self.target_net(next_states).max(1)[0]
            target_q = rewards + (1 - dones) * GAMMA * max_next_q
        
        # Pérdida y optimización
        loss = nn.MSELoss()(current_q.squeeze(), target_q)
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
        self.optimizer.step()
        
        # Actualizar red target periódicamente
        self.steps += 1
        if self.steps % TARGET_UPDATE == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())
        
        # Decaimiento de epsilon
        if self.epsilon > MIN_EPSILON:
            self.epsilon *= EPSILON_DECAY
    
    def save(self, filename):
        """Guarda el modelo"""
        torch.save({
            'policy_net': self.policy_net.state_dict(),
            'target_net': self.target_net.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'epsilon': self.epsilon
        }, filename)
    
    def load(self, filename):
        """Carga el modelo"""
        try:
            checkpoint = torch.load(filename)
            self.policy_net.load_state_dict(checkpoint['policy_net'])
            self.target_net.load_state_dict(checkpoint['target_net'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            self.epsilon = checkpoint['epsilon']
            print(f"Modelo cargado desde {filename}")
        except Exception as e:
            print(f"No se pudo cargar el modelo: {e}")

class RobotController:
    """Controlador del robot en Webots"""
    def __init__(self):
        try:
            # IMPORTANTE: Usar Supervisor en lugar de Robot
            self.robot = Supervisor()
            self.timestep = int(self.robot.getBasicTimeStep())
            print(f"Timestep: {self.timestep} ms")
            
            # Inicializar dispositivos
            self.init_devices()
            
            # Obtener posiciones de START y END
            self.start_node = self.robot.getFromDef('start')
            self.end_node = self.robot.getFromDef('end')
            
            # Posiciones
            self.start_position = None
            self.goal_position = None
            
            if self.start_node is not None:
                pos = self.start_node.getPosition()
                self.start_position = [pos[0], pos[1], pos[2]]
                print(f"Posición START: {self.start_position}")
            else:
                print("ADVERTENCIA: No se encontró el nodo 'start'")
                self.start_position = [-0.4, 0.0, 0.0]
            
            if self.end_node is not None:
                pos = self.end_node.getPosition()
                self.goal_position = [pos[0], pos[1], pos[2]]
                print(f"Posición END (objetivo): {self.goal_position}")
            else:
                print("ADVERTENCIA: No se encontró el nodo 'end'")
                self.goal_position = [0.5, 0.0, 0.5]
            
            # Agente DQN
            self.agent = DQNAgent(STATE_SIZE, ACTION_SIZE)
            
            # Variables de episodio
            self.episode = 0
            self.total_reward = 0
            self.max_velocity = 6.28
            
            # Variables previas
            self.prev_distance = None
            
            print("Controlador inicializado correctamente")
            
        except Exception as e:
            print(f"Error al inicializar controlador: {e}")
            sys.exit(1)
    
    def init_devices(self):
        """Inicializa los dispositivos del robot"""
        try:
            # Motores
            self.left_motor = self.robot.getDevice('left wheel motor')
            self.right_motor = self.robot.getDevice('right wheel motor')
            
            if self.left_motor is None or self.right_motor is None:
                raise Exception("No se pudieron obtener los motores")
            
            # Configurar motores para velocidad
            self.left_motor.setPosition(float('inf'))
            self.right_motor.setPosition(float('inf'))
            self.left_motor.setVelocity(0.0)
            self.right_motor.setVelocity(0.0)
            print("Motores inicializados")
            
            # Sensores de distancia
            self.distance_sensors = []
            for i in range(8):
                sensor = self.robot.getDevice(f'ps{i}')  # E-puck usa 'ps' no 'ds'
                if sensor is None:
                    sensor = self.robot.getDevice(f'ds{i}')  # Intentar con 'ds'
                
                if sensor is not None:
                    sensor.enable(self.timestep)
                    self.distance_sensors.append(sensor)
            
            if len(self.distance_sensors) == 0:
                print("ADVERTENCIA: No se encontraron sensores de distancia")
            else:
                print(f"Sensores de distancia inicializados: {len(self.distance_sensors)}")
            
            # Nodo del robot
            self.robot_node = self.robot.getSelf()
            if self.robot_node is None:
                print("ADVERTENCIA: No se pudo obtener el nodo del robot")
            else:
                print("Nodo del robot obtenido")
                
        except Exception as e:
            print(f"Error al inicializar dispositivos: {e}")
            raise
    
    def get_state(self):
        """Obtiene el estado actual del entorno"""
        try:
            # Leer sensores de distancia
            sensor_values = []
            for sensor in self.distance_sensors:
                value = sensor.getValue()
                # Normalizar valores (sensores E-puck típicamente 0-4096)
                normalized = np.clip(value / 4096.0, 0.0, 1.0)
                sensor_values.append(normalized)
            
            # Rellenar si faltan sensores
            while len(sensor_values) < 8:
                sensor_values.append(0.0)
            
            # Posición del robot
            distance_to_goal = 1.0
            angle_to_goal = 0.0
            
            if self.robot_node is not None:
                try:
                    position = self.robot_node.getPosition()
                    # CORRECCIÓN: En Webots [X, Y, Z] donde Y es altura
                    robot_pos = np.array([position[0], position[2]])  # Usar X y Z para posición 2D
                    goal_pos = np.array([self.goal_position[0], self.goal_position[2]])  # X y Z del objetivo
                    
                    # Distancia al objetivo en el plano horizontal (X-Z)
                    distance_to_goal = np.linalg.norm(robot_pos - goal_pos)
                    distance_to_goal = np.clip(distance_to_goal / 2.0, 0.0, 1.0)
                    
                    # Ángulo al objetivo
                    to_goal = goal_pos - robot_pos
                    angle_to_goal = np.arctan2(to_goal[1], to_goal[0])  # Ángulo en plano X-Z
                    
                    # Orientación del robot
                    rotation = self.robot_node.getOrientation()
                    # La orientación es una matriz 3x3 aplanada: [R00, R01, R02, R10, R11, R12, R20, R21, R22]
                    # El vector de dirección del robot es la columna 0: [R00, R10, R20]
                    # Para 2D usamos R00 (componente X) y R20 (componente Z)
                    robot_angle = np.arctan2(rotation[6], rotation[0])  # arctan2(R20, R00)
                    
                    # Diferencia angular normalizada
                    angle_diff = angle_to_goal - robot_angle
                    angle_diff = np.arctan2(np.sin(angle_diff), np.cos(angle_diff))
                    angle_to_goal = angle_diff / np.pi
                    
                except Exception as e:
                    print(f"Error obteniendo posición: {e}")
            
            # Estado completo
            state = sensor_values + [distance_to_goal, angle_to_goal]
            return np.array(state[:STATE_SIZE], dtype=np.float32)
            
        except Exception as e:
            print(f"Error en get_state: {e}")
            return np.zeros(STATE_SIZE, dtype=np.float32)
    
    def execute_action(self, action):
        """Ejecuta la acción seleccionada"""
        try:
            if action == 0:  # Adelante
                left_speed = self.max_velocity
                right_speed = self.max_velocity
            elif action == 1:  # Girar izquierda
                left_speed = self.max_velocity * 0.3
                right_speed = self.max_velocity
            elif action == 2:  # Girar derecha
                left_speed = self.max_velocity
                right_speed = self.max_velocity * 0.3
            else:
                left_speed = 0.0
                right_speed = 0.0
            
            self.left_motor.setVelocity(left_speed)
            self.right_motor.setVelocity(right_speed)
            
        except Exception as e:
            print(f"Error ejecutando acción: {e}")
    
    def calculate_reward(self, state, next_state):
        """Calcula la recompensa"""
        reward = 0.0
        done = False
        
        try:
            # Sensores (primeros 8 valores)
            sensors = state[:8]
            next_sensors = next_state[:8]
            
            # Distancia al objetivo
            prev_distance = state[8] if len(state) > 8 else 1.0
            curr_distance = next_state[8] if len(next_state) > 8 else 1.0
            
            # Penalización por colisión (sensores muy activos)
            if np.max(next_sensors) > 0.7:
                reward -= 10.0
                done = True
            
            # Recompensa por acercarse al objetivo
            if curr_distance < prev_distance:
                reward += 3.0
            else:
                reward -= 1.0
            
            # Recompensa grande por alcanzar el objetivo
            if curr_distance < 0.05:
                reward += 100.0
                done = True
            
            # Penalización por tiempo
            reward -= 0.1
            
        except Exception as e:
            print(f"Error calculando recompensa: {e}")
            reward = -0.1
        
        return reward, done
    
    def check_robot_status(self):
        """Verifica si el robot se ha volcado o salido del entorno"""
        if self.robot_node is None:
            return False
        
        try:
            position = self.robot_node.getPosition()
            rotation = self.robot_node.getOrientation()
            
            # Verificar altura (si está muy arriba o muy abajo = problema)
            if position[1] < -0.1 or position[1] > 0.5:
                print(f"Robot fuera de rango vertical: Y={position[1]:.3f}")
                return False
            
            # Verificar si está dentro del arena (2x2 metros)
            if abs(position[0]) > 1.0 or abs(position[2]) > 1.0:
                print(f"Robot fuera del arena: X={position[0]:.3f}, Z={position[2]:.3f}")
                return False
            
            # Verificar si está volcado (vector Y debe apuntar hacia arriba)
            # rotation es una matriz 3x3, el vector Y es rotation[3], rotation[4], rotation[5]
            up_vector_y = rotation[4]  # Componente Y del vector "up" del robot
            
            if up_vector_y < 0.3:  # Si el robot está muy inclinado
                print(f"Robot volcado: up_y={up_vector_y:.3f}")
                return False
            
            return True
            
        except Exception as e:
            print(f"Error verificando estado del robot: {e}")
            return False
    
    def reset_robot(self):
        """Reinicia la posición del robot sobre el START"""
        try:
            if self.robot_node is not None and self.start_position is not None:
                # Primero detener el robot completamente
                self.left_motor.setVelocity(0.0)
                self.right_motor.setVelocity(0.0)
                
                # Colocar robot en posición START con pequeña variación aleatoria
                # IMPORTANTE: Usar altura correcta del E-puck (radio de rueda ~0.02m)
                x = self.start_position[0] + random.uniform(-0.04, 0.04)
                y = 0.0  # Altura del suelo (el E-puck se ajusta automáticamente)
                z = self.start_position[2] + random.uniform(-0.04, 0.04)
                
                translation_field = self.robot_node.getField('translation')
                rotation_field = self.robot_node.getField('rotation')
                
                if translation_field is not None:
                    translation_field.setSFVec3f([x, y, z])
                
                if rotation_field is not None:
                    # Orientación aleatoria en el plano horizontal
                    angle = random.uniform(0, 2 * np.pi)
                    rotation_field.setSFRotation([0, 1, 0, angle])
                
                # Reset de la física DESPUÉS de cambiar posición
                self.robot_node.resetPhysics()
                
                # Dar tiempo para estabilizarse
                for _ in range(3):
                    self.robot.step(self.timestep)
                
                print(f"Robot reposicionado en START: [{x:.3f}, {y:.3f}, {z:.3f}], ángulo: {angle:.2f} rad")
            else:
                print("ADVERTENCIA: No se puede resetear - robot_node o start_position es None")
                
        except Exception as e:
            print(f"Error reseteando robot: {e}")
        """Reinicia la posición del robot sobre el START"""
        try:
            if self.robot_node is not None and self.start_position is not None:
                # Primero detener el robot completamente
                self.left_motor.setVelocity(0.0)
                self.right_motor.setVelocity(0.0)
                
                # Colocar robot en posición START con pequeña variación aleatoria
                # IMPORTANTE: Usar altura correcta del E-puck (radio de rueda ~0.02m)
                x = self.start_position[0] + random.uniform(-0.04, 0.04)
                y = 0.0  # Altura del suelo (el E-puck se ajusta automáticamente)
                z = self.start_position[2] + random.uniform(-0.04, 0.04)
                
                translation_field = self.robot_node.getField('translation')
                rotation_field = self.robot_node.getField('rotation')
                
                if translation_field is not None:
                    translation_field.setSFVec3f([x, y, z])
                
                if rotation_field is not None:
                    # Orientación aleatoria en el plano horizontal
                    angle = random.uniform(0, 2 * np.pi)
                    rotation_field.setSFRotation([0, 1, angle, 0])
                
                # Reset de la física DESPUÉS de cambiar posición
                self.robot_node.resetPhysics()
                
                # Dar tiempo para estabilizarse
                for _ in range(3):
                    self.robot.step(self.timestep)
                
                print(f"Robot reposicionado en START: [{x:.3f}, {y:.3f}, {z:.3f}], ángulo: {angle:.2f} rad")
            else:
                print("ADVERTENCIA: No se puede resetear - robot_node o start_position es None")
                
        except Exception as e:
            print(f"Error reseteando robot: {e}")
    
    def run(self):
        """Loop principal del controlador"""
        print("=== Iniciando entrenamiento DQN ===")
        print(f"Estado: {STATE_SIZE}, Acciones: {ACTION_SIZE}")
        
        steps = 0
        max_steps = 500
        
        try:
            # Dar tiempo para que todo se inicialice
            for _ in range(10):
                if self.robot.step(self.timestep) == -1:
                    print("Simulación terminada durante inicialización")
                    return
            
            state = self.get_state()
            print(f"Estado inicial obtenido: {state.shape}")
            
            while self.robot.step(self.timestep) != -1:
                # Seleccionar y ejecutar acción
                action = self.agent.act(state)
                self.execute_action(action)
                
                # Esperar un paso de simulación
                if self.robot.step(self.timestep) == -1:
                    break
                
                # Obtener nuevo estado
                next_state = self.get_state()
                
                # Verificar estado del robot
                robot_ok = self.check_robot_status()
                if not robot_ok:
                    reward = -50.0  # Penalización grande por volcarse o salirse
                    done = True
                    print("Robot volcado o fuera de límites - Episodio terminado")
                else:
                    # Calcular recompensa
                    reward, done = self.calculate_reward(state, next_state)
                self.total_reward += reward
                
                # Almacenar experiencia
                self.agent.remember(state, action, reward, next_state, done)
                
                # Entrenar agente
                self.agent.replay()
                
                state = next_state
                steps += 1
                
                # Reiniciar episodio
                if done or steps >= max_steps:
                    self.episode += 1
                    print(f"Episodio {self.episode} | Pasos: {steps} | "
                          f"Recompensa: {self.total_reward:.2f} | "
                          f"Epsilon: {self.agent.epsilon:.3f}")
                    
                    # Guardar modelo periódicamente
                    if self.episode % 10 == 0:
                        self.agent.save(f'dqn_model_ep{self.episode}.pth')
                        print(f"Modelo guardado: dqn_model_ep{self.episode}.pth")
                    
                    # Reiniciar
                    self.reset_robot()
                    
                    # Esperar más pasos para que el robot se estabilice completamente
                    for _ in range(10):
                        if self.robot.step(self.timestep) == -1:
                            return
                    
                    # Verificar que el robot esté estable después del reset
                    if not self.check_robot_status():
                        print("Robot inestable después del reset, reintentando...")
                        self.reset_robot()
                        for _ in range(10):
                            if self.robot.step(self.timestep) == -1:
                                return
                    
                    state = self.get_state()
                    self.total_reward = 0
                    steps = 0
                    
        except KeyboardInterrupt:
            print("\nEntrenamiento interrumpido por el usuario")
            self.agent.save('dqn_model_final.pth')
            print("Modelo final guardado")
        except Exception as e:
            print(f"Error en el loop principal: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    try:
        controller = RobotController()
        controller.run()
    except Exception as e:
        print(f"Error fatal: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)