// File:          magia.cpp
// Date:
// Description:   Camera PID with smart curve handling
// Author:
// Modifications:

#include <webots/Robot.hpp>
#include <webots/Motor.hpp>
#include <webots/Camera.hpp>
#include <iostream>
#include <cmath>

#define TIME_STEP 64
#define MAX_SPEED 6.28
#define BASE_SPEED 3.5
#define SEARCH_SPEED 1.5  // Velocidad reducida al buscar línea

// PID
#define KP 0.015
#define KI 0.00005
#define KD 0.08

// Detección
#define BRIGHTNESS_THRESHOLD 60
#define MIN_PIXELS 15
#define MAX_LINE_WIDTH 60
#define MIN_LINE_WIDTH 5

using namespace webots;

int main(int argc, char **argv) {
  Robot *robot = new Robot();
  
  // Configurar motores
  Motor *left_motor = robot->getMotor("left wheel motor");
  Motor *right_motor = robot->getMotor("right wheel motor");
  
  left_motor->setPosition(INFINITY);
  right_motor->setPosition(INFINITY);
  left_motor->setVelocity(0.0);
  right_motor->setVelocity(0.0);
  
  // Configurar cámara
  Camera *camera = robot->getCamera("camera");
  if (camera == nullptr) {
    std::cerr << "ERROR: Cámara no encontrada" << std::endl;
    delete robot;
    return 1;
  }
  
  camera->enable(TIME_STEP);
  robot->step(TIME_STEP);
  
  int width = camera->getWidth();
  int height = camera->getHeight();
  int center_x = width / 2;
  
  std::cout << "=== Robot con Búsqueda Inteligente ===" << std::endl;
  std::cout << "Cámara: " << width << "x" << height << std::endl;
  
  // Variables del PID
  double error = 0.0;
  double previous_error = 0.0;
  double integral = 0.0;
  double last_valid_error = 0.0;
  int last_line_position = center_x;  // Última posición conocida de la línea
  int frames_without_line = 0;
  bool searching = false;
  
  while (robot->step(TIME_STEP) != -1) {
    
    const unsigned char *image = camera->getImage();
    if (image == nullptr) continue;
    
    // Escanear línea
    int scan_line = height * 4 / 5;
    
    // Detectar píxeles negros
    bool is_black[width];
    int black_count = 0;
    
    for (int x = 0; x < width; x++) {
      int pixel_index = (scan_line * width + x) * 4;
      
      unsigned char blue = image[pixel_index];
      unsigned char green = image[pixel_index + 1];
      unsigned char red = image[pixel_index + 2];
      
      int brightness = (red + green + blue) / 3;
      
      is_black[x] = (brightness < BRIGHTNESS_THRESHOLD);
      if (is_black[x]) black_count++;
    }
    
    // Encontrar segmentos negros válidos
    int best_segment_center = -1;
    int best_segment_width = 0;
    int closest_distance = width;
    
    int segment_start = -1;
    
    for (int x = 0; x <= width; x++) {
      bool current_black = (x < width) && is_black[x];
      
      if (current_black && segment_start == -1) {
        segment_start = x;
      } else if (!current_black && segment_start != -1) {
        int segment_width = x - segment_start;
        int segment_center = (segment_start + x - 1) / 2;
        int distance_to_center = abs(segment_center - center_x);
        
        bool valid_width = (segment_width >= MIN_LINE_WIDTH) && 
                          (segment_width <= MAX_LINE_WIDTH);
        
        if (valid_width && distance_to_center < closest_distance) {
          best_segment_center = segment_center;
          best_segment_width = segment_width;
          closest_distance = distance_to_center;
        }
        
        segment_start = -1;
      }
    }
    
    // Verificar detección
    bool line_detected = (best_segment_width > 0);
    
    double left_speed = 0.0;
    double right_speed = 0.0;
    
    if (line_detected) {
      // ===== LÍNEA ENCONTRADA =====
      searching = false;
      frames_without_line = 0;
      
      // Guardar posición de la línea
      last_line_position = best_segment_center;
      
      // Calcular error normal
      error = center_x - best_segment_center;
      last_valid_error = error;
      
      // PID normal
      integral += error;
      if (integral > 2000) integral = 2000;
      if (integral < -2000) integral = -2000;
      
      double derivative = error - previous_error;
      double correction = (KP * error) + (KI * integral) + (KD * derivative);
      
      double max_correction = BASE_SPEED * 0.9;
      if (correction > max_correction) correction = max_correction;
      if (correction < -max_correction) correction = -max_correction;
      
      // Velocidad adaptativa en curvas
      double speed_factor = 1.0;
      if (abs(error) > 30) {
        speed_factor = 0.7;
      }
      
      left_speed = BASE_SPEED * speed_factor - correction;
      right_speed = BASE_SPEED * speed_factor + correction;
      
      std::cout << "✓ Línea detectada | pos=" << best_segment_center 
                << " E=" << (int)error << std::endl;
      
    } else {
      // ===== LÍNEA PERDIDA =====
      frames_without_line++;
      searching = true;
      
      std::cout << "⚠ BÚSQUEDA activa | frames=" << frames_without_line;
      
      // Determinar dirección de búsqueda según última posición
      bool line_was_on_left = (last_line_position < center_x);
      bool line_was_on_right = (last_line_position >= center_x);
      
      // Reducir velocidad y girar hacia donde estaba la línea
      if (frames_without_line < 20) {
        // Búsqueda suave: girar hacia el último lado conocido
        if (line_was_on_left) {
          // Línea estaba a la izquierda -> girar a la izquierda
          left_speed = SEARCH_SPEED * 0.3;
          right_speed = SEARCH_SPEED;
          std::cout << " | Girando IZQUIERDA" << std::endl;
        } else {
          // Línea estaba a la derecha -> girar a la derecha
          left_speed = SEARCH_SPEED;
          right_speed = SEARCH_SPEED * 0.3;
          std::cout << " | Girando DERECHA" << std::endl;
        }
      } else if (frames_without_line < 40) {
        // Búsqueda más agresiva
        if (line_was_on_left) {
          left_speed = 0.5;
          right_speed = SEARCH_SPEED * 1.2;
          std::cout << " | Búsqueda agresiva IZQUIERDA" << std::endl;
        } else {
          left_speed = SEARCH_SPEED * 1.2;
          right_speed = 0.5;
          std::cout << " | Búsqueda agresiva DERECHA" << std::endl;
        }
      } else {
        // Búsqueda extrema: girar en el lugar
        if (line_was_on_left) {
          left_speed = 0.2;
          right_speed = SEARCH_SPEED * 1.5;
          std::cout << " | GIRO EN LUGAR (izq)" << std::endl;
        } else {
          left_speed = SEARCH_SPEED * 1.5;
          right_speed = 0.2;
          std::cout << " | GIRO EN LUGAR (der)" << std::endl;
        }
      }
      
      // Decaer integral durante búsqueda
      integral *= 0.9;
    }
    
    // Limitar velocidades
    if (left_speed > MAX_SPEED) left_speed = MAX_SPEED;
    if (left_speed < 0) left_speed = 0;
    if (right_speed > MAX_SPEED) right_speed = MAX_SPEED;
    if (right_speed < 0) right_speed = 0;
    
    // Aplicar velocidades
    left_motor->setVelocity(left_speed);
    right_motor->setVelocity(right_speed);
    
    previous_error = error;
  }
  
  delete robot;
  return 0;