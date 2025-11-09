#include <webots/Robot.hpp>
#include <webots/Motor.hpp>
#include <webots/Lidar.hpp>
#include <webots/GPS.hpp>
#include <webots/InertialUnit.hpp>
#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <cstdlib>
#include <ctime>

#define TIME_STEP 64
#define MIN_CLEARANCE 0.1
#define SPEED_FWD 3.0
#define SPEED_TURN 2.0

#define TURN_RANDOM_INTERVAL 200
#define TURN_RANDOM_LENGTH 30
#define TURN_AVOID_LENGTH 50

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

using namespace webots;
using namespace std;

enum BotMode { MOVE_FORWARD, EVADE, RANDOM_SPIN };

int main() {
  srand(time(NULL));
  Robot robot;

  // Limpiar archivos previos
  ofstream("map_points.txt", ios::trunc).close();
  ofstream("robot_pos.txt", ios::trunc).close();
  cout << "[INFO] Archivos limpiados. Iniciando robot." << endl;

  // ===== Inicialización de dispositivos =====
  Motor *left = robot.getMotor("left wheel motor");
  Motor *right = robot.getMotor("right wheel motor");
  left->setPosition(INFINITY);
  right->setPosition(INFINITY);
  left->setVelocity(0.0);
  right->setVelocity(0.0);

  Lidar *lidar = robot.getLidar("lidar");
  lidar->enable(TIME_STEP);
  lidar->enablePointCloud();

  int res = lidar->getHorizontalResolution();
  float fov = lidar->getFov();
  float maxR = lidar->getMaxRange();
  float minR = lidar->getMinRange();

  GPS *gps = robot.getGPS("gps");
  gps->enable(TIME_STEP);
  InertialUnit *imu = robot.getInertialUnit("imu");
  imu->enable(TIME_STEP);

  ofstream mapFile("map_points.txt", ios::app);
  cout << "[INFO] Exploración iniciada." << endl;

  BotMode state = MOVE_FORWARD;
  int timer = 0;
  int turnDir = 1; // 1 = derecha, -1 = izquierda

  while (robot.step(TIME_STEP) != -1) {
    const float *ranges = lidar->getRangeImage();
    const double *gpsVal = gps->getValues();
    const double *imuVal = imu->getRollPitchYaw();

    double posX = gpsVal[0];
    double posY = gpsVal[2];
    double yaw = -imuVal[2]; // orientación global

    ofstream robotPos("robot_pos.txt", ios::trunc);
    robotPos << posX << " " << posY << endl;
    robotPos.close();

    double refAngle = yaw + (M_PI / 2.0);
    bool leftBlocked = false;
    bool rightBlocked = false;

    int mid = res / 2;
    int sectorHalf = res / 4;
    int startIdx = mid - sectorHalf;
    int endIdx = mid + sectorHalf;

    for (int i = 0; i < res; ++i) {
      float dist = ranges[i];
      if (dist > maxR * 0.98 || dist < minR)
        continue;

      bool inFront = (i >= startIdx && i <= endIdx);
      if (inFront && dist < MIN_CLEARANCE) {
        if (i < mid)
          rightBlocked = true;
        else
          leftBlocked = true;
      }

      // Mapeo global
      float beam = ((float)i / (res - 1)) * fov - (fov / 2.0f);
      double globalAngle = refAngle + beam;
      double px = posX + dist * cos(globalAngle);
      double py = posY + dist * sin(globalAngle);
      mapFile << px << " " << py << endl;
    }

    double vL = 0.0, vR = 0.0;

    switch (state) {
      case MOVE_FORWARD:
        vL = SPEED_FWD;
        vR = SPEED_FWD;

        if (leftBlocked || rightBlocked) {
          state = EVADE;
          timer = TURN_AVOID_LENGTH;
          turnDir = rightBlocked ? -1 : 1;
          cout << "[STATE] -> EVADE (" << (turnDir > 0 ? "Right" : "Left") << ")" << endl;
        } else if (timer > TURN_RANDOM_INTERVAL) {
          state = RANDOM_SPIN;
          timer = TURN_RANDOM_LENGTH;
          turnDir = (rand() % 2 == 0) ? 1 : -1;
          cout << "[STATE] -> RANDOM_SPIN (" << (turnDir > 0 ? "Right" : "Left") << ")" << endl;
        } else {
          timer++;
        }
        break;

      case EVADE:
        vL = SPEED_TURN * turnDir;
        vR = -SPEED_TURN * turnDir;
        if (timer-- <= 0) {
          state = MOVE_FORWARD;
          timer = 0;
          cout << "[STATE] -> MOVE_FORWARD" << endl;
        }
        break;

      case RANDOM_SPIN:
        vL = SPEED_TURN * turnDir;
        vR = -SPEED_TURN * turnDir;
        if (leftBlocked || rightBlocked) {
          state = EVADE;
          timer = TURN_AVOID_LENGTH;
          cout << "[STATE] -> EVADE (interrumpido)" << endl;
        } else if (timer-- <= 0) {
          state = MOVE_FORWARD;
          timer = 0;
          cout << "[STATE] -> MOVE_FORWARD" << endl;
        }
        break;
    }

    left->setVelocity(vL);
    right->setVelocity(vR);
  }

  mapFile.close();
  cout << "[INFO] Mapeo completado. Archivos actualizados." << endl;
  return 0;
}
