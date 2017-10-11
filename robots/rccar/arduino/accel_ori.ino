nclude <Wire.h>
#include <Adafruit_Sensor.h>
#include <Adafruit_LSM303_U.h>
#include <Adafruit_BMP085_U.h>
#include <Adafruit_Simple_AHRS.h>

// Create sensor instances.
Adafruit_LSM303_Accel_Unified accel(30301);
Adafruit_LSM303_Mag_Unified   mag(30302);
Adafruit_BMP085_Unified       bmp(18001);

// Create simple AHRS algorithm using the above sensors.
Adafruit_Simple_AHRS          ahrs(&accel, &mag);

// Create event and orientatin for accel/orientation
sensors_event_t event;
sensors_vec_t orientation;

float acc_x = 0;
float x_val = 0;
float acc_y = 0;
float acc_z = 0;
float h = 0;
float p = 0;
float r = 0;
float count = 0;

// Gets the ori and accell and prints them to the serial port
void display_accel_orientation(void)
{
  accel.getEvent(&event);
  if (ahrs.getOrientation(&orientation))
  {
    Serial.print("(imu,");
    Serial.print(acc_x); Serial.print(",");
    Serial.print(acc_y / count); Serial.print(",");
    Serial.print(acc_z / count); Serial.print(",");
    Serial.print(h / count); Serial.print(",");
    Serial.print(p / count); Serial.print(",");
    Serial.print(r / count); Serial.println(")");
  }
}

void update_data(void)
{
  if (accel.getEvent(&event) and (ahrs.getOrientation(&orientation))) {
    x_val = event.acceleration.x;
    if (abs(x_val) > abs(acc_x)){
      acc_x = x_val;
    }
    // acc_x += event.acceleration.x;
    acc_y += event.acceleration.y;
    acc_z += event.acceleration.z;
    h += orientation.heading;
    p += orientation.pitch;
    r += orientation.roll;
    
    count += 1;
  }
}

void setup()
{
  Serial.begin(115200);
  // Initialize the sensors.
  accel.begin();
  mag.begin();
}

void loop(void)
{
  int i = 0;
  while (i < 50) {
    update_data();
    i += 1;
    delay(1);
  }
  display_accel_orientation();
  acc_x = 0;
  acc_y = 0;
  acc_z = 0;
  h = 0;
  p = 0;
  r = 0;
  count = 0;
}
