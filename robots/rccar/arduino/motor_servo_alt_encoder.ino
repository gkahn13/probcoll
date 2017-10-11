/*
Code written for Arduino Uno.

The purpose is to read PWM signals coming out of a radio receiver
and either relay them unchanged to ESC/servo or substitute signals from a host system.

The steering servo and ESC(motor) are controlled with PWM signals. These meaning of these signals
may vary with time, and across devices. To get uniform behavior, the user calibrates with each session.
The host system deals only in percent control signals that are assumed to be based on calibrated PWM
signals. Thus, 0 should always mean 'extreme left', 49 should mean 'straight ahead', and 99 'extreme right'
in the percent signals, whereas absolute values of the PWM can vary for various reasons.

24 April 2016
*/
#include "PinChangeInterrupt.h" // Adafruit library
#include <Servo.h> // Arduino library

// These come from the radio receiver via three black-red-white ribbons.
#define PIN_SERVO_IN 11
#define PIN_MOTOR_IN 10
#define PIN_BUTTON_IN 12

// These go out to ESC (motor controller) and steer servo via black-red-white ribbons.
#define PIN_SERVO_OUT 9
#define PIN_MOTOR_OUT 8

// On-board LED, used to signal error state
#define PIN_LED_OUT 13

#define STATE_HUMAN_FULL_CONTROL            1
#define STATE_LOCK                          2
#define STATE_CAFFE_CAFFE_STEER_HUMAN_MOTOR 3
#define STATE_CAFFE_HUMAN_STEER_HUMAN_MOTOR 5
#define STATE_CAFFE_CAFFE_STEER_CAFFE_MOTOR 6
#define STATE_CAFFE_HUMAN_STEER_CAFFE_MOTOR 7
#define STATE_LOCK_CALIBRATE                4
#define STATE_ERROR                         -1
// Now for the sensors
#define STATE_GPS                           "'gps'"
#define STATE_GYRO                          "'gyro'"
#define STATE_SONAR                         "'sonar'"
#define STATE_ENCODER                       "'encoder'"
//
/////////////////////


/////////////////////
// Servo constants
//
// These define extreme min an max values that should never be broken.
#define SERVO_MAX   2000
#define MOTOR_MAX   SERVO_MAX
#define BUTTON_MAX  SERVO_MAX
#define SERVO_MIN   500
#define MOTOR_MIN   SERVO_MIN
#define BUTTON_MIN  SERVO_MIN
//
/////////////////////


/////////////////////
// The hand-held radio controller has two buttons. Pressing the upper or lower
// allows for reaching separate PWM levels: ~ 1710, 1200, 1000, and 888
// These are used for different control states.
// #define BUTTON_A 1710 // Human in full control of driving
// #define BUTTON_B 1200 // Lock state
// #define BUTTON_C 964  // Caffe steering, human on accelerator
// #define BUTTON_D 850  // Calibration of steering and motor control ranges
#define BUTTON_A 1900 //1988 // Human in full control of driving 
#define BUTTON_B 1700 // Lock state
#define BUTTON_C 1424  // Caffe steering, human on accelerator
#define BUTTON_D 870  // Calibration of steering and motor control ranges
#define BUTTON_DELTA 50 // range around button value that is considered in that value
//
/////////////////////


// Below are variables that hold ongoing signal data. I try to initalize them to
// to sensible values, but they will immediately be reset in the running program.
//
// These volatile variables are set by interrupt service routines
// tied to the servo and motor input pins. These values will be reset manually in the
// STATE_LOCK_CALIBRATE state, so conservative values are given here.
volatile int servo_null_pwm_value = 1500;
volatile int servo_max_pwm_value  = 1600;
volatile int servo_min_pwm_value  = 1400;
volatile int motor_null_pwm_value = 1528;
volatile int motor_max_pwm_value  = 1600;
volatile int motor_min_pwm_value  = 1400;
// These are three key values indicating current incoming signals.
// These are set in interrupt service routines.
volatile int button_pwm_value = 1210;
volatile int servo_pwm_value = servo_null_pwm_value;
volatile int motor_pwm_value = motor_null_pwm_value;
// These are used to interpret interrupt signals.
volatile unsigned long int button_prev_interrupt_time = 0;
volatile unsigned long int servo_prev_interrupt_time  = 0;
volatile unsigned long int motor_prev_interrupt_time  = 0;
volatile unsigned long int state_transition_time_ms = 0;
// Some intial conditions, putting the system in lock state.
volatile int state = STATE_LOCK;
volatile int previous_state = 0;
// Variable to receive caffe data and format it for output.
unsigned long int caffe_last_int_read_time;
int caffe_mode = -3;
float caffe_servo_percent = 49.;
float caffe_motor_percent = 49.;
int caffe_servo_pwm_value = servo_null_pwm_value;
int caffe_motor_pwm_value = motor_null_pwm_value;
int caffe_last_written_servo_pwm_value = servo_null_pwm_value;
int caffe_last_written_motor_pwm_value = motor_null_pwm_value;
// Values written to serial
volatile int written_servo_pwm_value = servo_null_pwm_value;
volatile int written_motor_pwm_value = motor_null_pwm_value;
// The host computer is not to worry about PWM values. These variables hold percent values
// that are passed up to host.
float servo_percent = 49.;
float motor_percent = 49.;
int servo_percent_int = 4900;
int motor_percent_int = 4900;

// Servo classes. ESC (motor) is treated as a servo for signaling purposes.
Servo servo;
Servo motor; 

////////////// ENCODER //////////////////
//PIN's definition
#include "RunningAverage.h"
#define encoder0PinA  2
#define encoder0PinB  3

RunningAverage times(12);

volatile int encoder0Pos = 0;
volatile boolean PastA = 0;
volatile boolean PastB = 0;
volatile unsigned long int a = 0;
volatile unsigned long int b = 0;
volatile unsigned long int t1 = micros();
volatile unsigned long int t2 = 0;
volatile unsigned long int last_t2 = 0;
volatile unsigned long int new_time = 0;
volatile unsigned long int dt = 0;
volatile float rate_1 = 0.0;

void encoder_setup() 
{
  //Serial.begin(9600);
  pinMode(encoder0PinA, INPUT);
  //turn on pullup resistor
  //digitalWrite(encoder0PinA, HIGH); //ONLY FOR SOME ENCODER(MAGNETIC)!!!! 
  pinMode(encoder0PinB, INPUT); 
  //turn on pullup resistor
  //digitalWrite(encoder0PinB, HIGH); //ONLY FOR SOME ENCODER(MAGNETIC)!!!! 
  PastA = (boolean)digitalRead(encoder0PinA); //initial value of channel A;
  PastB = (boolean)digitalRead(encoder0PinB); //and channel B

//To speed up even more, you may define manually the ISRs
// encoder A channel on interrupt 0 (arduino's pin 2)
  attachInterrupt(0, doEncoderA, CHANGE);
// encoder B channel pin on interrupt 1 (arduino's pin 3)
  attachInterrupt(1, doEncoderB, CHANGE); 

  times.clear();
}

void encoder_loop()
{ 
  if (a > 0) {
    new_time = micros() - last_t2;
    rate_1 = (1000.0 * 1000.0 / 16.0) * (a / (a * times.getAverage() + new_time));
  } else {
    rate_1 = 0.0;
  }
}

void doEncoderA()
{
  t2 = micros();
  if (last_t2 > 0) {
    times.addValue((t2 - last_t2));
    if (a < 12) {
      a += 1;
    }
  }
  last_t2 = t2;
}

void doEncoderB()
{
     b += 1;
}
//
///////////////////

////////////////////////////////////////
//
void setup()
{
  // Establishing serial communication with host system. The best values for these parameters
  // is an open question. At 9600 baud rate, data can be missed.
  Serial.begin(115200);
  Serial.setTimeout(5);

  // Setting up three input pins
  pinMode(PIN_BUTTON_IN, INPUT_PULLUP);
  pinMode(PIN_SERVO_IN, INPUT_PULLUP);
  pinMode(PIN_MOTOR_IN, INPUT_PULLUP);

  // LED out
  pinMode(PIN_LED_OUT, OUTPUT);

  // Attach interrupt service routines to pins. A change in signal triggers interrupts.
  attachPinChangeInterrupt(digitalPinToPinChangeInterrupt(PIN_BUTTON_IN),
    button_interrupt_service_routine, CHANGE);
  attachPinChangeInterrupt(digitalPinToPinChangeInterrupt(PIN_SERVO_IN),
    servo_interrupt_service_routine, CHANGE);
  attachPinChangeInterrupt(digitalPinToPinChangeInterrupt(PIN_MOTOR_IN),
    motor_interrupt_service_routine, CHANGE);

  // Attach output pins to ESC (motor) and steering servo.
  servo.attach(PIN_SERVO_OUT); 
  motor.attach(PIN_MOTOR_OUT); 

  encoder_setup(); 
}
//
////////////////////////////////////////






////////////////////////////////////////
//
void button_interrupt_service_routine(void) {
  volatile unsigned long int m = micros();
  volatile unsigned long int dt = m - button_prev_interrupt_time;
  button_prev_interrupt_time = m;
  // Human in full control of driving
  if (dt>BUTTON_MIN && dt<BUTTON_MAX) {
    button_pwm_value = dt;
    if (abs(button_pwm_value-BUTTON_A)<BUTTON_DELTA) {
      if (state == STATE_ERROR) return;
      if (state != STATE_HUMAN_FULL_CONTROL) {
        previous_state = state;
        state = STATE_HUMAN_FULL_CONTROL;
        state_transition_time_ms = m/1000.0;
      }
    }
    // Lock state
    else if (abs(button_pwm_value-BUTTON_B)<BUTTON_DELTA) {
      if (state == STATE_ERROR) return;
      if (state != STATE_LOCK) {
        previous_state = state;
        state = STATE_LOCK;
        state_transition_time_ms = m/1000.0;
      }
    }
    // Caffe steering, human on accelerator
    else if (abs(button_pwm_value-BUTTON_C)<BUTTON_DELTA) {
      if (state == STATE_ERROR) return;
      if (state != STATE_CAFFE_CAFFE_STEER_HUMAN_MOTOR && state != STATE_CAFFE_HUMAN_STEER_HUMAN_MOTOR &&
          state != STATE_CAFFE_CAFFE_STEER_CAFFE_MOTOR && state != STATE_CAFFE_HUMAN_STEER_CAFFE_MOTOR) {
        previous_state = state;
        // state = STATE_CAFFE_CAFFE_STEER_HUMAN_MOTOR;
        state = STATE_CAFFE_CAFFE_STEER_CAFFE_MOTOR;
        state_transition_time_ms = m/1000.0;
      }
    }
    // Calibration of steering and motor control ranges
    else if (abs(button_pwm_value-BUTTON_D)<BUTTON_DELTA) {
      if (state != STATE_LOCK_CALIBRATE) {
        previous_state = state;
        state = STATE_LOCK_CALIBRATE;
        state_transition_time_ms = m/1000.0;
        servo_null_pwm_value = servo_pwm_value;
        servo_max_pwm_value = servo_null_pwm_value;
        servo_min_pwm_value = servo_null_pwm_value;
        motor_null_pwm_value = motor_pwm_value;
        motor_max_pwm_value = motor_null_pwm_value;
        motor_min_pwm_value = motor_null_pwm_value;
        caffe_servo_pwm_value = servo_null_pwm_value;
        caffe_motor_pwm_value = motor_null_pwm_value;
        caffe_last_written_servo_pwm_value = servo_null_pwm_value;
      }
      if (servo_pwm_value > servo_max_pwm_value) {
        servo_max_pwm_value = servo_pwm_value;
      }
      if (servo_pwm_value < servo_min_pwm_value) {
        servo_min_pwm_value = servo_pwm_value;
      }
      if (motor_pwm_value > motor_max_pwm_value) {
        motor_max_pwm_value = motor_pwm_value;
      }
      if (motor_pwm_value < motor_min_pwm_value) {
        motor_min_pwm_value = motor_pwm_value;
      }
      written_servo_pwm_value = servo_pwm_value;
      written_motor_pwm_value = motor_pwm_value; 
    }
  }
}
//
////////////////////////////////////////




////////////////////////////////////////
//
// Servo interrupt service routine. This would be very short except that the human can take
// control from Caffe, and Caffe can take back control if steering left in neutral position.
void servo_interrupt_service_routine(void) {
  volatile unsigned long int m = micros();
  volatile unsigned long int dt = m - servo_prev_interrupt_time;
  servo_prev_interrupt_time = m;
  if (state == STATE_ERROR) return; // no action if in error state
  if (dt>SERVO_MIN && dt<SERVO_MAX) {
    servo_pwm_value = dt;
    if (state == STATE_HUMAN_FULL_CONTROL || state == STATE_LOCK) {
      servo.writeMicroseconds(servo_pwm_value);
      written_servo_pwm_value = servo_pwm_value;
    }
    else if (state == STATE_CAFFE_HUMAN_STEER_HUMAN_MOTOR || state == STATE_CAFFE_HUMAN_STEER_CAFFE_MOTOR) {
      // If steer is close to neutral, let Caffe take over.
      if (abs(servo_pwm_value-servo_null_pwm_value)<=30 ){
        previous_state = state;
        if (state == STATE_CAFFE_HUMAN_STEER_HUMAN_MOTOR) {
          state = STATE_CAFFE_CAFFE_STEER_HUMAN_MOTOR;
        } else {
          state = STATE_CAFFE_CAFFE_STEER_CAFFE_MOTOR;
        }
        state_transition_time_ms = m/1000.0;
        //servo.writeMicroseconds((caffe_servo_pwm_value+servo_pwm_value)/2);
      }
      else {
        // twice as sensitive so can reach all steering angles
        int adjusted_servo_pwm_value = 2*(servo_pwm_value - servo_null_pwm_value) + caffe_last_written_servo_pwm_value;
        adjusted_servo_pwm_value = min(adjusted_servo_pwm_value, servo_max_pwm_value);
        adjusted_servo_pwm_value = max(adjusted_servo_pwm_value, servo_min_pwm_value);
        servo.writeMicroseconds(adjusted_servo_pwm_value);
        written_servo_pwm_value = adjusted_servo_pwm_value;
      }
    }
    // If human makes strong steer command, let human take over.
    else if (state == STATE_CAFFE_CAFFE_STEER_HUMAN_MOTOR || state == STATE_CAFFE_CAFFE_STEER_CAFFE_MOTOR) {
      if (abs(servo_pwm_value-servo_null_pwm_value)>70) {
        previous_state = state;
        if (state == STATE_CAFFE_CAFFE_STEER_HUMAN_MOTOR) {
          state = STATE_CAFFE_HUMAN_STEER_HUMAN_MOTOR;
        } else {
          state = STATE_CAFFE_HUMAN_STEER_CAFFE_MOTOR;
        }
        state_transition_time_ms = m/1000.0;
        //servo.writeMicroseconds(servo_pwm_value);   
      }
      else {
        servo.writeMicroseconds(caffe_servo_pwm_value);
        written_servo_pwm_value = caffe_servo_pwm_value;
        caffe_last_written_servo_pwm_value = caffe_servo_pwm_value;
      }
    }
    else {
//      servo.writeMicroseconds(servo_null_pwm_value);
//      written_servo_pwm_value = servo_null_pwm_value;
    }
  } 
}
//
////////////////////////////////////////




////////////////////////////////////////
//
// Motor interrupt service routine. This is simple because right now only human controls motor.
void motor_interrupt_service_routine(void) {
  volatile unsigned long int m = micros();
  volatile unsigned long int dt = m - motor_prev_interrupt_time;
  motor_prev_interrupt_time = m;
  // Locking out in error state has bad results -- cannot switch off motor manually if it is on.
  // if (state == STATE_ERROR) return;
  if (dt>MOTOR_MIN && dt<MOTOR_MAX) {
    motor_pwm_value = dt;
    if (state == STATE_HUMAN_FULL_CONTROL || state == STATE_LOCK) {
      motor.writeMicroseconds(motor_pwm_value);
      written_motor_pwm_value = motor_pwm_value;
    }
    else if (state == STATE_CAFFE_HUMAN_STEER_HUMAN_MOTOR || state == STATE_CAFFE_CAFFE_STEER_HUMAN_MOTOR) {
      if (abs(motor_pwm_value - motor_null_pwm_value) <= 30) {
        // If motor is close to neutral, let caffe take over
        previous_state = state;
        if (state == STATE_CAFFE_HUMAN_STEER_HUMAN_MOTOR) {
          state = STATE_CAFFE_HUMAN_STEER_CAFFE_MOTOR;
        } else {
          state = STATE_CAFFE_CAFFE_STEER_CAFFE_MOTOR;
        }
      } else {
        // TODO: multiply by 2?
        int adjusted_motor_pwm_value = (motor_pwm_value - motor_null_pwm_value) + caffe_last_written_motor_pwm_value;
        adjusted_motor_pwm_value = min(adjusted_motor_pwm_value, motor_max_pwm_value);
        adjusted_motor_pwm_value = max(adjusted_motor_pwm_value, motor_min_pwm_value);
        motor.writeMicroseconds(adjusted_motor_pwm_value);
        written_motor_pwm_value = adjusted_motor_pwm_value;
      }
    }
    else if (state == STATE_CAFFE_CAFFE_STEER_CAFFE_MOTOR || state == STATE_CAFFE_HUMAN_STEER_CAFFE_MOTOR) {
      if (abs(motor_pwm_value - motor_null_pwm_value) > 70) {
        // If human makes strong motor command, let human take over
        previous_state = state;
        if (state == STATE_CAFFE_CAFFE_STEER_CAFFE_MOTOR) {
          state = STATE_CAFFE_CAFFE_STEER_HUMAN_MOTOR;
        } else {
          state = STATE_CAFFE_HUMAN_STEER_HUMAN_MOTOR;        
        }
      } else {
        motor.writeMicroseconds(caffe_motor_pwm_value);
        written_motor_pwm_value = caffe_motor_pwm_value;
        caffe_last_written_motor_pwm_value = caffe_motor_pwm_value;
      }    
    }
    else {
//      motor.writeMicroseconds(motor_null_pwm_value);
//      written_motor_pwm_value = motor_null_pwm_value;
    }
  } 
}
//
////////////////////////////////////////




////////////////////////////////////////
//
int check_for_error_conditions(void) {
// Check state of all of these variables for out-of-bound conditions
  // If in calibration state, ignore potential errors in order to attempt to correct.
  if (state == STATE_LOCK_CALIBRATE) return(1);
  if (
    safe_pwm_range(servo_null_pwm_value) &&
    safe_pwm_range(servo_max_pwm_value) &&
    safe_pwm_range(servo_min_pwm_value) &&
    safe_pwm_range(motor_null_pwm_value) &&
    safe_pwm_range(motor_max_pwm_value) &&
    safe_pwm_range(motor_min_pwm_value) &&
    safe_pwm_range(servo_pwm_value) &&
    safe_pwm_range(motor_pwm_value) &&
    safe_pwm_range(button_pwm_value) &&
    button_prev_interrupt_time >= 0 &&
    servo_prev_interrupt_time >= 0 &&
    motor_prev_interrupt_time >= 0 &&
    state_transition_time_ms >= 0 && 
    safe_percent_range(caffe_servo_percent) &&
    safe_percent_range(caffe_motor_percent) &&
    safe_percent_range(servo_percent) &&
    safe_percent_range(motor_percent) &&
    safe_pwm_range(caffe_servo_pwm_value) &&
    safe_pwm_range(caffe_motor_pwm_value) &&
    caffe_last_int_read_time >= 0 &&
    caffe_mode >= -3 && caffe_mode <= 9 &&
    state >= -1 && state <= 100 &&
    previous_state >= -1 && previous_state <= 100
    
  ) return(1);
  else {
    if (state != STATE_ERROR) {
      // On first entering error state, attempt to null steering and motor
      servo_pwm_value = servo_null_pwm_value;
      motor_pwm_value = motor_null_pwm_value;
      servo.writeMicroseconds(servo_null_pwm_value);
      motor.writeMicroseconds(motor_null_pwm_value);
    }
    state = STATE_ERROR;
    return(0);
  }
}
int safe_pwm_range(int p) {
  if (p < SERVO_MIN) return 0;
  if (p > SERVO_MAX) return 0;
  return(1);
}
int safe_percent_range(int p) {
  if (p > 99.99) return 0;
  if (p < 0) return 0;
  return(1);
}
//
////////////////////////////////////////





////////////////////////////////////////
//
void loop() {
  check_for_error_conditions();
  // Try to read the "caffe_int" sent by the host system (there is a timeout on serial reads, so the Arduino
  // doesn't wait long to get one -- in which case the caffe_int is set to zero.)
  long int caffe_int = Serial.parseInt();
  // If it is received, decode it to yield three control values (i.e., caffe_mode, caffe_servo_percent, and caffe_motor_percent)
  if (caffe_int > 0) {
      caffe_last_int_read_time = micros()/1000;
      caffe_mode = (int) (caffe_int / 100000000);
      servo_percent_int = (int) ((caffe_int-caffe_mode*100000000)/10000);
      motor_percent_int = (int) ((caffe_int-servo_percent_int*10000-caffe_mode*100000000));
      caffe_servo_percent = ((float) servo_percent_int) / 100.;
      caffe_motor_percent = ((float) motor_percent_int) / 100.;
    } else if (caffe_int < 0) {
      caffe_mode = (int) (caffe_int/100000000);
  }
  
  // if (caffe_int > 0) {
  //   caffe_last_int_read_time = micros()/1000;
  //   caffe_mode = caffe_int/10000;
  //   caffe_servo_percent = (caffe_int-caffe_mode*10000)/100;
  //   caffe_motor_percent = (caffe_int-caffe_servo_percent*100-caffe_mode*10000);
  // } else if (caffe_int < 0) {
  //   caffe_mode = caffe_int/10000;
  // }
  
  // Turn the caffe_servo_percent and caffe_motor_percent values from 0 to 99 values into PWM values that can be sent 
  // to ESC (motor) and servo.
  if (caffe_mode > 0) {
    if (caffe_servo_percent >= 49) {
      caffe_servo_pwm_value = (caffe_servo_percent-49)/50.0 * (servo_max_pwm_value - servo_null_pwm_value) + servo_null_pwm_value;
    }
    else {
      caffe_servo_pwm_value = (caffe_servo_percent - 50)/50.0 * (servo_null_pwm_value - servo_min_pwm_value) + servo_null_pwm_value;
    }
    if (caffe_motor_percent >= 49) {
      caffe_motor_pwm_value = (caffe_motor_percent-49)/50.0 * (motor_max_pwm_value - motor_null_pwm_value) + motor_null_pwm_value;
    }
    else {
      caffe_motor_pwm_value = (caffe_motor_percent - 50)/50.0 * (motor_null_pwm_value - motor_min_pwm_value) + motor_null_pwm_value;
    }
  }
  else {
    caffe_servo_pwm_value = servo_null_pwm_value;
    caffe_motor_pwm_value = motor_null_pwm_value;
  }
  // Compute command signal percents from signals from the handheld radio controller
  // to be sent to host computer, which doesn't bother with PWM values
  if (written_servo_pwm_value >= servo_null_pwm_value) {
    servo_percent = 49+50.0*(written_servo_pwm_value-servo_null_pwm_value)/(servo_max_pwm_value-servo_null_pwm_value);
  }
  else {
    servo_percent = 49 - 49.0*(servo_null_pwm_value-written_servo_pwm_value)/(servo_null_pwm_value-servo_min_pwm_value);
  }
  if (written_motor_pwm_value >= motor_null_pwm_value) {
    motor_percent = 49+50.0*(written_motor_pwm_value-motor_null_pwm_value)/(motor_max_pwm_value-motor_null_pwm_value);
  }
  else {
    motor_percent = 49 - 49.0*(motor_null_pwm_value-written_motor_pwm_value)/(motor_null_pwm_value-motor_min_pwm_value);
  }

  encoder_loop();

  int debug = false;
  if (debug) {
    Serial.print("(");
    Serial.print(state);
    Serial.print(",[");
    Serial.print(button_pwm_value);
    Serial.print(",");
    Serial.print(servo_pwm_value);    
    Serial.print(",");
    Serial.print(motor_pwm_value);
    Serial.print("],[");
    Serial.print(servo_min_pwm_value);
    Serial.print(",");
    Serial.print(servo_null_pwm_value);
    Serial.print(",");
    Serial.print(servo_max_pwm_value);  
    Serial.print("],[");
    Serial.print(motor_min_pwm_value);
    Serial.print(",");
    Serial.print(motor_null_pwm_value);
    Serial.print(",");
    Serial.print(motor_max_pwm_value);
    Serial.print("],");
    Serial.print(caffe_mode);
    Serial.print(",");
    Serial.print(caffe_servo_percent);
    Serial.print(",");
    Serial.print(caffe_motor_percent);
    Serial.print(",");
    Serial.print(caffe_servo_pwm_value);
    Serial.print(",");
    Serial.print(caffe_motor_pwm_value);
    Serial.print(",");
    Serial.print(servo_percent);
    Serial.print(",");
    Serial.print(motor_percent);
    Serial.print(",");
    Serial.print(millis() - state_transition_time_ms);
    Serial.println(")");
  }
  else {
    // Send data string which looks like a python tuple.
    Serial.print("(");
    Serial.print(state);
    Serial.print(",");
    Serial.print(servo_percent);
    Serial.print(",");
    Serial.print(motor_percent);
    Serial.print(",");
    Serial.print(rate_1);
    // Serial.print(",");
    // Serial.print((millis() - state_transition_time_ms)/1000); //one second resolution
    Serial.println(")");
  }
  delay(25); // How long should this be? Note, this is in ms, whereas most other times are in micro seconds.
  // Blink LED if in error state.
  if (state == STATE_ERROR) {
    digitalWrite(PIN_LED_OUT, HIGH);
    delay(100);
    digitalWrite(PIN_LED_OUT, LOW);
    delay(100);
  }
}

