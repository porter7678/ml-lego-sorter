import RPi.GPIO as GPIO
import time

class Servo:

    def __init__(self, left_label, right_label):
        self.left_label = left_label
        self.right_label = right_label
        self.degree_to_duty_cycle = {
            0: 12.0,
            45: 9.0,
            65: 8.0,
            90: 6.5,
            110: 4.5,
            135: 4.0,
            180: 1.0,
        }

        GPIO.setmode(GPIO.BOARD)
        servo_pin = 12
        GPIO.setup(servo_pin, GPIO.OUT)
        self.pwm = GPIO.PWM(servo_pin, 50)
        self.pwm.start(self.degree_to_duty_cycle[90])

    def rotate(self, degrees):
        self.pwm.ChangeDutyCycle(self.degree_to_duty_cycle[degrees])
        time.sleep(1.0)
        print('done rotating')

    def move_left(self):
        print('Moving left')
        self.rotate(110)

    def move_right(self):
        print('Moving right')
        self.rotate(65)

    def move_arm(self, label):
        if label == self.left_label:
            self.move_left()
        elif label == self.right_label:
            self.move_right()
        else:
            print('Servo did not recognize label:', label)

    def cleanup(self):
        self.pwm.stop()
        GPIO.cleanup()
        print('\nGPIO sucessfully cleaned up')


if __name__ == '__main__':
    servo = Servo()
    servo.move_left()
    time.sleep(2)
    servo.move_right()
    time.sleep(2)
    servo.move_left()
    time.sleep(2)
    servo.move_right()
    time.sleep(2)

    servo.cleanup()
