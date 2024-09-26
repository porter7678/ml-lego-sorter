import os
import threading
import time

from gpiozero import AngularServo
from gpiozero.pins.pigpio import PiGPIOFactory


class MyServo:

    def __init__(self, left_label, right_label):
        self.left_label = left_label
        self.right_label = right_label
        self.curr_direction = None

        # pin 12 == gpio 18
        gpio_num = 18

        # Use environment variables for pigpio address and port
        pigpio_host = os.getenv("PIGPIO_ADDR", "localhost")
        pigpio_port = int(os.getenv("PIGPIO_PORT", 8888))
        factory = PiGPIOFactory(host=pigpio_host, port=pigpio_port)

        self.servo = AngularServo(
            gpio_num,
            pin_factory=factory,
            min_pulse_width=0.5 / 1000,
            max_pulse_width=2.5 / 1000,
        )
        self.stop()

    def move_left(self):
        self.servo.angle = -30
        time.sleep(0.2)
        self.stop()
        self.curr_direction = "left'"

    def move_right(self):
        self.servo.angle = 30
        time.sleep(0.2)
        self.stop()
        self.curr_direction = "right"

    def move_arm(self, label):
        if label == self.left_label and self.curr_direction != "left":
            self.move_left()
        elif label == self.right_label and self.curr_direction != "right":
            self.move_right()
        else:
            print("Servo did not recognize label:", label)

    def stop(self):
        self.servo.detach()


class ServoThread(threading.Thread):
    def __init__(self, servo):
        super().__init__()
        self.servo = servo
        self.command_queue = []
        self.running = True

    def run(self):
        while self.running:
            if self.command_queue:
                label = self.command_queue.pop(0)
                self.servo.move_arm(label)
            time.sleep(0.1)  # Avoid busy waiting

    def add_command(self, label):
        self.command_queue.append(label)

    def stop(self):
        self.servo.stop()
        self.running = False
        print("Successfully cleaned up servo thread")


if __name__ == "__main__":
    servo = MyServo(None, None)

    servo.move_left()
    time.sleep(2)
    servo.move_right()
    time.sleep(2)
    servo.move_left()
    time.sleep(2)
    servo.move_right()
    time.sleep(2)
    servo.stop()
