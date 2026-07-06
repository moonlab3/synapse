from std_msgs.msg import Float64MultiArray

class ActionFormatter:
    def __init__(self, muscle_option):
        self.muscle_option = muscle_option

    def format_action(self, raw_action):
        msg = Float64MultiArray()
        return msg