# parsing command line arguments
import argparse
# decoding camera images
import base64
# matrix math
import numpy as np
# real-time server
import socketio
# concurrent networking
import eventlet
# web server gateway interface
import eventlet.wsgi
# image manipulation
from PIL import Image
# web framework
from flask import Flask
# input output
from io import BytesIO

import torch
from torchvision import transforms

# helper class
import utils

# initialize our server
sio = socketio.Server()
# our flask (web) app
app = Flask(__name__)
# init our model and image array as empty
model = None


class SimplePIController:
    def __init__(self, Kp, Ki):
        self.Kp = Kp
        self.Ki = Ki
        self.set_point = 0.
        self.error = 0.
        self.integral = 0.

    def set_desired(self, desired):
        self.set_point = desired

    def update(self, measurement):
        # proportional error
        self.error = self.set_point - measurement

        # integral error
        self.integral += self.error

        return self.Kp * self.error + self.Ki * self.integral


controller = SimplePIController(0.1, 0.002)
set_speed = 10
controller.set_desired(set_speed)


class CNN_end_to_end_driving(torch.nn.Module):
    def __init__(self):
        """
        Create a model based on Nvidia paper : https://arxiv.org/pdf/1604.07316.pdf
        """
        super().__init__() 
        
        self.conv1 = torch.nn.Conv2d(in_channels=3, out_channels=24, kernel_size=5, stride=2, padding=0, bias=True)
        self.conv2 = torch.nn.Conv2d(in_channels=24, out_channels=36, kernel_size=5, stride=2, padding=0, bias=True)
        self.conv3 = torch.nn.Conv2d(in_channels=36, out_channels=48, kernel_size=5, stride=2, padding=0, bias=True)
        self.conv4 = torch.nn.Conv2d(in_channels=48, out_channels=64, kernel_size=3, stride=1, padding=0, bias=True)
        self.conv5 = torch.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=0, bias=True)

        self.linear1 = torch.nn.Linear(in_features=1152, out_features=1164, bias=True) # cross-check the in-features
        # self.linear1 = torch.nn.Linear(in_features=1152, out_features=100, bias=True) 
        # self.linear1 = torch.nn.Linear(in_features=3840, out_features=1164, bias=True) # cross-check the in-features
        self.linear2 = torch.nn.Linear(in_features=1164, out_features=100, bias=True) 
        self.linear3 = torch.nn.Linear(in_features=100, out_features=50, bias=True) 
        self.linear4 = torch.nn.Linear(in_features=50, out_features=10, bias=True) 
        self.linear5 = torch.nn.Linear(in_features=10, out_features=1, bias=True) 

        self.ReLU = torch.nn.ReLU()
        
        # self.ReLU = torch.nn.ReLU()

    def forward(self, x):
        
        x = self.conv1(x)
        x = self.ReLU(x)
        x = self.conv2(x)
        x = self.ReLU(x)
        x = self.conv3(x)
        x = self.ReLU(x)
        x = self.conv4(x)
        x = self.ReLU(x)
        x = self.conv5(x)
        x = self.ReLU(x)

        x = x.reshape(x.size(0), -1) # flatten the image
        # print(f'x.size after flatteneing : {x.size()}')

        x = self.linear1(x)
        x = self.ReLU(x)
        x = self.linear2(x)
        x = self.ReLU(x)
        x = self.linear3(x)
        x = self.ReLU(x)
        x = self.linear4(x)
        x = self.ReLU(x)
        x = self.linear5(x)

        return x


to_tensor_trans = transforms.ToTensor()



# registering event handler for the server
@sio.on('telemetry')
def telemetry(sid, data):

    global speed_limit

    if data:
        print("getting data")
        # The current steering angle of the car
        steering_angle = float(data["steering_angle"])
        # The current throttle of the car, how hard to push peddle
        throttle = float(data["throttle"])
        # The current speed of the car
        speed = float(data["speed"])
        # The current image from the center camera of the car
        image = Image.open(BytesIO(base64.b64decode(data["image"])))
        try:
            image = np.asarray(image)  # from PIL image to numpy array
            image = utils.preprocess(image)  # apply the preprocessing
            image = image/255.0 - 0.5
            image = np.float32(image)

            img_tensor = to_tensor_trans(image)

            img_tensor = img_tensor.resize(1, 3, 66, 200)

            model.eval()
            preds = model(img_tensor)
            steering_angle = float(preds.item())
            throttle = controller.update(float(speed))

            print(f'--------\nstr angle = {steering_angle} \nthrottle={throttle}\nspeed={speed}---------- ')
            # send command for steering angle and throttle
            send_control(steering_angle, throttle)

        except Exception as e:
            print(e)
    else:

        sio.emit('manual', data={}, skip_sid=True)


@sio.on('connect')
def connect(sid, environ):
    print("connect ", sid)
    send_control(0, 0)


def send_control(steering_angle, throttle):
    sio.emit(
        "steer",
        data={
            'steering_angle': steering_angle.__str__(),
            'throttle': throttle.__str__()
        },
        skip_sid=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Remote Driving')
    parser.add_argument(
        'model',
        type=str,
        help='Path to model h5 file. Model should be on the same path.'
    )
    args = parser.parse_args()

    # load model
    model = CNN_end_to_end_driving()
    model.load_state_dict(torch.load(args.model, map_location=torch.device('cpu')))

    # wrap Flask application with engineio's middleware
    app = socketio.Middleware(sio, app)

    # deploy as an eventlet WSGI server
    eventlet.wsgi.server(eventlet.listen(('', 4567)), app)
