from picamera import PiCamera
from time import sleep
#things to define dates for file versioning
import time

#kontrol kamera
camera = PiCamera(resolution=(1280, 720))
camera.iso = 100
sleep(2) #allow camera to settle

camera.shutter_speed = camera.exposure_speed
camera.exposure_mode = 'auto'
g = camera.awb_gains
camera.awb_mode = 'tungsten'
camera.awb_gains = g


camera.capture_sequence([
"halseya-" + str(time.strftime('%H:%M:%S')) + ".jpeg",
"halseyb-" + str(time.strftime('%H:%M:%S')) + ".jpeg",
"halseyc-" + str(time.strftime('%H:%M:%S')) + ".jpeg",
])

camera.start_preview()
camera.vflip = True
camera.hflip = True


sleep(2)

