# OpenMV (Nicla Vision) MicroPython script
# Streams JPEG frames over USB serial: [4-byte little-endian length] + [JPEG bytes]

import sensor
import time
import pyb
import ustruct


def setup_camera() -> None:
	sensor.reset()
	sensor.set_pixformat(sensor.RGB565)
	sensor.set_framesize(sensor.QVGA)  # 320x240 for stable throughput
	sensor.set_auto_whitebal(True)
	sensor.set_auto_gain(False)
	sensor.skip_frames(time=1500)


def send_frame(usb: pyb.USB_VCP) -> None:
	img = sensor.snapshot()
	img = img.compress(quality=85)
	buf = img.bytes()
	size = len(buf)
	usb.send(ustruct.pack("<I", size))
	usb.send(buf)


def main() -> None:
	setup_camera()
	usb = pyb.USB_VCP()
	clock = time.clock()
	while True:
		clock.tick()
		if usb.isconnected():
			try:
				send_frame(usb)
			except Exception:
				pass


main()


