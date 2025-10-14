import argparse
import sys
import time
from typing import Optional

import cv2
import numpy as np


def open_serial(port: str, baud: int, dtr: Optional[bool], rts: Optional[bool]):
    import serial
    ser = serial.Serial(port=port, baudrate=baud, timeout=0.5)
    try:
        if dtr is not None:
            ser.setDTR(dtr)
        if rts is not None:
            ser.setRTS(rts)
        ser.reset_input_buffer()
        ser.reset_output_buffer()
    except Exception:
        pass
    return ser


def read_exact(ser, n: int, deadline: float) -> bytes | None:
    buf = bytearray()
    while len(buf) < n and time.time() < deadline:
        chunk = ser.read(n - len(buf))
        if chunk:
            buf.extend(chunk)
        else:
            time.sleep(0.001)
    return bytes(buf) if len(buf) == n else None


def request_one_frame(ser, baud: int) -> np.ndarray | None:
    # Send 'S' command to request one frame
    ser.write(b'S')
    ser.flush()

    # Read magic 'SJPG'
    deadline = time.time() + 1.0
    magic = b"SJPG"
    got = bytearray()
    while time.time() < deadline:
        b = ser.read(1)
        if not b:
            continue
        got += b
        if len(got) > 4:
            got.pop(0)
        if bytes(got) == magic:
            break
    else:
        return None

    # Read 4-byte little-endian length
    hdr = read_exact(ser, 4, time.time() + 0.5)
    if not hdr:
        return None
    length = int.from_bytes(hdr, 'little', signed=False)
    if length <= 0 or length > 2_000_000:
        return None

    # Compute deadline based on baud
    bytes_per_sec = max(baud // 10, 1)
    deadline = time.time() + min(3.0, max(0.5, length / bytes_per_sec + 0.3))
    payload = read_exact(ser, length, deadline)
    if not payload:
        return None

    img = cv2.imdecode(np.frombuffer(payload, dtype=np.uint8), cv2.IMREAD_COLOR)
    return img


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Request and display a single frame from ESP32-S3-CAM over serial")
    p.add_argument('--port', required=True)
    p.add_argument('--baud', type=int, default=460800)
    p.add_argument('--dtr', type=int, choices=[0,1], default=None)
    p.add_argument('--rts', type=int, choices=[0,1], default=None)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    dtr = None if args.dtr is None else bool(args.dtr)
    rts = None if args.rts is None else bool(args.rts)

    try:
        ser = open_serial(args.port, args.baud, dtr, rts)
    except Exception as e:
        print(f"Failed to open serial: {e}")
        sys.exit(1)

    try:
        frame = request_one_frame(ser, args.baud)
        if frame is None:
            print('Failed to get frame')
            sys.exit(2)
        cv2.imshow('Single Frame', frame)
        print('Press any key in the image window to exit...')
        cv2.waitKey(0)
    finally:
        try:
            ser.close()
        except Exception:
            pass
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()


