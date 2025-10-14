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


def read_exact(ser, n: int, deadline: float) -> Optional[bytes]:
    buf = bytearray()
    while len(buf) < n and time.time() < deadline:
        chunk = ser.read(n - len(buf))
        if chunk:
            buf.extend(chunk)
        else:
            time.sleep(0.001)
    return bytes(buf) if len(buf) == n else None


def read_frame_sjpg(ser, baud: int, max_len: int = 2_000_000) -> Optional[np.ndarray]:
    # Resync on magic 'SJPG'
    magic = b"SJPG"
    window = bytearray()
    sync_deadline = time.time() + 1.0
    while time.time() < sync_deadline:
        b = ser.read(1)
        if not b:
            continue
        window += b
        if len(window) > 4:
            window.pop(0)
        if bytes(window) == magic:
            break
    else:
        return None

    # Read 4-byte little-endian length
    hdr = read_exact(ser, 4, time.time() + 0.3)
    if not hdr:
        return None
    length = int.from_bytes(hdr, "little", signed=False)
    if length <= 0 or length > max_len:
        return None

    # Read JPEG payload with dynamic deadline (baud ~= bits/s, 10 bits per byte)
    bytes_per_sec = max(baud // 10, 1)
    read_time = length / bytes_per_sec + 0.25
    payload = read_exact(ser, length, time.time() + min(2.5, max(0.5, read_time)))
    if not payload:
        return None

    img = cv2.imdecode(np.frombuffer(payload, dtype=np.uint8), cv2.IMREAD_COLOR)
    return img


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Minimal SJPG serial viewer")
    p.add_argument("--port", required=True, help="Serial port, e.g., COM10 or /dev/ttyACM0")
    p.add_argument("--baud", type=int, default=921600)
    p.add_argument("--dtr", type=int, choices=[0, 1], default=None, help="DTR line (1=True, 0=False)")
    p.add_argument("--rts", type=int, choices=[0, 1], default=None, help="RTS line (1=True, 0=False)")
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

    last_log = 0.0
    try:
        while True:
            try:
                frame = read_frame_sjpg(ser, args.baud)
                if frame is None:
                    now = time.time()
                    if now - last_log > 1.0:
                        print(f"Waiting for frame on {args.port} @ {args.baud}...")
                        last_log = now
                    continue
                cv2.imshow("Serial Viewer", frame)
                if cv2.waitKey(1) & 0xFF == 27:
                    break
            except Exception as e:
                print(f"Serial error: {e}. Reconnecting...")
                try:
                    ser.close()
                except Exception:
                    pass
                time.sleep(0.5)
                ser = open_serial(args.port, args.baud, dtr, rts)
    finally:
        try:
            ser.close()
        except Exception:
            pass
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()


