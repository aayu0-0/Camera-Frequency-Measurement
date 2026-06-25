import cv2
import cv2.aruco as aruco
import numpy as np
from collections import deque
import time

# =====================================
# PENDULUM TRACKER + FFT ANALYZER
# =====================================
#
# Features:
# - ArUco detection
# - Horizontal (X) displacement tracking
# - Reference locking
# - Scale locking
# - Lost frame recovery
# - EMA smoothing
# - Live amplitude estimation
# - Live FFT frequency estimation
# - R = Reset reference
# - Q = Quit
#
# Designed for:
# Homemade pendulum experiments
#
# =====================================

MARKER_SIZE_MM = 50.0
FFT_WINDOW = 512

EMA_ALPHA = 0.20
MAX_LOST_FRAMES = 5
MAX_STEP_MM = 20.0

# =====================================
# ARUCO
# =====================================

aruco_dict = aruco.getPredefinedDictionary(
    aruco.DICT_4X4_50
)

aruco_params = aruco.DetectorParameters()

detector = aruco.ArucoDetector(
    aruco_dict,
    aruco_params
)

# =====================================
# CAMERA
# =====================================

cap = cv2.VideoCapture(0)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

if not cap.isOpened():
    print("Camera not found")
    exit()

fps = cap.get(cv2.CAP_PROP_FPS)

if fps <= 0 or fps > 200:
    fps = 30.0

print(f"Using FPS = {fps}")

# =====================================
# VIDEO RECORDING
# =====================================

fourcc = cv2.VideoWriter_fourcc(*'mp4v')

video_writer = cv2.VideoWriter(
    "pendulum_analysis.mp4",
    fourcc,
    fps,
    (1280, 720)
)

recording = True

# =====================================
# STATE
# =====================================

reference_x = None
scale_locked = None

last_cx = None
lost_frames = 0

filtered_disp_mm = None
prev_disp_mm = 0.0

current_freq = 0.0
current_amp_mm = 0.0

signal_buffer = deque(maxlen=FFT_WINDOW)

# =====================================
# LOOP
# =====================================

while True:

    ret, frame = cap.read()

    if not ret:
        break

    gray = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2GRAY
    )

    corners, ids, rejected = detector.detectMarkers(
        gray
    )

    detected = False

    # =====================================
    # MARKER DETECTED
    # =====================================

    if ids is not None and len(ids) > 0:

        detected = True
        lost_frames = 0

        aruco.drawDetectedMarkers(
            frame,
            corners,
            ids
        )

        corner = corners[0][0]

        cx = float(np.mean(corner[:, 0]))
        cy = float(np.mean(corner[:, 1]))

        last_cx = cx

        side_px = np.linalg.norm(
            corner[0] - corner[1]
        )

        if side_px > 20:

            # -------------------------
            # SCALE LOCK
            # -------------------------

            if scale_locked is None:

                scale_locked = (
                    side_px /
                    MARKER_SIZE_MM
                )

                print(
                    f"Scale locked at "
                    f"{scale_locked:.2f} px/mm"
                )

            px_per_mm = scale_locked

            # -------------------------
            # REFERENCE LOCK
            # -------------------------

            if reference_x is None:

                reference_x = cx

                print(
                    "Reference locked"
                )

            # -------------------------
            # DISPLACEMENT
            # -------------------------

            displacement_px = (
                cx - reference_x
            )

            displacement_mm = (
                displacement_px /
                px_per_mm
            )

            # -------------------------
            # JUMP REJECTION
            # -------------------------

            if abs(
                displacement_mm -
                prev_disp_mm
            ) > MAX_STEP_MM:

                displacement_mm = (
                    prev_disp_mm
                )

            prev_disp_mm = (
                displacement_mm
            )

            # -------------------------
            # EMA SMOOTHING
            # -------------------------

            if filtered_disp_mm is None:

                filtered_disp_mm = (
                    displacement_mm
                )

            else:

                filtered_disp_mm = (
                    EMA_ALPHA *
                    displacement_mm
                    +
                    (1.0 - EMA_ALPHA)
                    *
                    filtered_disp_mm
                )

            displacement_mm = (
                filtered_disp_mm
            )

            signal_buffer.append(
                displacement_mm
            )

            # -------------------------
            # AMPLITUDE
            # -------------------------

            if len(signal_buffer) > 20:

                arr = np.array(
                    signal_buffer,
                    dtype=np.float32
                )

                current_amp_mm = (
                    np.max(arr)
                    -
                    np.min(arr)
                ) / 2.0

            # -------------------------
            # FFT
            # -------------------------

            if len(signal_buffer) >= 128:

                signal = np.array(
                    signal_buffer,
                    dtype=np.float32
                )

                signal = (
                    signal -
                    np.mean(signal)
                )

                signal *= np.hamming(
                    len(signal)
                )

                fft = np.fft.rfft(
                    signal
                )

                magnitude = np.abs(
                    fft
                )

                freqs = (
                    np.fft.rfftfreq(
                        len(signal),
                        d=1.0 / fps
                    )
                )

                valid = (
                    freqs > 0.1
                )

                if np.any(valid):

                    idx = np.argmax(
                        magnitude[
                            valid
                        ]
                    )

                    current_freq = (
                        freqs[
                            valid
                        ][idx]
                    )

            # -------------------------
            # DRAW CENTER
            # -------------------------

            cv2.circle(
                frame,
                (
                    int(cx),
                    int(cy)
                ),
                6,
                (0,255,0),
                -1
            )

    # =====================================
    # LOST FRAME RECOVERY
    # =====================================

    else:

        lost_frames += 1

        if (
            last_cx is not None
            and
            reference_x is not None
            and
            scale_locked is not None
            and
            lost_frames <= MAX_LOST_FRAMES
        ):

            displacement_px = (
                last_cx -
                reference_x
            )

            displacement_mm = (
                displacement_px /
                scale_locked
            )

            signal_buffer.append(
                displacement_mm
            )

    # =====================================
    # HUD
    # =====================================

    color = (
        (0,255,0)
        if detected
        else (0,0,255)
    )

    cv2.circle(
        frame,
        (
            frame.shape[1]-30,
            30
        ),
        10,
        color,
        -1
    )

    cv2.putText(
        frame,
        f"Disp: {filtered_disp_mm if filtered_disp_mm is not None else 0:.2f} mm",
        (20,40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0,255,255),
        2
    )

    cv2.putText(
        frame,
        f"Amp: {current_amp_mm:.2f} mm",
        (20,80),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0,255,255),
        2
    )

    cv2.putText(
        frame,
        f"Freq: {current_freq:.3f} Hz",
        (20,120),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0,255,255),
        2
    )

    cv2.putText(
        frame,
        f"Lost Frames: {lost_frames}",
        (20,160),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255,255,255),
        2
    )

    if scale_locked is not None:

        cv2.putText(
            frame,
            f"Scale: {scale_locked:.2f} px/mm",
            (20,200),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255,255,0),
            2
        )

    cv2.putText(
        frame,
        "R = Reset Reference",
        (20, frame.shape[0]-50),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (200,200,200),
        1
    )

    cv2.putText(
        frame,
        "Q = Quit",
        (20, frame.shape[0]-20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (200,200,200),
        1
    )


    if recording:

        cv2.putText(
            frame,
            "REC",
            (frame.shape[1] - 90, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 255),
            2
        )

    if recording:
        video_writer.write(frame)
    cv2.imshow(
        "Pendulum Frequency Analyzer",
        frame
    )

    key = cv2.waitKey(1) & 0xFF

    if key == ord('q'):
        break

    elif key == ord('r'):

        reference_x = None

        print(
            "Reference reset"
        )

# =====================================
# CLEANUP
# =====================================

video_writer.release()

cap.release()

cv2.destroyAllWindows()

print("Saved video: pendulum_analysis.mp4")
