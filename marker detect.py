
import cv2
import cv2.aruco as aruco
import numpy as np

# =====================================
# ARUCO SETUP
# =====================================

ARUCO_DICT = aruco.getPredefinedDictionary(
    aruco.DICT_4X4_50
)

ARUCO_PARAMS = aruco.DetectorParameters()

ARUCO_PARAMS.adaptiveThreshWinSizeMin = 3
ARUCO_PARAMS.adaptiveThreshWinSizeMax = 53
ARUCO_PARAMS.adaptiveThreshWinSizeStep = 10

ARUCO_DETECTOR = aruco.ArucoDetector(
    ARUCO_DICT,
    ARUCO_PARAMS
)

MARKER_SIZE_MM = 50.0

# =====================================
# CAMERA
# =====================================

cap = cv2.VideoCapture(0)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

if not cap.isOpened():
    print("Camera not found")
    exit()

print("Press Q to quit")

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

    corners, ids, rejected = ARUCO_DETECTOR.detectMarkers(
        gray
    )

    detected = False

    # ---------------------------------
    # Draw rejected candidates
    # ---------------------------------

    for r in rejected:

        pts = r.astype(np.int32)

        cv2.polylines(
            frame,
            [pts],
            True,
            (0, 0, 255),
            1
        )

    # ---------------------------------
    # Marker detected
    # ---------------------------------

    if ids is not None and len(ids) > 0:

        detected = True

        aruco.drawDetectedMarkers(
            frame,
            corners,
            ids
        )

        corner = corners[0][0]

        cx = np.mean(corner[:, 0])
        cy = np.mean(corner[:, 1])

        side_px = np.linalg.norm(
            corner[0] - corner[1]
        )

        px_per_mm = side_px / MARKER_SIZE_MM

        cv2.circle(
            frame,
            (int(cx), int(cy)),
            6,
            (0, 255, 0),
            -1
        )

        cv2.putText(
            frame,
            f"ID: {ids[0][0]}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2
        )

        cv2.putText(
            frame,
            f"Size: {side_px:.1f}px",
            (20, 80),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 0),
            2
        )

        cv2.putText(
            frame,
            f"Scale: {px_per_mm:.2f} px/mm",
            (20, 120),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 0),
            2
        )

        cv2.putText(
            frame,
            f"Center: ({int(cx)}, {int(cy)})",
            (20, 160),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 0),
            2
        )

    # ---------------------------------
    # Status Indicator
    # ---------------------------------

    color = (0, 255, 0) if detected else (0, 0, 255)

    cv2.circle(
        frame,
        (frame.shape[1] - 25, 25),
        10,
        color,
        -1
    )

    status = "DETECTED" if detected else "NOT DETECTED"

    cv2.putText(
        frame,
        status,
        (frame.shape[1] - 180, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        color,
        2
    )

    cv2.putText(
        frame,
        f"Rejected: {len(rejected)}",
        (20, frame.shape[0] - 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 0, 255),
        2
    )

    cv2.imshow(
        "ArUco Detection Test",
        frame
    )

    key = cv2.waitKey(1) & 0xFF

    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()