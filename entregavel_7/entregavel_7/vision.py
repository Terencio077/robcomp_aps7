def segment_creepers(bgr):
    import cv2
    import numpy as np

    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)

    ranges = {
        'red': [
            ((0, 100, 70), (12, 255, 255)),
            ((160, 100, 70), (179, 255, 255))
        ],
        'green': [
            ((35, 70, 50), (90, 255, 255))
        ],
        'blue': [
            ((95, 75, 45), (135, 255, 255))
        ],
    }

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    min_area = bgr.shape[0] * bgr.shape[1] * 0.003
    creepers = []

    for color, intervals in ranges.items():
        mask = np.zeros(hsv.shape[:2], dtype=np.uint8)

        for lower, upper in intervals:
            mask |= cv2.inRange(
                hsv,
                np.array(lower, np.uint8),
                np.array(upper, np.uint8)
            )

        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        for contour in contours:
            if cv2.contourArea(contour) < min_area:
                continue

            moments = cv2.moments(contour)

            if moments['m00'] == 0:
                continue

            center = (
                int(moments['m10'] / moments['m00']),
                int(moments['m01'] / moments['m00'])
            )

            bbox = cv2.boundingRect(contour)

            creepers.append({
                'color': color,
                'center': center,
                'bbox': bbox
            })

    return creepers
