import math


COW_CLASSES = {'cow', 'vaca'}
WOLF_CLASSES = {'wolf', 'lobo', 'dog', 'zebra'}


def normalize_box(box):
    x1, y1, x2, y2 = map(float, box)
    return min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)


def objects_are_near(first, second):
    ax1, ay1, ax2, ay2 = normalize_box(first)
    bx1, by1, bx2, by2 = normalize_box(second)

    gap_x = max(0.0, ax1 - bx2, bx1 - ax2)
    gap_y = max(0.0, ay1 - by2, by1 - ay2)
    scale = max(1.0, ((ax2 - ax1) + (bx2 - bx1)) / 2)

    return math.hypot(gap_x, gap_y) <= 1.0 * scale


def danger_detected(detections):
    cows = [box for label, box in detections if label.lower() in COW_CLASSES]
    wolves = [box for label, box in detections if label.lower() in WOLF_CLASSES]

    return any(objects_are_near(cow, wolf) for cow in cows for wolf in wolves)


def associate_tags(tags, creepers):
    remaining = list(creepers)
    pairs = []

    for tag_id, tag_center in tags:
        if not remaining:
            break

        nearby = []

        for index, obj in enumerate(remaining):
            x, y, w, h = obj['bbox']
            margin_x = max(30, int(w * 0.35))
            margin_y = max(30, int(h * 0.35))

            if (x - margin_x <= tag_center[0] <= x + w + margin_x
                    and y - margin_y <= tag_center[1] <= y + h + margin_y):
                nearby.append(index)

        if not nearby:
            continue

        index = min(
            nearby,
            key=lambda i: abs(tag_center[0] - remaining[i]['center'][0])
            + abs(tag_center[1] - remaining[i]['center'][1])
        )

        obj = remaining.pop(index)
        pairs.append({
            'id': tag_id,
            'tag_center': tag_center,
            'color': obj['color'],
            'center': obj['center'],
            'bbox': obj['bbox']
        })

    return pairs


def segment_creepers(bgr):
    import cv2
    import numpy as np

    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)

    ranges = {
        'red': [
            ((0, 100, 65), (12, 255, 255)),
            ((170, 100, 65), (179, 255, 255))
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


def detect_tagged_creepers(bgr, tags):
    import cv2
    import numpy as np

    height, width = bgr.shape[:2]
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)

    half_width = max(24, int(width * 0.09))
    start_offset = max(18, int(height * 0.12))
    end_offset = max(70, int(height * 0.42))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

    ranges = {
        'red': [
            ((0, 135, 70), (10, 255, 255)),
            ((170, 135, 70), (179, 255, 255))
        ],
        'green': [
            ((35, 75, 50), (90, 255, 255))
        ],
        'blue': [
            ((95, 140, 55), (135, 255, 255))
        ],
    }

    result = []

    for tag_id, (cx, cy) in tags:
        x1 = max(0, cx - half_width)
        x2 = min(width, cx + half_width)
        y1 = min(height, cy + start_offset)
        y2 = min(height, cy + end_offset)

        if x2 - x1 < 20 or y2 - y1 < 35:
            continue

        roi = hsv[y1:y2, x1:x2]
        best = None

        for color, intervals in ranges.items():
            mask = np.zeros(roi.shape[:2], dtype=np.uint8)

            for lower, upper in intervals:
                mask |= cv2.inRange(
                    roi,
                    np.array(lower, dtype=np.uint8),
                    np.array(upper, dtype=np.uint8)
                )

            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

            contours, _ = cv2.findContours(
                mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )

            if not contours:
                continue

            largest = max(contours, key=cv2.contourArea)
            area = cv2.contourArea(largest)
            coverage = cv2.countNonZero(mask) / mask.size

            if coverage < 0.12 or area < mask.size * 0.04:
                continue

            if best is not None and coverage <= best[0]:
                continue

            moments = cv2.moments(largest)

            if moments['m00'] == 0:
                continue

            center = (
                x1 + int(moments['m10'] / moments['m00']),
                y1 + int(moments['m01'] / moments['m00'])
            )

            best = (coverage, color, center)

        if best is not None:
            result.append({
                'id': tag_id,
                'tag_center': (cx, cy),
                'color': best[1],
                'center': best[2]
            })

    return result