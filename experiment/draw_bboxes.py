import cv2


def draw_bbox(img, coordinates):
    for c in coordinates:
        t, l, b, r = c['bbox']
        label = c['label']
        font = cv2.FONT_HERSHEY_SIMPLEX

        # fontScale
        fontScale = 1

        # Blue color in BGR
        color = (0, 0, 255)

        # Line thickness of 2 px
        thickness = 2

        img = cv2.putText(img, label, (int(t), int(r)), font,
                            fontScale, color, thickness, cv2.LINE_AA)

        img = cv2.rectangle(img, (int(t), int(l)), (int(b), int(r)), (255, 23, 1), 2)

    cv2.namedWindow('img', cv2.WINDOW_NORMAL)
    cv2.imshow('img', img)
    cv2.waitKey(0)
