import cv2 as cv
import numpy as np

print(cv.__version__)

# 동영상 파일 경로 및 시작 위치
video_path = 'line_detect_night.mp4'
cap = cv.VideoCapture(video_path)
fourcc = cv.VideoWriter_fourcc(*'mp4v')
out = cv.VideoWriter('output.mp4', fourcc, 30.0, (768,432))  # 파일명, 코덱, FPS, 프레임 크기


fps = cap.get(cv.CAP_PROP_FPS)

# start_sec = (60*60*2)+(60*32)
start_sec = 40
start_frame = int(start_sec * fps)
cap.set(cv.CAP_PROP_POS_FRAMES, start_frame)

# 이전 라인 정보 저장
old_thetaQ = np.array([0.0, 0.0])
old_rhoQ = np.array([0.0, 0.0])
old_countQ = np.array([0.0, 0.0])

def drawLine(rho, theta, img, color, thick):
    x0 = rho * np.cos(theta)
    y0 = rho * np.sin(theta)
    x1 = int(x0 - 1000*np.sin(theta))
    y1 = int(y0 + 1000*np.cos(theta))
    x2 = int(x0 + 1000*np.sin(theta))
    y2 = int(y0 - 1000*np.cos(theta))
    cv.line(img, (x1, y1), (x2, y2), color, thick)
    return [x1, y1], [x2, y2]

def mode_select(img_hls, roi_rect):
    l_channel = img_hls[:,:,1]
    x, y, w, h = roi_rect
    roi_l = l_channel[y:y+h, x:x+w]
    avg_l = np.mean(roi_l)

    if avg_l < 100:
        return "night"
    elif avg_l >= 150:
        return "bright"
    else:
        return "sunny"

def dynamic_threshold(img, roi_rect):
    hls = cv.cvtColor(img, cv.COLOR_BGR2HLS)
    env = mode_select(hls, roi_rect)
    print(env)
    if env == "night":
        return (120, 255), (180, 255), 80
    elif env == "sunny":
        return (150, 255), (200, 255), 100
    else:  # bright
        return (180, 255), (220, 255), 120

def image_filtering(img_bgr, l_thresh, y_thresh, s_thresh):
    lab = cv.cvtColor(img_bgr, cv.COLOR_BGR2LAB)
    l, a, b = cv.split(lab)
    clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(16,16))
    l_clahe = clahe.apply(l)
    enhanced_lab = cv.merge((l_clahe, a, b))
    enhanced_lab = cv.cvtColor(enhanced_lab, cv.COLOR_LAB2BGR)

    enhanced_lab = cv.bitwise_and(enhanced_lab, enhanced_lab, mask=roi_mask)

    hsv = cv.cvtColor(enhanced_lab, cv.COLOR_BGR2HSV)
    mask_white = cv.inRange(hsv, (0, 0, 150), (179, 40, 255))
    mask_yellow = cv.inRange(hsv, (18, 120, 150), (30, 255, 255))
    mask_blue = cv.inRange(hsv, (80, 40, 80), (130, 255, 255))
    combined_mask = cv.bitwise_or(mask_white, mask_yellow)
    combined_mask = cv.bitwise_or(combined_mask, mask_blue)

    gray = cv.cvtColor(enhanced_lab, cv.COLOR_BGR2GRAY)
    masked_gray = cv.bitwise_and(gray, gray, mask=combined_mask)

    # LAB 색공간 (조명 변화에 강함)
    lab = cv.cvtColor(enhanced_lab, cv.COLOR_BGR2YUV)
    l_channel = lab[:,:,0]
    _, lab_binary = cv.threshold(l_channel, l_thresh[0], 255, cv.THRESH_BINARY)

    # YUV 색공간 (도로 환경에 효과적)
    yuv = cv.cvtColor(enhanced_lab, cv.COLOR_BGR2YUV)
    y_channel = yuv[:,:,0]
    _, yuv_binary = cv.threshold(y_channel, y_thresh[0], 255, cv.THRESH_BINARY)

    # 가중 결합
    combined = cv.bitwise_and(lab_binary, yuv_binary)
    combined = cv.bitwise_or(combined, masked_gray)
    
    # Sobel 최적화
    sobelx = cv.Sobel(combined, cv.CV_64F, 1, 0, ksize=3)
    abs_sobel = cv.convertScaleAbs(sobelx)
    _, sobel_binary = cv.threshold(abs_sobel, s_thresh, 255, cv.THRESH_BINARY)
    cv.imshow('combined', sobel_binary)

    return combined

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # 1. 프레임 축소 (40%)
    scale_percent = 40
    width = int(frame.shape[1] * scale_percent / 100)
    height = int(frame.shape[0] * scale_percent / 100)
    frame = cv.resize(frame, (width, height), interpolation=cv.INTER_AREA)

    # 2. ROI 설정 (하단 사다리꼴)
    h, w = frame.shape[:2]
    roi_pts = np.array([
        [int(w*0.44), int(h*0.6)],
        [int(w*0.0), int(h*0.85)],
        [int(w*0.0), int(h*0.95)],
        [int(w*1.0), int(h*0.95)],
        [int(w*1.0), int(h*0.85)],
        [int(w*0.56), int(h*0.6)],
        [int(w*0.52), int(h*0.6)],
        [int(w*0.78), int(h*0.95)],
        [int(w*0.22), int(h*0.95)],
        [int(w*0.48), int(h*0.6)],
    ], dtype=np.int32)

    syROI=roi_pts[0][1]
    eyROI=roi_pts[1][1]
    
    roi_mask = np.zeros((h, w), np.uint8)
    cv.fillPoly(roi_mask, [roi_pts], 255)
    x, y, w_roi, h_roi = cv.boundingRect(roi_pts)
    roi_rect = (x, y, w_roi, h_roi)
    cv.imshow('roi_mask',roi_mask)

    # 3. 환경별 임계값 계산
    l_thresh, y_thresh, sobel_thresh = dynamic_threshold(frame, roi_rect)

    # 4. 차선 후보 이진화
    lane_binary = image_filtering(frame, l_thresh, y_thresh, sobel_thresh)
    lane_binary = cv.bitwise_and(lane_binary, roi_mask)

    # 5. 엣지 검출 & 허프 라인
    edges = cv.Canny(lane_binary, 50, 200)
    lines = cv.HoughLines(edges, 1, np.pi/180, 30)
    imgHough = frame.copy()

    # 6. 직선 평균화 및 시각화
    thetaQ = np.array([0.0, 0.0])
    rhoQ = np.array([0.0, 0.0])
    countQ = np.array([0.0, 0.0])
    pt1 = np.array([[0, 0], [0, 0]])
    pt2 = np.array([[0, 0], [0, 0]])
    if lines is not None:
        for line in lines:
            rho, theta = line[0]
            if np.pi*45/180 < theta < np.pi*65/180:
                thetaQ[0] += theta
                rhoQ[0] += rho
                countQ[0] += 1
            elif np.pi*115/180 < theta < np.pi*135/180:
                thetaQ[1] += theta
                rhoQ[1] += rho
                countQ[1] += 1

    for i in range(2):
        if countQ[i] == 0:
            countQ[i]=old_countQ[i]
            thetaQ[i]=old_thetaQ[i]
            rhoQ[i]=old_rhoQ[i]
        if countQ[i] > 0:
            avg_theta = thetaQ[i] / countQ[i]
            avg_rho = rhoQ[i] / countQ[i]
            p1, p2 = drawLine(avg_rho, avg_theta, imgHough, (0,0,255), 2)
            pt1[i], pt2[i] = p1, p2
            # Detecting the lane
            retval, pt1[i], pt2[i] = cv.clipLine((roi_pts[1][0], syROI, roi_pts[3][0], eyROI-syROI), p1, p2)
            cv.rectangle(imgHough, [roi_pts[1][0], syROI], [roi_pts[3][0], eyROI], (0, 128, 0), 2)
            old_thetaQ[i]=thetaQ[i]
            old_rhoQ[i]=rhoQ[i]
            old_countQ[i]=countQ[i]

    # 7. 차선 영역 시각화
    imgOut = np.zeros_like(frame)
    cv.rectangle(imgHough, pt1[0], pt2[0], (0, 128, 256), 2)
    cv.rectangle(imgHough, pt1[1], pt2[1], (128, 0, 0), 2)
    lane_pts = np.array([pt2[0], pt1[0], pt2[1], pt1[1]], np.int32)
    cv.fillConvexPoly(imgOut, lane_pts, (255, 0, 0))
    overlapImage = cv.addWeighted(frame, 0.6, imgOut, 0.4, 0)
 
    # 8. Bird's Eye View 변환
    pts1 = np.float32([
        [int(w*0.0), int(h*0.95)],
        [int(w*0.3), int(h*0.6)],
        [int(w*0.7), int(h*0.6)],
        [int(w*1.0), int(h*0.95)],
    ])
    pts2 = np.float32([
        [0, 540], [0, 0], [960, 0], [960, 540]
    ])
    M = cv.getPerspectiveTransform(pts1, pts2)
    perspectiveImg = cv.warpPerspective(frame, M, (960, 540))

    # 9. 결과 출력
    cv.imshow('Overlap Lane', overlapImage)
    cv.imshow('perspectiveImg', perspectiveImg)
    out.write(overlapImage)
    if cv.waitKey(1) & 0xFF == ord('q'):
        break
out.release()
cap.release()
cv.destroyAllWindows()
