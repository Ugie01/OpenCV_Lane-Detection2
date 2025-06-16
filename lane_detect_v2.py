import cv2 as cv
import numpy as np
import time

# OpenCV 버전 확인
print(cv.__version__)

# 동영상 파일 및 출력 설정
video_path = 'line_detect_night.mp4'
cap = cv.VideoCapture(video_path)
fourcc = cv.VideoWriter_fourcc(*'mp4v')  # MP4 코덱
out = cv.VideoWriter('output.mp4', fourcc, 30.0, (640,360))  # 출력 파일 설정

# 시작 프레임 계산
fps = cap.get(cv.CAP_PROP_FPS)
start_sec = 0
start_frame = int(start_sec * fps)
cap.set(cv.CAP_PROP_POS_FRAMES, start_frame)

scale_percent = 100
# 이전 프레임 차선 정보 저장 변수
old_thetaQ = np.array([0.0, 0.0])
old_rhoQ = np.array([0.0, 0.0])
old_countQ = np.array([0.0, 0.0])

prevTime = 0 #이전 시간을 저장할 변수
fps_filtered = 1
frame_count = 0
start = time.time()

def drawLine(rho, theta, img, color, thick):
    """허프 변환 결과를 이미지에 선으로 그리는 함수"""
    x0 = rho * np.cos(theta)
    y0 = rho * np.sin(theta)
    x1 = int(x0 - 1000*np.sin(theta))
    y1 = int(y0 + 1000*np.cos(theta))
    x2 = int(x0 + 1000*np.sin(theta))
    y2 = int(y0 - 1000*np.cos(theta))
    cv.line(img, (x1, y1), (x2, y2), color, thick)
    return [x1, y1], [x2, y2]

def mode_select(img_hls):
    """조명 환경 분류 함수"""
    l_channel = img_hls[:,:,1]
    x, y, w, h = roi_rect
    roi_l = l_channel[y:y+h, x:x+w]
    avg_l = np.mean(roi_l)
    if avg_l < 100: return "night"
    elif avg_l >= 150: return "bright"
    else: return "sunny"

def dynamic_threshold(img):
    """동적 임계값 설정 함수"""
    hls = cv.cvtColor(img, cv.COLOR_BGR2HLS)
    env = mode_select(hls)
    print(env)
    if env == "night":
        return (160, 255), (180, 255), 80
    elif env == "sunny":
        return (180, 255), (200, 255), 100
    else:
        return (200, 255), (220, 255), 120

# 종합 이미지 처리 함수
def image_filtering(img_bgr, l_thresh, y_thresh, s_thresh):

    # ROI 마스킹
    cv.imshow('CLAHE befor', img_bgr)

    # CLAHE 기반 대비 향상
    lab = cv.cvtColor(img_bgr, cv.COLOR_BGR2LAB)
    l, a, b = cv.split(lab)
    clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(16,16))
    l_clahe = clahe.apply(l)
    enhanced_lab = cv.merge((l_clahe, a, b))
    enhanced_lab = cv.cvtColor(enhanced_lab, cv.COLOR_LAB2BGR)
    cv.imshow("CLAHE after", enhanced_lab)

    # HSV 색상 필터링
    hsv = cv.cvtColor(enhanced_lab, cv.COLOR_BGR2HSV)
    mask_white = cv.inRange(hsv, (0,0,150), (179,40,255))
    mask_yellow = cv.inRange(hsv, (18,120,150), (30,255,255))
    mask_blue = cv.inRange(hsv, (80,40,80), (130,255,255))
    combined_mask = cv.bitwise_or(mask_white, mask_yellow)
    combined_mask = cv.bitwise_or(combined_mask, mask_blue)
    cv.imshow("combined_mask", combined_mask)

    # LAB/YUV 임계값 처리
    lab_binary = cv.inRange(cv.cvtColor(enhanced_lab, cv.COLOR_BGR2LAB)[:,:,0], l_thresh[0], 255)
    yuv_binary = cv.inRange(cv.cvtColor(enhanced_lab, cv.COLOR_BGR2YUV)[:,:,0], y_thresh[0], 255)
    cv.imshow("lab_binary", lab_binary)
    cv.imshow("yuv_binary", yuv_binary)

    # 결과 결합
    combined = cv.bitwise_and(lab_binary, yuv_binary)
    masked = cv.bitwise_and(combined, combined, mask=combined_mask)

    cv.imshow("masked", masked)
    # Sobel 엣지 검출
    sobelx = cv.Sobel(masked, cv.CV_64F, 1, 0, ksize=3)
    abs_sobel = cv.convertScaleAbs(sobelx)
    _, sobel_binary = cv.threshold(abs_sobel, s_thresh, 255, cv.THRESH_BINARY)
    cv.imshow("sobel_binary", sobel_binary)

    return sobel_binary

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    frame_count += 1
    # 프레임 전처리
    w = int(frame.shape[1] * scale_percent / 100)
    h = int(frame.shape[0] * scale_percent / 100)
    frame = cv.resize(frame, (w, h), interpolation=cv.INTER_AREA)

    # ROI 영역 정의
    # h, w = frame.shape[:2]
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

    # 차로 필터링
    l_thresh, y_thresh, sobel_thresh = dynamic_threshold(frame)
    # roi_frame = cv.bitwise_and(frame, frame, mask=roi_mask)
    lane_binary = image_filtering(frame, l_thresh, y_thresh, sobel_thresh)
    lane_binary = cv.bitwise_and(lane_binary, roi_mask)
    cv.imshow("ROI_lane_binary", lane_binary)

    # 허프 변환 기반 차선 검출
    edges = cv.Canny(lane_binary, 50, 200)

    lines = cv.HoughLines(edges, 1, np.pi/180, 30)
    imgHough = frame.copy()

    # 차선 정보 평균화 및 시각화
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

    # 차선 영역 시각화
    imgOut = np.zeros_like(frame)
    cv.rectangle(imgHough, pt1[0], pt2[0], (0, 128, 256), 2)
    cv.rectangle(imgHough, pt1[1], pt2[1], (128, 0, 0), 2)
    lane_pts = np.array([pt2[0], pt1[0], pt2[1], pt1[1]], np.int32)
    cv.fillConvexPoly(imgOut, lane_pts, (255, 0, 0))
    overlapImage = cv.addWeighted(frame, 0.6, imgOut, 0.4, 0)

    # Bird's Eye View 변환
    pts1 = np.float32([
        [int(w*0.0), int(h*0.95)],
        [int(w*0.3), int(h*0.6)],
        [int(w*0.7), int(h*0.6)],
        [int(w*1.0), int(h*0.95)],
    ])
    pts2 = np.float32([
        [0, h], [0, 0], [w, 0], [w, h]
    ])
    M = cv.getPerspectiveTransform(pts1, pts2)
    perspectiveImg = cv.warpPerspective(frame, M, (w, h))

    # FPS 계산
    curTime = time.time()
    sec = curTime - prevTime
    prevTime = curTime
    fps = 1/(sec)
    alpha = 0.9  # 이전 값에 대한 가중치
    fps_filtered = alpha * fps_filtered + (1 - alpha) * fps  # fps_filtered는 0으로 초기화

    # FPS 값을 영상에 표시
    fps_text = "FPS: {:.2f}".format(fps_filtered)
    cv.putText(overlapImage, fps_text, (10, 30), cv.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

    # 결과 출력
    # cv.imshow('frame', frame)
    cv.imshow('Overlap Lane', overlapImage)
    # cv.imshow('Birds Eye View', perspectiveImg)
    out.write(overlapImage)
    if cv.waitKey(1) & 0xFF == ord('q'):
        break

end = time.time()
avg_fps = frame_count / (end - start)
print(f"Average FPS: {avg_fps:.2f}")
out.release()
cap.release()
cv.destroyAllWindows()
