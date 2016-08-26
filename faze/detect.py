# Detecting faces, landmarks and transforms

import dlib, cv2
import numpy as np

JAW_IDX = list(np.arange(0, 17))
FACE_IDX = list(np.arange(17, 68))
MOUTH_IDX = list(np.arange(48, 61))

RIGHT_EYE_IDX = list(np.arange(36, 42))
LEFT_EYE_IDX = list(np.arange(42, 48))

NOSE_IDX = list(np.arange(27, 36))
LEFT_EYE_BROW_IDX = list(np.arange(22, 27))
RIGHT_EYE_BROW_IDX = list(np.arange(17, 22))

FACTOR_Rm = 0.5
FACTOR_Rn = 0.5

detector = dlib.get_frontal_face_detector()
predictor = None

def faces(img):
    # img: image
    return detector(img, 1)

def landmarks(img, roi, path):
    # img: image
    # roi: ROI of the face (x1, y1, x2, y2) [ Dlib rectangle ]
    # path: /path/to/dlib/shape_predictor.dat

    global predictor
    predictor = dlib.shape_predictor(path)
    shape = predictor(img, roi)
    return shape

def face_pose(img, roi, pts):
    # img: image
    # roi: ROI of the face (x1, y1, x2, y2) [ Dlib rectangle ]
    # pts: Dlib's 68 keypoints w.r.t original image coords ( Numpy array )

    # Computing required points
    mid_eye = (pts[36] + pts[39] + pts[42] + pts[45])/4.0
    face_c = np.array([roi.left() + roi.right(), roi.top() + roi.bottom()])/2.0
    nose_tip = pts[30]
    mouth_c = (pts[48] + pts[54])/2.0
    nose_base = pts[33]

    # print nose_base, nose_tip

    # Computing required distances
    mid_eye_mouth_d = np.linalg.norm(mid_eye - mouth_c)
    nose_base_tip_d = np.linalg.norm(nose_base - nose_tip)

    # print mid_eye_mouth_d, nose_base_tip_d

    def find_sigma(d1, d2, Rn, theta):
        dz = 0
        m1 = (d1*d1)/(d2*d2)
        m2 = np.cos(theta)**2
        Rn2 = Rn**2
        if m2 == 1:
            dz = np.sqrt(Rn2/( m1 + Rn2 ))
        else:
            dz = np.sqrt(((Rn2)-m1-2*m2*(Rn2) + np.sqrt(((m1-(Rn2))*(m1-(Rn2))) + 4*m1*m2*(Rn2)))/(2*(1-m2)*(Rn2)))
        # print dz
        return np.arccos(dz)

    # Computing required angles
    t = mid_eye - nose_base
    symm_x = np.pi - np.arctan2(t[1], t[0])
    t = nose_tip - nose_base
    tau = np.pi - np.arctan2(t[1], t[0])
    theta = np.abs(tau - symm_x)
    sigma = find_sigma(nose_base_tip_d, mid_eye_mouth_d, FACTOR_Rn, theta)

    # print np.degrees(symm_x), np.degrees(tau), theta, sigma

    # Computing face pose
    normal = np.zeros(3)
    # print sigma, tau, theta, symm_x
    sin_sigma = np.sin(sigma)
    normal[0] = sin_sigma*np.cos(tau)
    normal[1] = -sin_sigma*np.sin(tau)
    normal[2] = -np.cos(sigma)

    # print normal

    n02 = normal[0]**2
    n12 = normal[1]**2
    n22 = normal[2]**2

    pitch = np.arccos(np.sqrt((n02 + n22)/(n02 + n12 + n22)))
    if nose_tip[1] - nose_base[1] < 0:
        pitch = -pitch

    yaw = np.arccos(np.abs(normal[2]/np.linalg.norm(normal)))
    if nose_tip[0] - nose_base[0] < 0:
        yaw = -yaw

    # print yaw, pitch
    return normal, yaw, pitch
