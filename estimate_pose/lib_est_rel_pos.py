import cv2
import numpy as np
import torch 
import time


def estimate_relative_pose(img1, img2):
    h, w, c = img1.shape
    dim = (w,h)
    base_order = 1  # Base sphere resolution
    sample_order = 8  # Determines sample resolution (10 = 2048 x 4096)
    scale_factor = 1.0  # How much to scale input equirectangular image by
    points = 12000
    opt, mode, sphered = get_descriptor('sift')

    pts1, desc1 = process_image_to_keypoints(img1, None, scale_factor, base_order, sample_order, opt, mode)
    pts2, desc2 = process_image_to_keypoints(img2, None, scale_factor, base_order, sample_order, opt, mode)

    pts1, pts2, desc1, desc2 = sort_key(pts1, pts2, desc1, desc2, points)

    if len(pts1.shape) == 1:
        pts1 = pts1.reshape(1,-1)

    if len(pts2.shape) == 1:
        pts2 = pts2.reshape(1,-1)

    s_pts1, s_pts2, x1, x2 = matched_points(pts1, pts2, desc1, desc2, '100p', opt, 'ratio')
    x1,x2 = coord_3d(x1, dim), coord_3d(x2, dim)

    cam = get_cam_pose_by_ransac_8pa(x1.copy().T,x2.copy().T)
    return cam


def estimate_relative_pose_from_matches(x1, x2, dim):
    x1,x2 = coord_3d(x1, dim), coord_3d(x2, dim)

    cam = get_cam_pose_by_ransac_8pa(x1.copy().T,x2.copy().T)
    return cam


def get_descriptor(descriptor):
    if descriptor == 'sift':
        return 'sift', 'erp', 512


def process_image_to_keypoints(img, corners, scale_factor, base_order, sample_order, opt, mode):
    if mode == 'erp':
        tangent_image_kp, tangent_image_desc = keypoint_equirectangular(img, opt)
    return tangent_image_kp, np.transpose(tangent_image_desc,[1,0])


def keypoint_equirectangular(img, opt ='superpoint', crop_degree=0):
    if opt == 'sift':
        erp_kp_details = computes_sift_keypoints(img)

    erp_kp = erp_kp_details[0]
    erp_desc = erp_kp_details[1]

    crop_h = compute_crop(img.shape[-2:], crop_degree)

    mask = (erp_kp[:, 1] > crop_h) & (erp_kp[:, 1] < img.shape[1] - crop_h)
    erp_kp = erp_kp[mask]
    erp_desc = erp_desc[mask]

    return erp_kp, erp_desc


def compute_crop(image_shape, crop_degree=0):
    """Compute padding space in an equirectangular images"""
    crop_h = 0
    if crop_degree > 0:
        crop_h = image_shape[0] // (180 / crop_degree)

    return crop_h


def computes_sift_keypoints(img):
    sift = cv2.SIFT_create(nfeatures=10000)
    keypoints, desc = sift.detectAndCompute(img, None)
    if len(keypoints) > 0:
        return format_keypoints(keypoints, desc)
    return None


def format_keypoints(keypoints, desc):
    coords = torch.tensor([kp.pt for kp in keypoints])
    responsex = torch.tensor([kp.response for kp in keypoints])
    responsey = torch.tensor([kp.response for kp in keypoints])
    desc = torch.from_numpy(desc)
    return torch.cat((coords, responsex.unsqueeze(1), responsey.unsqueeze(1)), -1), desc


def sort_key(pts1, pts2, desc1, desc2, points):
    ind1 = np.argsort(pts1[:,2].numpy(),axis = 0)[::-1]
    ind2 = np.argsort(pts2[:,2].numpy(),axis = 0)[::-1]

    max1 = np.min([points,ind1.shape[0]])
    max2 = np.min([points,ind2.shape[0]])

    ind1 = ind1[:max1]
    ind2 = ind2[:max2]

    pts1 = pts1[ind1.copy(),:]
    pts2 = pts2[ind2.copy(),:]

    desc1 = desc1[:,ind1.copy()]
    desc2 = desc2[:,ind2.copy()]

    pts1 = np.concatenate((pts1[:,:2], np.ones((pts1.shape[0],1))), axis = 1 )
    pts2 = np.concatenate((pts2[:,:2], np.ones((pts2.shape[0],1))), axis = 1 )

    desc1 = np.transpose(desc1,[1,0]).numpy()
    desc2 = np.transpose(desc2,[1,0]).numpy()

    return pts1, pts2, desc1, desc2


def matched_points(pts1, pts2, desc1, desc2, opt, args_opt, match='ratio'):

    if opt[-1] == 'p':
        porce = int(opt[:-1])
        n_key = int(porce/100 * pts1.shape[0])
    else:
        n_key = int(opt)

    s_pts1  = pts1.copy()[:n_key,:]
    s_pts2  = pts2.copy()[:n_key,:]
    s_desc1 = desc1.copy().astype('float32')[:n_key,:]
    s_desc2 = desc2.copy().astype('float32')[:n_key,:]

    if  'orb' in args_opt:
        s_desc1 = s_desc1.astype(np.uint8)
        s_desc2 = s_desc2.astype(np.uint8)

    if match == '2-cross':
        if 'orb' in args_opt:
            bf = cv2.BFMatcher(cv2.NORM_HAMMING, True)
        else:
            bf = cv2.BFMatcher(cv2.NORM_L2, True)
        matches = bf.match(s_desc1, s_desc2)
    elif match == 'ratio':
        if 'orb' in args_opt:
            bf = cv2.BFMatcher(cv2.NORM_HAMMING, False)
        else:
            bf = cv2.BFMatcher(cv2.NORM_L2, False)
        matches = bf.knnMatch(s_desc1,s_desc2, k=2)
        good = []
        for m,n in matches:
            if m.distance < 0.75*n.distance:
                good.append(m)
        matches = good

    M = np.zeros((2,len(matches)))
    for ind, match in zip(np.arange(len(matches)),matches):
        M[0,ind] = match.queryIdx
        M[1,ind] = match.trainIdx

    num_M = M.shape[1]

    return s_pts1, s_pts2, s_pts1[M[0,:].astype(int),:3], s_pts2[M[1,:].astype(int),:3]


# def coord_3d(X,dim):
#     phi   = X[:,1]/dim[1] * np.pi     # phi
#     theta = X[:,0]/dim[0] * 2 * np.pi         # theta
#     R = np.stack([(np.sin(phi) * np.cos(theta)).T,(np.sin(phi) * np.sin(theta)).T,np.cos(phi).T], axis=1)

#     return R

def coord_3d(points, wh):
    width, height = wh
    x, y = points[:,0], points[:,1]
        
    # Normalize to [0, 1] range
    u = x / width
    v = y / height
    
    # Convert to spherical coordinates
    # theta = - 2 * np.pi * u + np.pi       # longitude (0 to 2π)
    theta = 2 * np.pi * u       # longitude (0 to 2π)
    phi = np.pi * v - np.pi/2   # latitude (0 to π)
    
    x = np.cos(phi) * np.sin(theta)
    y = np.sin(phi)
    z = np.cos(phi) * np.cos(theta)

    vectors = np.stack([x, y, z], 1)
    
    return np.array(vectors)


def get_cam_pose_by_ransac_8pa(x1, x2, get_E = False, I = "8PA"):
    ransac = RANSAC_8PA()
    ransac.post_function_evaluation = get_cam_pose_by_8pa
    ransac.I = I
    if I == "8PA":
        ransac.min_super_set = 8
    elif I == "5PA":
        ransac.min_super_set = 5
    if get_E == False:
        cam_final = ransac.get_cam_pose(
            bearings_1=x1,
            bearings_2=x2)
        return cam_final
    else:
        E, cam_final = ransac.get_cam_pose(
            bearings_1=x1,
            bearings_2=x2, get_E = True)
        return E, cam_final


def get_cam_pose_by_8pa(x1, x2):
    """
    Returns a camera pose by 8PA
    """
    g8p = EightPointAlgorithmGeneralGeometry()
    cam_pose = g8p.recover_pose_from_matches(
        x1=x1.copy(),
        x2=x2.copy(),
    )

    return cam_pose


def vector2skew_matrix(vector):
    """
    Converts a vector [3,] into a matrix [3, 3] for cross product operations. v x v' = [v]v' where [v] is a skew representation of v
    :param vector: [3,]
    :return: skew matrix [3, 3]
    """
    assert len(vector.shape) < 2

    skew_matrix = np.zeros((3, 3))
    skew_matrix[1, 0] = vector[2]
    skew_matrix[2, 0] = -vector[1]
    skew_matrix[0, 1] = -vector[2]
    skew_matrix[2, 1] = vector[0]
    skew_matrix[0, 2] = vector[1]
    skew_matrix[1, 2] = -vector[0]

    return skew_matrix.copy()

def spherical_normalization(array):
    assert array.shape[0] in (3, 4)
    if array.shape.__len__() < 2:
        array = array.reshape(-1, 1)
    norm = np.linalg.norm(array[0:3, :], axis=0)
    return array[0:3, :] / norm


class EightPointAlgorithmGeneralGeometry:
    """
    This class wraps the main functions for the general epipolar geometry solution
    using bearing points which don't necessary lie on a homegeneous plane.
    This implementation aims to find the Essential matrix for both perspective and spherical projection.
    """


    def compute_essential_matrix(self, x1, x2, return_sigma=False, I = "8PA"):
        """
        This function compute the Essential matrix of a pair of Nx3 points. The points must be matched each other
        from two geometry views (Epipolar constraint). This general function doesn't assume a homogeneous
        representation of points.
        :param x1: Points from the 1st frame (n, 3) [x, y, z]
        :param x2: Points from the 2st frame (n, 3) [x, y, z]
        :return: Essential Matrix (3,3)
        """

        assert x1.shape == x2.shape, f"Shapes do not match {x1.shape} != {x2.shape}"
        assert x1.shape[0] in [3, 4], f"PCL out of shape {x1.shape} != (3, n) or (4, n)"
        if I == "8PA":
            A = self.building_matrix_A(x1, x2)

            #! compute linear least square solution
            U, Sigma, V = np.linalg.svd(A)
            E = V[-1].reshape(3, 3)

            #! constraint E
            #! making E rank 2 by setting out the last singular value
            U, S, V = np.linalg.svd(E)
            S[2] = 0
            E = np.dot(U, np.dot(np.diag(S), V))
        if return_sigma:
            return E / np.linalg.norm(E), Sigma
        return E / np.linalg.norm(E)

    @staticmethod
    def building_matrix_A(x1, x2):
        """
        Build an observation matrix A of the linear equation AX=0. This function doesn't assume
        homogeneous coordinates on a plane for p1s, and p2s
        :param x1:  Points from the 1st frame (n, 3) [x, y, z]
        :param x2: Points from the 2st frame (n, 3) [x, y, z]
        :return:  Matrix (n x 9)
        """
        A = np.array([
            x1[0, :] * x2[0, :], x1[0, :] * x2[1, :], x1[0, :] * x2[2, :],
            x1[1, :] * x2[0, :], x1[1, :] * x2[1, :], x1[1, :] * x2[2, :],
            x1[2, :] * x2[0, :], x1[2, :] * x2[1, :], x1[2, :] * x2[2, :]
        ]).T

        return A

    @staticmethod
    def get_the_four_cam_solutions_from_e(E, x1, x2):
        """
        This function computes the four relative transformation poses T(4, 4)
        from x1 to x2 given an Essential matrix (3,3)
        :param E: Essential matrix
        :param x1: points in camera 1 (3, n) or (4, n)
        :param x2: points in camera 2 (3, n) or (4, n)
        """
        assert x1.shape == x2.shape
        assert x1.shape[0] == 3

        U, S, V = np.linalg.svd(E)
        if np.linalg.det(np.dot(U, V)) < 0:
            V = -V

        #! create matrix W and Z (Hartley's Book pp 258)
        W = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])

        t = U[:, 2].reshape(1, -1).T

        #! 4 possible transformations
        transformations = [
            np.vstack((np.hstack((U @ W.T @ V, t)), [0, 0, 0, 1])),
            np.vstack((np.hstack((U @ W.T @ V, -t)), [0, 0, 0, 1])),
            np.vstack((np.hstack((U @ W @ V, t)), [0, 0, 0, 1])),
            np.vstack((np.hstack((U @ W @ V, -t)), [0, 0, 0, 1])),
        ]
        return transformations

    def recover_pose_from_e(self, E, x1, x2):
        transformations = self.get_the_four_cam_solutions_from_e(E, x1, x2)
        return self.select_camera_pose(transformations, x1=x1, x2=x2)

    def select_camera_pose(self, transformations, x1, x2):
        """
        Selects the best transformation between a list of four posible tranformation given an
        essential matrix.
        :param transformations: list of (4x4) transformations
        :param x1: points in camera 1 (3, n) or (4, n)
        :param x2: points in camera 2 (3, n) or (4, n)
        """
        residuals = np.zeros((len(transformations),))
        # sample = np.random.randint(0, x1.shape[1],  int(x1.shape[1]*0.8))
        # x1 = np.copy(x1[:, sample])
        # x2 = np.copy(x2[:, sample])
        for idx, M in enumerate(transformations):
            pt1_3d = self.triangulate_points_from_cam_pose(cam_pose=M,
                                                           x1=x1,
                                                           x2=x2)
            # pt2_3d = np.linalg.inv(M).dot(pt1_3d)
            pt2_3d = M @ pt1_3d

            x1_hat = spherical_normalization(pt1_3d)
            x2_hat = spherical_normalization(pt2_3d)

            # ! Dot product xn and xn_hat must be close to 1
            closest_projections_cam2 = (np.sum(x2 * x2_hat, axis=0)) > 0.98
            closest_projections_cam1 = (np.sum(x1 * x1_hat, axis=0)) > 0.98
            residuals[idx] = np.sum(closest_projections_cam1) + np.sum(
                closest_projections_cam2)

        return transformations[residuals.argmax()]

    @staticmethod
    def get_e_from_cam_pose(cam_pose):
        t_x = vector2skew_matrix(cam_pose[0:3, 3] /
                                 np.linalg.norm(cam_pose[0:3, 3]))
        e = t_x.dot(cam_pose[0:3, 0:3])
        return e / np.linalg.norm(e)

    def recover_pose_from_matches(self, x1, x2):
        """
        Returns the a relative camera pose by using LSQ method (Higgins 1981)
        """
        e = self.compute_essential_matrix(x1, x2)
        return self.recover_pose_from_e(e, x1, x2)

    @staticmethod
    def triangulate_points_from_cam_pose(cam_pose, x1, x2):
        '''
        Triagulate 4D-points based on the relative camera pose and pts1 & pts2 matches
        :param Mn: Relative pose (4, 4) from cam1 to cam2
        :param x1: (3, n)
        :param x2: (3, n)
        :return:
        '''

        assert x1.shape[0] == 3
        assert x1.shape == x2.shape

        cam_pose = np.linalg.inv(cam_pose)
        landmarks_x1 = []
        for p1, p2 in zip(x1.T, x2.T):
            p1x = vector2skew_matrix(p1.ravel())
            p2x = vector2skew_matrix(p2.ravel())

            A = np.vstack(
                (np.dot(p1x,
                        np.eye(4)[0:3, :]), np.dot(p2x, cam_pose[0:3, :])))
            U, D, V = np.linalg.svd(A)
            landmarks_x1.append(V[-1])

        landmarks_x1 = np.asarray(landmarks_x1).T
        landmarks_x1 = landmarks_x1 / landmarks_x1[3, :]
        return landmarks_x1

    @staticmethod
    def projected_error(**kwargs):
        """
        This residual loss is introduced as projected distance in:
        A. Pagani and D. Stricker,
        “Structure from Motion using full spherical panoramic cameras,”
        ICCV 2011
        """
        E_dot_x1 = np.matmul(kwargs["e"].T, kwargs["x1"])
        E_dot_x2 = np.matmul(kwargs["e"], kwargs["x2"])
        dst = np.sum(kwargs["x1"] * E_dot_x2, axis=0)
        return dst / (np.linalg.norm(kwargs["x1"]) *
                    np.linalg.norm(E_dot_x2))

    @staticmethod
    def algebraic_error(**kwargs):
        E_dot_x2 = np.matmul(kwargs["e"], kwargs["x2"])
        dst = np.sum(kwargs["x1"] * E_dot_x2, axis=0)
        return dst

    def __projectedDistance(E, x1, x2):
        Ex1 = x1.dot(E.T)
        return np.abs(np.einsum('ij,ij->i',x2,Ex1))/np.linalg.norm(Ex1, axis=1)

    @staticmethod
    def projected_error2(**kwargs):
        Ex1 = kwargs["x1"].T.dot(kwargs["e"])
        A = np.abs(np.einsum('ij,ij->i',kwargs["x2"].T,Ex1))/(np.linalg.norm(Ex1, axis=1)+1e-5)

        Ex2 = kwargs["x2"].T.dot(kwargs["e"].T)
        B = np.abs(np.einsum('ij,ij->i',kwargs["x1"].T,Ex2))/(np.linalg.norm(Ex2, axis=1)+1e-5)

        return (A+B)/2.



def get_ransac_iterations(p_success=0.99,
                          outliers=0.5,
                          min_constraint=8):
    return int(
        np.log(1 - p_success) / np.log(1 - (1 - outliers) ** min_constraint)) + 1


class RANSAC_8PA:

    def __init__(self):
        self.residual_threshold = 1e-1
        self.probability_success = 0.99
        self.expected_inliers = 0.5
        self.solver = EightPointAlgorithmGeneralGeometry()
        self.max_trials = get_ransac_iterations(
            p_success=self.probability_success,
            outliers=1 - self.expected_inliers,
            min_constraint=8
        )
        self.num_samples = 0
        self.best_model = None
        self.best_evaluation = np.inf
        self.best_inliers = None
        self.best_inliers_num = 0
        self.counter_trials = 0
        self.time_evaluation = np.inf
        self.post_function_evaluation = None
        self.min_super_set = 5
        self.I = None

    def estimate_essential_matrix(self, sample_bearings1, sample_bearings2, function):
        bearings = dict(
            x1=sample_bearings1,
            x2=sample_bearings2
        )
        return self.solver.get_e_from_cam_pose(function(**bearings))

    def run(self, bearings_1, bearings_2):
        assert bearings_1.shape == bearings_2.shape
        assert bearings_1.shape[0] is 3
        self.num_samples = bearings_1.shape[1]

        random_state = np.random.RandomState(1000)
        self.time_evaluation = 0
        aux_time = time.time()
        for self.counter_trials in range(1000):

            initial_inliers = random_state.choice(self.num_samples, self.min_super_set, replace=False)
            sample_bearings1 = bearings_1[:, initial_inliers]
            sample_bearings2 = bearings_2[:, initial_inliers]

            # * Estimation
            e_hat = self.solver.compute_essential_matrix(
                x1=sample_bearings1,
                x2=sample_bearings2,
                return_sigma=False,
                I = self.I                
            )

            # * Evaluation
            sample_residuals = self.solver.projected_error2(
                e=e_hat,
                x1=bearings_1,
                x2=bearings_2
            )
            sample_evaluation = np.sum(sample_residuals ** 2)

            # * Selection
            sample_inliers = np.abs(sample_residuals) < self.residual_threshold
            sample_inliers_num = np.sum(sample_inliers)

            # * Loop Control
            lc_1 = sample_inliers_num > self.best_inliers_num
            lc_2 = sample_inliers_num == self.best_inliers_num
            lc_3 = sample_evaluation < self.best_evaluation
            if lc_1 or (lc_2 and lc_3):
                # + Update best performance
                self.best_model = e_hat.copy()
                self.best_inliers_num = sample_inliers_num.copy()
                self.best_evaluation = sample_evaluation.copy()
                self.best_inliers = sample_inliers.copy()

            if self.counter_trials >= self._dynamic_max_trials():
                break

        best_bearings_1 = bearings_1[:, self.best_inliers]
        best_bearings_2 = bearings_2[:, self.best_inliers]

        # * Estimating final model using only inliers
        self.best_model = self.estimate_essential_matrix(
            sample_bearings1=best_bearings_1,
            sample_bearings2=best_bearings_2,
            function=self.post_function_evaluation
            # ! predefined function used for post-evaluation
        )
        self.time_evaluation += time.time() - aux_time
        # * Final Evaluation
        sample_residuals = self.solver.projected_error(
            e=self.best_model,
            x1=best_bearings_1,
            x2=best_bearings_2
        )
        self.best_evaluation = np.sum(sample_residuals ** 2)

        # * Final Selection
        sample_inliers = sample_residuals < self.residual_threshold
        self.best_inliers_num = np.sum(sample_inliers)
        return self.best_model, self.best_inliers

    def get_cam_pose(self, bearings_1, bearings_2,get_E = False):
        self.run(
            bearings_1=bearings_1,
            bearings_2=bearings_2
        )
        cam_pose = self.solver.recover_pose_from_e(
            E=self.best_model,
            x1=bearings_1[:, self.best_inliers],
            x2=bearings_2[:, self.best_inliers]
        )
        if get_E == False:
            return cam_pose
        else:
            return self.best_model, cam_pose

    def _dynamic_max_trials(self):
        if self.best_inliers_num == 0:
            return np.inf

        nom = 1 - self.probability_success
        if nom == 0:
            return np.inf

        inlier_ratio = self.best_inliers_num / float(self.num_samples)
        denom = 1 - inlier_ratio ** 8
        if denom == 0:
            return 1
        elif denom == 1:
            return np.inf

        nom = np.log(nom)
        denom = np.log(denom)
        if denom == 0:
            return 0

        return int(np.ceil(nom / denom))
