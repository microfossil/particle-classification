import cv2
import numpy as np

class Morphology:
    def __init__(self):
        self.height = 0
        self.width = 0
        self.area = 0.0
        self.perimeter = 0.0
        self.convex_area = 0.0
        self.convex_perimeter = 0.0
        self.circle_area = 0.0
        self.circle_radius = 0.0
        self.major_axis_length = 0.0
        self.minor_axis_length = 0.0
        self.mean_diameter = 0.0
        self.eccentricity = 0.0
        self.angle = 0.0
        self.solidity = 0.0
        self.circularity = 0.0
        self.roundness = 0.0
        self.convex_perimeter_to_perimeter_ratio = 0.0
        self.convex_area_to_area_ratio = 0.0
        self.mean = 0.0
        self.stddev = 0.0
        self.stddev_invariant = 0.0
        self.skew = 0.0
        self.kurtosis = 0.0
        self.moment5 = 0.0
        self.moment6 = 0.0
        self.aspect_ratio = 0.0
        self.equivalent_diameter = 0.0
        self.perimeter_to_area_ratio = 0.0
        self.area_to_bounding_rectangle_area = 0.0
        self.equivalent_spherical_diameter = 0.0
        self.elongation = 0.0
        self.husmoment1 = 0.0
        self.husmoment2 = 0.0
        self.husmoment3 = 0.0
        self.husmoment4 = 0.0
        self.husmoment5 = 0.0
        self.husmoment6 = 0.0
        self.husmoment7 = 0.0

    def convert_to_mm(self, pixels_per_mm):
        pixels2_per_mm2 = pixels_per_mm * pixels_per_mm

        if pixels_per_mm == 0:
            return Morphology()

        self.mean_diameter /= pixels_per_mm
        self.area /= pixels2_per_mm2
        self.perimeter /= pixels_per_mm
        self.convex_area /= pixels2_per_mm2
        self.convex_perimeter /= pixels_per_mm
        self.circle_area /= pixels2_per_mm2
        self.circle_radius /= pixels_per_mm
        self.major_axis_length /= pixels_per_mm
        self.minor_axis_length /= pixels_per_mm

        return self



class MorphologyProcessor:
    @staticmethod
    def calculate_morphology(rgb, greyscale, contour):
        m = Morphology()

        # Exit if mask is non-existent or too small
        if contour is None or len(contour) < 5:
            return m

        contour = contour.astype(int)

        # Create binary image from contour
        binary = np.zeros(greyscale.shape, dtype=np.uint8)
        cv2.drawContours(binary, [contour], -1, 1, thickness=cv2.FILLED)
        binaryF = binary.astype(np.float32)

        m.area = cv2.contourArea(contour)

        if m.area < 25:
            return m

        m.width = greyscale.shape[1]
        m.height = greyscale.shape[0]
        m.mean_diameter = np.sqrt(m.area / np.pi) * 2

        # IMAGE INTENSITY MOMENTS
        # Mean(0) and standard deviation(1)
        mean, stddev = cv2.meanStdDev(greyscale, mask=binary)
        m.mean = mean[0][0]
        m.stddev = stddev[0][0]
        m.stddev_invariant = m.stddev / m.mean

        variance = np.power(m.stddev, 2)
        # num_pixels_in_mask = cv2.sumElems(binaryF)[0]
        # print(contour)
        # print(num_pixels_in_mask)
        num_pixels_in_mask = m.area

        greyscale_masked = cv2.multiply(binaryF, greyscale, dtype=cv2.CV_32FC1)

        # Mean
        m.mean = cv2.sumElems(greyscale_masked)[0] / num_pixels_in_mask

        # Subtract mean
        greyscale_masked = cv2.subtract(greyscale_masked, m.mean)
        greyscale_masked = cv2.multiply(greyscale_masked, binaryF, dtype=cv2.CV_32FC1)

        # Standard deviation
        temp = cv2.pow(greyscale_masked, 2)
        temp_sum = cv2.sumElems(temp)[0]
        m.stddev = np.sqrt(temp_sum / num_pixels_in_mask)

        # Skew
        temp = cv2.pow(greyscale_masked, 3)
        temp_sum = cv2.sumElems(temp)[0]
        m.skew = temp_sum / np.power(m.stddev, 3) / num_pixels_in_mask

        # Kurtosis
        temp = cv2.pow(greyscale_masked, 4)
        temp_sum = cv2.sumElems(temp)[0]
        m.kurtosis = temp_sum / np.power(m.stddev, 4) / num_pixels_in_mask

        # 5th moment
        temp = cv2.pow(greyscale_masked, 5)
        temp_sum = cv2.sumElems(temp)[0]
        m.moment5 = temp_sum / np.power(m.stddev, 5) / num_pixels_in_mask

        # 6th moment
        temp = cv2.pow(greyscale_masked, 6)
        temp_sum = cv2.sumElems(temp)[0]
        m.moment6 = temp_sum / np.power(m.stddev, 6) / num_pixels_in_mask

        # MASK DIMENSIONS
        # Area and perimeter
        m.perimeter = cv2.arcLength(contour, True)

        # Convex area and perimeter
        convex_contour = cv2.convexHull(contour)
        # print(hull)
        # convex_contour = np.array([contour[idx] for idx in hull])

        m.convex_area = cv2.contourArea(convex_contour)
        m.convex_perimeter = cv2.arcLength(convex_contour, True)
        m.convex_perimeter_to_perimeter_ratio = m.convex_perimeter / m.perimeter
        m.convex_area_to_area_ratio = m.convex_area / m.area

        # ELLIPSE AND CIRCLE
        # Enclosing circle
        (x, y), radius = cv2.minEnclosingCircle(contour)
        m.circle_radius = radius
        m.circle_area = np.pi * radius * radius

        # Approximating ellipse
        if len(contour) >= 5:
            (center, (l1, l2), angle) = cv2.fitEllipse(contour)
            m.major_axis_length = max(l1, l2)
            m.minor_axis_length = min(l1, l2)
            m.eccentricity = np.sqrt(1 - np.power(m.minor_axis_length / m.major_axis_length, 2))
            m.angle = angle
            m.roundness = 4.0 * m.area / np.pi / np.power(m.major_axis_length, 2)
            m.elongation = m.major_axis_length / m.minor_axis_length

        # COMMON MEASUREMENTS
        m.solidity = m.area / m.convex_area
        m.circularity = 4.0 * np.pi * m.area / np.power(m.perimeter, 2)
        m.perimeter_to_area_ratio = m.perimeter / m.area
        m.equivalent_diameter = np.sqrt(4.0 * m.area / np.pi)
        m.equivalent_spherical_diameter = 2.0 * np.sqrt(m.area / np.pi)

        bounding_rectangle = cv2.boundingRect(contour)
        m.aspect_ratio = bounding_rectangle[2] / bounding_rectangle[3]
        m.area_to_bounding_rectangle_area = m.area / (bounding_rectangle[2] * bounding_rectangle[3])

        hu_moments = cv2.HuMoments(cv2.moments(contour)).flatten()
        m.husmoment1 = hu_moments[0]
        m.husmoment2 = hu_moments[1]
        m.husmoment3 = hu_moments[2]
        m.husmoment4 = hu_moments[3]
        m.husmoment5 = hu_moments[4]
        m.husmoment6 = hu_moments[5]
        m.husmoment7 = hu_moments[6]

        return m