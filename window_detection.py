import cv2
import numpy


class Debug:
    def __init__(self, file_location: str):
        try:
            self.log_file = open(file_location, "w", encoding="utf-8")
        except FileNotFoundError:
            raise Exception("Logs file couldn't be opened.")

        # Create a debug level.
        class DebugLevels:
            INFO = 0
            ERROR = 1
            RESULT = 2
        self.debug_level = DebugLevels()
        self.debug_level_names = ["INFO", "ERROR", "RESULT"]

    def _save_and_return_message_(self, debug_level: int, message: str):
        log = f"[{self.debug_level_names[debug_level]}] {message}\n"
        self.log_file.write(log)
        return log

    def info(self, message):
        return self._save_and_return_message_(self.debug_level.INFO, message)

    def error(self, message):
        return self._save_and_return_message_(self.debug_level.ERROR, message)

    def result(self, message):
        return self._save_and_return_message_(self.debug_level.ERROR, message)


class WindowDetection:
    def __init__(self, image_file: numpy.ndarray, log_file_location: str = "logs.txt") -> None:
        self.image_uploaded = image_file
        self.debug = Debug(log_file_location)

        self.steps = []
        self.corners_by_contour = []
        self.corners_by_harris = []
        self.end_image = None

    def run(self):
        self.end_image = self.apply_binarization(self.image_uploaded)
        self.steps.append(self.end_image)
        self.save_image(self.end_image, "result0.png")

        # self.end_image = self.apply_gaussian_filter(self.end_image)
        # self.steps.append(self.end_image)
        # self.save_image(self.end_image, "result1.png")

        kernel_size = 25
        element = cv2.getStructuringElement(cv2.MORPH_RECT,
                                            (2 * kernel_size + 1, 2 * kernel_size + 1),
                                            (kernel_size, kernel_size))
        self.end_image = cv2.erode(self.end_image, element)
        self.save_image(self.end_image, "result1.png")

        self.end_image = cv2.dilate(self.end_image, element)
        self.save_image(self.end_image, "result2.png")

        self.corners_by_contour = self.apply_contour_find(self.end_image)
        self.corners_by_harris = self.apply_harris_corner_detection(self.end_image)

    def save_image(self, image: numpy.ndarray, file_location: str) -> None:
        self.debug.info("save_image(): Function started.")
        cv2.imwrite(file_location, image)
        self.debug.result(f"Image is saved to {file_location}.")
        self.debug.info("save_image(): Function ended.")

    def apply_binarization(self, image: numpy.ndarray) -> numpy.ndarray:
        """
        The function gets an image input, and apply Otsu's Method
        to find its binary representation.
        :param image: Original 3-channel image to perform Otsu's method.
        :return: The binary image as a numpy.ndarray.
        """
        self.debug.info("make_binary_image(): Function started.")
        grayscale_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, binary_image = cv2.threshold(grayscale_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        binary_image = cv2.bitwise_not(binary_image)
        self.debug.info("make_binary_image(): Function ended.")
        return binary_image

    def apply_gaussian_filter(self, image: numpy.ndarray, window_size=51, gaussian_sigma=5) -> numpy.ndarray:
        """
        Function returns the cv2 Image instance that
        has Gaussian blur with predefined window size
        and gaussian sigma ratio.
        :param image: Original image to perform blur onto.
        :param window_size: Default is 5.
        :param gaussian_sigma: Default is 5.
        :return: The blurred image as a numpy.ndarray.
        """
        self.debug.info("blurred_image(): Function started.")
        blurred_image = cv2.GaussianBlur(image, (window_size, window_size), gaussian_sigma)
        self.debug.info("blurred_image(): Function ended.")
        return blurred_image

    def apply_contour_find(self, image: numpy.ndarray):
        self.debug.info("apply_contour_find(): Function started.")
        _, thresh = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY_INV)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        self.debug.info(f"{len(contours)} counters found. Selecting the biggest one.")

        contour_wanted = contours[0]
        for contour in contours:
            if cv2.contourArea(contour_wanted) <= cv2.contourArea(contour):
                contour_wanted = contour
        self.debug.info("The biggest contour is selected.")

        epsilon = 0.01 * cv2.arcLength(contour_wanted, True)
        approx = cv2.approxPolyDP(contour_wanted, epsilon, True)
        self.debug.info("Approximated the polygon, returning the corners.")

        self.debug.info("apply_contour_find(): Function ended.")
        return approx

    def apply_harris_corner_detection(self, image: numpy.ndarray):
        self.debug.info("apply_harris_corner_detection(): Function started.")
        corners = []

        dst = cv2.cornerHarris(image, 10, 7, 0.04)
        self.debug.info("Corner Harris method is applied, and corners are detected.")
        thresh = 0.25 * dst.max()
        self.debug.info(f"Threshold for corner extraction is {thresh}. Selection is starting.")
        for j in range(0, dst.shape[0]):
            for i in range(0, dst.shape[1]):
                if dst[j, i] > thresh:
                    corners.append((i, j))
                    self.debug.result(f"Corner is found and added to list: ({i}, {j})")

        self.debug.info("Algorithm started to delete nearby corners.")
        minimum_corners = []
        if len(corners) > 4:
            for corner in corners:
                if corner not in minimum_corners:
                    corner_x = corner[0]
                    corner_y = corner[1]

                    # Check if there is similar pixel in minimum array.
                    is_there_similar_pixel = False
                    for x_value in range(corner_x - 15, corner_x + 15):
                        for y_value in range(corner_y - 15, corner_y + 15):
                            if (x_value, y_value) in minimum_corners:
                                is_there_similar_pixel = True

                    if not is_there_similar_pixel:
                        minimum_corners.append(corner)

        self.debug.info("apply_harris_corner_detection(): Function ended.")
        return minimum_corners


# Main function
if __name__ == "__main__":
    image_to_test = cv2.imread("reference.jpg")
    window_detection = WindowDetection(image_to_test)
    window_detection.run()
