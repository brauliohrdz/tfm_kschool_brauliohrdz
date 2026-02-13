from abc import ABC

import cv2
import numpy as np


class Coords:
    def __init__(self, x: int = 0, y: int = 0):
        self.x = x
        self.y = y


class Region:
    def __init__(self, top_left: tuple = None, bottom_right: tuple = None):
        self.top_left = Coords(*top_left) if top_left else Coords()
        self.bottom_right = Coords(*bottom_right) if bottom_right else Coords()


class IDDocumentTemplate(ABC):

    image: np.ndarray = None
    spain_text: Region = None
    surname1: Region = None
    surname2: Region = None
    name: Region = None
    gender: Region = None
    nationality: Region = None
    date_of_birth: Region = None
    dni_num: Region = None
    dni_letter: Region = None
    validity_date: Region = None
    expedition_date: Region = None
    picture: Region = None

    def __init__(self, image: np.ndarray, debug: bool = False):
        self._image = image
        self._debug = debug

    def _crop_region(self, region_name: str) -> np.ndarray:
        region = getattr(self, region_name)
        if region is None:
            return np.ndarray([0, 0])
        return self._image[
            region.top_left.y : region.bottom_right.y,
            region.top_left.x : region.bottom_right.x,
        ]

    def _get_fields(self) -> list[str]:
        return [
            name
            for name in dir(self)
            if isinstance(
                getattr(self, name),
                Region,
            )
        ]

    def view_template(self) -> np.ndarray:
        templated_image = self._image.copy()
        for region_name in self._get_fields():
            print(region_name)
            region = getattr(self, region_name)
            cv2.rectangle(
                templated_image,
                (region.top_left.x, region.top_left.y),
                (region.bottom_right.x, region.bottom_right.y),
                (0, 255, 0),
                2,
            )
            print("done", region_name)
        if self._debug:
            cv2.imshow("Template", templated_image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        return templated_image

    def get_field(self, field_name: str) -> np.ndarray:
        return self._crop_region(f"{field_name}")


class DNI2DocumentTemplate(IDDocumentTemplate):

    spain_text = Region((5, 104), (275, 181))
    name = Region((275, 195), (550, 233))
    surname1 = Region((275, 94), (600, 135))
    surname2 = Region((275, 144), (600, 185))
    gender = Region((275, 250), (365, 290))
    nationality = Region((389, 250), (488, 290))
    date_of_birth = Region((275, 304), (506, 345))
    validity_date = Region((275, 401), (506, 445))
    dni_num = Region((15, 575), (240, 625))
    dni_letter = Region((236, 575), (270, 625))
    picture = Region((675, 250), (1010, 635))


class DNI2FaceTemplate(IDDocumentTemplate):

    picture = Region((675, 250), (1010, 635))
