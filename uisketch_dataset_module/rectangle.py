""" Class for defining rectangular bounding box """


class Rectangle:
    """
    Rectangle class with xmin, ymin, width, height, xmax and ymax values. Check
    whether a rectangle bounds or intersects another rectangle(s)
    """

    def __init__(self, xmin=0, ymin=0, width=0, height=0):
        self.xmin = xmin
        self.ymin = ymin

        self.xmax = xmin + width
        self.ymax = ymin + height

        self.width = width
        self.height = height

    def __str__(self):
        return (
            "Rectangle "
            f"(xmin:{self.xmin}, ymin:{self.ymin}, width:{self.width}, height:{self.height})"
        )

    def set_position(self, xmin, ymin):
        """Set position of the rectangle and calculate xmax, ymax

        Arguments:
            xmin {int} -- x min value (left top)
            ymin {int} -- y min value (left top)
        """
        self.xmin = xmin
        self.ymin = ymin

        self.xmax = xmin + self.width
        self.ymax = ymin + self.height

    def intersects(self, other):
        """Check whether this rectangle intersects with the other rectangle

        Arguments:
            other {Rectangle} -- instance of Rectangle class

        Returns:
            Boolean -- Returns True if this rectangle intersects with the other
            rectangle, else False
        """
        if self.xmin > other.xmax or self.xmax < other.xmin:
            return False
        if self.ymin > other.ymax or self.ymax < other.ymin:
            return False
        return True

    def intersects_any(self, rectangles):
        """Check whether this rectangle intersects with the list of rectangles

        Arguments:
            rectangles {array} -- List of Rectangles (Rectangle class)

        Returns:
            Boolean -- Returns True if this rectangle intersects with any of the
            other rectangles in the list, else False
        """
        is_intersecting = False

        for rectangle in rectangles:
            if self.intersects(rectangle):
                is_intersecting = True
                break

        return is_intersecting

    def bounds(self, other):
        """Check whether this rectangle bounds the other rectangle

        Arguments:
            other {Rectangle} -- instance of Rectangle class

        Returns:
            Boolean -- Returns True if this rectangle bounds the other
            rectangle, else False
        """
        if (
            other.xmin >= self.xmin
            and other.ymin >= self.ymin
            and other.xmax <= self.xmax
            and other.ymax <= self.ymax
        ):
            return True
        return False
