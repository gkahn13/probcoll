import abc

import numpy as np
import matplotlib
from matplotlib import pyplot as plt

epsilon = 1e-5

class Point:
    """ Allows comparing 2d points """
    def __init__(self, p):
        """
        :param p: list/np.array
        """
        self.p = np.array(p)

    def __eq__(self, other):
        """
        :param other: list/np.array/Point
        """
        if isinstance(other, Point):
            return np.linalg.norm(self.p - other.p) < epsilon
        elif isinstance(other, list) or isinstance(other, np.array):
            return np.linalg.norm(self.p - np.array(other)) < epsilon
        return False

    def __hash__(self):
        return 0

class Object2d:
    __metaclass__ = abc.ABCMeta

    def __init__(self):
        pass

    @abc.abstractmethod
    def closest_point_to(self, x):
        """
        Finds closest point on object to x
        :param x: 2d list or np.array
        :return: signed distance, 2d np.array
        """
        raise NotImplementedError("Must be implemented in subclass")

    def is_within(self, dist, x):
        """
        Checks if object is with dist of point x
        :type dist: float
        :param x: 2d list or np.array
        :return: bool
        """
        return self.closest_point_to(x)[0] <= dist

    @abc.abstractmethod
    def intersection(self, seg):
        """
        Finds intersection with segment (returns intersection closer to seg.p0)
        :type seg: Segment
        :return: (None, None) if no intersection, else (intersection, normal)
        """
        raise NotImplementedError("Must be implemented in subclass")

    @abc.abstractmethod
    def plot(self, ax, color='r'):
        """
        :param ax: pyplot axis
        :param color: character or (r,g,b) [0,1]
        """
        raise NotImplementedError("Must be implemented in subclass")

class Circle(Object2d):
    def __init__(self, center, radius):
        self.center = np.array(center)
        self.radius = radius

    def closest_point_to(self, x):
        dist_to_center = np.linalg.norm(x - self.center)

        towards_x = x - self.center
        towards_x /= np.linalg.norm(towards_x)

        pt = self.center + self.radius * towards_x
        sd = dist_to_center - self.radius

        return sd, pt

    def intersection(self, seg):
        """
        http://stackoverflow.com/questions/1073336/circle-line-segment-collision-detection-algorithm
        t^2 (d.dot(d)) + t (2*d.dot(f)) + (f.dot(f) - r*r)
        """
        d = seg.p1 - seg.p0
        f = seg.p0 - self.center

        a = d.dot(d)
        b = 2*d.dot(f)
        c = f.dot(f) - self.radius * self.radius

        discriminant = b*b - 4*a*c
        if discriminant < 0:
            return None, None
        else:
            t1 = (-b - np.sqrt(discriminant)) / (2*a)
            t2 = (-b + np.sqrt(discriminant)) / (2*a)

            if (t1 >= 0) and (t1 <= 1):
                intersection = seg.p0 + t1 * d
                normal = intersection - self.center
                normal /= np.linalg.norm(normal)
                return intersection, normal
            elif (t2 >= 0) and (t2 <= 1):
                intersection = seg.p0 + t2 * d
                normal = intersection - self.center
                normal /= np.linalg.norm(normal)
                return intersection, normal
            else:
                return None, None

    def plot(self, ax, color='r', facecolor=None):
        if facecolor is None:
            facecolor = color
        patch = ax.add_artist(plt.Circle(self.center, self.radius, edgecolor=color, facecolor=facecolor))
        ax.draw_artist(patch)

class Triangle(Object2d):
    def __init__(self, a, b, c):
        self.a, self.b, self.c = np.array(a), np.array(b), np.array(c)

    def closest_point_to(self, x):
        """
        Finds closest point on triangle to x
        by checking distances to the triangle edges

        :param p: 2d list or np.array
        :return signed distance, closest point 2d np.array
        """
        min_pt, min_dist = None, np.inf
        for s in self.segments:
            s_min_dist, s_min_pt = s.closest_point_to(x)
            if s_min_dist < min_dist:
                min_dist = s_min_dist
                min_pt = s_min_pt

        sd = np.copysign(min_dist, -1 if self.is_inside(x) else 1)
        return sd, min_pt

    def is_inside(self, x):
        """
        :param x: 2d list or np.array
        :return True if x is inside, else False
        """
        total_area = self.area
        area0 = Triangle(self.a, self.b, x).area
        area1 = Triangle(self.b, self.c, x).area
        area2 = Triangle(self.c, self.a, x).area

        is_correct_area = np.abs(total_area - (area0 + area1 + area2)) < epsilon

        return is_correct_area

    def intersection(self, other):
        raise Exception('Need to implement normal')
        closest_dist, closest_intersection = np.inf, None
        for seg in self.segments:
            intersection = other.intersection(seg)
            if intersection is not None:
                dist = np.linalg.norm(intersection - other.p0)
                if dist < closest_dist:
                    closest_dist = dist
                    closest_intersection = intersection

        return closest_intersection

    @property
    def area(self):
        """
        :return float area
        """
        a, b, c = self.a, self.b, self.c
        return np.abs((c[0]*(a[1] - b[1]) + a[0]*(b[1] - c[1]) + b[0]*(c[1] - a[1])) / 2.0)

    @property
    def vertices(self):
        """
        :return Triangle corners
        """
        return (self.a, self.b, self.c)

    @property
    def segments(self):
        """
        :return [edge0, edge1, edge2]
        """
        return (Segment(self.a, self.b), Segment(self.b, self.c), Segment(self.c, self.a))

    @property
    def halfspaces(self):
        """
        halfspaces of edges pointing outwards
        :return [hyp0, hyp1, hyp2]
        """
        segments = (Segment(self.a, self.b), Segment(self.b, self.c), Segment(self.c, self.a))
        other_points = (self.c, self.a, self.b)

        hspaces = list()
        for segment, other_point in zip(segments, other_points):
            origin = (segment.p0 + segment.p1)/2.

            colinear = segment.p1 - segment.p0
            normal = np.array([-colinear[1], colinear[0]])
            if np.dot(normal, other_point - origin) < 0:
                hspaces.append(Halfspace(origin, normal))
            else:
                hspaces.append(Halfspace(origin, -normal))

        return hspaces

    def plot(self, ax, color='r'):
        poly = matplotlib.patches.Polygon([self.a, self.b, self.c], closed=True, fill=True)
        poly.set_color(color)
        patch = ax.add_artist(poly)
        ax.draw_artist(patch)

    def __eq__(self, other):
        """
        :param other: Triangle
        :return True if other's points within epsilon distance
        """
        points = sorted(sorted([self.a, self.b, self.c], key=lambda x: x[0]), key=lambda x: x[1])
        other_points = sorted(sorted([other.a, other.b, other.c], key=lambda x: x[0]), key=lambda x: x[1])

        for point, other_point in zip(points, other_points):
            if np.linalg.norm(point - other_point) > epsilon:
                return False

        return True

    def __hash__(self):
        return 0

    @staticmethod
    def random(min_x, max_x, min_y, max_y):
        return Triangle([np.random.uniform(min_x, max_x), np.random.uniform(min_y, max_y)],
                        [np.random.uniform(min_x, max_x), np.random.uniform(min_y, max_y)],
                        [np.random.uniform(min_x, max_x), np.random.uniform(min_y, max_y)])


class Segment(Object2d):
    def __init__(self, p0, p1):
        self.p0, self.p1 = np.array(p0), np.array(p1)

    def closest_point_to(self, x):
        """
        min_{0<=t<=1} ||t*(p1-p0) + p0 - x||_{2}^{2}

        :param x: 2d list or np.array
        :return distance, 2d np.array of closest point on segment to x
        """
        x = np.array(x)
        v = self.p1 - self.p0
        b = self.p0 - x

        t = -np.dot(v, b) / np.dot(v, v)
        if (0 <= t <= 1):
            closest = t*(self.p1 - self.p0) + self.p0
            return np.linalg.norm(x - closest), closest
        else:
            if np.linalg.norm(x - self.p0) < np.linalg.norm(x - self.p1):
                return np.linalg.norm(x - self.p0), self.p0
            else:
                return np.linalg.norm(x - self.p1), self.p1

    def intersection(self, other):
        """
        Finds intersection point with another segment
        :param other: Segment
        :return None if no intersection, else [x,y] of intersection
        """
        p0_other, p1_other = other.p0, other.p1

        # for speed, check if axis-aligned separation
        if (max(self.p0[0], self.p1[0]) < min(p0_other[0], p1_other[0])) or \
            (max(p0_other[0], p1_other[0]) < min(self.p0[0], self.p1[0])) or \
            (max(self.p0[1], self.p1[1]) < min(p0_other[1], p1_other[1])) or \
            (max(p0_other[1], p1_other[1]) < min(self.p0[1], self.p1[1])):
            return None

        # w = p1 - p0
        # v = p1_other - p0_other
        # s*w + p0 = t*v + p_other

        w = self.p1 - self.p0
        v = p1_other - p0_other

        A = np.vstack((w,v)).T
        b = p0_other - self.p0

        if np.abs(np.linalg.det(A)) < epsilon:
            return None

        soln = np.linalg.solve(A, b)
        s, t = soln[0], -soln[1]

        intersection = s*w + self.p0

        if ((-epsilon <= s) and (s <= 1+epsilon) and (-epsilon <= t) and (t <= 1+epsilon)):
            return intersection, None
        else:
            return None, None

    def angle(self, other):
        """
        Finds vector angle between this and other
        :param other: Segment
        :return float [-pi, pi]
        """
        a = self.p1 - self.p0
        b = other.p1 - other.p0

        theta = np.arctan2(a[0]*b[1] - a[1]*b[0], a[0]*b[0] + a[1]*b[1])
        return theta

    def is_endpoint(self, point):
        """
        :param point: 2d np.array
        :return True if point is an endpoint of the segment
        """
        return np.linalg.norm(self.p0 - point) < epsilon or np.linalg.norm(self.p1 - point) < epsilon

    def is_parallel(self, other):
        """
        :param other: Segment
        :return True if segments are parallel
        """
        slope = self.p1 - self.p0
        slope /= np.linalg.norm(slope)

        other_slope = other.p1 - other.p0
        other_slope /= np.linalg.norm(other_slope)

        return np.linalg.norm(slope - other_slope) < epsilon or np.linalg.norm(slope + other_slope) < epsilon

    def plot(self, ax, color='r'):
        ax.plot([self.p0[0], self.p1[0]], [self.p0[1], self.p1[1]], color=color)

    def __eq__(self, other):
        """
        :param other: Segment
        :return True if other's points within epsilon distance
        """
        return (np.linalg.norm(self.p0 - other.p0) < epsilon and np.linalg.norm(self.p1 - other.p1) < epsilon) or \
                 (np.linalg.norm(self.p0 - other.p1) < epsilon and np.linalg.norm(self.p1 - other.p0) < epsilon)

    def __hash__(self):
        return 0


class Halfspace:
    def __init__(self, origin, normal):
        self.origin = np.array(origin, dtype=float)
        self.normal = np.array(normal, dtype=float)
        self.normal /= np.linalg.norm(self.normal)

    def contains(self, x):
        """
        :param x: 2d point as list or np.array
        :return True if x forms acute angle with plane normal, else False
        """
        return np.dot(self.normal, np.array(x) - self.origin) >= epsilon

    def plot(self, axes, color='r'):
        """
        Plots the normal

        :param axes: pyplot axes
        :param color: character or (r,g,b) [0,1]
        """
        x_list = [self.origin[0], self.origin[0] + self.normal[0]]
        y_list = [self.origin[1], self.origin[1] + self.normal[1]]
        axes.plot(x_list, y_list, color=color)

