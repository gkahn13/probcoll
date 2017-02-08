import numpy as np

import transformations as tft

import geometry2d

epsilon = 1e-5

class Point:
    """ Allows comparing 3d points """
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

class RectangularPrism:
    def __init__(self, a0, a1, b0, b1, c0, c1, d0, d1):
        """
        A rectangular prism with rectangle a0, b0, c0, d0
        and end rectangle a1, b1, c1, d1 arranged as
        b0 --- a0       b1 --- a1
        |      |        |      |
        |      |        |      |
        c0 --- d0       c1 --- d1
        """
        self.a0 = np.array(a0)
        self.b0 = np.array(b0)
        self.c0 = np.array(c0)
        self.d0 = np.array(d0)
        self.a1 = np.array(a1)
        self.b1 = np.array(b1)
        self.c1 = np.array(c1)
        self.d1 = np.array(d1)

    @staticmethod
    def from_aabb(pos, ext):
        """
        Create from axis-aligned bounding box
        :param pos: center position
        :param ext: extents of sides
        :return: RectangularPrism
        """
        pos, ext = np.array(pos), np.array(ext)

        a0 = pos + [ext[0], -ext[1], ext[2]]
        b0 = pos + [ext[0], ext[1], ext[2]]
        c0 = pos + [-ext[0], ext[1], ext[2]]
        d0 = pos + [-ext[0], -ext[1], ext[2]]
        a1 = pos + [ext[0], -ext[1], -ext[2]]
        b1 = pos + [ext[0], ext[1], -ext[2]]
        c1 = pos + [-ext[0], ext[1], -ext[2]]
        d1 = pos + [-ext[0], -ext[1], -ext[2]]

        return RectangularPrism(a0, a1, b0, b1, c0, c1, d0, d1)

    def is_inside(self, p):
        """
        :param p: 3d point as list or np.array
        :return True if p is inside the pyramid, else False
        """
        p = np.array(p)

        halfspaces = self.halfspaces
        return np.min([h.contains(p) for h in halfspaces])

    def distance_to(self, p):
        """
        Finds signed distance to point p

        :param p: 3d list or np.array
        :return float distance
        """
        sign = 1 # -1 if self.is_inside(p) else 1
        dist = min([tri.distance_to(p) for tri in self.triangles])
        return sign * dist

    @property
    def triangles(self):
        """
        :return: list of triangles composing the rectangular prism
        """
        # triangles = [Triangle(self.a0, self.a1, self.d1), # right side
        #              Triangle(self.d1, self.d0, self.a0)]

        triangles = [Triangle(self.a0, self.b0, self.c0), # 1st rectangle
                     Triangle(self.c0, self.d0, self.a0),
                     Triangle(self.a0, self.a1, self.d1), # right side
                     Triangle(self.d1, self.d0, self.a0),
                     Triangle(self.a0, self.a1, self.b1), # top side
                     Triangle(self.b1, self.b0, self.a0),
                     Triangle(self.b0, self.b1, self.c1), # left side
                     Triangle(self.c1, self.c0, self.b0),
                     Triangle(self.c0, self.c1, self.d1), # bottom side
                     Triangle(self.d1, self.d0, self.c0),
                     Triangle(self.a1, self.b1, self.c1), # 2nd rectangle
                     Triangle(self.c1, self.d1, self.a1)]

        return triangles

    @property
    def halfspaces(self):
        """
        :return list of halfspaces representing outward-pointing faces
        """
        a0, b0, c0, d0 = self.a0, self.b0, self.c0, self.d0
        a1, b1, c1, d1 = self.a1, self.b1, self.c1, self.d1

        origins = [(a0+b0+c0+d0)/4.0, # small rectangle
                   (a0+a1+d0+d1)/4.0, # right side
                   (a0+a1+b0+b1)/4.0, # top side
                   (b0+b1+c0+c1)/4.0, # left side
                   (c0+c1+d0+d1)/4.0, # bottom side
                   (a1+b1+c1+d1)/4.0] # big rectangle

        normals = [-np.cross(b0-c0, d0-c0),
                   -np.cross(a1-a0, d0-a0),
                   -np.cross(b0-a0, a1-a0),
                   -np.cross(c1-c0, b0-c0),
                   -np.cross(d1-d0, c0-d0),
                   -np.cross(a1-d1, c1-d1)]

        normals = [n/np.linalg.norm(n) for n in normals]

        center = (a0+b0+c0+d0+a1+b1+c1+d1)/8.0
        hspaces = [Halfspace(origin, normal) for origin, normal in zip(origins, normals)]
        for hspace in hspaces:
            if not hspace.contains(center):
                hspace.normal *= -1

        return hspaces

    def clip_triangle(self, triangle):
        """
        Clips triangle against faces (http://www.cs.uu.nl/docs/vakken/gr/2011/Slides/08-pipeline2.pdf)
        :param triangle: Triangle
        :return list of Triangle
        """
        triangles = [triangle]
        for i, h in enumerate(self.halfspaces):
            new_triangles = list()
            # clip all triangles against the halfspace
            splitting_halfspaces = [h]
            if i == 0:
                # keep part in other half
                splitting_halfspaces.append(h.complement)
            for halfspace in splitting_halfspaces:
                for tri in triangles:
                    tri_segments = tri.segments
                    clipped_segments = filter(lambda x: x is not None, [halfspace.clip_segment(segment) for segment in tri_segments])
                    if len(clipped_segments) == 2:
                        new_triangles.append(Triangle(clipped_segments[0].p0, clipped_segments[1].p0, clipped_segments[0].p1))
                    elif len(clipped_segments) == 3:
                        crossing_segments = list()
                        for seg, clipped_seg in zip(tri_segments, clipped_segments):
                            if not (halfspace.contains(seg.p0) and halfspace.contains(seg.p1)):
                                crossing_segments.append(clipped_seg)
                        assert len(crossing_segments) == 0 or len(crossing_segments) == 2

                        if len(crossing_segments) == 2:
                            new_triangles.append(Triangle(crossing_segments[0].p0, crossing_segments[0].p1, crossing_segments[1].p0))
                            new_triangles.append(Triangle(crossing_segments[0].p1, crossing_segments[1].p0, crossing_segments[1].p1))
                        else:
                            new_triangles.append(tri)

            triangles = new_triangles

        return triangles

    def plot(self, rave_env, fill=False, with_sides=True, color=(1,0,0), alpha=0.25):
        """
        :param rave_env: RaveEnv instance
        :param fill: if True, colors the faces
        :param with_sides: if True, plots side edges too
        :param color: (r,g,b) [0,1]
        :param alpha: if fill is True, alpha of faces
        """
        a0, b0, c0, d0 = self.a0, self.b0, self.c0, self.d0
        a1, b1, c1, d1 = self.a1, self.b1, self.c1, self.d1

        if with_sides:
            rave_env.plot_segment(a0, a1, color=color)
            rave_env.plot_segment(b0, b1, color=color)
            rave_env.plot_segment(c0, c1, color=color)
            rave_env.plot_segment(d0, d1, color=color)

        rave_env.plot_segment(a0, b0, color=color)
        rave_env.plot_segment(b0, c0, color=color)
        rave_env.plot_segment(c0, d0, color=color)
        rave_env.plot_segment(d0, a0, color=color)
        rave_env.plot_segment(a1, b1, color=color)
        rave_env.plot_segment(b1, c1, color=color)
        rave_env.plot_segment(c1, d1, color=color)
        rave_env.plot_segment(d1, a1, color=color)

        if fill:
            if with_sides:
                rave_env.plot_triangle((a0,a1,d1), color=color, alpha=alpha)
                rave_env.plot_triangle((a0,d0,d1), color=color, alpha=alpha)

                rave_env.plot_triangle((a0,a1,b1), color=color, alpha=alpha)
                rave_env.plot_triangle((a0,b0,b1), color=color, alpha=alpha)

                rave_env.plot_triangle((b0,b1,c1), color=color, alpha=alpha)
                rave_env.plot_triangle((b0,c0,c1), color=color, alpha=alpha)

                rave_env.plot_triangle((c0,c1,d1), color=color, alpha=alpha)
                rave_env.plot_triangle((c0,d0,d1), color=color, alpha=alpha)

            rave_env.plot_triangle((a0,b0,d0), color=color, alpha=alpha)
            rave_env.plot_triangle((b0,c0,d0), color=color, alpha=alpha)

            rave_env.plot_triangle((a1,b1,d1), color=color, alpha=alpha)
            rave_env.plot_triangle((b1,c1,d1), color=color, alpha=alpha)

class Triangle:
    def __init__(self, a, b, c):
        self.a, self.b, self.c = np.array(a), np.array(b), np.array(c)

    def align_with(self, target):
        """
        Aligns the normal of this triangle to target

        :param target: 3d list or np.array
        :return (rotated triangle, rotation matrix)
        """
        target = np.array(target)
        source = np.cross(self.b - self.a, self.c - self.a)
        source /= np.linalg.norm(source)

        rotation = np.eye(3)

        dot = np.dot(source, target)
        if not np.isnan(dot):
            angle = np.arccos(dot)
            if not np.isnan(angle):
                cross = np.cross(source, target)
                cross_norm = np.linalg.norm(cross)
                if not np.isnan(cross_norm) and not cross_norm < epsilon:
                    cross = cross / cross_norm
                    rotation = tft.rotation_matrix(angle, cross)[:3,:3]

        return (Triangle(np.dot(rotation, self.a),
                        np.dot(rotation, self.b),
                        np.dot(rotation, self.c)),
                rotation)

    def closest_point_to(self, p):
        """
        Find distance to point p
        by rotating and projecting
        then return that closest point unrotated

        :param p: 3d list or np.array
        :return 3d np.array of closest point
        """
        p = np.array(p)
        # align with z-axis so all triangle have same z-coord
        tri_rot, rot = self.align_with([0,0,1])
        tri_rot_z = tri_rot.a[-1]
        p_rot = np.dot(rot, p)

        p_2d = p_rot[:2]
        tri_2d = geometry2d.Triangle(tri_rot.a[:2], tri_rot.b[:2], tri_rot.c[:2])

        if tri_2d.is_inside(p_2d):
            # projects onto triangle, so return difference in z
            return np.dot(np.linalg.inv(rot), np.array(list(p_2d) + [tri_rot_z]))
        else:
            closest_pt_2d = tri_2d.closest_point_to(p_2d)[1]

            closest_pt_3d = np.array(list(closest_pt_2d) + [tri_rot_z])

            return np.dot(np.linalg.inv(rot), closest_pt_3d)

    def distance_to(self, p):
        """
        Find distance to point p
        by rotating and projecting

        :param p: 3d list or np.array
        :return float distance
        """
        closest_pt = self.closest_point_to(p)
        return np.linalg.norm(p - closest_pt)

    def intersection(self, segment):
        """
        Determine point where line segment intersects this triangle
        - find intersection of segment with hyperplane
        - if intersection is in the triangle, return it

        :param segment: 3d line segment
        :return 3d np.array if intersection, else None
        """
        intersection = self.hyperplane.intersection(segment)
        if intersection is not None and np.linalg.norm(intersection - self.closest_point_to(intersection)) < epsilon:
            return intersection

        return None

    def closest_point_on_segment(self, segment): # TODO: incorrect
        hyperplane = self.hyperplane
        intersection = hyperplane.intersection(segment)

        if intersection is not None:
            return intersection
        else:
            if self.distance_to(segment.p0) < self.distance_to(segment.p1):
                return segment.p0
            else:
                return segment.p1

    @property
    def vertices(self):
        """
        :return list of np.array points
        """
        return [self.a, self.b, self.c]

    @property
    def segments(self):
        """
        :return list of Segment
        """
        return [Segment(self.a, self.b), Segment(self.b, self.c), Segment(self.c, self.a)]

    @property
    def hyperplane(self):
        """
        Returns hyperplane that this triangle is embedded in

        :return Hyperplane
        """
        origin = (self.a+self.b+self.c)/3.
        normal = np.cross(self.a-self.b, self.a-self.c)
        return Hyperplane(origin, normal)

    @property
    def area(self):
        """
        :return area of the triangle
        """
        tri_rot, rot = self.align_with([0,0,1])
        tri_2d = geometry2d.Triangle(tri_rot.a[:2], tri_rot.b[:2], tri_rot.c[:2])
        return tri_2d.area

class Segment:
    def __init__(self, p0, p1):
        self.p0, self.p1 = np.array(p0), np.array(p1)

    def closest_point_to(self, x):
        """
        min_{0<=t<=1} ||t*(p1-p0) + p0 - x||_{2}^{2}

        :param x: 3d list or np.array
        :return 3d np.array of closest point on segment to x
        """
        x = np.array(x)
        v = self.p1 - self.p0
        b = self.p0 - x

        t = -np.dot(v, b) / np.dot(v, v)
        if (0 <= t <= 1):
            closest = t*(self.p1 - self.p0) + self.p0
            return closest
        else:
            if np.linalg.norm(x - self.p0) < np.linalg.norm(x - self.p1):
                return self.p0
            else:
                return self.p1

    def distance_to(self, x):
        """
        Finds distance of closest point on segment to x
        :param x: 3d list or np.array
        :return float distance
        """
        return np.linalg.norm(np.array(x) - self.closest_point_to(x))

    def intersection(self, other):
        """
        Finds intersection point with another segment
        :param other: Segment
        :return None if no intersection, else [x,y] of intersection
        """
        p0_other, p1_other = other.p0, other.p1

        # w = p1 - p0
        # v = p1_other - p0_other
        # s*w + p0 = t*v + p0_other

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
            return intersection
        else:
            return None

    @property
    def length(self):
        return np.linalg.norm(self.p1 - self.p0)

class Line:
    def __init__(self, p0, p1):
        self.p0, self.p1 = np.array(p0), np.array(p1)

    def closest_points(self, other):
        """
        Finds the closest points on this line and other line to each other
        :param other: Line
        :return point on self, point on other
        """
        p0_other, p1_other = other.p0, other.p1

        # w = p1 - p0
        # v = p1_other - p0_other
        # s*w + p0 = t*v + p0_other

        w = self.p1 - self.p0
        v = p1_other - p0_other

        A = np.vstack((w,v)).T
        b = p0_other - self.p0

        #soln = np.linalg.solve(A, b)
        soln = np.linalg.pinv(A).dot(b)
        s, t = soln[0], -soln[1]

        return s*w + self.p0, t*v + p0_other

    def distance_to(self, other):
        """
        Finds minimum distance between this line and other
        :param other: Line
        :return distance
        """
        p_self, p_other = self.closest_points(other)
        return np.linalg.norm(p_self - p_other)

class Halfspace:
    def __init__(self, origin, normal):
        self.origin = origin
        self.normal = normal

    def contains(self, x):
        """
        :param x: 3d point as list or np.array
        :return True if x forms acute angle with plane normal, else False
        """
        return np.dot(self.normal, np.array(x) - self.origin) >= 0

    def clip_segment(self, segment):
        """
        :param segment Segment
        :return None if seg is not in halfspace, otherwise new Segment of part in halfspace
        """
        contains_p0 = self.contains(segment.p0)
        contains_p1 = self.contains(segment.p1)

        if contains_p0 and contains_p1:
            return Segment(segment.p0, segment.p1)
        elif not contains_p0 and not contains_p1:
            return None
        else:
            intersection = self.hyperplane.intersection(segment)
            assert intersection is not None
            if contains_p0:
                return Segment(intersection, segment.p0)
            else:
                return Segment(intersection, segment.p1)

    @property
    def hyperplane(self):
        """
        :return Hyperplane that defines Halfspace
        """
        return Hyperplane(self.origin, self.normal)

    @property
    def complement(self):
        """
        :return Halfspace corresponding to other half
        """
        return Halfspace(self.origin, -self.normal)


class Hyperplane:
    def __init__(self, origin, normal):
        self.origin = origin
        self.normal = normal

    def intersection(self, segment):
        """
        Finds intersection with a line segment

        :segment segment: 2d line segment
        :return 2d np.array, or None if no intersection
        """
        p0, p1 = segment.p0, segment.p1

        # x = t*(p1 - p0) + p0
        # n'*(x - origin) = 0
        # combine to get
        # n'*(t*(p1-p0) + p0 - origin) = 0
        # solve for t

        v = p1 - p0
        w = p0 - self.origin
        t = -np.dot(self.normal, w)/np.dot(self.normal, v)

        if 0-epsilon <= t <= 1+epsilon:
            return t*(p1-p0) + p0
        else:
            return None
