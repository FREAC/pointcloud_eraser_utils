# definitions for some implementations, defaults to a gaussian right now if you call fill_void_polygon
import laspy
import numpy as np
import os
import math
from dataclasses import dataclass
from scipy.spatial import KDTree
from scipy.interpolate import (
    CloughTocher2DInterpolator,
    LinearNDInterpolator,
    NearestNDInterpolator,
)
from CSF import CSF, VecInt
import geopandas as gp
from shapely.geometry import Polygon, MultiPolygon
import shapely
from typing_extensions import Literal
from io import BytesIO
import pyproj
import numpy as np
from KDEpy import TreeKDE
from warnings import warn
from copy import deepcopy
from types import NoneType
from typing_extensions import Any, NamedTuple
from collections import namedtuple
import time
from metpy.interpolate import natural_neighbor_to_points
from z_order import z_order_3d


@dataclass
class PointSampling:
    """A class encapsulating some rough point statistics , mean , std , var"""

    mean: float
    std: float
    var: float
    n: int


@dataclass
class PolygonExcision:
    """A class encapsulating the results of a point excision operation (removing points and reinterpolating based on a polygon)"""

    excised_cloud: np.ndarray  # (N,3)
    replacement_points: np.ndarray
    excised_points: np.ndarray
    excised_indexes: (
        np.ndarray
    )  # the indexes that were removed from the ORIGINAL cloud, you can use this to remove attributes besides goemetry and refill in
    retained_indexes: np.ndarray


LasPoly = namedtuple("LasPoly", ["lasdata", "poly"])


@dataclass
class DimInterp:
    """A dataclass that holds reinterpolated dimension values. 'cutout_cloud_dimension' is the dimension values for the modified cloud without reinterp values.
    'interpolated_values' are the interpolated values for the proposed patch.
    """

    cutout_cloud_dimension: np.ndarray
    interpolated_values: np.ndarray
    dim_name: str
    info: laspy.DimensionInfo


def parse_las_and_poly(
    las_file: str | BytesIO | laspy.LasData,
    poly_mask: str | BytesIO | shapely.Geometry | shapely.GeometryCollection,
) -> LasPoly:
    """parse and return lasdata and excison polygons from a few diff sources

    Args:
        las_file (str | BytesIO | laspy.LasData): data
        poly_mask (str | BytesIO | shapely.Geometry | shapely.GeometryCollection): polys

    Returns:
        LasPoly: the LasData and the poly array
    """

    orig_las_data: laspy.LasData | None = None
    removal_polygons: np.ndarray[shapely.Geometry] | None = None

    if isinstance(las_file, str) or isinstance(las_file, BytesIO):
        orig_las_data = laspy.read(las_file)
    else:
        orig_las_data = las_file

    assert isinstance(
        orig_las_data, laspy.LasData
    ), "well , this isnt a LasData object - that's wrong."

    if isinstance(poly_mask, str) or isinstance(poly_mask, BytesIO):
        gdf: gp.GeoDataFrame = gp.read_file(poly_mask)

        las_crs: pyproj.CRS = orig_las_data.header.parse_crs()

        if not las_crs.is_exact_same(gdf.crs):
            gdf = gdf.to_crs(las_crs)

        removal_polygons = gdf.geometry.array.to_numpy()  # (N,)
    else:

        assert isinstance(poly_mask, shapely.GeometryCollection) or isinstance(
            poly_mask, shapely.Geometry
        ), "'poly_mask' not a shapely geom or readable, thats not right."

        if isinstance(poly_mask, shapely.GeometryCollection):
            removal_polygons = np.array(list(poly_mask.geoms))
        else:
            removal_polygons = np.array([poly_mask])

    shapely.prepare(removal_polygons)

    return LasPoly._make([orig_las_data, removal_polygons])


def avg_dist_across_axis(pc: np.ndarray, axis: int = 0) -> PointSampling:
    """Computes the average distance from every unique coordinate
    to its nearest non-self unique neighbor across a given axis.

    Args:
        pc (np.ndarray): the (N, 3) pointcloud
        axis (int, optional): the axis to compute the stats over. Defaults to 0.

    Returns:
        tuple: (PointSampling, distances)
    """
    assert axis in (0, 1, 2), f"invalid axis: {axis} specified"

    kdtree = KDTree(data=pc)

    _, indices = kdtree.query(x=pc, k=2, workers=max(1, os.cpu_count() // 2))

    nn = pc[indices[..., 1]]

    delta = np.abs(pc[..., axis] - nn[..., axis])

    mean_dist = np.mean(delta)
    std_dist = np.std(delta)
    variance = std_dist**2

    return PointSampling(mean=mean_dist, std=std_dist, var=variance, n=len(pc))


def find_exemplar_patch(
    pc: np.ndarray, area_bounds: np.ndarray, return_points: bool = False
) -> np.ndarray:
    """finds a patch of points in 'pc' adjacent to 'area_bounds' that can be used as a sub possibly for 'area_bounds', 'area_bounds' is expected to be the bbox of the place we want to find a similar nearby exemplar for. This is done just using the nearest contiguous patch of same size as 'bounds' , if can't find one raises error. returns bounds of new patch as [minx,miny,maxx,maxy]

    Args:
        pc (np.ndarray): the full pc to search in
        area_bounds (np.ndarray): the bounds of the area we want to find exemplar for
        return_points (bool): whether to return the points themselves. Defaults to True.

    Returns:
        np.ndarray: bounds of exemplar patch of same size as 'area_bounds' or the points inside those bounds themselves if 'return_points' is true.
    """

    pc_bounds = np.array(
        [pc[..., 0].min(), pc[..., 1].min(), pc[..., 0].max(), pc[..., 1].max()]
    )

    diffs = np.array(
        [  # check all four adjacent sides
            area_bounds[2] - area_bounds[0],
            area_bounds[3] - area_bounds[1],
            -1 * (area_bounds[2] - area_bounds[0]),
            -1 * (area_bounds[3] - area_bounds[1]),
        ]
    )  # (x , y) diff

    for idx, diff in enumerate(
        diffs
    ):  # move the square to the 4 areas around the cutout bounds and see of any of these is fully inside the bit

        bounds = None

        if idx % 2 == 0:  # is x
            bounds = np.where(
                np.arange(len(area_bounds)) % 2 == 0, area_bounds - diff, area_bounds
            )
        else:
            bounds = np.where(
                np.arange(len(area_bounds)) % 2 != 0, area_bounds - diff, area_bounds
            )

        poly_bounds = shapely.geometry.box(*bounds)

        if shapely.contains(shapely.geometry.box(*pc_bounds), poly_bounds):
            if not return_points:
                return bounds
            else:
                pts = pc[
                    shapely.intersects_xy(geom=poly_bounds, x=pc[..., 0], y=pc[..., 1])
                ].copy()

                return pts

    raise AssertionError(
        "None of the adjacent bbox squares of the same size as bbox are inside the pointcloud."
    )


def compute_point_intensity_mesh(
    mesh: np.ndarray,
    cloud: np.ndarray,
    search_radius: float = 1,
    return_kde: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """computes an intensity map over a 2d mesh and a 3d pointcloud by casting all points to 2D and then checking NN for each point in the 2D mesh

    Args:
        mesh (np.ndarray): 2D mesh as (N , 2)
        cloud (np.ndarray): a pointcloud
        search_radius (float, optional): radius to search for . Defaults to 0.5
        return_kde (bool): Whether to also return a gaussian KDE evaulated at each mesh point as a 3rd item in the return tuple
    Returns:
        tople[np.ndarray , np.ndarray] | tople[np.ndarray , np.ndarray , np.ndarray]
    """
    assert (
        mesh.shape[-1] == 2
    ), "Mesh expected to be 2D grid of points of shape (X * Y , 2), similar to output of np.meshgrid but flattened"

    pc = cloud[..., 0:2]

    if return_kde:

        scale_x = pc[..., 0].min()
        scale_y = pc[..., 1].min()

        normed_pc = pc - np.array([scale_x, scale_y])

        m_scale_x = mesh[..., 0].min()
        m_scale_y = mesh[..., 1].min()

        normed_mesh = mesh - np.array([m_scale_x, m_scale_y])

        kde = (
            TreeKDE(kernel="gaussian", bw=round(search_radius * 0.33, 2))
            .fit(data=normed_pc)
            .evaluate(normed_mesh)
        )

    kdtree = KDTree(data=pc[..., 0:2])

    res = kdtree.query_ball_point(
        x=mesh,
        r=search_radius,
        return_length=True,
        workers=max(1, os.cpu_count() // 2),
    )

    return (
        (
            mesh,
            res / (math.pi * (search_radius**2)),
        )
        if not return_kde
        else (mesh, res / (math.pi * (search_radius**2)), kde)
    )


def build_grid(points: np.ndarray, grid_size: float = 0.1) -> np.ndarray:
    """Builds a grid based on the bounds of a reference array of shape at least ( N , 2 ), builds 2D grid

    Args:
        points (np.ndarray): (N, 2) or (N,3) array
        grid_size (float, optional): size of grid in coordinate units. Defaults to 0.1.

    Returns:
        np.ndarray : (N , 2)
    """

    assert (
        points.shape[-1] >= 2
    ), "The final dimension of points must be >= 2 (X, Y). Will take 0 & 1 dims as x & y"
    assert len(points.shape) == 2, "Need a (N , d) matrix."
    assert grid_size > 0, "Grid size must be positive."

    min_x, max_x = points[..., 0].min(), points[..., 0].max()
    min_y, max_y = points[..., 1].min(), points[..., 1].max()

    start_x = np.floor(min_x / grid_size) * grid_size
    end_x = np.ceil(max_x / grid_size) * grid_size

    start_y = np.floor(min_y / grid_size) * grid_size
    end_y = np.ceil(max_y / grid_size) * grid_size

    eps = grid_size * 0.1
    x_coords = np.arange(start_x, end_x + eps, grid_size)
    y_coords = np.arange(start_y, end_y + eps, grid_size)

    return np.stack(np.meshgrid(x_coords, y_coords)).reshape(2, -1).transpose(1, 0)


def anisotropic_kde_xy(
    points, query_points=None, k_neighbors=15, bandwidth=1.0, epsilon=1e-4
) -> Any:
    r"""fits 2d anisotropic KDE to 'points' and optionally queries the KDE at 'query_points' or returns eval and sample functions.

    equation is roughly:
    .. math::
    \hat{f}(\mathbf{x}) = \frac{1}{n(2\pi) \alpha^2} \sum_{i=1}^{n} \frac{1}{|\mathbf{\Sigma}_i|^{1/2}} \exp \left( -\frac{1}{2\alpha^2} (\mathbf{x} - \mathbf{x}_i)^T \mathbf{\Sigma}_i^{-1} (\mathbf{x} - \mathbf{x}_i) \right)

    Args:
        points (_type_): points to use for KDE
        query_points (_type_, optional): query. Defaults to None.
        k_neighbors (int, optional): num neighbors for coavriance matrix. Defaults to 15.
        bandwidth (float, optional): bandwidth param. Defaults to 1.0.
        epsilon (_type_, optional): stability. Defaults to 1e-4.

    Raises:
        ValueError: _description_

    Returns:
        Any
    """

    points = np.asarray(points)
    points_xy = points[:, :2]
    N, dim = points_xy.shape

    tree = KDTree(points_xy)
    _, indices = tree.query(points_xy, k=k_neighbors)

    covariances = np.zeros((N, dim, dim))
    inv_covariances = np.zeros((N, dim, dim))
    norm_constants = np.zeros(N)

    for i in range(N):
        neighbors = points_xy[indices[i]]
        centered = neighbors - np.mean(neighbors, axis=0)

        cov = np.cov(centered, rowvar=False)
        cov += np.eye(dim) * epsilon

        # Scale by bandwidth squared
        covariances[i] = cov * (bandwidth**2)

        # Precalculate inverse and determinant
        inv_covariances[i] = np.linalg.inv(covariances[i])
        det = np.linalg.det(covariances[i])

        # Precalculate the denominator of the Gaussian
        norm_constants[i] = np.sqrt(((2 * np.pi) ** dim) * det)

    def evaluate(q_points):
        q_points = np.atleast_2d(q_points)

        if q_points.shape[1] >= 2:
            q_points = q_points[:, :2]
        else:
            raise ValueError("Query points must have at least X and Y coordinates.")

        M = q_points.shape[0]
        densities = np.zeros(M)

        max_dist = 2 + (4 * bandwidth)
        query_indices = tree.query_ball_point(q_points, r=max_dist)

        for j in range(M):
            nearby_idx = query_indices[j]
            if not nearby_idx:
                continue

            local_pts = points_xy[nearby_idx]
            diff = q_points[j] - local_pts

            inv_covs = inv_covariances[nearby_idx]
            norm_consts = norm_constants[nearby_idx]

            mahalanobis = np.einsum("ki,kij,kj->k", diff, inv_covs, diff)

            probs = np.exp(-0.5 * mahalanobis) / norm_consts
            densities[j] = np.sum(probs)

        return densities / N

    def sample(n_samples=1):
        """Generates random samples from the fitted KDE distribution."""

        chosen_indices = np.random.choice(N, size=n_samples, replace=False)

        samples = np.zeros((n_samples, dim))

        for i, idx in enumerate(chosen_indices):
            mean = points_xy[idx]
            cov = covariances[idx]
            samples[i] = np.random.multivariate_normal(mean, cov)

        return samples

    if query_points is not None:
        return evaluate(query_points)

    return evaluate, sample


def z_order_cloud(
    las_file: str | BytesIO | laspy.LasData, output_path: str | None = None
) -> laspy.LasData:
    """modifies a cloud to have a z order on its points based on the xyz geometry

    Args:
        las_file (str | BytesIO | laspy.LasData): the las data to use or path to las/laz file.
        output_path (str | None, optional): path to write the order file to. Defaults to None.

    Returns:
        laspy.LasData: The modified las data. Also mods the params in place if is a lasdata object.
    """

    orig_las_data: laspy.LasData | None = None

    if isinstance(las_file, str) or isinstance(las_file, BytesIO):
        orig_las_data = laspy.read(las_file)
    else:
        orig_las_data = las_file

    order = z_order_3d(
        x=orig_las_data.X - orig_las_data.X.min(),
        y=orig_las_data.Y - orig_las_data.Y.min(),
        z=orig_las_data.Z - orig_las_data.Z.min(),
    )

    sort_indexes = np.argsort(order)

    orig_las_data.points.array = orig_las_data.points.array[sort_indexes]

    if output_path != None:
        orig_las_data.write(output_path)

    return orig_las_data


class PointCloudModifier:
    """This is just a thin and (admittedly poor) wrapper around implemented methods for object removal/modification in pointclouds."""

    def __init__(self):
        return

    @staticmethod
    def generate_points_across_bounds_kde(
        ground_neighborhood: np.ndarray,
        bounds: np.ndarray,
        num: int = None,
        bandwidth: float | None = None,
        k_n: int = None,
    ) -> np.ndarray:
        """fit a kde to neighborhood and sample from it. Return points inside 'bounds' like 'neighborhood'. In 2D. Basically we just make xmin and ymin of neighborhood map to xmin,ymin of bounds, so you may want to make the two similar in size for full coverage.

        Args:
            ground_neighborhood (np.ndarray): (N,2), probably should make this a square neighborhood greater than or = to 'bounds' in size... , ideally just do whole pc for this since we use an exemplar.
            bounds (np.ndarray): bbox to return sampled points inside of. expected [xmin , ymin , xmax , ymax].
            num (int): number of points to sample, defaults to the density of neighborhood avg. Defaults to None.


        Returns:
            np.ndarray: The sampled points shifted inside 'bounds'.
        """

        if bandwidth == None:  # a hueristic for bandwidth
            x_s = avg_dist_across_axis(pc=ground_neighborhood, axis=0)
            y_s = avg_dist_across_axis(pc=ground_neighborhood, axis=1)

            bandwidth = min(x_s.mean, y_s.mean) / 1.5

        if k_n is None:
            k_n = 10

        adjacent_exemplar_points = find_exemplar_patch(
            pc=ground_neighborhood, area_bounds=bounds, return_points=True
        )

        evaluate, sample = anisotropic_kde_xy(
            points=adjacent_exemplar_points,
            query_points=None,
            bandwidth=bandwidth,
            k_neighbors=k_n,
        )

        if num == None:
            inten = compute_point_intensity_mesh(
                mesh=build_grid(
                    points=adjacent_exemplar_points, grid_size=1
                ),  # does not contain the removed points... no hole, correct density
                cloud=adjacent_exemplar_points,
            )

            avg = np.mean(inten[1])

            expectation = round(avg * shapely.area(shapely.geometry.box(*bounds)))

            num = expectation

        samples_xy = sample(n_samples=num)

        neighborhood_xmin = samples_xy[..., 0].min()
        neighborhood_ymin = samples_xy[..., 1].min()

        # translation to align bottom left corner
        translate = np.array(
            [bounds[0] - neighborhood_xmin, bounds[1] - neighborhood_ymin]
        )

        # align bottom left
        samples_xy = samples_xy + translate

        return samples_xy

    @staticmethod
    def generate_points_across_bounds_exemplar(
        ground_neighborhood: np.ndarray, bounds: np.ndarray
    ) -> np.ndarray:
        """grabs an adjacent exemplar to 'bounds' if one exists. raises aerror if there is not enough adacent space in pointcloud to get an exemplar adjacent to 'bounds'. Returns points shiifted into coordinate space of 'bounds'.

        Args:
            ground_neighborhood (np.ndarray): pointcloud neighborhood to consider, should be able to accomodate another area of size 'bounds' directly adjacent to bounds, else we wont be albe to get an adjacent exemplar.
            bounds (np.ndarray): bbox of patch

        Raises:
            AssertionError: When an adjacent exemplar to bounds cannot be found due to pointcloud neighborhood size.

        Returns:
            np.ndarray: the adjacent area shifted into the coordinate system of bounds
        """

        patch = find_exemplar_patch(
            pc=ground_neighborhood, area_bounds=bounds, return_points=True
        )[..., 0:2]

        translate = np.array(
            [bounds[0] - patch[..., 0].min(), bounds[1] - patch[..., 1].min()]
        )

        return patch + translate

    @staticmethod
    def estimate_scan_vectors(
        pc: np.ndarray,
        gps_times: np.ndarray,
        window_size: int = 21,
        normalize: bool = False,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Computes instantaneous scan direction vectors using spatio-temporal
        regression on time-ordered sliding windows. An approximation of dxy/dt.

        Args:
            pc (np.ndarray): Point cloud array of shape (N, 3).
            gps_times (np.ndarray): GPS timestamps array of shape (N,).
            window_size (int): Temporal window size (must be odd).
            normalize (bool): Whether to return unit vectors. Defaults to False.

        Returns:
            tuple[np.ndarray, np.ndarray]:
                - Centroids of shape (N_windows, 3)
                - Unit scan vectors of shape (N_windows, 3).
        """
        assert window_size % 2 == 1, "window_size must be an odd integer."

        sort_idx = np.argsort(gps_times)
        t_sorted = gps_times[sort_idx]
        pc_sorted = pc[sort_idx]

        pc_windows = np.lib.stride_tricks.sliding_window_view(
            pc_sorted, window_shape=(window_size, 3)
        )[:, 0, :, :]

        t_windows = np.lib.stride_tricks.sliding_window_view(
            t_sorted, window_shape=window_size
        )

        # temporal and spatial means per window
        t_mean = np.mean(t_windows, axis=1, keepdims=True)  # (num_windows, 1)
        pc_mean = np.mean(pc_windows, axis=1, keepdims=True)  # (num_windows, 1, 3)

        # center coordinates and time
        dt = t_windows - t_mean  # (num_windows, window_size)
        d_pc = pc_windows - pc_mean  # (num_windows, window_size, 3)

        # spatial-cemporal cross-covariance Vector (v = sum(dt * d_pc) / sum(dt^2))
        numerator = np.sum(dt[..., np.newaxis] * d_pc, axis=1)  # (num_windows, 3)
        denominator = np.sum(dt**2, axis=1, keepdims=True)  # (num_windows, 1)

        v_scan = numerator / np.maximum(denominator, 1e-12)

        norms = np.linalg.norm(v_scan, axis=1, keepdims=True)

        centroids = pc_mean.squeeze(axis=1)

        return centroids, (v_scan / np.maximum(norms, 1e-12)) if normalize else v_scan

    @staticmethod
    def generate_points_across_bounds_sampled(
        ground_neighborhood: np.ndarray,
        bounds: np.ndarray,
        add_noise: bool = True,
        winnnow: bool = True,
    ) -> np.ndarray:
        """generates XY synthetic points over 'bounds' that are similar to the point densities/variances found in 'neighborhood' in the x/y direction. This really only works well for more or less gridded data like from a drum scanner.
        Returns (n , 2) x/y points. You should only pass ground filtered areas.

        Args:
            neighborhood (np.ndarray): The legitimate points, expected to be ground points.
            bounds (np.ndarray): The bounds of the void we need to fill inside neighborhood simplified to a bounds array, xmin,ymin,xmax,ymax
            add_noise (bool): If true, sample neighborhood points and add noise to the generated grid similar to those points. Defaults to True.
            winnow (bool): If true, winnow points to match neighborhood intensities over 1 unit grid using circular neighborhood of area 1. Defaults to True.

        Returns:
            np.ndarray: (n , 2)
        """

        assert (
            bounds[0] >= ground_neighborhood[..., 0].min()
            and bounds[1] >= ground_neighborhood[..., 1].min()
            and bounds[2] <= ground_neighborhood[..., 0].max()
            and bounds[3] <= ground_neighborhood[..., 1].max()
        ), "Your bounding box is outside of the neighborhood bounds."

        generator = np.random.default_rng(round(time.process_time()))

        local_point_x_stats = avg_dist_across_axis(pc=ground_neighborhood, axis=0)
        local_point_y_stats = avg_dist_across_axis(pc=ground_neighborhood, axis=1)

        # we generate starting at the x_min side and move to x_max of bounds
        # will have to be a bit careful about the edge near x_max, perhaps avg point is closer than 2std from mean we translate line back until it is no longer the case?

        x_lines = np.arange(bounds[0], bounds[2], local_point_x_stats.mean)
        y_lines = np.arange(bounds[1], bounds[3], local_point_y_stats.mean)

        mesh = (
            np.stack(np.meshgrid(x_lines, y_lines), axis=0)
            .reshape(2, -1)
            .transpose(1, 0)
        )  # (n , 2)

        if add_noise:
            point_x_procession = (
                generator.normal(
                    loc=local_point_x_stats.mean,
                    scale=local_point_x_stats.std,
                    size=len(mesh),
                )
                - local_point_x_stats.mean
            )
            point_y_procession = (
                generator.normal(
                    loc=local_point_y_stats.mean,
                    scale=local_point_y_stats.std,
                    size=len(mesh),
                )
                - local_point_y_stats.mean
            )

            mesh[..., 0] = mesh[..., 0] + point_x_procession
            mesh[..., 1] = mesh[..., 1] + point_y_procession

        if winnnow:
            local_ground_point_intensity = compute_point_intensity_mesh(
                mesh=build_grid(points=ground_neighborhood, grid_size=1),
                cloud=ground_neighborhood,
                search_radius=(1 / math.pi),
            )

            avg_inten = np.mean(local_ground_point_intensity[1])

            expected_num_objs_rem = round(
                avg_inten * shapely.area(shapely.geometry.box(*bounds))
            )

            indexes_to_keep = generator.choice(
                a=np.arange(0, len(mesh) - 1, 1),
                size=expected_num_objs_rem,
                replace=False,
            )

            mesh = mesh[indexes_to_keep]

        return mesh

    @staticmethod
    def get_ground_points(
        pc: np.ndarray, cloth_res: int = 2, rigidness: int = 2
    ) -> np.ndarray[np.int32]:
        """Generates a ground point classification for a pointcloud. Uses cloth simulation filtering. Returns an array of ground point indexes for the passed 'pc' array / file.

        Args:
            pc (np.ndarray): the pointcloud area to classify ground
            cloth_res (int, optional): cloth resolution in crs units, use a crs with metres. Defaults to 2.
            rigidness (int, optional): cloth rigidness, higher means a stiffer cloth and less detail. Defaults to 2.

        Returns:
            np.ndarray[np.int32]: an array of indexes of ground points in 'neighborhood'
        """

        csf = CSF()
        csf.params.cloth_resolution = cloth_res
        csf.params.rigidness = rigidness
        csf.params.bSloopSmooth = False

        csf.setPointCloud(np.ascontiguousarray(pc))

        ground_ind = VecInt()
        non_ground = VecInt()

        csf.do_filtering(
            groundIndexes=ground_ind, offGroundIndexes=non_ground, exportCloth=False
        )

        return np.array(ground_ind, dtype=np.int32)

    @staticmethod
    def interpolate_z_axis(
        neighborhood: np.ndarray,
        query_points: np.ndarray,
        method: Literal["linear", "cubic", "nn"],
    ) -> np.ndarray:
        """Interpolates new z axis values for query points in a neighborhood using various interp algorithms.

        NOTE: when a query point falls outside of the convex hull of the neighborhood (if that happens) then we will reinterpolate a nan using nearest neighbor interpolation based on the nearest successful interpolated value.

        Args:
            neighborhood (np.ndarray): the beighborhood to consider for interpolation
            query_points (np.ndarray): the points to evaluate an interpolated z value at
            method (Literal[&quot;linear&quot; , &quot;cubic&quot; , &quot;nn&quot; ]): the interpolation method to use, 'nn' is natural neighbors.

        Raises:
            ValueError: invalid interp method

        Returns:
            np.ndarray: the reinterpolated z-axis for the query points as (N,)
        """

        z_vals = None

        if method == "linear":
            # do linear
            # z-axis interp
            interp = LinearNDInterpolator(
                points=neighborhood[..., 0:2],
                values=neighborhood[..., -1],
            )

            z_vals = interp(query_points[..., 0], query_points[..., 1])
        elif method == "cubic":
            # z-axis interp
            interp = CloughTocher2DInterpolator(
                points=neighborhood[..., 0:2],
                values=neighborhood[..., -1],
            )

            z_vals = interp(query_points[..., 0], query_points[..., 1])
        elif method == "nn":
            z_vals = natural_neighbor_to_points(
                points=neighborhood[..., 0:2],
                values=neighborhood[..., -1],
                xi=query_points[..., 0:2],
            )

        else:
            raise ValueError(f"Unknown interpolation method: {method}")

        if np.isnan(
            z_vals
        ).any():  # may have nans outside of convex hull, edge.... maybe just fill with nearest query

            nan_idxs = np.isnan(z_vals)

            nn = NearestNDInterpolator(
                x=query_points[..., 0:2][~nan_idxs], y=z_vals[~nan_idxs]
            )

            resampled_z = nn(query_points[..., 0:2][nan_idxs])

            z_vals[nan_idxs] = resampled_z

        return z_vals

    def fill_void_polygon(
        self,
        las_file: str | BytesIO | laspy.LasData,
        removal_polygon: str | BytesIO | shapely.Geometry | shapely.GeometryCollection,
        interp_method: Literal["cubic", "linear", "nn"] = "linear",
        fill_method: Literal["kde", "exemplar"] = "kde",
        buffer_dist: float | None = 20,
        ground_class: int = 2,
    ) -> PolygonExcision:
        """Extracts the points bounded by the input polygon(s) and then reinterpolates the holes in the ground surface using interpolated ground points to try and match surrounding terrain. Attributes are not modifed, only raw geometry is returned.

        Args:
            las_file (str | laspy.LasData): path to a .las or .laz file you want to use, or readable object-like or lasdata
            removal_polygon (str | BytesIO | shapely.Geometry | shapely.GeometryCollection): path (or readable) to a geopandas-compatible file containing mask polygon(s) where we want to remove points inside those polygons, reccomneded to be in same CRS as 'las_file' but will attempt to cast to 'las_file' CRS if mismatch. Does not support 3D. Will not cast CRS if passed shapely geom.
            interp_method (Literal[&quot;cubic&quot; , &quot;linear&quot; , &quot;nn&quot;], optional): The method used to close the resulting gap in ground points, 'cubic' is a cubic interpolation, 'linear' is linear surface reconstruction that fits interpolated qhull to the polygon to remove surrounding ground points, fast and pretty good on flat terrain. 'nn' is natural neighbor interpolation across the gap. Defaults to "linear".
            fill_method (Literal[&quot;kde&quot; , &quot;exemplar&quot;], optional): The point generation method to use in the xy plane, both generate points based on the first available local exemplar of the same size and adjacent to the bounds of the cutout polygon(s) buffered by 'buffer_dist'. 'kde' is a remix of that geometry using anisotropic kernel density estimation. 'exemplar' is just a cut and paste of the xy geometry from a neighbor patch.
            buffer_dist (float | None ): The max distance (in crs units) around a cutout geometry to use when interpolating, smaller means faster while sacrificing context. Do not make too large as it gets used to create the bounds for the exemplar geom for kde and direct patching. Default assumes crs in metres. Defaults to 50.
            ground_class (int | None): Class value for ground points in 'las_file'.

        Returns:
            PolygonExcision: The pointcloud with all points inside the 2D geometry(s) summarily removed from it, followed by the replacement 2D interpolated points and the old points - as well as the indexes removed in the original cloud. All of shape (N,3), except indexes (N,)
        """

        assert (
            isinstance(las_file, str)
            or isinstance(las_file, BytesIO)
            or isinstance(las_file, laspy.LasData)
        ), f"'las_file' of unexpected type {type(las_file)}"
        assert (
            isinstance(removal_polygon, str)
            or isinstance(removal_polygon, BytesIO)
            or isinstance(removal_polygon, shapely.Geometry)
            or isinstance(removal_polygon, shapely.GeometryCollection)
        ), f"polygon is of unexpected type {type(removal_polygon)}"

        las_data: laspy.LasData | None = None
        geoms: np.ndarray[Polygon | MultiPolygon] | None = None

        las_data, geoms = parse_las_and_poly(
            las_file=las_file, poly_mask=removal_polygon
        )

        full_ground_points = (
            las_data.xyz[las_data.classification == ground_class]
            if ground_class is not None
            else las_data.xyz
        )

        points: np.ndarray = (
            las_data.xyz
            if ground_class == None
            else las_data.xyz[las_data.classification == ground_class]
        )  # (N,3)

        for geom in geoms:
            ge: Polygon | MultiPolygon = geom

            poly_bounds = ge.bounds  # (minx, miny, maxx, maxy)

            assert (
                poly_bounds[0] >= points[..., 0].min()
                and poly_bounds[1] >= points[..., 1].min()
                and poly_bounds[2] <= points[..., 0].max()
                and poly_bounds[3] <= points[..., 1].max()
            ), "Your mask polygon's bounding box is outside of the las_data bounds."

        check_intersection_poly = deepcopy(
            geoms
        )  # the full polygonal representation of our area

        shapely.prepare(check_intersection_poly)

        keep_mask = (
            shapely.intersects_xy(
                check_intersection_poly[:, None], x=points[..., 0], y=points[..., 1]
            )
        ) == False

        keep_mask = np.all(a=keep_mask, axis=0)  # any intersection with a polygon

        # drop intersecting point records
        points = points[keep_mask]

        if keep_mask[keep_mask == False].size >= 0.5 * keep_mask.size:
            warn(
                "You are removing 50% or more of the points in your pointcloud...",
                category=RuntimeWarning,
            )

        removal_bounds: np.ndarray | None = shapely.bounds(geoms)

        # use the whole pointcloud as each geom's neighborhood
        las_bounds = (
            las_data.header.x_min,
            las_data.header.y_min,
            las_data.header.x_max,
            las_data.header.y_max,
        )

        # ok so the bboxes of our neighborhoods have been decided, lets grab those points inside those bboxes
        reinterpolated_points: np.ndarray | None = None

        # TODO: reinterpolate using provided params and tune reinterpolation method to have reanoable defaults on flatter terrain, may want to preform reinterpolation based on some  idk
        for idx, bounds in enumerate(removal_bounds):

            neighborhood_bounds_poly = (
                shapely.buffer(shapely.geometry.box(*bounds), buffer_dist)
                if buffer_dist != None
                else shapely.geometry.box(*las_bounds)
            )

            shapely.prepare(neighborhood_bounds_poly)

            neighborhood_points_mask = shapely.intersects_xy(
                geom=neighborhood_bounds_poly,
                x=points[..., 0],
                y=points[..., 1],
            )

            neighborhood_ground_points = points[neighborhood_points_mask]

            # extract the

            patch = (
                self.generate_points_across_bounds_kde(
                    ground_neighborhood=full_ground_points, bounds=bounds
                )
                if fill_method == "kde"
                else self.generate_points_across_bounds_exemplar(
                    ground_neighborhood=full_ground_points, bounds=bounds
                )
            )

            fill = np.zeros((len(patch), 3))
            fill[..., 0:2] = patch

            fill[..., -1] = self.interpolate_z_axis(
                neighborhood=neighborhood_ground_points,
                query_points=fill,
                method=interp_method,
            )

            reinterpolated_points = (
                np.concat([reinterpolated_points, fill])
                if not isinstance(reinterpolated_points, NoneType)
                else fill
            )

        reinterpolation_keep_mask = shapely.intersects_xy(
            check_intersection_poly[:, None],
            x=reinterpolated_points[..., 0],
            y=reinterpolated_points[..., 1],
        )

        reinterpolated_points = reinterpolated_points[
            np.any(reinterpolation_keep_mask, axis=0)
        ]

        full_cloud_mask = shapely.intersects_xy(
            geom=check_intersection_poly, x=las_data.xyz[..., 0], y=las_data.xyz[..., 1]
        )

        return PolygonExcision(
            excised_cloud=las_data.xyz[~full_cloud_mask],
            replacement_points=reinterpolated_points,
            excised_points=las_data.xyz[full_cloud_mask],
            excised_indexes=np.arange(len(las_data.xyz))[full_cloud_mask],
            retained_indexes=np.arange(len(las_data.xyz))[~full_cloud_mask],
        )
        # TODO: buffer dist includes the area to be removed....

    @staticmethod
    def estimate_ground_attribute_nn(
        las_data: laspy.LasData,
        excision: PolygonExcision,
        attr_name: str,
        use_only_ground: bool = True,
        ground_class: int = 2,
        zero_out_dim: bool = False,
    ) -> DimInterp:
        """interpolates a dimension of the las_data specified by 'attr_name' using nearest neighbor interpolation based on surrounding points. Returns dim values for the rest of the cloud (sans the patch, so the cutout cloud dim values) as well as interpolated values for the patch.

        Args:
            las_data (laspy.LasData): the las data to use
            excision (PolygonExcision): the previously excised polygon return and associated
            attr_name (str): the LasData attribute to reinterpolate based on nn queries
            use_only_ground (bool, optional): whether to use only ground classified points for this or simply the nearest neighbor. Defaults to True.
            ground_class (int, optional): the class vakue for ground in LasData. Defaults to 2.
            zero_out_dim (bool , optional): instead just 0 out the intersecting polygon dimensions..

        Returns:
            DimInterp: the reinterpolated attribute spanning the full range of the reinterp cloud (excised_cloud + replacement_points) as a data object
        """

        unmod_indexes = np.arange(len(las_data.xyz))

        if use_only_ground:
            unmod_indexes = unmod_indexes[las_data.classification == ground_class]

        unmod_attr_values = las_data[attr_name][unmod_indexes]

        if zero_out_dim:  # just skip the reinterp and pass a 0 array as the fill back
            return DimInterp(
                cutout_cloud_dimension=np.delete(
                    las_data[attr_name], excision.excised_indexes
                ),
                interpolated_values=np.zeros(
                    shape=unmod_attr_values.shape, dtype=unmod_attr_values.dtype
                ),
                dim_name=attr_name,
            )

        # geom to attribute value
        nn_interp = NearestNDInterpolator(
            x=las_data.xyz[unmod_indexes], y=unmod_attr_values
        )

        nn_interp = nn_interp(excision.replacement_points).astype(
            unmod_attr_values.dtype
        )

        return DimInterp(
            cutout_cloud_dimension=np.delete(
                las_data[attr_name], excision.excised_indexes
            ),
            interpolated_values=nn_interp,
            dim_name=attr_name,
            info=las_data.header.point_format.dimension_by_name(attr_name),
        )

    @staticmethod
    def get_z_bounds(geom):
        """Returns (min_z, max_z) for any 3D Shapely geometry."""
        z_coords = shapely.get_coordinates(geom, include_z=True)[:, 2]
        return z_coords.min(), z_coords.max()

    def summary_poly_removal(
        self,
        las_file: str | BytesIO | laspy.LasData,
        poly_mask: str | BytesIO | shapely.Geometry | shapely.GeometryCollection,
        output_path: str | None = None,
    ) -> laspy.LasData:
        """Summarily removes ALL geometry and attribute entries associated with a polygon(s) found in the input and returns a new LasData object and/or writes to a new file. Supports 3D polygons (xyz) but assumes they are horizontal in orientation.

        Args:
            las_file (str | BytesIO | LasData): the las data to use, unmodified.
            removal_polygon (str | BytesIO | shapely.Geometry): the polygon(s) to remove all data inside of.
            output_path (str | None, optional): The output path to use. Defaults to None.

        Returns:
            laspy.LasData: The modifed LasData, if 'output_path' is set will write to a new file as well
        """

        orig_las_data: laspy.LasData | None = None
        removal_polygons: np.ndarray[shapely.Geometry] | None = None

        assert isinstance(output_path, str) or isinstance(
            output_path, NoneType
        ), "invalid output argument"

        orig_las_data, removal_polygons = parse_las_and_poly(
            las_file=las_file, poly_mask=poly_mask
        )

        intersects_mask: np.ndarray[np.bool] | None = np.zeros(
            len(orig_las_data.xyz), dtype=np.bool
        )

        for geom in removal_polygons:

            xy_mask = shapely.intersects_xy(
                geom=geom, x=orig_las_data.xyz[..., 0], y=orig_las_data.xyz[..., 1]
            )

            if shapely.has_z(geom):

                min_z, max_z = self.get_z_bounds(geom)

                z_mask = np.where(
                    (orig_las_data.xyz[..., 2] >= min_z)
                    & (orig_las_data.xyz[..., 2] <= max_z),
                    True,
                    False,
                )

                mask = xy_mask & z_mask  # np.all(np.array([xy_mask , z_mask]) , axis=0)

                intersects_mask = (
                    intersects_mask | mask
                )  # np.any(np.array([intersects_mask , mask]) , axis=0)
            else:
                intersects_mask = (
                    intersects_mask | xy_mask
                )  # np.any(np.array([intersects_mask , xy_mask]) , axis=0)

        # now have 2 | 3 D intersection mask, just filter out the points

        keep_mask = ~intersects_mask

        filtered_points = orig_las_data.points[keep_mask]

        new_file = laspy.LasData(orig_las_data.header)

        new_file.points = laspy.ScaleAwarePointRecord(
            filtered_points.array.copy(),
            point_format=orig_las_data.header.point_format,
            scales=orig_las_data.header.scales,
            offsets=orig_las_data.header.offsets,
        )

        if output_path != None:
            new_file.write(destination=output_path)

        new_file.update_header()
        return new_file

    def z_flatten_poly(
        self,
        las_file: str | BytesIO | laspy.LasData,
        poly_mask: str | BytesIO | shapely.Geometry | shapely.GeometryCollection,
        output_path: str | None = None,
        ground_class: int = 2,
        mark_as_withhed: bool = True,
        mark_as_ground_surface: bool = True,
        zero_attributes: bool = False,
        mark_as_first_return: bool = True,
    ) -> laspy.LasData:
        """Takes all geometry within a polygon and flattens it to the expected ground surface, optionally 0's all attributes inside the polygon. Ground surface expectation is done using linear TIN interpolation on existing ground points.

        Args:
            las_file (str | BytesIO | laspy.LasData): las data to use
            poly_mask (str | BytesIO | shapely.Geometry | shapely.GeometryCollection): the polygons to flatten inside of, suppoorts horizontally oriented 3D.
            output_path (str | None, optional): the path to write a new modified las file to. Defaults to None.
            ground_class (int): the class value for ground points in the file
            mark_as_withhed (bool): mark the flattend points as withheld. Defaults to True.
            mark_as_ground_surface (bool): whether to mark the 'classification' of the flattend area/points as the same class as 'ground_class'. Defaults to True.
            zero_attributes (bool, optional): whether to 0 out all the attributes of the flattened points, overrides 'mark_as_ground' but not 'mark_as_withheld'. Defaults to False.
            mark_as_first_return (bool , optional): whether to mark flattened points as first returns or leave as original.
        Returns:
            laspy.LasData: the modifed point cloud
        """

        orig_las_data: laspy.LasData | None = None
        removal_polygons: np.ndarray[shapely.Geometry] | None = None

        assert isinstance(output_path, str) or isinstance(
            output_path, NoneType
        ), "invalid output argument"

        orig_las_data, removal_polygons = parse_las_and_poly(
            las_file=las_file, poly_mask=poly_mask
        )

        intersects_mask: np.ndarray[np.bool] | None = np.zeros(
            len(orig_las_data.xyz), dtype=np.bool
        )

        for geom in removal_polygons:

            xy_mask = shapely.intersects_xy(
                geom=geom, x=orig_las_data.xyz[..., 0], y=orig_las_data.xyz[..., 1]
            )

            if shapely.has_z(geom):

                min_z, max_z = self.get_z_bounds(geom)

                z_mask = np.where(
                    (orig_las_data.xyz[..., 2] >= min_z)
                    & (orig_las_data.xyz[..., 2] <= max_z),
                    True,
                    False,
                )

                mask = xy_mask & z_mask  # np.all(np.array([xy_mask , z_mask]) , axis=0)

                intersects_mask = (
                    intersects_mask | mask
                )  # np.any(np.array([intersects_mask , mask]) , axis=0)
            else:
                intersects_mask = (
                    intersects_mask | xy_mask
                )  # np.any(np.array([intersects_mask , xy_mask]) , axis=0)

        # grab those points inside our polys
        intersecting_points = orig_las_data.points[intersects_mask]

        # now lets cast all of those points inside our file in the poly(s) onto this surface
        new_z_values = self.interpolate_z_axis(
            neighborhood=orig_las_data.xyz[
                orig_las_data.classification == ground_class
            ],
            query_points=orig_las_data.xyz[intersects_mask],
            method="linear",
        )

        intersecting_points.z = new_z_values

        # now have the new Z values, let see about zeroing out all the other features in our intersecting points copy

        if mark_as_ground_surface:
            intersecting_points["classification"] = np.array([ground_class])

        if zero_attributes:
            for dim in orig_las_data.header.point_format.dimensions:
                if not dim.name in ("X", "Y", "Z"):  # not geometry
                    intersecting_points[dim.name] = np.array([0])

        if mark_as_first_return:
            intersecting_points["return_number"] = np.array([1])
            intersecting_points["number_of_returns"] = np.array([1])

        if mark_as_withhed:
            intersecting_points["withheld"] = np.array([1])

        # now that everything is done, lets modify the original records

        modified_cloud = laspy.LasData(header=orig_las_data.header)

        modified_cloud.points = laspy.ScaleAwarePointRecord(
            orig_las_data.points.array.copy(),
            point_format=orig_las_data.header.point_format,
            scales=orig_las_data.header.scales,
            offsets=orig_las_data.header.offsets,
        )

        modified_cloud.points.array[intersects_mask] = (
            intersecting_points.array
        )  # = intersecting_points

        modified_cloud.update_header()

        if output_path != None:
            modified_cloud.write(output_path)

        return modified_cloud

    def excise_and_patch_polygon_stat(
        self,
        las_file: str | BytesIO | laspy.LasData,
        poly_mask: str | BytesIO | shapely.Geometry | shapely.GeometryCollection,
        output_path: str | None = None,
        ground_class: int = 2,
        interp_method: Literal["cubic", "linear", "nn"] = "linear",
        fill_method: Literal["kde", "exemplar"] = "kde",
        buffer_dist: float | None = None,
        mark_as_withhed: bool = True,
        mark_as_ground_surface: bool = True,
        zero_attributes: bool = False,
        reorder: bool = True,
        use_only_ground: bool = True,
    ) -> laspy.LasData:
        """Excises sumarily and refills using 'interp_method' for ground surface (z) and 'fill_method' for xy geometry.
        If 'zero_attributes' is set, will zero out all non-geom attributes of the modifed area, save for return_number and number_of_returns, and withheld if set.
        Else will interpolate attributes using nearest neighbor interpolation on surrounding ground surface, because attribute reinterpolation is done rather naively,
        this means that you probably shouldn't use this method on clouds with RGB values or other highly variable features.
        But for clouds with geometry and other standard features like intensity, this may produce satisfactory results.

        Args:
            las_file (str | BytesIO | laspy.LasData): The las/laz file to use
            poly_mask (str | BytesIO | shapely.Geometry | shapely.GeometryCollection): The polygon(s) to cutout. Does not support 3D.
            output_path (str | None, optional): optional path to write modified las/laz to. Defaults to None.
            ground_class (int, optional): the ground class of the input/output file. Defaults to 2.
            interp_method (Literal[&quot;cubic&quot;, &quot;linear&quot;, &quot;nn&quot;], optional): interpolation method for the 'z' axis of the reinterpolated ground surface. Defaults to "linear".
            fill_method (Literal[&quot;kde&quot;, &quot;exemplar&quot;], optional): the fill method for how reinterpolated ground is placed in the xy plane. 'kde' uses an anisotropic KDE of a surrounding exemplar ground surface, this amounts to a slight remix on local geometry. 'exemplar' is just a straight cut and paste of surrounding xy ground geometry. Defaults to "kde".
            buffer_dist (float | None, optional): the neighborhood distance around a cutout polygon to consider for ground surface interpolation, lower gives less context for the 'z' axis interpolation but runs faster as we fit a ground surface over less points. Defaults to None, the whole cloud.
            mark_as_withhed (bool, optional): mark the reinterpolated points as withheld or not. Defaults to True.
            mark_as_ground_surface (bool, optional): mark the reinterpolated points as 'ground_class' or not. Defaults to True.
            zero_attributes (bool, optional): zero all attributes besides geometry inside the cutout polygons (except 'withheld' if set). Defaults to False.
            reorder (bool , optional): reorder the geometry and attributes of the cloud according to z-order. This masks the patch in the order of the cloud and provides good performance in other software (indexing).
            use_only_ground (bool , optional): only use ground geometry for attribute interpolation, probably best to leave this as True. Defaults to True.
        Returns:
            laspy.LasData: the modified las data
        """

        assert isinstance(output_path, str) or isinstance(
            output_path, NoneType
        ), f"output_path is of unrecognized type: {type(output_path)}"
        assert isinstance(
            ground_class, int
        ), f"ground_class is unrecognized type: {type(ground_class)}"

        pars = parse_las_and_poly(las_file=las_file, poly_mask=poly_mask)

        las_data: laspy.LasData = pars.lasdata
        removal_polygons: np.ndarray[shapely.Geometry] = pars.poly

        geom_removed: PolygonExcision = self.fill_void_polygon(
            las_file=las_data,
            removal_polygon=poly_mask,
            interp_method=interp_method,
            fill_method=fill_method,
            buffer_dist=buffer_dist,
            ground_class=ground_class,
        )

        # ok , we went over the goemetry, now lets interpolate some features using nearest neighbor.

        reinterpolated_dims: dict[str, DimInterp] = {}

        for dim in las_data.header.point_format.dimensions:

            if dim.name in ("X", "Y", "Z"):
                continue

            interp_dim = self.estimate_ground_attribute_nn(
                las_data=las_data,
                excision=geom_removed,
                attr_name=dim.name,
                use_only_ground=use_only_ground,
                ground_class=ground_class,
                zero_out_dim=zero_attributes,
            )

            reinterpolated_dims[dim.name] = interp_dim

        # now have reinterpolated dims, patch all of it together

        # TODO: use dxy/dt for gps_times where available
        new_header = laspy.LasHeader(
            version=las_data.header.version, point_format=las_data.header.point_format
        )
        new_header.scales = las_data.header.scales
        new_header.offsets = las_data.header.offsets
        new_header.vlrs = las_data.header.vlrs
        new_header.evlrs = las_data.header.evlrs

        new_las_data = laspy.LasData(header=new_header)

        # ok, so lets do the geometry and then the attributes

        new_geom = np.concat(
            [geom_removed.excised_cloud, geom_removed.replacement_points], axis=0
        )

        new_las_data.x = new_geom[..., 0]
        new_las_data.y = new_geom[..., 1]
        new_las_data.z = new_geom[..., 2]

        if mark_as_withhed:
            reinterpolated_dims["withheld"].interpolated_values = np.ones_like(
                reinterpolated_dims["withheld"].interpolated_values
            )

        if (
            mark_as_ground_surface
        ):  # should all be ground surface anyway lol but to be sure lol, if using other geom...
            reinterpolated_dims["classification"].interpolated_values = np.zeros_like(
                reinterpolated_dims["classification"].interpolated_values
            )
            reinterpolated_dims["classification"].interpolated_values.fill(ground_class)

        # set as first/only returns either way
        reinterpolated_dims["number_of_returns"].interpolated_values = np.ones_like(
            reinterpolated_dims["number_of_returns"].interpolated_values
        )
        reinterpolated_dims["return_number"].interpolated_values = np.ones_like(
            reinterpolated_dims["return_number"].interpolated_values
        )

        # attributes
        for dim in reinterpolated_dims.keys():

            new_las_data[dim] = np.concat(
                [
                    reinterpolated_dims[dim].cutout_cloud_dimension,
                    reinterpolated_dims[dim].interpolated_values,
                ],
                axis=0,
            )

        if reorder:
            z_order_index = z_order_3d(
                x=new_las_data.X - new_las_data.X.min(),
                y=new_las_data.Y - new_las_data.Y.min(),
                z=new_las_data.Z - new_las_data.Z.min(),
            )

            sort_idxs = np.argsort(a=z_order_index)

            new_las_data.points.array = new_las_data.points.array[sort_idxs]

        new_las_data.update_header()

        if output_path != None:
            new_las_data.write(output_path)

        return new_las_data
