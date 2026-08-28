# PointCloudModifier

A library for point cloud object removal, surface interpolation, and void filling operations. Supports reading, modifying, and exporting `.las`/`.laz` files via `laspy`.

## API Documentation

### Point Cloud Manipulation

The main API of this repo is managed through a single **PointCloudModifier** instance, most of the methods listed below exist on an instance of **PointCloudModifier**.
Some methods, like **z_order_cloud** (which reorders the points in your point-cloud to conform to a z order for subsequent performance on disk) exist as standalone functions as well, but are mostly undocumented as the main functionality lies with removing sections of a point-cloud according to a polygon, flattening geometry of a point-cloud to ground surface inside a polygon as well as some minor methods to remove and replace geometry inside removal polygons if so desired.

an example usage, note that masks can contain multiple polygons and that any filetype supported by geopandas is also supported for mask polygons:
```python
point_modifier = utils.PointCloudModifier()

b = point_modifier.z_flatten_poly(las_file=file , poly_mask="path_to_poly(s).shp , zero_attributes=False  , output_path="s.laz")
```

**`summary_poly_removal(las_file, poly_mask, output_path=None) -> laspy.LasData`**
Removes all points and associated attributes within provided 2D/3D Shapely polygons.
*   **Args:** `las_file` (Path/LasData), `poly_mask` (Shapely Geometry/Collection), `output_path` (Optional export path).

**`z_flatten_poly(las_file, poly_mask, output_path=None, ...) -> laspy.LasData`**
Flattens all points within a polygon to the expected ground surface using linear TIN interpolation. Optionally resets attributes or marks points as withheld.
*   **Args:** `las_file`, `poly_mask`, `output_path`, `ground_class` (Default: 2), `mark_as_withhed` (bool), `mark_as_ground_surface` (bool), `zero_attributes` (bool), `mark_as_first_return` (bool).

**`excise_and_patch_polygon_stat(las_file, poly_mask, output_path=None, ...) -> laspy.LasData`**
Excises points within a 2D polygon and refills the void using interpolated ground surfaces (Z) and generated geometry (XY). Reinterpolates or zeroes attributes and optionally Z-order sorts the output cloud.
*   **Args:** `las_file`, `poly_mask`, `output_path`, `ground_class` (Default: 2), `interp_method` (`linear`|`cubic`|`nn`), `fill_method` (`kde`|`exemplar`), `buffer_dist`, `mark_as_withhed` (bool), `mark_as_ground_surface` (bool), `zero_attributes` (bool), `reorder` (bool), `use_only_ground` (bool).

### Attribute Interpolation

**`interpolate_z_axis(neighborhood, query_points, method) -> np.ndarray`**
Interpolates Z-axis values for 2D query points using surrounding neighborhood data.
*   **Args:** `neighborhood` (N,3 array), `query_points` (N,2 array), `method` (`linear`|`cubic`|`nn`).

**`estimate_ground_attribute_nn(las_data, excision, attr_name, use_only_ground=True, ground_class=2, zero_out_dim=False) -> DimInterp`**
Reconstructs missing point attributes (e.g., intensity, classification) across an excised region using nearest-neighbor interpolation.

### Utility

**`get_ground_points(pc, cloth_res=2, rigidness=2) -> np.ndarray[np.int32]`**
Executes Cloth Simulation Filtering (CSF) to classify and extract ground point indices from a point cloud.
