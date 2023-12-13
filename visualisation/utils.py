from shapely.geometry import Polygon
from shapely import normalize
from rasterio.warp import transform_geom
from rasterio.crs import CRS
from shapely.geometry import shape

def pol_from_points(t):
    x1, y1, x2, y2 = t
    return normalize(Polygon(((x1,y1),(x1,y2),(x2,y2),(x2, y1), (x1, y1))))

def wrap_pol(pol, epsg1, epsg2):
    
    return shape(transform_geom(CRS.from_epsg(epsg1), CRS.from_epsg(epsg2), pol))

def make_square(polygon):
    centroid = polygon.centroid
    envelope = polygon.envelope
    minx, miny, maxx, maxy = envelope.bounds

    width = maxx - minx
    height = maxy - miny

    side_length = max(width, height)
    cx, cy = centroid.x, centroid.y
    half_side = side_length / 4

    return Polygon([
    (cx - half_side, cy - half_side),
    (cx + half_side, cy - half_side),
    (cx + half_side, cy + half_side),
    (cx - half_side, cy + half_side)
])



