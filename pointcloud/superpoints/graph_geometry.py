"""Point-supported, hole-filled footprints for pre-ID mask graph candidates."""
from __future__ import annotations

import numpy as np
import shapely
from scipy.ndimage import binary_closing,binary_fill_holes
from rasterio.features import shapes
from rasterio.transform import Affine
from shapely.geometry import shape


def footprint(xy, resolution=.5):
    xy=np.asarray(xy,np.float64)
    if not len(xy):
        return shapely.GeometryCollection()
    origin=np.floor(xy.min(0)/resolution)*resolution
    grid=np.floor((xy-origin)/resolution).astype(np.int32)
    occupancy=np.zeros((grid[:,1].max()+5,grid[:,0].max()+5),bool)
    occupancy[grid[:,1]+2,grid[:,0]+2]=True
    filled=binary_fill_holes(binary_closing(occupancy,structure=np.ones((3,3)))|occupancy)
    transform=Affine(resolution,0,origin[0]-2*resolution,
                     0,resolution,origin[1]-2*resolution)
    pieces=[shape(geom) for geom,value in shapes(filled.astype(np.uint8),mask=filled,
                                                 transform=transform) if value]
    return shapely.union_all(pieces) if pieces else shapely.GeometryCollection()


def graph_crown_matches(arrays,graph,reference):
    """Best crown IoU per candidate, using only training/validation reference."""
    truth=list(reference.geometry)
    identifiers=(reference['treeID'].to_numpy() if 'treeID' in reference else
                 reference['tree_id'].to_numpy())
    index=shapely.STRtree(truth) if truth else None
    world=arrays['coord'][:,:2].astype(np.float64)+arrays['source_origin'][:2]
    best=np.zeros(len(graph['feature']),np.float32)
    matched=np.zeros(len(best),np.int64)
    area=np.zeros(len(best),np.float32)
    for i in range(len(best)):
        a,b=graph['offset'][i:i+2]
        geom=footprint(world[graph['point_index'][a:b]])
        area[i]=geom.area
        if index is None or geom.is_empty:
            continue
        for j in index.query(geom,predicate='intersects'):
            other=truth[int(j)]
            intersection=geom.intersection(other).area
            iou=intersection/max(geom.area+other.area-intersection,1e-9)
            if iou>best[i]:
                best[i]=iou
                matched[i]=identifiers[int(j)]
    return best,matched,area
