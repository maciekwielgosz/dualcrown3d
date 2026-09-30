"""Reversible, evidence-gated split/merge of vote instances by point masks."""
import numpy as np
from scipy.spatial import cKDTree

from pointcloud.dual_fusion import groups, vote_instances, _fuse, instance_records


def revise_anchors(xyz, vote_labels, mask_labels, mask_confidence, config):
    labels=vote_labels.copy()
    changes=np.zeros(len(labels),np.uint8)
    vote_groups={int(labels[m[0]]):m for m in groups(labels)}
    mask_groups={int(mask_labels[m[0]]):m for m in groups(mask_labels)}
    quality={k:float(mask_confidence[m].mean()) for k,m in mask_groups.items()}
    threshold=config.get('revision_confidence', .4)
    minimum=config.get('minimum_voxels',12)
    next_id=int(labels.max(initial=0))
    mode=config.get('revision_mode','both')
    # Splits require two independently supported, spatially separated masks.
    if mode in ('both','split'):
        for identifier,members in vote_groups.items():
            found,count=np.unique(mask_labels[members],return_counts=True)
            parts=[]
            for mid,n in zip(found,count):
                if mid==0 or n<minimum or quality[int(mid)]<threshold: continue
                if n/len(mask_groups[int(mid)])<.65: continue
                parts.append(members[mask_labels[members]==mid])
            if len(parts)<2 or sum(map(len,parts))/len(members)<.65:continue
            centers=np.asarray([np.median(xyz[p],axis=0) for p in parts])
            distance=np.linalg.norm(centers[:,None]-centers[None],axis=2)
            distance+=np.eye(len(parts))*1e9
            if distance.min()<1.:continue
            # Supported points define new components; nearby residuals follow
            # the closest 3D support, never just an XY crown footprint.
            distances=np.column_stack([cKDTree(xyz[p]).query(xyz[members])[0] for p in parts])
            chosen=distances.argmin(1)
            chosen[distances.min(1)>config.get('revision_residual_radius',2.)]=int(np.argmax(list(map(len,parts))))
            for part in range(len(parts)):
                if part==0:new_id=identifier
                else:next_id+=1;new_id=next_id
                labels[members[chosen==part]]=new_id
            changes[members]=6
    # A coherent mask may replace several fragmented vote instances. Require
    # mutual coverage so a large bridge mask cannot merge unrelated neighbours.
    if mode in ('both','merge'):
        current={int(labels[p[0]]):p for p in groups(labels)}
        for mid,members in sorted(mask_groups.items(),key=lambda item:-quality[item[0]]):
            if quality[mid]<max(threshold,.5):continue
            ids,count=np.unique(labels[members],return_counts=True)
            ids,count=ids[ids>0],count[ids>0]
            if len(ids)<2:continue
            parts=[current[int(i)] for i in ids]
            if any((changes[p]==6).any() for p in parts):continue
            if any(n/len(p)<.75 for n,p in zip(count,parts)):continue
            if count.sum()/len(members)<.75:continue
            centers=np.array([np.median(xyz[p,:2],axis=0) for p in parts])
            if np.linalg.norm(centers[:,None]-centers[None],axis=2).max()>3.:continue
            all_members=np.concatenate(parts)
            labels[all_members]=ids.min();changes[all_members]=7
            for identifier in ids: current.pop(int(identifier))
            current[int(ids.min())]=all_members
    return labels,changes


def adaptive_consensus(arrays,raw,mask_labels,mask_confidence,config):
    votes=vote_instances(arrays,raw,config['vote_cluster_config'])
    anchors,changes=revise_anchors(arrays['coord'],votes,mask_labels,mask_confidence,config)
    labels,confidence,_,source=_fuse(arrays,raw,anchors,raw['tree_probability'],config,mask_labels,
        anchor_source=3,new_source=5,mask_support=mask_labels>0,
        complement_probability=raw['point_probability'],complement_confidence=mask_confidence)
    take=(source==3)&(changes>0);source[take]=changes[take]
    # Export requires consecutive scene IDs, including after merges.
    unique=np.unique(labels[labels>0]);dense=np.zeros(int(labels.max(initial=0))+1,np.uint32)
    dense[unique]=np.arange(1,len(unique)+1,dtype=np.uint32);labels=dense[labels]
    return labels,confidence,instance_records(arrays,labels,confidence),source
