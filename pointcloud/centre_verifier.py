"""Stage 2b: a learned verifier over the candidate tree centres.

The vote-field detector scores each cell from a local 3-D view. Whether a
candidate is a real tree also depends on things only visible at the level of the
whole candidate set: is a stronger candidate claiming the same votes, does the
candidate sit in the middle of its voters or on their flank, is it a fragment of
a stage-1 tree. The verifier scores every candidate (detections and stage-1
centres alike) from such features, which lets the second pass add, relocate and
reject in one learned decision. With a perfect verifier the epoch-16 detector
would reach 0.66 point SB-PQ on dense validation against 0.45 without one.
"""
from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree

from pointcloud.vote_field import ASSIGN, SCALE, centre_limits

FEATURE_NAMES = (
    'score', 'extent', 'height', 'is_stage1_centre', 'has_detection',
    'n1_dist_over_extent', 'n1_dist_over_its_extent', 'n1_score', 'n1_score_delta', 'n1_height_delta',
    'n2_dist_over_extent', 'n2_score', 'n3_dist_over_extent', 'n3_score',
    'stronger_within_extent', 'candidates_within_extent', 'candidates_within_2_extent', 'covered_by_stronger',
    'votes_05', 'votes_1', 'votes_extent', 'votes_1_share', 'tree_probability', 'vote_spread',
    'voter_height_mean', 'voter_height_std', 'voter_height_max', 'voter_radius', 'voter_offcentre', 'voter_offcentre_ratio',
    'voter_travel', 'exclusive_votes', 'exclusive_share',
    'stage1_assigned_share', 'stage1_purity', 'stage1_capture', 'stage1_labels', 'stage1_centre_distance',
    'stage1_centre_height_delta', 'local_density')


def build_candidates(detection, anchors, minimum_score=.1, xy=1., z=3.):
    """Detections plus the stage-1 centres no detection sits on.

    Returns centres [K, 3], scores, extents (height rule for stage-1 centres),
    ``is_anchor`` and ``has_detection`` flags.
    """
    keep = detection['scores'] >= minimum_score
    centres, scores, extents = detection['centres'][keep], detection['scores'][keep], detection['extents'][keep]
    anchors = np.asarray(anchors, np.float32).reshape(-1, 3)
    lonely = np.ones(len(anchors), bool)
    if len(centres) and len(anchors):
        tree = cKDTree(centres[:, :2])
        for i, anchor in enumerate(anchors):
            near = tree.query_ball_point(anchor[:2], xy)
            lonely[i] = not (near and (np.abs(centres[near, 2] - anchor[2]) <= z).any())
    extra = anchors[lonely]
    rule = centre_limits(extra, np.full(len(extra), np.nan, np.float32)) / 1.25 if len(extra) else np.zeros(0, np.float32)
    return dict(centres=np.concatenate((centres, extra)).astype(np.float32).reshape(-1, 3),
                scores=np.concatenate((scores, detection['query_scores'][lonely])).astype(np.float32),
                extents=np.concatenate((extents, rule)).astype(np.float32),
                is_anchor=np.concatenate((np.zeros(len(centres), bool), np.ones(len(extra), bool))))


def candidate_features(arrays, votes, probability, spread, stage1_labels, candidates, anchors):
    """Feature matrix [K, F] in the order of ``FEATURE_NAMES``."""
    centres, scores, extents = candidates['centres'], candidates['scores'], candidates['extents']
    K = len(centres)
    features = np.zeros((K, len(FEATURE_NAMES)), np.float32)
    if not K:
        return features
    xyz = arrays['coord']
    use = np.flatnonzero((probability >= ASSIGN['probability']) & (xyz[:, 2] >= .5))
    scaled_votes = votes[use] * SCALE
    vote_tree = cKDTree(scaled_votes)
    scaled = centres * SCALE
    centre_tree = cKDTree(scaled)
    column = {name: i for i, name in enumerate(FEATURE_NAMES)}

    def put(name, values):
        features[:, column[name]] = values
    put('score', scores)
    put('extent', extents)
    put('height', centres[:, 2])
    put('is_stage1_centre', candidates['is_anchor'])
    put('has_detection', ~candidates['is_anchor'])
    k = min(4, K)
    distance, neighbour = centre_tree.query(scaled, k=k)
    distance, neighbour = distance.reshape(K, k), neighbour.reshape(K, k)
    for rank, prefix in enumerate(('n1', 'n2', 'n3'), 1):
        if rank < k:
            d, j = distance[:, rank], neighbour[:, rank]
            put(f'{prefix}_dist_over_extent', d / extents)
            put(f'{prefix}_score', scores[j])
            if prefix == 'n1':
                put('n1_dist_over_its_extent', d / extents[j])
                put('n1_score_delta', scores[j] - scores)
                put('n1_height_delta', centres[j, 2] - centres[:, 2])
        else:
            put(f'{prefix}_dist_over_extent', 9.)
    for i, near in enumerate(centre_tree.query_ball_point(scaled, extents)):
        near = [j for j in near if j != i]
        features[i, column['candidates_within_extent']] = len(near)
        features[i, column['stronger_within_extent']] = sum(scores[j] > scores[i] for j in near)
    for i, near in enumerate(centre_tree.query_ball_point(scaled, 2 * extents)):
        near = [j for j in near if j != i]
        features[i, column['candidates_within_2_extent']] = len(near)
        features[i, column['covered_by_stronger']] = sum(
            scores[j] > scores[i] and np.linalg.norm(scaled[j] - scaled[i]) <= extents[j] for j in near)
    # exclusive assignment: every vote to its nearest candidate
    _, owner = centre_tree.query(scaled_votes)
    exclusive = np.bincount(owner, minlength=K)
    labels = np.asarray(stage1_labels, np.int64)[use]
    sizes = np.bincount(labels, minlength=int(labels.max(initial=0)) + 1)
    points, travel = xyz[use], np.linalg.norm(votes[use, :2] - xyz[use, :2], axis=1)
    anchors = np.asarray(anchors, np.float32).reshape(-1, 3)
    anchor_tree = cKDTree(anchors[:, :2]) if len(anchors) else None
    density_tree = cKDTree(points[:, :2])
    for i in range(K):
        inside = np.asarray(vote_tree.query_ball_point(scaled[i], float(extents[i])), np.int64)
        row = features[i]
        row[column['votes_extent']] = np.log1p(len(inside))
        row[column['exclusive_votes']] = np.log1p(exclusive[i])
        if len(inside):
            d = np.linalg.norm(scaled_votes[inside] - scaled[i], axis=1)
            row[column['votes_05']] = np.log1p((d <= .5).sum())
            row[column['votes_1']] = np.log1p((d <= 1.).sum())
            row[column['votes_1_share']] = (d <= min(1., .5 * extents[i])).mean()
            row[column['tree_probability']] = probability[use[inside]].mean()
            row[column['vote_spread']] = spread[use[inside]].mean()
            voters = points[inside]
            row[column['voter_height_mean']] = voters[:, 2].mean()
            row[column['voter_height_std']] = voters[:, 2].std()
            row[column['voter_height_max']] = voters[:, 2].max()
            centroid = voters[:, :2].mean(0)
            radius = np.sqrt(((voters[:, :2] - centroid) ** 2).sum(1).mean())
            off = np.linalg.norm(centroid - centres[i, :2])
            row[column['voter_radius']] = radius
            row[column['voter_offcentre']] = off
            row[column['voter_offcentre_ratio']] = off / max(radius, .25)
            row[column['voter_travel']] = travel[inside].mean()
            row[column['exclusive_share']] = (owner[inside] == i).mean()
            own = labels[inside]
            assigned = own > 0
            row[column['stage1_assigned_share']] = assigned.mean()
            if assigned.any():
                counts = np.bincount(own[assigned])
                dominant = counts.argmax()
                row[column['stage1_purity']] = counts[dominant] / assigned.sum()
                row[column['stage1_capture']] = counts[dominant] / max(sizes[dominant], 1)
                row[column['stage1_labels']] = (counts > 0).sum()
        if anchor_tree is not None:
            gap, nearest = anchor_tree.query(centres[i, :2])
            row[column['stage1_centre_distance']] = min(gap, 10.)
            row[column['stage1_centre_height_delta']] = anchors[nearest, 2] - centres[i, 2]
        else:
            row[column['stage1_centre_distance']] = 10.
        row[column['local_density']] = np.log1p(len(density_tree.query_ball_point(centres[i, :2], 5.)) / 78.5)
    return np.nan_to_num(features, nan=0., posinf=9., neginf=-9.)


def candidate_targets(candidates, true_centres, arrays, votes, xy=1., z=3.):
    """Label per candidate: 1 = the best-scoring candidate on a true centre, 0 = not,
    plus a weight of 0 for candidates in unannotated area."""
    centres, scores = candidates['centres'], candidates['scores']
    target = np.zeros(len(centres), np.float32)
    weight = np.ones(len(centres), np.float32)
    if not len(centres):
        return target, weight
    truth = np.asarray(true_centres, np.float32).reshape(-1, 3)
    if len(truth):
        tree = cKDTree(truth[:, :2])
        used = set()
        for i in np.argsort(-scores):
            best, best_distance = -1, np.inf
            for j in tree.query_ball_point(centres[i, :2], xy):
                gap = np.linalg.norm(truth[j, :2] - centres[i, :2])
                if j not in used and abs(truth[j, 2] - centres[i, 2]) <= z and gap < best_distance:
                    best, best_distance = j, gap
            if best >= 0:
                used.add(best)
                target[i] = 1.
    vote_tree = cKDTree(votes[:, :2])
    known = arrays['tree_id'] >= 0
    for i, voters in enumerate(vote_tree.query_ball_point(centres[:, :2], 1.)):
        if target[i] == 0 and (not voters or known[voters].mean() < .5):
            weight[i] = 0.
    return target, weight


def combine_scores(rule, probability, candidates):
    """Final candidate score.

    ``verifier``: the verifier probability. ``mean``: its mean with the detector
    heat, where a stage-1 centre without a detection counts as heat 0 (it can then
    never pass a threshold above 0.5). ``mean_heat``: the same mean, but such a
    centre uses the heat sampled at its position.
    """
    probability = np.asarray(probability, np.float32)
    if rule == 'verifier':
        return probability
    if rule == 'mean':
        return .5 * (probability + np.where(candidates['is_anchor'], 0., candidates['scores']).astype(np.float32))
    if rule == 'mean_heat':
        return .5 * (probability + candidates['scores'].astype(np.float32))
    raise ValueError(f'unknown verifier rule: {rule}')


def verified_detection(candidates, probability):
    """Candidate set in the shape ``select_field`` expects, scored by the verifier."""
    return dict(centres=candidates['centres'], scores=np.asarray(probability, np.float32), extents=candidates['extents'],
                query_scores=np.zeros(0, np.float32))


def verify(classifier, arrays, votes, probability, spread, stage1_labels, detection, anchors, rule='mean'):
    """Detections + stage-1 centres -> verified detection dict and the stage-1 flag per candidate."""
    candidates = build_candidates(detection, anchors)
    if not len(candidates['centres']):
        return verified_detection(candidates, np.zeros(0, np.float32)), candidates['is_anchor']
    features = candidate_features(arrays, votes, probability, spread, stage1_labels, candidates, anchors)
    scores = combine_scores(rule, classifier.predict_proba(features)[:, 1], candidates)
    return verified_detection(candidates, scores), candidates['is_anchor']
