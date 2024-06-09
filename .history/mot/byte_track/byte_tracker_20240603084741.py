import numpy as np
from .kalman_filter import KalmanFilter
from . import matching
from .basetrack import BaseTrack, TrackState


class STrack(BaseTrack):
    def __init__(self, tlwh, score, idx=None, feature=None):
        self._tlwh = np.asarray(tlwh, dtype=np.float32)
        self.is_activated = False
        self.last_det_idx = idx
        self.score = score
        self.tracklet_len = 0
        self.curr_feat = feature  # current feature for ReID

    def predict(self):
        pass  

    @staticmethod
    def multi_predict(stracks):
        pass  

    def activate(self, frame_id):
        self.track_id = self.next_id()
        self.tracklet_len = 0
        self.state = TrackState.Tracked
        self.is_activated = True if frame_id == 1 else False
        self.frame_id = frame_id
        self.start_frame = frame_id

    def re_activate(self, new_track, frame_id, new_id=False):
        self.tracklet_len = 0
        self.state = TrackState.Tracked
        self.is_activated = True
        self.last_det_idx = new_track.last_det_idx
        self.frame_id = frame_id
        if new_id:
            self.track_id = self.next_id()
        self.score = new_track.score
        self.curr_feat = new_track.curr_feat

    def update(self, new_track, frame_id):
        self.frame_id = frame_id
        self.tracklet_len += 1
        self.state = TrackState.Tracked
        self.is_activated = True
        self.last_det_idx = new_track.last_det_idx
        self.score = new_track.score
        self.curr_feat = new_track.curr_feat

    @property
    def tlwh(self):
        return self._tlwh.copy()

    @property
    def tlbr(self):
        ret = self.tlwh.copy()
        ret[2:] += ret[:2]
        return ret

    @staticmethod
    def tlwh_to_xyah(tlwh):
        ret = np.asarray(tlwh).copy()
        ret[:2] += ret[2:] / 2
        ret[2] /= ret[3]
        return ret

    def to_xyah(self):
        return self.tlwh_to_xyah(self.tlwh)

    @staticmethod
    def tlbr_to_tlwh(tlbr):
        ret = np.asarray(tlbr).copy()
        ret[2:] -= ret[:2]
        return ret

    @staticmethod
    def tlwh_to_tlbr(tlwh):
        ret = np.asarray(tlwh).copy()
        ret[2:] += ret[:2]
        return ret

    def __repr__(self):
        return 'OT_{}_({}-{})'.format(self.track_id, self.start_frame, self.end_frame)


class BYTETracker(object):
    def __init__(self, args, frame_rate=30):
        self.tracked_stracks = []
        self.lost_stracks = []
        self.removed_stracks = []
        self.frame_id = 0
        self.args = args
        self.det_thresh = args.det_thresh
        self.buffer_size = int(frame_rate / 30.0 * args.track_buffer)
        self.max_time_lost = self.buffer_size

    def update(self, output_results, features, img_info=None, img_size=None):
        self.frame_id += 1
        activated_stracks = []
        refind_stracks = []
        lost_stracks = []
        removed_stracks = []

        if output_results.shape[1] == 5:
            scores = output_results[:, 4]
            bboxes = output_results[:, :4]
        else:
            output_results = output_results.cpu().numpy()
            scores = output_results[:, 4] * output_results[:, 5]
            bboxes = output_results[:, :4]

        if img_info and img_size:
            img_h, img_w = img_info[0], img_info[1]
            scale = min(img_size[0] / float(img_h), img_size[1] / float(img_w))
            bboxes /= scale

        remain_inds = scores > self.args.track_thresh
        inds_low = scores > 0.1
        inds_high = scores < self.args.track_thresh

        inds_second = np.logical_and(inds_low, inds_high)
        dets_second = bboxes[inds_second]
        dets = bboxes[remain_inds]
        scores_keep = scores[remain_inds]
        scores_second = scores[inds_second]
        det_idxes_keep = np.arange(output_results.shape[0])[remain_inds]
        det_idxes_second = np.arange(output_results.shape[0])[inds_second]
        features_keep = features[remain_inds]
        features_second = features[inds_second]

        if len(dets) > 0:
            detections = [STrack(STrack.tlbr_to_tlwh(tlbr), s, idx, feat) for (tlbr, s, idx, feat) in zip(dets, scores_keep, det_idxes_keep, features_keep)]
        else:
            detections = []

        unconfirmed = []
        tracked_stracks = []
        for track in self.tracked_stracks:
            if not track.is_activated:
                unconfirmed.append(track)
            else:
                tracked_stracks.append(track)

        strack_pool = joint_stracks(tracked_stracks, self.lost_stracks)

       
        strack_pool_tlbrs = [track.tlbr for track in strack_pool]
        detections_tlbrs = [det.tlbr for det in detections]

        
        reid_dists = matching.embedding_distance(strack_pool, detections)
        
        
        iou_dists = matching.iou_distance(strack_pool_tlbrs, detections_tlbrs)
        
        # 追蹤時使用ReID 和IoU 的 weights ㄉ比例，
        lambda_reid = 0.96
        lambda_iou = 0.04
        
        dists = lambda_reid * reid_dists + lambda_iou * iou_dists
        
        if not self.args.mot20:
            dists = matching.fuse_score(dists, detections)
            
        matches, u_track, u_detection = matching.linear_assignment(dists, thresh=self.args.match_thresh)

        for itracked, idet in matches:
            track = strack_pool[itracked]
            det = detections[idet]
            if track.state == TrackState.Tracked:
                track.update(detections[idet], self.frame_id)
                activated_stracks.append(track)
            else:
                track.re_activate(det, self.frame_id, new_id=False)
                refind_stracks.append(track)

        if len(dets_second) > 0:
            detections_second = [STrack(STrack.tlbr_to_tlwh(tlbr), s, idx, feat) for (tlbr, s, idx, feat) in zip(dets_second, scores_second, det_idxes_second, features_second)]
        else:
            detections_second = []
        r_tracked_stracks = [strack_pool[i] for i in u_track if strack_pool[i].state == TrackState.Tracked]
        
        
        r_tracked_stracks_tlbrs = [track.tlbr for track in r_tracked_stracks]
        detections_second_tlbrs = [det.tlbr for det in detections_second]
        
        
        reid_dists_second = matching.embedding_distance(r_tracked_stracks, detections_second)
        
        
        iou_dists_second = matching.iou_distance(r_tracked_stracks_tlbrs, detections_second_tlbrs)
        
        
        dists_second = lambda_reid * reid_dists_second + lambda_iou * iou_dists_second
        
        matches, u_track, u_detection_second = matching.linear_assignment(dists_second, thresh=0.5)
        for itracked, idet in matches:
            track = r_tracked_stracks[itracked]
            det = detections_second[idet]
            if track.state == TrackState.Tracked:
                track.update(det, self.frame_id)
                activated_stracks.append(track)
            else:
                track.re_activate(det, self.frame_id, new_id=False)
                refind_stracks.append(track)

        for it in u_track:
            track = r_tracked_stracks[it]
            if not track.state == TrackState.Lost:
                track.mark_lost()
                lost_stracks.append(track)

        detections = [detections[i] for i in u_detection]
        
        
        unconfirmed_tlbrs = [track.tlbr for track in unconfirmed]
        detections_tlbrs = [det.tlbr for det in detections]
        
        
        reid_dists_unconfirmed = matching.embedding_distance(unconfirmed, detections)
        
        
        iou_dists_unconfirmed = matching.iou_distance(unconfirmed_tlbrs, detections_tlbrs)
        
        
        dists_unconfirmed = lambda_reid * reid_dists_unconfirmed + lambda_iou * iou_dists_unconfirmed
        
        if not self.args.mot20:
            dists_unconfirmed = matching.fuse_score(dists_unconfirmed, detections)
            
        matches, u_unconfirmed, u_detection = matching.linear_assignment(dists_unconfirmed, thresh=0.7)
        for itracked, idet in matches:
            unconfirmed[itracked].update(detections[idet], self.frame_id)
            activated_stracks.append(unconfirmed[itracked])
        for it in u_unconfirmed:
            track = unconfirmed[it]
            track.mark_removed()
            removed_stracks.append(track)

        for inew in u_detection:
            track = detections[inew]
            if track.score < self.det_thresh:
                continue
            track.activate(self.frame_id)
            activated_stracks.append(track)

        for track in self.lost_stracks:
            if self.frame_id - track.end_frame > self.max_time_lost:
                track.mark_removed()
                removed_stracks.append(track)

        self.tracked_stracks = [t for t in self.tracked_stracks if t.state == TrackState.Tracked]
        self.tracked_stracks = joint_stracks(self.tracked_stracks, activated_stracks)
        self.tracked_stracks = joint_stracks(self.tracked_stracks, refind_stracks)
        self.lost_stracks = sub_stracks(self.lost_stracks, self.tracked_stracks)
        self.lost_stracks.extend(lost_stracks)
        self.lost_stracks = sub_stracks(self.lost_stracks, self.removed_stracks)
        self.removed_stracks.extend(removed_stracks)

        self.tracked_stracks, self.lost_stracks = remove_duplicate_stracks(self.tracked_stracks, self.lost_stracks)

        return self.tracked_stracks

def joint_stracks(tlista, tlistb):
    exists = {}
    res = []
    for t in tlista:
        exists[t.track_id] = 1
        res.append(t)
    for t in tlistb:
        tid = t.track_id
        if not exists.get(tid, 0):
            exists[tid] = 1
            res.append(t)
    return res

def sub_stracks(tlista, tlistb):
    stracks = {}
    for t in tlista:
        stracks[t.track_id] = t
    for t in tlistb:
        tid = t.track_id
        if stracks.get(tid, 0):
            del stracks[tid]
    return list(stracks.values())

def remove_duplicate_stracks(stracksa, stracksb):
    pdist = matching.iou_distance(stracksa, stracksb)
    pairs = np.where(pdist < 0.15)
    dupa, dupb = list(), list()
    for p, q in zip(*pairs):
        timep = stracksa[p].frame_id - stracksa[p].start_frame
        timeq = stracksb[q].frame_id - stracksb[q].start_frame
        if timep > timeq:
            dupb.append(q)
        else:
            dupa.append(p)
    resa = [t for i, t in enumerate(stracksa) if not i in dupa]
    resb = [t for i, t in enumerate(stracksb) if not i in dupb]
    return resa, resb