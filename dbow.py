import numpy as np
import cv2
try:
    import pydbow3 as _py
except Exception:
    _py = None


class ORB:
    @staticmethod
    def from_cv_descriptor(desc):
        if desc is None:
            return None
        arr = np.asarray(desc)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        return arr.astype(np.uint8)


class Vocabulary:
    def __init__(self, images=None, k=10, L=5):
        if _py is None:
            raise ImportError("pydbow3 is not installed")
        self._voc = _py.Vocabulary("", k, L)
        if images:
            orb = cv2.ORB_create()
            features = []
            for img in images:
                kps, descs = orb.detectAndCompute(img, None)
                if descs is None:
                    descs_mat = np.empty((0, 32), dtype=np.uint8)
                else:
                    descs_mat = np.asarray(descs).astype(np.uint8)
                features.append(descs_mat)
            self._voc.create(features)

    def descs_to_bow(self, descs):
        features = []
        for d in descs:
            if d is None:
                continue
            arr = np.asarray(d)
            if arr.ndim == 1:
                arr = arr.reshape(1, -1)
            features.append(arr.astype(np.uint8))
        return self._voc.transform(features)

    def save(self, path):
        self._voc.save(path)

    def load(self, path):
        self._voc.load(path)
        return self


class Database:
    def __init__(self, vocabulary=None):
        if _py is None:
            raise ImportError("pydbow3 is not installed")
        self._db = _py.Database()
        if vocabulary is not None:
            if isinstance(vocabulary, Vocabulary):
                self._db.setVocabulary(vocabulary._voc, False, 0)
            else:
                self._db.setVocabulary(vocabulary, False, 0)

    def add(self, descs):
        if descs is None:
            return
        # If given a numpy array (n, d)
        if isinstance(descs, np.ndarray):
            self._db.add(descs.astype(np.uint8))
            return
        arrs = []
        for d in descs:
            if d is None:
                continue
            a = np.asarray(d)
            if a.ndim == 1:
                a = a.reshape(1, -1)
            arrs.append(a.astype(np.uint8))
        if not arrs:
            return
        stacked = np.vstack(arrs)
        self._db.add(stacked)

    def query(self, descs, max_results=1):
        if descs is None:
            return []
        arrs = []
        for d in descs:
            if d is None:
                continue
            a = np.asarray(d)
            if a.ndim == 1:
                a = a.reshape(1, -1)
            arrs.append(a.astype(np.uint8))
        if not arrs:
            return []
        stacked = np.vstack(arrs)
        return self._db.query(stacked, max_results)


# Fallback pure-Python implementation when pydbow3 isn't available
if _py is None:
    class Vocabulary:
        def __init__(self, images=None, k=10, L=5):
            self.k = k
            self.centers = None
            if images:
                orb = cv2.ORB_create()
                descs_all = []
                for img in images:
                    kps, descs = orb.detectAndCompute(img, None)
                    if descs is None:
                        continue
                    descs_all.append(np.asarray(descs).astype(np.float32))
                if descs_all:
                    all_descs = np.vstack(descs_all)
                    # kmeans with OpenCV
                    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
                    attempts = 3
                    flags = cv2.KMEANS_PP_CENTERS
                    if all_descs.shape[0] < k:
                        # fewer descriptors than clusters: pad centers with zeros
                        centers = np.zeros((k, all_descs.shape[1]), dtype=np.float32)
                        centers[: all_descs.shape[0], :] = all_descs
                        self.centers = centers
                    else:
                        compactness, labels, centers = cv2.kmeans(all_descs, k, None, criteria, attempts, flags)
                        self.centers = centers
                else:
                    self.centers = np.zeros((k, 32), dtype=np.float32)

        def descs_to_bow(self, descs):
            if self.centers is None:
                return np.zeros(self.k, dtype=np.float32)
            # descs is list of 1xd arrays
            words = np.zeros(self.k, dtype=np.float32)
            for d in descs:
                if d is None:
                    continue
                a = np.asarray(d).astype(np.float32)
                if a.ndim == 2 and a.shape[0] == 1:
                    a = a.reshape(-1)
                if a.size == 0:
                    continue
                # find nearest center
                diffs = self.centers - a
                dists = np.sum(diffs * diffs, axis=1)
                idx = int(np.argmin(dists))
                words[idx] += 1.0
            return words

        def save(self, path):
            import pickle
            with open(path, "wb") as f:
                pickle.dump(self.centers, f)

        def load(self, path):
            import pickle
            with open(path, "rb") as f:
                self.centers = pickle.load(f)
            return self

    class Database:
        def __init__(self, vocabulary=None):
            self.vocabulary = vocabulary
            self.bows = []
            self.descriptors = []

        def add(self, descs):
            bow = self.vocabulary.descs_to_bow(descs)
            self.bows.append(bow)
            # store stacked descriptors
            arrs = []
            for d in descs:
                if d is None:
                    continue
                a = np.asarray(d)
                if a.ndim == 1:
                    a = a.reshape(1, -1)
                arrs.append(a)
            if arrs:
                self.descriptors.append(np.vstack(arrs))
            else:
                self.descriptors.append(np.empty((0, 32), dtype=np.uint8))

        def query(self, descs, max_results=1):
            qbow = self.vocabulary.descs_to_bow(descs)
            scores = []
            q = qbow.astype(np.float32)
            if q.sum() > 0:
                q = q / q.sum()
            for b in self.bows:
                bb = b.astype(np.float32)
                if bb.sum() > 0:
                    bb = bb / bb.sum()
                scores.append(float(np.dot(q, bb)))
            return np.array(scores)

        def __getitem__(self, idx):
            return self.bows[idx]

        def save(self, path):
            import pickle
            with open(path, 'wb') as f:
                pickle.dump({'bows': self.bows, 'descriptors': self.descriptors}, f)

        def load(self, path):
            import pickle
            with open(path, 'rb') as f:
                data = pickle.load(f)
            self.bows = data.get('bows', [])
            self.descriptors = data.get('descriptors', [])
            return self

