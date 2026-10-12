"""Sparse incidence lookups and causal target-specific event clocks."""
from __future__ import annotations
from collections import defaultdict
import logging
import numpy as np


def clocks(data, assignments, banned=None, fixed_newness=None):
    n = len(assignments)
    valid = np.zeros(n, bool)
    fresh = np.zeros(n, bool)
    gap = np.zeros(n, np.float64)
    predecessor = np.full(n, -1, np.int64)
    previous, new_events = {}, {}
    order = np.lexsort((data["event_id"], data["first_packet_time"]))
    for i in order:
        cid = int(data["connection"][i])
        time = data["first_packet_time"][i]
        fresh[i] = cid not in previous if fixed_newness is None else fixed_newness[i]
        key = (int(data["source_node"][i]), int(data["destination_node"][i]), int(assignments[i]))
        prior = new_events.get(key) if fresh[i] else previous.get(cid)
        if prior is not None:
            valid[i] = True
            predecessor[i] = prior
            gap[i] = time - data["first_packet_time"][prior]
            if gap[i] < -1e-12:
                raise ValueError("Noncausal event clock")
        if banned is None or not banned[i]:
            previous[cid] = i
            if fresh[i]:
                new_events[key] = i
    return dict(gap=gap, gap_valid=valid, new_connection=fresh, gap_predecessor=predecessor)


class Incidence:
    def __init__(self, ids, seconds, events):
        groups = defaultdict(list)
        order = np.lexsort((events, seconds))
        for i in order:
            groups[int(ids[i])].append(int(i))
        self.groups = {k: np.asarray(v, np.int64) for k, v in groups.items()}
        self.times = {k: seconds[v] for k, v in self.groups.items()}
        self.positions = np.empty(len(ids), np.int64)
        self.ids = ids
        for rows in self.groups.values():
            self.positions[rows] = np.arange(len(rows))

    def between(self, key, lower, upper):
        key = int(key)
        if key not in self.groups:
            return np.empty(0, np.int64)
        times = self.times[key]
        lo, hi = np.searchsorted(times, (lower, upper), side="left")
        return self.groups[key][lo:hi]

    def span(self, key, lower, upper):
        key = int(key)
        if key not in self.groups:
            return np.empty(0, np.int64), 0, 0
        lo, hi = np.searchsorted(self.times[key], (lower, upper), side='left')
        return self.groups[key], int(lo), int(hi)


class GraphIndex:
    def __init__(self, data, cfg, banned=None, cache_candidates=False):
        self.data, self.cfg, self.banned = data, cfg, banned
        self.seconds = data["timestamp_second"]
        self.src = Incidence(data["source_endpoint"], self.seconds, data["event_id"])
        self.dst = Incidence(data["destination_endpoint"], self.seconds, data["event_id"])
        self.conn = Incidence(data["connection"], self.seconds, data["event_id"])
        self.cache = {} if cache_candidates else None
        self.historical_width = cfg["history_neighbors_per_relation"] * 4
        self.current_width = cfg["current_neighbors_per_relation"] * 4
        self.private_width = cfg["same_connection_neighbors"] + cfg["reverse_connection_neighbors"]

    def pools(self, i, current):
        key = (int(i), bool(current))
        if self.cache is not None and key in self.cache:
            return self.cache[key]
        sec = int(self.seconds[i])
        lower, upper = (sec, sec + self.cfg.get('slice_seconds', 1)) if current else (sec - self.cfg["history_seconds"], sec)
        se, de = self.data["source_endpoint"][i], self.data["destination_endpoint"][i]
        raw = [self.src.between(se, lower, upper), self.dst.between(de, lower, upper),
               self.dst.between(se, lower, upper), self.src.between(de, lower, upper)]
        pools = []
        own, reverse = self.data["connection"][i], self.data["reverse_connection"][i]
        for ids in raw:
            if current:
                mask = ids != i
            else:
                # Own and reverse chains have their own private channel.
                mask = ((self.data["connection"][ids] != own) &
                        (self.data["connection"][ids] != reverse))
            if self.banned is not None:
                mask &= ~self.banned[ids]
            pools.append(ids[mask])
        if self.cache is not None:
            self.cache[key] = pools
        return pools

    def private(self, i):
        sec = int(self.seconds[i])
        lower = sec - self.cfg["connection_history_seconds"]
        own = self.conn.between(self.data["connection"][i], lower, sec)
        reverse = self.conn.between(self.data["reverse_connection"][i], lower, sec)
        if self.banned is not None:
            own, reverse = own[~self.banned[own]], reverse[~self.banned[reverse]]
        own = own[-self.cfg["same_connection_neighbors"]:]
        reverse = reverse[~np.isin(reverse, own)][-self.cfg["reverse_connection_neighbors"]:]
        return own, reverse

    def sample_relations(self, i, current, rng, k):
        """Sample filtered ranks without scanning every incident edge per target.

        Matches pools()+choice exactly, including pre-deduplication counts.
        Anonymous indices are used only for incidence lookup.
        """
        if self.banned is not None:
            pools = self.pools(i, current)
            taken, chosen_rows, counts = set(), [], []
            for pool in pools:
                counts.append(len(pool))
                if taken:
                    pool = pool[~np.isin(pool, tuple(taken))]
                chosen = pool if len(pool) <= k else pool[rng.choice(len(pool), k, replace=False)]
                chosen = chosen[np.argsort(self.data['event_id'][chosen], kind='stable')]
                chosen_rows.append(chosen)
                taken.update(map(int, chosen))
            return chosen_rows, counts
        sec = int(self.seconds[i])
        lower, upper = (sec, sec + self.cfg.get('slice_seconds', 1)) if current else (sec - self.cfg['history_seconds'], sec)
        if current:
            excluded = np.asarray([i], np.int64)
        else:
            own = self.conn.between(self.data['connection'][i], lower, upper)
            rev = self.conn.between(self.data['reverse_connection'][i], lower, upper)
            excluded = np.unique(np.concatenate([own, rev]))
        se, de = self.data['source_endpoint'][i], self.data['destination_endpoint'][i]
        lookups = ((self.src, se), (self.dst, de), (self.dst, se), (self.src, de))
        taken, chosen_rows, counts = [], [], []
        for incidence, key in lookups:
            group, lo, hi = incidence.span(key, lower, upper)
            if hi == lo:
                counts.append(0)
                chosen_rows.append(np.empty(0, np.int64))
                continue
            def positions(candidates):
                candidates = np.asarray(candidates, np.int64)
                selected = candidates[incidence.ids[candidates] == key]
                pos = incidence.positions[selected]
                return pos[(pos >= lo) & (pos < hi)] - lo
            blocked = positions(excluded)
            counts.append(hi - lo - len(blocked))
            if taken:
                blocked = np.union1d(blocked, positions(taken))
            blocked.sort()
            available = hi - lo - len(blocked)
            rank = np.arange(available) if available <= k else rng.choice(available, k, replace=False)
            rank = rank.copy()
            if len(blocked):
                # For sorted excluded positions b[j], b[j]-j is the first
                # available rank shifted by that exclusion. Consecutive
                # exclusions share a boundary; side='right' skips all of them.
                rank += np.searchsorted(blocked - np.arange(len(blocked)), rank, side='right')
            chosen = group[lo + rank]
            chosen = chosen[np.argsort(self.data['event_id'][chosen], kind='stable')]
            chosen_rows.append(chosen)
            taken.extend(map(int, chosen))
        return chosen_rows, counts

    def bank(self, rows, rng, include_current=True, score_seed=None, fast=None):
        if fast is None:
            # Normal training already caches small candidate lists. The rank
            # sampler serves large uncached inference graphs.
            fast = self.cache is None
        rows = np.asarray(rows, np.int64)
        n = len(rows)
        widths = dict(private=self.private_width, historical=self.historical_width, current=self.current_width)
        result = {name: np.full((n, w), -1, np.int64) for name, w in widths.items()}
        result.update({name + "_type": np.zeros((n, w), np.int64) for name, w in widths.items()})
        result["historical_counts"] = np.zeros((n, 4), np.int64)
        result["current_counts"] = np.zeros((n, 4), np.int64)
        for p, i in enumerate(rows):
            # Inference choices belong to an event, not a global RNG stream:
            # an unrelated row elsewhere must not shift this target's sampling.
            row_rng = rng if score_seed is None else np.random.default_rng(
                np.random.SeedSequence([int(score_seed), int(self.data["event_id"][i])]))
            own, rev = self.private(i)
            result["private"][p, :len(own)] = own
            result["private_type"][p, :len(own)] = 4
            start = self.cfg["same_connection_neighbors"]
            result["private"][p, start:start + len(rev)] = rev
            result["private_type"][p, start:start + len(rev)] = 5
            for name, current in (("historical", False), ("current", True)):
                if current and not include_current:
                    continue
                k = self.cfg["current_neighbors_per_relation"] if current else self.cfg["history_neighbors_per_relation"]
                if fast:
                    choices, counts = self.sample_relations(i, current, row_rng, k)
                    result[name + '_counts'][p] = counts
                    for rel, chosen in enumerate(choices):
                        result[name][p, rel * k:rel * k + len(chosen)] = chosen
                        result[name + '_type'][p, rel * k:rel * k + len(chosen)] = rel
                    continue
                taken = set()
                for rel, pool in enumerate(self.pools(i, current)):
                    result[name + "_counts"][p, rel] = len(pool)
                    if not len(pool):
                        continue
                    if taken:
                        pool = pool[~np.isin(pool, tuple(taken))]
                    chosen = pool if len(pool) <= k else pool[row_rng.choice(len(pool), k, replace=False)]
                    # Stable ordering; numeric anonymous IDs never break ties.
                    chosen = chosen[np.argsort(self.data["event_id"][chosen], kind="stable")]
                    result[name][p, rel * k:rel * k + len(chosen)] = chosen
                    result[name + "_type"][p, rel * k:rel * k + len(chosen)] = rel
                    taken.update(map(int, chosen))
        for name in widths:
            ids = result[name]
            valid = ids >= 0
            safe = np.maximum(ids, 0)
            if name == 'private' and self.cfg.get('private_time_policy') == 'past_relative_v1':
                # Only intervals among observed past nodes. Referencing the
                # target arrival would directly reveal the predicted gap.
                times = self.data['first_packet_time'][safe]
                origin = np.max(np.where(valid, times, -np.inf), axis=1)
                origin[~valid.any(1)] = 0
                dt = np.maximum(0., origin[:, None] - times)
            else:
                dt = np.abs(self.data["first_packet_time"][rows, None] - self.data["first_packet_time"][safe])
            result[name + "_dt"] = (np.log1p(dt) / np.log(65)).astype(np.float32)
            result[name + "_dt"][~valid] = 0
            if name != "current" and valid.any():
                assert (self.seconds[safe][valid] < np.broadcast_to(self.seconds[rows, None], ids.shape)[valid]).all()
            if name == "current" and valid.any():
                assert (self.seconds[safe][valid] == np.broadcast_to(self.seconds[rows, None], ids.shape)[valid]).all()
            assert not ((ids == rows[:, None]) & valid).any()
        return result


def subset_bank(bank, positions):
    return {k: v[positions] for k, v in bank.items()}


def fixed_bank(index, n, cfg):
    rng = np.random.default_rng(cfg["seed"] + 500)
    result = None
    for start in range(0, n, cfg["score_batch_size"]):
        end = min(n, start + cfg["score_batch_size"])
        part = index.bank(np.arange(start, end), rng, score_seed=cfg["seed"] + 500)
        if result is None:
            result = {}
            for key, value in part.items():
                dtype = value.dtype
                if key.endswith('_type'):
                    dtype = np.uint8
                elif value.dtype == np.int64 and n < np.iinfo(np.int32).max:
                    dtype = np.int32
                result[key] = np.empty((n, *value.shape[1:]), dtype=dtype)
        for key, value in part.items():
            result[key][start:end] = value
        if n > 100000 and (end == n or start // 25000 != end // 25000):
            logging.getLogger(__name__).info('Graph queries %d/%d', end, n)
    return result


def cached_bank(cache, cfg, data=None):
    """Build anonymous indices once, including across concurrent CPU workers."""
    import fcntl
    from pathlib import Path
    from .data import load
    from viot_nids_v7.data import write_json
    cache = Path(cache)
    folder = cache / 'score_bank'
    folder.mkdir(parents=True, exist_ok=True)
    with (folder / 'build.lock').open('a') as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        if not (folder / 'complete.json').exists():
            if data is None:
                data, _ = load(cache)
            values = fixed_bank(GraphIndex(data, cfg), len(data['row_id']), cfg)
            for name, array in values.items():
                np.save(folder / (name + '.npy'), array, allow_pickle=False)
            write_json(folder / 'complete.json', dict(rows=len(data['row_id']),
                sampling_policy='event_seed_v1', exact_reference_choices=True,
                private_time_policy=cfg.get('private_time_policy')))
    return {p.stem: np.load(p, mmap_mode='r') for p in folder.glob('*.npy')}
