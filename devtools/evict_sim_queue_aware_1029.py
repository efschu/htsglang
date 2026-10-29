"""Offline simulation: radix-cache eviction policies under multi-agent load (27b-queue-evict-1029).

Pure Python, no GPU. Uses the REAL strategy classes from flliper.srt.mem_cache.evict_policy
(LRUStrategy, SLRUStrategy, QueueAwareStrategy) on a segment radix tree that mirrors the radix
cache's eviction contract: only unlocked leaves are evictable, a parent becomes a candidate once its
last present child left (heap re-push, as radix_cache.evict / full_component._peel_heap do).

Scheduler model = the 27B D side as the code has it: FCFS, no match of the waiting queue in
calc_priority (supports_fast_match_prefix() is False everywhere), a request touches its prefix at
intake (init_next_round_input in _add_request_to_queue) and when the admission loop reaches it; a
head that does not fit blocks the queue. Arms:
  lru          the default (radix_eviction_policy=lru)
  slru         SLRUStrategy (protected after 2 hits)
  lru_touch    LRU + sglang upstream PR #41593: every >=0.5 s the first 128 waiters' prefixes are
               touched back to front (head newest), no hit_count change
  queue_aware  QueueAwareStrategy
  oracle       Belady with the FUTURE KNOWN (next arrival of every conversation) -- an upper bound,
               not implementable

Metrics per admitted request: prompt tokens, tokens served from cache, newly prefilled tokens.

Usage:  python3 devtools/evict_sim_queue_aware_1029.py [--quick] [--replay CSV]
"""

from __future__ import annotations

import argparse
import heapq
import math
import random
import sys
from typing import Callable, Dict, List, Optional, Tuple

import msgspec

sys.path.insert(0, "python")
from flliper.srt.mem_cache.evict_policy import (  # noqa: E402
    EvictionStrategy,
    LRUStrategy,
    QueueAwareStrategy,
    SLRUStrategy,
)

TOUCH_INTERVAL_S = 0.5  # PR #41593 WAITING_PREFIX_REFRESH_INTERVAL_S
TOUCH_MAX_QUEUE = 128  # PR #41593 WAITING_QUEUE_PREFIX_MATCH_MAX


class SimConfig(msgspec.Struct, frozen=True):
    pool_tokens: int
    seats: int = 6
    prefill_tok_s: float = 4000.0
    step_s: float = 0.033
    step_s_per_seat: float = 0.002
    horizon_s: float = 3600.0


class Result(msgspec.Struct):
    requests: int = 0
    prompt_tokens: int = 0
    cached_tokens: int = 0
    prefill_tokens: int = 0
    evict_runs: int = 0

    def hit(self) -> float:
        return self.cached_tokens / max(1, self.prompt_tokens)


# ----------------------------------------------------------------------------------------- tree
class Node:
    __slots__ = ("id", "parent", "children", "length", "present", "lock_ref",
                 "last_access_time", "hit_count", "users")

    def __init__(self, *, nid: int, parent: Optional["Node"], length: int):
        self.id = nid
        self.parent = parent
        self.children: Dict[object, Node] = {}
        self.length = length
        self.present = False
        self.lock_ref = 0
        self.last_access_time = 0
        self.hit_count = 0
        self.users: set = set()


class Tree:
    def __init__(self, *, pool_tokens: int, strategy: EvictionStrategy):
        self.root = Node(nid=0, parent=None, length=0)
        self.root.present = True
        self.pool = pool_tokens
        self.used = 0
        self.clock = 0
        self.next_id = 1
        self.strategy = strategy
        self.evictable: set = set()
        self.evict_runs = 0

    def tick(self) -> int:
        self.clock += 1
        return self.clock

    def node_for(self, *, path: List[Tuple[object, int]]) -> List[Node]:
        """Nodes of a path, created (not present) where missing."""
        out, cur = [], self.root
        for key, length in path:
            nxt = cur.children.get(key)
            if nxt is None:
                nxt = Node(nid=self.next_id, parent=cur, length=length)
                self.next_id += 1
                cur.children[key] = nxt
            out.append(nxt)
            cur = nxt
        return out

    def match(self, *, nodes: List[Node], count_hit: bool) -> Tuple[int, Node]:
        cached, last = 0, self.root
        for n in nodes:
            if not n.present:
                break
            n.last_access_time = self.tick()
            if count_hit:
                n.hit_count += 1
            cached += n.length
            last = n
        return cached, last

    def _refresh(self, n: Node) -> None:
        leaf = (n is not self.root and n.present and n.lock_ref == 0
                and not any(c.present for c in n.children.values()))
        if leaf:
            self.evictable.add(n)
        else:
            self.evictable.discard(n)

    def lock(self, *, nodes: List[Node], delta: int) -> None:
        for n in nodes:
            n.lock_ref += delta
            self._refresh(n)

    def materialize(self, *, nodes: List[Node]) -> None:
        for n in nodes:
            if not n.present:
                n.present = True
                self.used += n.length
                n.last_access_time = self.tick()
                n.hit_count += 1
                self._refresh(n)
                self._refresh(n.parent)

    def evict(self, *, need: int) -> int:
        self.evict_runs += 1
        self.strategy.note_eviction_run()
        heap = [(self.strategy.get_priority(n), n.id, n) for n in self.evictable]
        heapq.heapify(heap)
        freed = 0
        while freed < need and heap:
            _, _, x = heapq.heappop(heap)
            if x not in self.evictable:
                continue
            x.present = False
            self.evictable.discard(x)
            self.used -= x.length
            freed += x.length
            p = x.parent
            self._refresh(p)
            if p in self.evictable:
                heapq.heappush(heap, (self.strategy.get_priority(p), p.id, p))
        return freed


# ------------------------------------------------------------------------------------- requests
class Req:
    __slots__ = ("rid", "agent", "path", "nodes", "prompt", "out", "arrival", "last_node",
                 "best_match_node", "out_key")

    def __init__(self, *, rid, agent, path, nodes, prompt, out, arrival, out_key):
        self.rid = rid
        self.path = path
        self.agent = agent
        self.nodes = nodes
        self.prompt = prompt
        self.out = out
        self.arrival = arrival
        self.out_key = out_key
        self.last_node = None
        self.best_match_node = None


class OracleStrategy(EvictionStrategy):
    """Belady with known future: evict the node whose next use lies furthest ahead."""

    def __init__(self, *, next_use: Callable[[object], float]):
        self.next_use = next_use

    def get_priority(self, node) -> Tuple[float, int]:
        if not node.users:
            return (-math.inf, node.last_access_time)
        return (-min(self.next_use(a) for a in node.users), node.last_access_time)


def make_strategy(*, name: str, next_use: Callable[[object], float]) -> EvictionStrategy:
    if name in ("lru", "lru_touch"):
        return LRUStrategy()
    if name == "slru":
        return SLRUStrategy()
    if name == "queue_aware":
        return QueueAwareStrategy(log_every_runs=0)
    if name == "oracle":
        return OracleStrategy(next_use=next_use)
    raise ValueError(name)


# ------------------------------------------------------------------------------------ scheduler
class Server:
    """FCFS D-side admission with seats; prefill serialized, decode a fixed per-token step."""

    def __init__(self, *, cfg: SimConfig, tree: Tree, arm: str):
        self.cfg = cfg
        self.tree = tree
        self.arm = arm
        self.queue: List[Req] = []
        self.running = 0
        self.reserved = 0
        self.prefill_free_at = 0.0
        self.last_touch = -math.inf
        self.result = Result()

    def arrive(self, *, req: Req) -> None:
        # intake: _add_request_to_queue -> init_next_round_input -> match_prefix (touches the path)
        _, last = self.tree.match(nodes=req.nodes, count_hit=False)
        req.last_node = req.best_match_node = last
        self.queue.append(req)

    def schedule(self, *, now: float) -> List[Tuple[float, Req]]:
        self.tree.strategy.note_waiting_queue(self.queue)
        if self.arm == "lru_touch" and now - self.last_touch >= TOUCH_INTERVAL_S:
            self.last_touch = now
            for req in reversed(self.queue[:TOUCH_MAX_QUEUE]):
                _, last = self.tree.match(nodes=req.nodes, count_hit=False)
                req.last_node = req.best_match_node = last
        started = []
        while self.running < self.cfg.seats and self.queue:
            fin = self._try_admit(req=self.queue[0], now=now)
            if fin is None:
                break
            started.append((fin, self.queue.pop(0)))
        return started

    def _try_admit(self, *, req: Req, now: float) -> Optional[float]:
        cached, last = self.tree.match(nodes=req.nodes, count_hit=True)
        req.last_node = req.best_match_node = last
        matched = [n for n in req.nodes if n.present]  # contiguous: leaves-only eviction
        rest = req.nodes[len(matched):]
        need = sum(n.length for n in rest) + req.out
        self.tree.lock(nodes=matched, delta=1)  # add_one_req: _lock_node(req.last_node)
        free = self.tree.pool - self.tree.used - self.reserved
        if free < need:
            self.tree.evict(need=need - free)
            free = self.tree.pool - self.tree.used - self.reserved
        if free < need:
            self.tree.lock(nodes=matched, delta=-1)
            return None
        self.tree.materialize(nodes=rest)
        self.tree.lock(nodes=rest, delta=1)
        self.reserved += req.out
        self.running += 1
        new = req.prompt - cached
        r = self.result
        r.requests += 1
        r.prompt_tokens += req.prompt
        r.cached_tokens += cached
        r.prefill_tokens += new
        start = max(now, self.prefill_free_at)
        self.prefill_free_at = start + new / self.cfg.prefill_tok_s
        step = self.cfg.step_s + self.cfg.step_s_per_seat * self.running
        return self.prefill_free_at + req.out * step

    def finish(self, *, req: Req) -> List[Node]:
        self.reserved -= req.out
        self.running -= 1
        self.tree.lock(nodes=req.nodes, delta=-1)
        out_nodes = self.tree.node_for(path=req.path + [(req.out_key, req.out)])
        self.tree.materialize(nodes=out_nodes[-1:])
        return out_nodes


# ------------------------------------------------------------------------------------ workloads
class AgentParams(msgspec.Struct, frozen=True):
    agents: int = 12
    sys_len: int = 8000
    in_median: float = 1000.0
    out_median: float = 350.0
    ctx_cap: int = 60000
    turns_lo: int = 8
    turns_hi: int = 40
    think_fast_p: float = 0.70  # tool call returns
    think_fast_s: float = 2.0
    think_mid_p: float = 0.25
    think_mid_s: float = 20.0
    think_long_lo_s: float = 120.0  # rest: user away
    think_long_hi_s: float = 900.0
    new_conv_s: float = 30.0
    fork_every: int = 0  # >0: agents 0, team_size, ... are leads that spawn subagents every N turns
    team_size: int = 4
    sub_turns_lo: int = 3
    sub_turns_hi: int = 10


class Agent:
    __slots__ = ("aid", "path", "nodes", "conv", "turn", "turns_left", "next_time", "lead",
                 "subs_open", "is_sub")

    def __init__(self, *, aid):
        self.aid = aid
        self.path: List[Tuple[object, int]] = []
        self.nodes: List[Node] = []
        self.conv = 0
        self.turn = 0
        self.turns_left = 0
        self.next_time = math.inf
        self.lead = None
        self.subs_open = 0
        self.is_sub = False


class AgentWorkload:
    """Closed-loop agents: next turn = whole history + new input, after a think/tool delay."""

    def __init__(self, *, p: AgentParams, seed: int):
        self.p = p
        self.rng = random.Random(seed)
        self.agents: Dict[object, Agent] = {}
        self.rid = 0

    def _lognorm(self, *, median: float, sigma: float, lo: int, hi: int) -> int:
        return int(min(hi, max(lo, self.rng.lognormvariate(math.log(median), sigma))))

    def _think(self) -> float:
        u, p = self.rng.random(), self.p
        if u < p.think_fast_p:
            return self.rng.expovariate(1.0 / p.think_fast_s)
        if u < p.think_fast_p + p.think_mid_p:
            return self.rng.expovariate(1.0 / p.think_mid_s)
        return self.rng.uniform(p.think_long_lo_s, p.think_long_hi_s)

    def start(self, *, push) -> None:
        for i in range(self.p.agents):
            if self.p.fork_every and i % self.p.team_size != 0:
                continue  # subagents are spawned by their lead
            a = self.agents.setdefault(i, Agent(aid=i))
            self._new_conv(agent=a, base=[("sys", self.p.sys_len)], turns=None)
            self._schedule(agent=a, at=self.rng.uniform(0, 10), push=push)

    def _new_conv(self, *, agent: Agent, base, turns: Optional[int]) -> None:
        agent.conv += 1
        agent.turn = 0
        agent.path = list(base)
        agent.nodes = []
        agent.turns_left = turns or self.rng.randint(self.p.turns_lo, self.p.turns_hi)

    def _schedule(self, *, agent: Agent, at: float, push) -> None:
        agent.next_time = at
        push(at, "arrive", agent)

    def make_request(self, *, agent: Agent, now: float, tree: Tree) -> Req:
        agent.turn += 1
        inp = self._lognorm(median=self.p.in_median, sigma=1.0, lo=50, hi=16000)
        out = self._lognorm(median=self.p.out_median, sigma=0.8, lo=16, hi=4000)
        tag = (agent.aid, agent.conv, agent.turn)
        path = agent.path + [(("in",) + tag, inp)]
        nodes = tree.node_for(path=path)
        for n in nodes:
            n.users.add(agent.aid)
        agent.nodes = nodes
        agent.next_time = now
        self.rid += 1
        return Req(rid=self.rid, agent=agent.aid, path=path, nodes=nodes,
                   prompt=sum(L for _, L in path), out=out, arrival=now, out_key=("out",) + tag)

    def on_finish(self, *, req: Req, out_nodes: List[Node], now: float, push) -> None:
        a = self.agents[req.agent]
        out_nodes[-1].users.add(a.aid)
        a.path = req.path + [(req.out_key, req.out)]
        a.nodes = out_nodes
        a.turns_left -= 1
        ctx = sum(L for _, L in a.path)
        if a.turns_left > 0 and ctx < self.p.ctx_cap:
            if self.p.fork_every and not a.is_sub and a.turn % self.p.fork_every == 0:
                self._spawn_subs(lead=a, now=now, push=push)
                return
            self._schedule(agent=a, at=now + self._think(), push=push)
            return
        self._end_conv(agent=a, now=now, push=push)

    def _end_conv(self, *, agent: Agent, now: float, push) -> None:
        for n in agent.nodes:
            n.users.discard(agent.aid)
        agent.next_time = math.inf
        if agent.is_sub:
            lead = self.agents[agent.lead]
            lead.subs_open -= 1
            if lead.subs_open == 0:
                self._schedule(agent=lead, at=now + self._think(), push=push)
            return
        self._new_conv(agent=agent, base=[("sys", self.p.sys_len)], turns=None)
        self._schedule(agent=agent, at=now + self.rng.expovariate(1.0 / self.p.new_conv_s), push=push)

    def _spawn_subs(self, *, lead: Agent, now: float, push) -> None:
        lead.next_time = now + 120.0  # oracle approximation: the lead waits for its subagents
        for j in range(1, self.p.team_size):
            sid = ("sub", lead.aid, lead.conv, lead.turn, j)
            s = self.agents.setdefault(sid, Agent(aid=sid))
            s.is_sub, s.lead = True, lead.aid
            turns = self.rng.randint(self.p.sub_turns_lo, self.p.sub_turns_hi)
            self._new_conv(agent=s, base=lead.path, turns=turns)
            lead.subs_open += 1
            self._schedule(agent=s, at=now + self.rng.uniform(0.1, 2.0), push=push)

    def next_use(self, aid) -> float:
        return self.agents[aid].next_time


class ReplayWorkload:
    """Open-loop replay of the 27B front-log trace; conversations = session_rek chains (heuristic).

    Path of a request = shared system segment + the chain's earlier segments + a new segment with
    the remaining prompt tokens. A prompt shorter than the chain's context starts a new branch
    (compaction / fresh conversation on the same chain id)."""

    def __init__(self, *, csv_path: str, sys_len: int, max_prompt: int):
        self.rows = _read_replay(csv_path=csv_path, max_prompt=max_prompt)
        self.sys_len = sys_len
        self.chains: Dict[str, Agent] = {}
        self.next_arrival: Dict[str, List[float]] = {}
        for r in self.rows:
            self.next_arrival.setdefault(r["chain"], []).append(r["t"])
        self.rid = 0

    def start(self, *, push) -> None:
        for r in self.rows:
            push(r["t"], "arrive", r)

    def make_request(self, *, agent, now: float, tree: Tree) -> Req:
        r = agent
        a = self.chains.setdefault(r["chain"], Agent(aid=r["chain"]))
        a.turn += 1
        ctx = sum(L for _, L in a.path)
        if not a.path or r["prompt"] < ctx:
            for n in a.nodes:  # the old branch is never asked for again (oracle bookkeeping)
                n.users.discard(a.aid)
            a.conv += 1
            a.path = [("sys", self.sys_len)] if r["prompt"] > self.sys_len else []
            ctx = sum(L for _, L in a.path)
        path = a.path + [(("in", a.aid, a.conv, a.turn), max(1, r["prompt"] - ctx))]
        nodes = tree.node_for(path=path)
        for n in nodes:
            n.users.add(a.aid)
        out_key = ("out", a.aid, a.conv, a.turn)
        # the chain's next prompt extends THIS prompt + its output, whether or not it finished yet
        a.path = path + [(out_key, r["out"])]
        a.nodes = nodes
        self.rid += 1
        return Req(rid=self.rid, agent=a.aid, path=path, nodes=nodes,
                   prompt=sum(L for _, L in path), out=r["out"], arrival=now, out_key=out_key)

    def on_finish(self, *, req: Req, out_nodes: List[Node], now: float, push) -> None:
        a = self.chains[req.agent]
        if a.nodes is req.nodes:  # else a newer request of the chain already covers this node
            out_nodes[-1].users.add(a.aid)
            a.nodes = out_nodes
        # the head of next_arrival stays the request's OWN (past) arrival while it waits/runs
        self.next_arrival[a.aid].pop(0)

    def next_use(self, aid) -> float:
        q = self.next_arrival[aid]
        return q[0] if q else math.inf


def _read_replay(*, csv_path: str, max_prompt: int) -> List[dict]:
    rows = []
    with open(csv_path) as f:
        lines = [ln for ln in f if not ln.startswith("#")]
    head = lines[0].strip().split(",")
    for ln in lines[1:]:
        d = dict(zip(head, ln.strip().split(",")))
        prompt, out = int(d["prompt_tokens"]), int(d["output_tokens"])
        if prompt < 200 or out <= 1 or prompt > max_prompt:
            continue  # probes / calibration prompts / larger than the simulated pool
        rows.append({"t": float(d["t_arrival"]), "prompt": prompt, "out": out,
                     "chain": d["session_rek"]})
    rows.sort(key=lambda r: r["t"])
    return rows


# --------------------------------------------------------------------------------------- driver
def run_sim(*, cfg: SimConfig, arm: str, workload) -> Result:
    events: List[tuple] = []
    seq = [0]

    def push(t, kind, obj):
        seq[0] += 1
        heapq.heappush(events, (t, seq[0], kind, obj))

    tree = Tree(pool_tokens=cfg.pool_tokens,
                strategy=make_strategy(name=arm, next_use=workload.next_use))
    server = Server(cfg=cfg, tree=tree, arm=arm)
    workload.start(push=push)
    while events:
        t, _, kind, obj = heapq.heappop(events)
        if t > cfg.horizon_s:
            break
        if kind == "arrive":
            server.arrive(req=workload.make_request(agent=obj, now=t, tree=tree))
        else:
            out_nodes = server.finish(req=obj)
            workload.on_finish(req=obj, out_nodes=out_nodes, now=t, push=push)
        for fin, req in server.schedule(now=t):
            push(fin, "finish", req)
    server.result.evict_runs = tree.evict_runs
    return server.result


# --------------------------------------------------------------------------------------- report
ARMS = ("lru", "slru", "lru_touch", "queue_aware", "oracle")

SCENARIOS: Dict[str, AgentParams] = {
    # 12 independent agents on 6 seats, shared system prompt
    "W1_unabh_12": AgentParams(agents=12),
    # agent teams: a lead spawns 3 subagents from its CURRENT history (growing shared prefix)
    "W2_fork_teams": AgentParams(agents=16, fork_every=4, team_size=4),
    # congestion: 24 agents on 6 seats, mostly fast tool calls -> long waiting queue
    "W3_stau_24": AgentParams(agents=24, think_fast_p=0.9, think_mid_p=0.09),
    # MegaCapybara-like: 8 agents, long contexts, few seats relative to agents
    "W4_lang_8": AgentParams(agents=8, in_median=3000.0, ctx_cap=120000, sys_len=18289),
}


def _mean_2se(xs: List[float]) -> Tuple[float, float]:
    m = sum(xs) / len(xs)
    if len(xs) < 2:
        return m, 0.0
    var = sum((x - m) ** 2 for x in xs) / (len(xs) - 1)
    return m, 2.0 * math.sqrt(var / len(xs))


def _row(*, label: str, by_arm: Dict[str, List[Result]]) -> str:
    """Paired against lru (same seed): hit = cached/prompt in points, served = admitted prompt
    tokens in %, prefill/served = newly prefilled tokens per served prompt token in %."""
    base = by_arm["lru"]
    cells = []
    for arm in ARMS:
        rs = by_arm[arm]
        hit = sum(r.hit() for r in rs) / len(rs)
        served = sum(r.prompt_tokens for r in rs) / len(rs)
        d_hit = _mean_2se([100.0 * (r.hit() - b.hit()) for r, b in zip(rs, base)])
        d_srv = _mean_2se([100.0 * (r.prompt_tokens / max(1, b.prompt_tokens) - 1.0)
                           for r, b in zip(rs, base)])
        cells.append(f"{arm:11s} hit {100*hit:5.1f}%  dHit {d_hit[0]:+5.1f}±{d_hit[1]:4.1f}pt  "
                     f"served {served/1e6:6.2f}M  dServed {d_srv[0]:+5.1f}±{d_srv[1]:4.1f}%")
    qa, tc = by_arm["queue_aware"], by_arm["lru_touch"]
    d_hit = _mean_2se([100.0 * (q.hit() - t.hit()) for q, t in zip(qa, tc)])
    d_srv = _mean_2se([100.0 * (q.prompt_tokens / max(1, t.prompt_tokens) - 1.0) for q, t in zip(qa, tc)])
    cells.append(f"qa-vs-touch dHit {d_hit[0]:+5.1f}±{d_hit[1]:4.1f}pt  dServed {d_srv[0]:+5.1f}±{d_srv[1]:4.1f}%")
    return f"{label}\n    " + "\n    ".join(cells)


def run_synthetic(*, seeds: int, pools: List[int], horizon_s: float) -> None:
    for name, p in SCENARIOS.items():
        for pool in pools:
            cfg = SimConfig(pool_tokens=pool, horizon_s=horizon_s)
            by_arm = {arm: [run_sim(cfg=cfg, arm=arm, workload=AgentWorkload(p=p, seed=s))
                            for s in range(seeds)] for arm in ARMS}
            print(_row(label=f"{name} pool={pool//1000}k seeds={seeds}", by_arm=by_arm), flush=True)


def run_replay(*, csv_path: str, pools: List[int], sys_len: int) -> None:
    for pool in pools:
        cfg = SimConfig(pool_tokens=pool, horizon_s=1e9)
        by_arm = {}
        for arm in ARMS:
            wl = ReplayWorkload(csv_path=csv_path, sys_len=sys_len, max_prompt=int(pool * 0.9))
            by_arm[arm] = [run_sim(cfg=cfg, arm=arm, workload=wl)]
        n = len(ReplayWorkload(csv_path=csv_path, sys_len=sys_len, max_prompt=int(pool * 0.9)).rows)
        print(_row(label=f"REPLAY pool={pool//1000}k requests={n}", by_arm=by_arm), flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--replay", default="")
    ap.add_argument("--replay-only", action="store_true")
    args = ap.parse_args()
    if not args.replay_only:
        if args.quick:
            run_synthetic(seeds=2, pools=[128_000], horizon_s=1200.0)
        else:
            run_synthetic(seeds=40, pools=[96_000, 160_000, 256_000, 384_000], horizon_s=3600.0)
    if args.replay:
        run_replay(csv_path=args.replay, pools=[180_000, 256_000, 384_000, 512_000, 768_000],
                   sys_len=18289)


if __name__ == "__main__":
    main()
