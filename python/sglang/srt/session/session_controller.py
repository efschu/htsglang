# Copyright 2023-2024 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

from __future__ import annotations

import logging
import time
import uuid
from array import array
from typing import TYPE_CHECKING, Dict, Optional, Sequence

from sglang.srt.managers.io_struct import (
    CloseSessionReqInput,
    OpenSessionReqInput,
    OpenSessionReqOutput,
    TokenizedGenerateReqInput,
)
from sglang.srt.managers.schedule_batch import FINISH_ABORT, Req
from sglang.srt.utils.common import log_info_on_rank0

if TYPE_CHECKING:
    from sglang.srt.mem_cache.base_prefix_cache import BasePrefixCache

logger = logging.getLogger(__name__)


def _concat(head, tail):
    """Type-preserving concatenation of two token sequences.

    ``array("q") + list`` is a TypeError, and the streaming session's token
    arrays ARE ``array("q")`` -- the in-place share path builds them that way.
    The copy path used to reach for ``+`` and therefore only ever worked on
    list-typed histories; #410's rewind is the first caller that takes the
    copy path on a streaming session, which is where that surfaced.
    ``extend`` accepts any iterable, so a slice of the head keeps its type and
    the tail can be either.
    """
    out = head[:]
    out.extend(tail)
    return out


class SessionReqNode:
    def __init__(
        self,
        req: Req,
        parent: Optional[SessionReqNode] = None,
        children=None,
    ):
        self.req = req
        self.parent = parent
        if parent is not None:
            parent.children.append(self)
        self.children = [] if not children else children

    def clear_children(self, req_dict):
        for req_node in self.children:
            req_node.clear(req_dict)
        self.children = []

    def clear(self, req_dict):
        for req_node in self.children:
            req_node.clear(req_dict)

        if self.req.finished_reason is None:
            self.req.to_finish = FINISH_ABORT()
        del req_dict[self.req.rid]

    def abort(self):
        if self.req.finished_reason is None:
            self.req.to_finish = FINISH_ABORT()

    def __str__(self):
        return self._str_helper(self.req.rid)

    def _str_helper(self, prefix=""):
        if len(self.children) == 0:
            return prefix + "\n"
        else:
            origin_prefix = prefix
            prefix += " -- " + self.children[0].req.rid
            ret = self.children[0]._str_helper(prefix)
            for child in self.children[1:]:
                prefix = " " * len(origin_prefix) + " \\- " + child.req.rid
                ret += child._str_helper(prefix)
            return ret


class Session:
    def __init__(
        self,
        capacity_of_str_len: int,
        session_id: Optional[str] = None,
        streaming: bool = False,
        timeout: Optional[float] = None,
    ):
        self.session_id = session_id if session_id is not None else uuid.uuid4().hex
        self.capacity_of_str_len = capacity_of_str_len
        self.streaming = streaming
        self.timeout = timeout
        self.last_active_time: float = time.monotonic()
        self.req_nodes: Dict[str, SessionReqNode] = {}
        self.close_on_finish: bool = False
        self._inflight: bool = False
        # Token-array lengths of last_req as of its finish_req. The share path
        # appends speculatively beyond these; only finish_req confirms them, so
        # _share_token_arrays trims back first (heals aborted turns).
        self.committed_origin_len: Optional[int] = None
        self.committed_unpadded_len: Optional[int] = None
        self.committed_fill_len: Optional[int] = None
        # #410 checkpoint rewind / branch: a token-level splice point the NEXT
        # turn starts from, consumed once. It reuses SessionParams.offset --
        # the mechanism that already rewrites session history
        # (``context[:offset] + new_tokens``) -- rather than adding a second
        # way to do the same thing. An explicit per-request offset always
        # wins; this is the server-side default when the client sends none.
        self.pending_rewind_offset: Optional[int] = None

    def is_timed_out(self) -> bool:
        if self.timeout is None:
            return False
        return time.monotonic() - self.last_active_time > self.timeout

    @staticmethod
    def _strip_bos_token(req: TokenizedGenerateReqInput, tokenizer) -> None:
        """Trim a leading BOS on an appended turn; shift mm offsets to match."""
        if not (
            tokenizer is not None
            and req.input_ids
            and req.input_ids[0] == tokenizer.bos_token_id
        ):
            return
        req.input_ids = req.input_ids[1:]
        if req.mm_inputs:
            for item in req.mm_inputs.mm_items:
                if item.offsets:
                    if any(s == 0 for s, _ in item.offsets):
                        logging.warning(
                            "mm_item offset starts at 0 (BOS position), "
                            "clamping to 0 after BOS strip"
                        )
                    item.offsets = [
                        (max(0, s - 1), max(0, e - 1)) for s, e in item.offsets
                    ]

    def _share_token_arrays(self, last_req: Req, new_input_ids):
        """Plain streaming append: reuse last_req's token arrays in place.

        Trims each array back to its committed length first — an earlier turn
        may have appended its tokens and then aborted before finish_req, and
        req_nodes still points at last_req, so anything beyond the committed
        lengths is unconfirmed. Then extends with last turn's output and the
        new input. Returns (input_ids, input_ids_unpadded, carry_fill);
        carry_fill (== the new origin) spares the first fill_ids rebuild.
        """
        out_tail = last_req.output_ids[: last_req.sampling_params.max_new_tokens]

        input_ids = last_req.origin_input_ids
        del input_ids[self.committed_origin_len :]
        if last_req.origin_input_ids_unpadded is input_ids:
            input_ids_unpadded = input_ids
        else:
            input_ids_unpadded = last_req.origin_input_ids_unpadded
            del input_ids_unpadded[self.committed_unpadded_len :]

        carry_fill = last_req.full_untruncated_fill_ids
        if (
            not isinstance(carry_fill, array)
            or carry_fill is input_ids
            or carry_fill is input_ids_unpadded
        ):
            # Unexpected type or aliased with an origin array (extending it
            # below would double-append): let _refresh_fill_ids rebuild.
            carry_fill = None
        else:
            del carry_fill[self.committed_fill_len :]
            baked = len(carry_fill) - len(input_ids)
            if 0 <= baked <= len(out_tail):
                carry_fill.extend(out_tail[baked:])
                carry_fill.extend(new_input_ids)
            else:
                carry_fill = None

        input_ids.extend(out_tail)
        input_ids.extend(new_input_ids)
        if input_ids_unpadded is not input_ids:
            input_ids_unpadded.extend(out_tail)
            input_ids_unpadded.extend(new_input_ids)
        return input_ids, input_ids_unpadded, carry_fill

    @staticmethod
    def _concat_token_arrays(
        last_req: Req,
        req: TokenizedGenerateReqInput,
        session_params,
        offset: Optional[int] = None,
    ):
        """Copy-based assembly for replace/offset/drop_previous_output turns.

        ``offset`` is the EFFECTIVE splice point (the request's own, or the
        session's pending #410 rewind offset), resolved by the caller so this
        function has one notion of where history is cut.
        """
        out_tail = last_req.output_ids[: last_req.sampling_params.max_new_tokens]

        input_ids = _concat(last_req.origin_input_ids, out_tail)
        if session_params.drop_previous_output:
            input_ids = last_req.origin_input_ids[:]
        if offset:
            input_ids = _concat(input_ids[:offset], req.input_ids)
        else:
            input_ids = _concat(input_ids, req.input_ids)

        input_ids_unpadded = _concat(last_req.origin_input_ids_unpadded, out_tail)
        if session_params.drop_previous_output:
            input_ids_unpadded = last_req.origin_input_ids_unpadded[:]
        if offset:
            input_ids_unpadded = _concat(input_ids_unpadded[:offset], req.input_ids)
        else:
            input_ids_unpadded = _concat(input_ids_unpadded, req.input_ids)
        return input_ids, input_ids_unpadded

    def create_req(
        self,
        req: TokenizedGenerateReqInput,
        tokenizer,
        vocab_size: int,
        eos_token_ids=None,
    ):
        assert req.session_params is not None
        self.last_active_time = time.monotonic()
        session_params = req.session_params
        # #410: the session's pending rewind/branch splice point applies to
        # this turn unless the client named its own. Consumed here, so a
        # rewind moves the session once and the turns after it append
        # normally.
        effective_offset = session_params.offset
        if not effective_offset:
            effective_offset = self.pending_rewind_offset
        self.pending_rewind_offset = None

        last_req_node = None
        last_req = None
        abort = False
        abort_message = ""
        if self.streaming:
            # Streaming sessions: only simple appends allowed; reject otherwise.
            if self._inflight:
                abort = True
                abort_message = "Streaming session already has an active request."
            elif session_params.replace:
                abort = True
                abort_message = "Streaming sessions do not support replace."
            elif session_params.drop_previous_output:
                abort = True
                abort_message = (
                    "Streaming sessions do not support drop_previous_output."
                )
            elif session_params.offset and session_params.offset != 0:
                # A CLIENT-supplied offset stays refused on a streaming
                # session. The #410 rewind offset is not refused here: it is
                # set by the scheduler, only while the session has no request
                # in flight, and it takes the copy path below like any other
                # history rewrite.
                abort = True
                abort_message = "Streaming sessions do not support offset."
            elif self.req_nodes:
                assert len(self.req_nodes) == 1
                # Peek (don't pop) the single req_node. req_nodes is updated
                # only in finish_req after the request completes successfully.
                [last_req_node] = self.req_nodes.values()
                last_req = last_req_node.req
        elif session_params.replace:
            if session_params.rid is None:
                for _, req_node in self.req_nodes.items():
                    req_node.clear(self.req_nodes)
            else:
                if session_params.rid not in self.req_nodes:
                    abort = True
                    abort_message = "Invalid request session id"
                else:
                    last_req_node = self.req_nodes[session_params.rid]
                    last_req_node.abort()
                    last_req = last_req_node.req
                    last_req_node.clear_children(self.req_nodes)
        else:
            if session_params.rid is not None:
                if session_params.rid not in self.req_nodes:
                    abort = True
                    abort_message = "Invalid request session id"
                else:
                    last_req_node = self.req_nodes[session_params.rid]
                    last_req = last_req_node.req
                    if not last_req.finished():
                        abort = True
                        abort_message = "Session request is appending to a request that hasn't finished."
                        logging.warning(abort_message)

        carry_fill = None
        if last_req is not None:
            self._strip_bos_token(req, tokenizer)
            # In-place sharing is only safe for the plain streaming append:
            # streaming sessions allow a single inflight request, last_req has
            # finished, and the committed_* lengths recorded by finish_req let
            # _share_token_arrays trim away tokens appended by an aborted turn.
            # offset / drop_previous_output rewrite history and must copy.
            can_share_token_arrays = (
                self.streaming
                and self.committed_origin_len is not None
                and not session_params.drop_previous_output
                and not effective_offset
            )
            if can_share_token_arrays:
                input_ids, input_ids_unpadded, carry_fill = self._share_token_arrays(
                    last_req, req.input_ids
                )
            else:
                input_ids, input_ids_unpadded = self._concat_token_arrays(
                    last_req, req, session_params, effective_offset
                )
        else:
            input_ids = req.input_ids
            input_ids_unpadded = req.input_ids

        new_req = Req(
            rid=req.rid,
            origin_input_text=None,
            origin_input_ids=input_ids,
            origin_input_ids_unpadded=input_ids_unpadded,
            sampling_params=req.sampling_params,
            lora_id=req.lora_id,
            session=self,
            custom_logit_processor=req.custom_logit_processor,
            stream=req.stream,
            return_logprob=req.return_logprob,
            top_logprobs_num=req.top_logprobs_num,
            token_ids_logprob=req.token_ids_logprob,
            vocab_size=vocab_size,
            eos_token_ids=eos_token_ids,
            require_reasoning=req.require_reasoning,
            return_hidden_states=req.return_hidden_states,
            return_routed_experts=req.return_routed_experts,
            routed_experts_start_len=req.routed_experts_start_len,
            priority=req.priority,
            routing_key=req.routing_key,
            extra_key=req.extra_key,
            http_worker_ipc=req.http_worker_ipc,
            time_stats=req.time_stats,
        )
        if last_req is not None:
            new_req.multimodal_inputs = last_req.multimodal_inputs
        new_req.tokenizer = tokenizer
        if carry_fill is not None:
            new_req.full_untruncated_fill_ids = carry_fill

        if abort:
            new_req.set_finish_with_abort(abort_message)
        elif self.streaming:
            # req_nodes is NOT updated here — finish_req() handles it.
            self._inflight = True
        else:
            new_req_node = SessionReqNode(new_req, last_req_node)
            self.req_nodes[req.rid] = new_req_node

        return new_req

    def finish_req(self, req):
        """Update req_nodes after a streaming request finishes successfully."""
        self._inflight = False
        if self.req_nodes:
            [prev_node] = self.req_nodes.values()
            prev_node.req.session = None
            self.req_nodes.clear()
        self.req_nodes[req.rid] = SessionReqNode(req)
        # Confirm this req's token arrays as the session's rollback point.
        self.committed_origin_len = len(req.origin_input_ids)
        self.committed_unpadded_len = len(req.origin_input_ids_unpadded)
        self.committed_fill_len = len(req.full_untruncated_fill_ids)

    def abort_req(self):
        """Clear inflight flag on abort (req_nodes stays unchanged)."""
        self._inflight = False


class SessionController:
    def __init__(self, tree_cache: BasePrefixCache):
        self.sessions: Dict[str, Session] = {}
        self._last_reap_time: float = 0.0
        self.tree_cache = tree_cache

    def __contains__(self, session_id: str) -> bool:
        return session_id in self.sessions

    def get(self, session_id: str) -> Optional[Session]:
        return self.sessions.get(session_id)

    def open(self, recv_req: OpenSessionReqInput) -> OpenSessionReqOutput:
        session_id = recv_req.session_id
        if session_id in self.sessions:
            logger.warning(f"session id {session_id} already exist, cannot open.")
            return OpenSessionReqOutput(session_id=session_id, success=False)
        elif session_id is None:
            logger.warning("session id is None, cannot open.")
            return OpenSessionReqOutput(session_id=session_id, success=False)
        else:
            self.sessions[session_id] = Session(
                recv_req.capacity_of_str_len,
                session_id,
                streaming=bool(recv_req.streaming),
                timeout=recv_req.timeout,
            )
            log_info_on_rank0(
                logger, f"Session opened: {session_id} (active={len(self.sessions)})"
            )
            return OpenSessionReqOutput(session_id=session_id, success=True)

    # -- #410 checkpoint branch / rewind ------------------------------------
    #
    # Both are expressed as a token-level splice point on the session, which
    # is the mechanism SessionParams.offset already implements. Neither copies
    # a KV page: the branch's first turn matches the checkpoint prefix in the
    # radix tree and shares the nodes that hold it, and the tree splits at the
    # point where the branch diverges. Copy-on-restore is what the radix tree
    # does anyway.

    def _checkpoint_node(self, session: Session) -> SessionReqNode:
        if not session.req_nodes:
            raise ValueError(
                f"session {session.session_id!r} has no completed turn to "
                "branch from or rewind to"
            )
        if len(session.req_nodes) > 1:
            raise ValueError(
                f"session {session.session_id!r} holds "
                f"{len(session.req_nodes)} request nodes; branch/rewind of a "
                "non-streaming session tree is not supported -- close the "
                "branches you do not want first"
            )
        [node] = session.req_nodes.values()
        return node

    def branch_from(
        self,
        *,
        parent_session_id: str,
        checkpoint_tokens: Sequence[int],
        new_session_id: Optional[str] = None,
    ) -> str:
        """Open a new session that continues from a checkpoint of another.

        The new session references the SAME last request node as the parent
        and carries the checkpoint's token count as its splice point, so its
        first turn assembles ``parent_context[:len(checkpoint)] + new_tokens``.
        Because a splice point is set, that turn takes the copy path -- the
        parent's token arrays are never mutated by its child, which is the one
        way a branch could corrupt the session it came from.
        """
        parent = self.sessions.get(parent_session_id)
        if parent is None:
            raise ValueError(
                f"unknown parent session {parent_session_id!r}; a branch is "
                "taken from a live server-side session"
            )
        node = self._checkpoint_node(parent)
        session_id = new_session_id or uuid.uuid4().hex
        if session_id in self.sessions:
            raise ValueError(f"session id {session_id} already exists, cannot branch")
        child = Session(
            parent.capacity_of_str_len,
            session_id,
            streaming=parent.streaming,
            timeout=parent.timeout,
        )
        # Share the parent's node object: the branch reads its token arrays
        # and never writes them (the splice point forces the copy path).
        child.req_nodes[node.req.rid] = SessionReqNode(node.req)
        child.committed_origin_len = parent.committed_origin_len
        child.committed_unpadded_len = parent.committed_unpadded_len
        child.committed_fill_len = parent.committed_fill_len
        child.pending_rewind_offset = len(checkpoint_tokens)
        self.sessions[session_id] = child
        log_info_on_rank0(
            logger,
            f"Session {session_id} branched from {parent_session_id} at "
            f"{len(checkpoint_tokens)} tokens (active={len(self.sessions)})",
        )
        return session_id

    def rewind_to(self, session_id: str, checkpoint_tokens: Sequence[int]) -> int:
        """Move a session's continuation point back to a checkpoint prefix.

        Only the splice point moves. The tokens beyond it stay in the radix
        tree, where they are ordinary cached pages: a later branch of the same
        checkpoint that regenerates the same continuation hits them, and the
        eviction policy reclaims them otherwise. Deleting them here would
        throw away a cache hit to save nothing.
        """
        session = self.sessions.get(session_id)
        if session is None:
            raise ValueError(f"unknown session {session_id!r}, cannot rewind")
        self._checkpoint_node(session)
        offset = len(checkpoint_tokens)
        session.pending_rewind_offset = offset
        session.last_active_time = time.monotonic()
        log_info_on_rank0(logger, f"Session {session_id} rewound to {offset} tokens")
        return offset

    def close(self, recv_req: CloseSessionReqInput):
        session_id = recv_req.session_id
        if session_id not in self.sessions:
            logger.warning(f"session id {session_id} does not exist, cannot delete.")
        else:
            self._close(session_id)

    def _close(self, session_id: str):
        session = self.sessions[session_id]
        req = None
        has_unfinished_request = False
        if session.streaming and session._inflight:
            has_unfinished_request = True
        elif session.streaming and session.req_nodes:
            assert len(session.req_nodes) == 1
            [last_node] = session.req_nodes.values()
            req = last_node.req
            if not req.finished():
                has_unfinished_request = True

        if has_unfinished_request:
            # An in-flight request is still decoding on this session's KV
            # memory. Freeing now would corrupt the scheduler. Mark the
            # session for deferred cleanup: the request keeps its session
            # reference so cache_finished_req takes the streaming path,
            # and we schedule release_session for after it completes.
            session.close_on_finish = True
            logger.info(
                "Deferring session close for %s (unfinished request)",
                session_id,
            )
            return

        # No owning request -- safe to release immediately.
        if session.streaming and session.req_nodes:
            req = next(iter(session.req_nodes.values())).req
            req.session = None

        # Release multimodal features held by session requests.
        # Session reqs skip the normal mm cleanup path (scheduler and
        # output_processor) so features stay alive until the session closes.
        seen_mm = set()
        for node in session.req_nodes.values():
            mm = node.req.multimodal_inputs
            if mm is not None and id(mm) not in seen_mm:
                seen_mm.add(id(mm))
                mm.release_features()
            node.req.multimodal_inputs = None

        self.tree_cache.release_session(session_id)
        del self.sessions[session_id]
        log_info_on_rank0(
            logger, f"Session closed: {session_id} (active={len(self.sessions)})"
        )

    def maybe_reap(self, now: float, interval: float = 1.0):
        # reap sessions every second
        if now - self._last_reap_time > interval:
            self._last_reap_time = now

            # Finish deferred closes for sessions whose requests completed.
            pending = [
                sid
                for sid, session in self.sessions.items()
                if session.close_on_finish and self._all_requests_finished(session)
            ]
            for sid in pending:
                log_info_on_rank0(
                    logger, f"Deferred close ready for session {sid}, releasing."
                )
                # Reset close_on_finish so _close proceeds with the release.
                self.sessions[sid].close_on_finish = False
                self._close(sid)

            timed_out = [
                sid for sid, session in self.sessions.items() if session.is_timed_out()
            ]
            for sid in timed_out:
                log_info_on_rank0(logger, f"Session {sid} timed out, closing.")
                self._close(sid)

    @staticmethod
    def _all_requests_finished(session: Session) -> bool:
        if not session.req_nodes:
            return True
        return all(node.req.finished() for node in session.req_nodes.values())

    @staticmethod
    def adjust_mm_offsets(recv_req: TokenizedGenerateReqInput, req: Req, image_inputs):
        # For session requests, adjust mm_inputs offsets by the prefix length.
        # Session.create_req prepends previous context to origin_input_ids,
        # so offsets from the new prompt need to be shifted.
        if len(recv_req.input_ids) >= len(req.origin_input_ids):
            return
        prefix_len = len(req.origin_input_ids) - len(recv_req.input_ids)
        for mm_item in image_inputs.mm_items:
            if mm_item.offsets:
                mm_item.offsets = [
                    (start + prefix_len, end + prefix_len)
                    for start, end in mm_item.offsets
                ]
