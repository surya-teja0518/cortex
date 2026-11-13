"""Streamlit review list (MVP). Run:  streamlit run src/cortex/ui/app.py --server.address 127.0.0.1
Local-only by design. Set CORTEX_UI_PASSWORD to require a password before anything renders."""
import hmac
import os

import streamlit as st

from cortex.core.audit import AuditLog
from cortex.core.secrets import get_secret
from cortex.db.schema import get_session
from cortex.graph import Deps, build_graph, open_checkpointer, submit_decision
from cortex.core.config import load_config
from cortex.review import pending_approvals, record_skip, review_queue

st.set_page_config(page_title="CORTEX", layout="wide")

pw = os.getenv("CORTEX_UI_PASSWORD")
if pw:
    given = st.sidebar.text_input("Password", type="password")
    if not hmac.compare_digest(given.encode(), pw.encode()):
        st.warning("Enter password")
        st.stop()

uid = os.getenv("CORTEX_USER_ID", "surya")
session = get_session(get_secret("CORTEX_DB_URL", required=False) or "sqlite:///data/cortex.db")

st.title("CORTEX: review queue")
with st.sidebar:
    tracks = st.multiselect("Track", ["data", "database", "ai"], default=["data", "database", "ai"])
    city = st.text_input("Location contains", "")
    min_syn = st.slider("Min SYNAPSE score", 0, 100, 70)
    show_gated = st.checkbox("Include gated (title/YOE)", False)
    show_dups = st.checkbox("Include possible duplicates", False)
    _max_age = load_config().get("freshness", {}).get("max_age_days", 30)
    show_old = st.checkbox(f"Include postings older than {_max_age} days (likely closed)", False)
    statuses = st.multiselect("Status", ["to_apply", "applied", "closed", "not_accepting", "exp_mismatch"],
                              default=["to_apply"])

rows = review_queue(session, uid, tracks=tracks, city=city, min_synapse=min_syn,
                    include_gated=show_gated, include_dups=show_dups, statuses=tuple(statuses),
                    max_age_days=None if show_old else _max_age)
st.caption(f"{len(rows)} jobs. Applications are never auto-submitted: open the link and apply yourself.")
event = st.dataframe(rows, use_container_width=True, hide_index=True,
                     on_select="rerun", selection_mode="multi-row",
                     column_config={"url": st.column_config.LinkColumn("job link"),
                                    "flagged": st.column_config.CheckboxColumn("injection flag")})
picked = [rows[i]["id"] for i in event.selection.rows]
if st.button(f"Skip selected ({len(picked)})", disabled=not picked,
             help="Record 'not applying'. Skips are the negative labels for `cortex eval`."):
    audit = AuditLog()
    for jid in picked:
        record_skip(session, uid, jid, audit)
    st.rerun()  # skipped jobs drop out of the queue


# ---- Human approval gate (the ONLY place approvals happen in v1) ----
st.header("Pending approvals")
st.caption("Each resume below already passed Truth Guard. Approving records your decision; "
           "it never submits an application.")


@st.cache_resource
def _checkpointer():
    return open_checkpointer()


from cortex.cli import packet_builder

_builder, _why = packet_builder()
if _builder is None:
    st.warning(f"Packets disabled until fixed: {_why}. Approvals still work; run `cortex packet <job_id>` later.")
gate = build_graph(Deps(session_factory=lambda: session, audit=AuditLog(), packet=_builder), _checkpointer())
pending = pending_approvals(session, uid)
if not pending:
    st.info("Nothing waiting for approval.")
for p in pending:
    with st.expander(f"{p['company']} | {p['role']} (Truth Guard passed after {p['attempts']} attempt(s))"):
        if p["url"]:
            st.link_button("Open job posting", p["url"])
        st.json(p["resume"])
        note = st.text_input("Note (optional)", key=f"note-{p['job_id']}", max_chars=500)
        c1, c2 = st.columns(2)
        for col, label, decision in ((c1, "Approve", "approve"), (c2, "Reject", "reject")):
            if col.button(label, key=f"{decision}-{p['job_id']}"):
                out = submit_decision(gate, session, uid, p["job_id"], {"decision": decision, "note": note or None})
                if out.get("packet_error"):
                    st.session_state["packet_msg"] = f"Approved, but the packet failed: {out['packet_error']}"
                elif out.get("packet_path"):
                    st.session_state["packet_msg"] = f"Approved. Packet ready: {out['packet_path']}"
                st.rerun()

if st.session_state.get("packet_msg"):
    st.info(st.session_state.pop("packet_msg"))


# ---- Tracker: what happened after you applied (you report it; CORTEX never contacts anyone) ----
from datetime import date

from cortex.agents.tracker import (
    TRANSITIONS,
    TrackerError,
    followups,
    history,
    report,
    undo,
    update_status,
)
from cortex.core.config import load_config
from cortex.review import applications, ready_to_apply

st.header("Applications")
ready = ready_to_apply(session, uid)
if ready:
    st.subheader("Approved: apply, then mark it here")
for r in ready:
    c1, c2, c3 = st.columns([5, 2, 2])
    c1.write(f"**{r['company']}** | {r['role']}" + (f"  \n`{r['packet']}`" if r["packet"] else ""))
    if r["url"]:
        c2.link_button("Open posting", r["url"])
    if c3.button("I applied", key=f"applied-{r['job_id']}"):
        update_status(session, uid, r["job_id"], "applied", audit=AuditLog())
        st.rerun()

_tcfg = load_config().get("tracker", {})
_days, _stale = _tcfg.get("followup_days", 7), _tcfg.get("stale_days", 30)
_fups = followups(session, uid, date.today(), _days, _stale)  # noqa: DTZ011 (local date)
if _fups:
    st.subheader(f"Follow-ups due ({len(_fups)}): no update for {_days}-{_stale} days")
    st.dataframe(_fups, use_container_width=True, hide_index=True,
                 column_config={"url": st.column_config.LinkColumn("job link")})

apps = applications(session, uid)
_rep = report(session, uid, date.today())  # noqa: DTZ011 (local date)
st.caption(f"{_rep.applied_total} applications, response rate "
           + (f"{_rep.response_rate:.0%}" if _rep.response_rate is not None else "n/a")
           + " (interview or offer / applied)")
if apps:
    with st.expander(f"Update a status ({len(apps)} applications)"):
        st.dataframe(apps, use_container_width=True, hide_index=True,
                     column_config={"url": st.column_config.LinkColumn("job link")})
        movable = [a for a in apps if TRANSITIONS.get(a["status"])]
        if movable:
            pick = st.selectbox("Application", movable, format_func=lambda a: f"[{a['job_id']}] {a['company']} | "
                                f"{a['role']} ({a['status']})")
            new = st.selectbox("New status", sorted(TRANSITIONS[pick["status"]]))
            tnote = st.text_input("Note (optional)", max_chars=300, key="tracker-note")
            if st.button("Record"):
                try:
                    update_status(session, uid, pick["job_id"], new, note=tnote or None, audit=AuditLog())
                    st.rerun()
                except TrackerError as e:
                    st.error(str(e))
        undoable = [a for a in apps if any(e.status != "undo" for e in history(session, uid, a["job_id"]))]
        if undoable:
            upick = st.selectbox("Undo the last change for", undoable, key="undo-pick",
                                 format_func=lambda a: f"[{a['job_id']}] {a['company']} | {a['role']} ({a['status']})")
            if st.button("Undo last change"):
                try:
                    undo(session, uid, upick["job_id"], audit=AuditLog())
                    st.rerun()
                except TrackerError as e:
                    st.error(str(e))
