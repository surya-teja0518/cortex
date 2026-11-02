"""Streamlit review list (MVP). Run:  streamlit run src/cortex/ui/app.py --server.address 127.0.0.1
Local-only by design. Set CORTEX_UI_PASSWORD to require a password before anything renders."""
import hmac
import os

import streamlit as st

from cortex.core.audit import AuditLog
from cortex.core.secrets import get_secret
from cortex.db.schema import get_session
from cortex.graph import Deps, build_graph, open_checkpointer, submit_decision
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
    statuses = st.multiselect("Status", ["to_apply", "applied", "closed", "not_accepting", "exp_mismatch"],
                              default=["to_apply"])

rows = review_queue(session, uid, tracks=tracks, city=city, min_synapse=min_syn,
                    include_gated=show_gated, include_dups=show_dups, statuses=tuple(statuses))
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


gate = build_graph(Deps(session_factory=lambda: session, audit=AuditLog()), _checkpointer())
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
                submit_decision(gate, session, uid, p["job_id"], {"decision": decision, "note": note or None})
                st.rerun()
