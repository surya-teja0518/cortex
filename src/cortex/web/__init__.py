"""Hosted dashboard (Phase 3, 2026-11-19): FastAPI + HTMX, Google sign-in for ONE allowed address.

The phone/any-device replacement for the Streamlit dashboard. Every action calls the same Python code as the
CLI (review.record_skip, tracker.update_status / undo, graph.submit_decision), so the same rules apply wherever
a decision is made. Tailoring and packets stay on a computer: resume files are never hosted.
"""
