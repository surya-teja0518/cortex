"""Playwright adapter for the Workday fill helper (agents/workday_fill.py holds every decision).

Local only, headed (he sees and drives the browser), throwaway profile: no cookies or passwords are saved.
Install once on the Mac:  pip install -e ".[workday]" && python -m playwright install chromium
"""
from cortex.agents.workday_fill import FieldInfo

_COLLECT = """() => Array.from(document.querySelectorAll('input, textarea')).map((el, i) => {
  const anc = []; let p = el.parentElement;
  while (p && anc.length < 15) { const a = p.getAttribute('data-automation-id'); if (a) anc.push(a); p = p.parentElement; }
  let kind = el.tagName === 'TEXTAREA' ? 'textarea' : (el.type || 'text').toLowerCase();
  if (kind === 'email' || kind === 'tel' || kind === 'search') kind = 'text';
  if (!['text', 'textarea', 'file', 'password'].includes(kind)) kind = 'other';
  const visible = !!(el.offsetWidth || el.offsetHeight || el.getClientRects().length);
  const usable = kind === 'file' || (visible && !el.disabled && !el.readOnly);
  const label = (el.labels && el.labels[0] ? el.labels[0].innerText : el.getAttribute('aria-label') || '').trim();
  return {i, aid: el.getAttribute('data-automation-id') || '', kind: usable ? kind : 'other',
          empty: !el.value, anc, label};
})"""

_STILL_EMPTY = "(el) => !el.value"


class PlaywrightPage:
    def __init__(self, context):
        self._ctx = context

    def _page(self):
        pages = [p for p in self._ctx.pages if not p.is_closed()]
        return pages[-1] if pages else None

    def is_open(self) -> bool:
        return self._page() is not None

    def url(self) -> str:
        p = self._page()
        return p.url if p else ""

    def fields(self) -> list[FieldInfo]:
        p = self._page()
        if p is None:
            return []
        try:
            raw = p.evaluate(_COLLECT)
        except Exception:  # noqa: BLE001 — page navigating mid-read: just try again next round
            return []
        return [FieldInfo(r["i"], r["aid"], r["kind"], r["empty"], tuple(r["anc"]), r["label"]) for r in raw]

    def _el(self, index: int):
        return self._page().locator("input, textarea").nth(index)

    def type_into(self, index: int, value: str) -> None:
        el = self._el(index)
        if el.evaluate(_STILL_EMPTY):            # the DOM may have changed since fields(): never overwrite
            el.fill(value)

    def attach(self, index: int, path: str) -> None:
        self._el(index).set_input_files(path)


def open_browser(start_url: str | None):
    """Returns (playwright, browser, PlaywrightPage). Caller closes them."""
    try:
        from playwright.sync_api import sync_playwright
    except ImportError as e:
        raise RuntimeError('Playwright missing: pip install -e ".[workday]" && python -m playwright install chromium') \
            from e
    pw = sync_playwright().start()
    browser = pw.chromium.launch(headless=False)
    ctx = browser.new_context()                  # throwaway: nothing persists after you close it
    page = ctx.new_page()
    if start_url:
        page.goto(start_url)
    return pw, browser, PlaywrightPage(ctx)
