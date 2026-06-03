"""Tracking Overlay UI のブラウザ確認スクリプト（webapp-testing skill）。

動画版 Gradio を起動し、Advanced アコーディオンを開いて
- Tracking Overlay を生成 Checkbox
- Tracking Overlay (追跡確認用) Video 出力
の存在とスクリーンショットを確認する。
"""

from __future__ import annotations

import sys

from playwright.sync_api import sync_playwright

URL = "http://127.0.0.1:7871"
SHOT = "outputs/_ui_check_tracking_overlay.png"


def main() -> int:
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()
        page.goto(URL)
        # Gradio は websocket を張り続けるため networkidle は発火しない。
        page.wait_for_load_state("domcontentloaded")
        page.wait_for_selector("text=transparent-background", timeout=60000)
        page.wait_for_timeout(2000)

        # Advanced アコーディオンを開く（折りたたみ済みのため）
        for label in ("Advanced: 動画処理設定", "Advanced", "詳細"):
            handle = page.query_selector(f"text={label}")
            if handle is not None:
                handle.click()
                page.wait_for_timeout(500)
                break

        page.screenshot(path=SHOT, full_page=True)
        content = page.content()

        checks = {
            "Tracking Overlay を生成 (Checkbox label)": "Tracking Overlay を生成" in content,
            "Tracking Overlay (追跡確認用) (Video label)": "Tracking Overlay (追跡確認用)" in content,
            "overlay info text": "追従" in content or "確認" in content,
        }
        browser.close()

    ok = all(checks.values())
    for name, passed in checks.items():
        print(f"[{'OK' if passed else 'NG'}] {name}")
    print(f"screenshot: {SHOT}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
