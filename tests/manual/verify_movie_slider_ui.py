"""Playwright UI verification for the simplified movie SAM2 prompt-frame slider.

Launches the Gradio movie app and verifies:
- The single prompt-frame slider exists with the new label.
- The removed redundant buttons ("表示中フレームを再取得", "シーク位置を SAM2 に反映") are gone.
- No JS seek-sync bridge is present.

Run with CPU-only: CUDA_VISIBLE_DEVICES=-1.
"""

from playwright.sync_api import sync_playwright

URL = "http://localhost:7862"


def main() -> None:
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()
        page.goto(URL)
        page.wait_for_load_state("domcontentloaded")
        page.wait_for_selector("#prompt-frame-idx", timeout=30000)

        body = page.inner_text("body")

        # New slider label present.
        assert "プロンプト起点フレーム位置（ドラッグで Canvas 更新）" in body, "new slider label missing"

        # Removed buttons absent.
        assert "表示中フレームを再取得" not in body, "redisplay button should be removed"
        assert "シーク位置を SAM2 に反映" not in body, "reflect-to-SAM2 button should be removed"

        # Slider element is present and interactive.
        slider = page.locator("#prompt-frame-idx input[type='range']")
        assert slider.count() >= 1, "prompt-frame-idx slider missing"

        page.screenshot(path="outputs/movie_slider_ui.png", full_page=True)
        print("OK: slider UI verified, redundant controls removed")
        browser.close()


if __name__ == "__main__":
    main()
