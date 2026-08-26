"""诊断索引管理页:抓 console 错误 + 截图 + dump 目录表 DOM。"""
from playwright.sync_api import sync_playwright

msgs = []
with sync_playwright() as p:
    b = p.chromium.launch(headless=True)
    page = b.new_page()
    page.on("console", lambda m: msgs.append(f"[{m.type}] {m.text[:200]}"))
    page.on("pageerror", lambda e: msgs.append(f"[PAGEERROR] {str(e)[:300]}"))
    page.goto("http://127.0.0.1:8747")
    page.wait_for_load_state("networkidle")
    page.click("#tabManage")
    page.wait_for_timeout(2500)  # 等 pollStatus + folders fetch
    page.screenshot(path="logs/ui_manage.png", full_page=True)
    rows = page.locator("#folderRows tr").count()
    html = page.locator("#folderRows").inner_html()[:500]
    panels = page.evaluate("""() => ({
        barThumb: document.querySelector('#barThumb')?.style.width,
        kvTotal: document.querySelector('#kvTotal')?.textContent,
        hit: document.querySelector('#hit')?.textContent,
        idx: document.querySelector('#idx')?.textContent,
    })""")
    print("folderRows <tr> count:", rows)
    print("folderRows html head:", html.replace("\n", " ")[:400])
    print("panels:", panels)
    b.close()

print("--- console ---")
for m in msgs:
    print(m)
