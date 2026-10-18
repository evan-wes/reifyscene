// The README's viewer animation, step 1: screenshot the app's scene panel -- the frame, the
// scene graph at that frame and the presence timeline -- at every frame of a run.
//
//   reifyscene serve --runs RUN &                        # the app, at http://127.0.0.1:8765
//   node scripts/readme_media/capture_viewer.mjs RUN FRAMES_DIR [URL] [THEME]
//   python scripts/readme_media/viewer_animation.py FRAMES_DIR OUT.webp
//
// Needs Node and Playwright with a Chromium (`npm install playwright && npx playwright install
// chromium`, outside the repository; set PLAYWRIGHT to its module path if it is not on Node's
// path). A development tool only: nothing in the pipeline uses it. The page loads Cytoscape
// from cdnjs, so the machine needs network access. THEME is the colour scheme the browser asks
// for, "dark" (the default) or "light"; the app follows it.
import fs from "node:fs";
import path from "node:path";
import { createRequire } from "node:module";

const require = createRequire(import.meta.url);
const { chromium } = require(process.env.PLAYWRIGHT || "playwright");

const [run, out, url = "http://127.0.0.1:8765", theme = "dark"] = process.argv.slice(2);
if (!run || !out || !["dark", "light"].includes(theme)) {
  console.error("usage: capture_viewer.mjs RUN FRAMES_DIR [URL] [dark|light]");
  process.exit(2);
}
fs.mkdirSync(out, { recursive: true });
const browser = await chromium.launch();
const page = await browser.newPage({ viewport: { width: 1500, height: 1000 }, deviceScaleFactor: 1,
                                     colorScheme: theme });
await page.goto(`${url}/?run=${encodeURIComponent(run)}`);
await page.waitForFunction(() => typeof S !== "undefined" && S.cy && S.frames.length > 0);
const n = await page.evaluate(() => S.frames.length);
const panel = page.locator("section.scene-panel");
for (let i = 0; i < n; i++) {
  await page.evaluate((k) => setFrame(k), i);
  await page.waitForFunction(() => {          // this frame's image is in, and nothing renders
    const img = document.getElementById("frame-img");
    return img.complete && img.naturalWidth > 0 && img.src.includes(`/${S.frames[S.fi]}.jpg`)
      && document.getElementById("stage-spinner").hidden;
  }, null, { timeout: 60000 });
  await page.waitForTimeout(150);             // the graph's edges settle
  await panel.screenshot({ path: path.join(out, `${String(i).padStart(4, "0")}.png`) });
}
console.log(`captured ${n} frames of ${run} to ${out}`);
await browser.close();
