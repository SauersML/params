// Every ```mermaid block of a Markdown file through mermaid.parse, Mermaid's own parser, in Chrome (no layout: the
// renderer stops at 500 edges and teacher circuits reach ~1,000). Blocks are named by a preceding "<!-- name -->".
//   PUPPETEER_SKIP_DOWNLOAD=1 npm install --prefix DIR @mermaid-js/mermaid-cli@11.4.2
//   node experiments/oracle/mermaid_parse.mjs DIR CIRCUITS.md ["/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"]
// prints {"diagrams", "failed"} and the first failures; exits 1 on any.
import fs from 'fs';
import path from 'path';
import {createRequire} from 'module';
const [, , dir, mdPath, chrome = '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome'] = process.argv;
const require = createRequire(path.join(path.resolve(dir), 'node_modules', 'x.js'));
const puppeteer = (await import(require.resolve('puppeteer-core'))).default;
const md = fs.readFileSync(mdPath, 'utf8');
const blocks = [...md.matchAll(/<!-- (.*?) -->\n```mermaid\n([\s\S]*?)\n```/g)].map(m => [m[1], m[2]]);
const browser = await puppeteer.launch({executablePath: chrome, args: ['--no-sandbox'], headless: true});
const page = await browser.newPage();
await page.addScriptTag({path: path.join(path.resolve(dir), 'node_modules/mermaid/dist/mermaid.min.js')});
await page.evaluate(() => mermaid.initialize({startOnLoad: false, maxEdges: 100000, maxTextSize: 50000000}));
let failed = 0;
for (const [name, text] of blocks) {
  const err = await page.evaluate(async (t) => { try { await mermaid.parse(t); return null; } catch (e) { return String(e.message || e); } }, text);
  if (err) { failed += 1; if (failed <= 5) console.log(name, err.slice(0, 300)); }
}
console.log(JSON.stringify({diagrams: blocks.length, failed}));
await browser.close();
process.exit(failed ? 1 : 0);
