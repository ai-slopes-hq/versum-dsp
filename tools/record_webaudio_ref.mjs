// Records a Web Audio reference for Shaper<2049>: 4x WaveShaperNode with the fold curve sin(5*pi*x), on a 440 Hz
// sine (peak 0.8) and a 220 Hz saw (peak 0.8), 4096 samples at 48000 Hz. Needs Google Chrome (headless) and node.
// Usage: node tools/record_webaudio_ref.mjs > tests/fixtures/webaudio_shaper_ref.json
import { spawnSync } from 'node:child_process';
import { mkdtempSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';

const CHROME = process.env.CHROME || '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome';
const page = `<!doctype html><body><pre id="out">pending</pre><script>
const SR = 48000, N = 4096, T = 2049;
const curve = new Float32Array(T);
for (let i = 0; i < T; i++) curve[i] = Math.sin(5 * Math.PI * (i / (T - 1) * 2 - 1));
const sine = new Float32Array(N), saw = new Float32Array(N);
for (let i = 0; i < N; i++) {
  sine[i] = 0.8 * Math.sin(2 * Math.PI * 440 * i / SR);
  saw[i] = 0.8 * (2 * ((220 * i / SR) % 1) - 1);
}
async function render(input) {
  const ctx = new OfflineAudioContext(1, N, SR);
  const buf = ctx.createBuffer(1, N, SR);
  buf.copyToChannel(input, 0);
  const src = ctx.createBufferSource(); src.buffer = buf;
  const ws = ctx.createWaveShaper(); ws.curve = curve; ws.oversample = '4x';
  src.connect(ws); ws.connect(ctx.destination); src.start();
  return (await ctx.startRendering()).getChannelData(0);
}
(async () => {
  const a = await render(sine), b = await render(saw);
  const f = (x) => '[' + Array.from(x).join(',') + ']';
  document.getElementById('out').textContent = JSON.stringify({sr: SR, curve: 'sin(5*pi*x), 2049 points', curve_values: '__C__', sine_in: '__SI__', sine_out: '__SO__', saw_in: '__WI__', saw_out: '__WO__'})
    .replace('"__C__"', f(curve)).replace('"__SI__"', f(sine)).replace('"__SO__"', f(a)).replace('"__WI__"', f(saw)).replace('"__WO__"', f(b));
})();
</script></body>`;
const dir = mkdtempSync(join(tmpdir(), 'wa-ref-'));
const file = join(dir, 'ref.html');
writeFileSync(file, page);
const r = spawnSync(CHROME, ['--headless=new', '--disable-gpu', '--no-sandbox', '--virtual-time-budget=20000', '--dump-dom', 'file://' + file], { encoding: 'utf8', maxBuffer: 1 << 28 });
const m = /<pre id="out">([\s\S]*?)<\/pre>/.exec(r.stdout || '');
if (!m || m[1] === 'pending') { console.error('no result', r.stderr); process.exit(1); }
const json = m[1].replace(/&quot;/g, '"').replace(/&amp;/g, '&');
JSON.parse(json);
process.stdout.write(json + '\n');
