// The era switch of a site page and the pure site logic of the viewer (src/shapesmith/web/static/viewer.js), run
// with `node --test`. The page tests run the viewer in a node vm with a minimal DOM: the element ids of index.html,
// scripts appended to document.head are read from the file system relative to the page as from file://.
import assert from 'node:assert/strict';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { createRequire } from 'node:module';
import test from 'node:test';
import vm from 'node:vm';

const require = createRequire(import.meta.url);
const STATIC = new URL('../../src/shapesmith/web/static/', import.meta.url);
const SCRIPT = fs.readFileSync(new URL('viewer.js', STATIC), 'utf8');
const INDEX = fs.readFileSync(new URL('index.html', STATIC), 'utf8');
const viewer = require('../../src/shapesmith/web/static/viewer.js');

const both = ['png', 'pdf'];

/** A gallery manifest with the given channels -> variables of variant "emb" (one region, one category). */
function gallery(era, channels, extra) {
  const plots = {};
  for (const [channel, variables] of Object.entries(channels)) {
    plots[channel] = { nominal: { inclusive: Object.fromEntries(variables.map((v) => [v, both])) } };
  }
  return {
    schema: 1, generator: 'shapesmith test', kind: 'control', title: 'run ' + era, description: 'about ' + era,
    updated: '2026-10-19T10:00:00', site: { key: era },
    channels: Object.keys(channels).map((key) => ({ key: key, label: key })),
    regions: [{ key: 'nominal', label: 'nominal' }], categories: [{ key: 'inclusive', label: 'inclusive' }],
    variables: ['m_vis', 'pt_1', 'eta_1'].map((key) => ({ key: key, label: key })),
    variants: [{ key: 'emb', label: 'emb', sources: [{ channels: Object.keys(channels), stamp: 'S' + era }] }, ...(extra || [])],
    plots: { emb: plots },
    yields: {},
  };
}

const SITE = {
  schema: 1, generator: 'shapesmith test', level: 'site', kind: 'control', title: 'smrun2 · control plots', updated: '2026-10-19T11:00:00',
  eras: [
    { key: '2016preVFP', label: '2016preVFP', updated: '2026-10-19T09:00:00' },
    { key: '2017', label: '2017', updated: '2026-10-19T09:30:00' },
    { key: '2018', label: '2018', updated: '2026-10-19T10:00:00' },
  ],
};

/** A site directory: site.js and the manifest.js of the eras 2017 and 2018 (2016preVFP is listed but missing). */
function siteDirectory() {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), 'shapesmith-site-'));
  fs.writeFileSync(path.join(root, 'site.js'), 'window.SHAPESMITH_SITE = ' + JSON.stringify(SITE) + ';\n');
  const eras = {
    2017: gallery('2017', { et: ['m_vis', 'pt_1'], tt: ['m_vis'] }),
    2018: gallery('2018', { et: ['m_vis', 'pt_1', 'eta_1'], mt: ['m_vis', 'pt_1'], tt: ['m_vis'] }, [{ key: 'emb_ff', label: 'emb ff', sources: [] }]),
  };
  eras[2018].plots.emb_ff = { mt: { nominal: { inclusive: { pt_1: both } } } };
  for (const [era, m] of Object.entries(eras)) {
    fs.mkdirSync(path.join(root, era));
    fs.writeFileSync(path.join(root, era, 'manifest.js'), 'window.SHAPESMITH_GALLERY = ' + JSON.stringify(m) + ';\n');
  }
  return root;
}

// ---------------------------------------------------------------- a minimal DOM

class Element {
  constructor(document, tag) {
    this.ownerDocument = document;
    this.tagName = String(tag).toUpperCase();
    this.children = [];
    this.attributes = {};
    this.dataset = {};
    this.style = {};
    const classes = new Set();
    this.classList = { add: (c) => classes.add(c), remove: (c) => classes.delete(c), toggle: (c, on) => (on ?? !classes.has(c) ? classes.add(c) : classes.delete(c)), contains: (c) => classes.has(c) };
    this.listeners = {};
    this.hidden = false;
    this.disabled = false;
    this.value = '';
    this.className = '';
    this.parent = null;
    this.text = '';
  }

  get textContent() {
    return this.text + this.children.map((c) => c.textContent).join('');
  }

  set textContent(value) {
    this.children = [];
    this.text = String(value);
  }

  setAttribute(key, value) {
    this.attributes[key] = String(value);
    if (key.startsWith('data-')) this.dataset[key.slice(5).replace(/-(\w)/g, (_, c) => c.toUpperCase())] = String(value);
    if (key === 'id') this.ownerDocument.ids[value] = this;
  }

  getAttribute(key) {
    return Object.prototype.hasOwnProperty.call(this.attributes, key) ? this.attributes[key] : null;
  }

  removeAttribute(key) {
    delete this.attributes[key];
  }

  addEventListener(type, listener) {
    (this.listeners[type] = this.listeners[type] || []).push(listener);
  }

  dispatch(type, event) {
    for (const listener of this.listeners[type] || []) listener(Object.assign({ type: type, target: this, button: 0, preventDefault() {} }, event));
  }

  click() {
    this.dispatch('click');
  }

  append(...nodes) {
    for (const node of nodes) {
      const child = typeof node === 'string' ? Object.assign(new Element(this.ownerDocument, '#text'), { text: node }) : node;
      child.parent = this;
      this.children.push(child);
      if (this === this.ownerDocument.head && child.tagName === 'SCRIPT') this.ownerDocument.load(child);
    }
  }

  replaceChildren(...nodes) {
    this.children = [];
    this.text = '';
    this.append(...nodes);
  }

  remove() {
    if (this.parent) this.parent.children = this.parent.children.filter((c) => c !== this);
    this.parent = null;
  }

  replaceWith(node) {
    const parent = this.parent;
    if (!parent) return;
    parent.children[parent.children.indexOf(this)] = node;
    node.parent = parent;
  }

  focus() {
    this.ownerDocument.activeElement = this;
  }

  blur() {}

  select() {}

  querySelector() {
    return null;
  }

  all() {
    return [this, ...this.children.flatMap((c) => c.all())];
  }
}

/** The viewer on a page in `directory`: a site page (level "site") or a gallery page, at `hash`. */
function page(directory, level, hash) {
  const document = {
    ids: {}, readyState: 'complete', activeElement: null, title: '', scripts: [],
    addEventListener() {},
    getElementById(id) {
      return this.ids[id] || null;
    },
    createElement(tag) {
      return new Element(this, tag);
    },
    querySelectorAll() {
      return [];
    },
    load(script) { // a script appended to <head>: read relative to the page (no query), run, then onload/onerror
      this.scripts.push(script.src);
      setTimeout(() => {
        const file = path.join(directory, decodeURIComponent(script.src.split('?')[0]));
        if (!fs.existsSync(file)) return script.onerror && script.onerror();
        vm.runInContext(fs.readFileSync(file, 'utf8'), context);
        if (script.onload) script.onload();
      }, 0);
    },
  };
  document.documentElement = new Element(document, 'html');
  if (level) document.documentElement.dataset.level = level;
  document.head = new Element(document, 'head');
  document.body = new Element(document, 'body');
  for (const [, tag, id] of INDEX.matchAll(/<(\w+)[^>]*\sid="([^"]+)"/g)) {
    const node = document.createElement(tag);
    node.setAttribute('id', id);
    if (/\shidden[\s>]/.test(INDEX.slice(INDEX.indexOf('id="' + id + '"')).split('>')[0] + '>')) node.hidden = true;
  }
  const location = { protocol: 'file:', hash: hash ? '#' + hash.replace(/^#/, '') : '', href: '' };
  const window = { addEventListener() {}, isSecureContext: false };
  const context = vm.createContext(Object.assign(window, {
    window: window, document: document, location: location, navigator: {}, setTimeout: setTimeout, clearTimeout: clearTimeout,
    history: { replaceState(state, title, url) { location.hash = url; } },
    localStorage: { getItem() { return null; }, setItem() {} },
  }));
  if (level === 'site') vm.runInContext(fs.readFileSync(path.join(directory, 'site.js'), 'utf8'), context);
  else vm.runInContext(fs.readFileSync(path.join(directory, 'manifest.js'), 'utf8'), context);
  vm.runInContext(SCRIPT, context);
  return { document: document, window: window, location: location, $: (id) => document.ids[id] };
}

const settle = () => new Promise((resolve) => setTimeout(resolve, 20));
const pressed = (p, id) => p.$(id).children.filter((b) => b.getAttribute('aria-pressed') === 'true').map((b) => b.textContent);
const images = (p) => p.$('grid').all().filter((n) => n.tagName === 'IMG').map((n) => n.attributes.src);
const button = (p, id, text) => p.$(id).children.find((b) => b.textContent === text);

// ---------------------------------------------------------------- pure logic

test('site eras: the wanted one, else the most recent', () => {
  assert.deepEqual(viewer.siteEras(SITE).map((e) => e.key), ['2016preVFP', '2017', '2018']);
  assert.equal(viewer.pickEra(SITE, '2017'), '2017');
  assert.equal(viewer.pickEra(SITE, undefined), '2018');
  assert.equal(viewer.pickEra(SITE, '2016'), '2018');
  assert.equal(viewer.pickEra({ eras: 'junk' }, '2017'), null);
  assert.deepEqual(viewer.siteEras({ eras: [null, { key: '' }, { key: 'a' }] }).map((e) => e.key), ['a']);
});

test('era paths are relative to the site page, so that they work from file:// and over HTTP', () => {
  assert.equal(viewer.eraBase('2016preVFP'), '2016preVFP/');
  assert.equal(viewer.eraBase(null), '');
  assert.equal(viewer.eraManifestUrl(SITE, '2017'), '2017/manifest.js?v=20261019T093000');
  assert.equal(viewer.eraManifestUrl(SITE, '2017', 'manifest.json'), '2017/manifest.json?v=20261019T093000');
  assert.equal(viewer.eraManifestUrl(SITE, 'x y'), 'x%20y/manifest.js');
  const m = gallery('2018', { et: ['m_vis'] });
  const sel = { channel: 'et', region: 'nominal', category: 'inclusive', variable: 'm_vis' };
  assert.equal(viewer.imageUrl(m, 'emb', sel, 'png', '2018/'), '2018/data/emb/et/nominal/inclusive/m_vis.png?v=S2018');
  assert.equal(viewer.imageUrl(m, 'emb', sel, 'png'), 'data/emb/et/nominal/inclusive/m_vis.png?v=S2018');
});

test('the era round trips through the hash and leaves plain galleries alone', () => {
  const params = { era: '2016preVFP', ch: 'mt', reg: 'nominal', cat: 'inclusive', var: 'pt_1', cols: ['emb'] };
  const hash = viewer.serializeHash(params);
  assert.equal(hash, 'era=2016preVFP&ch=mt&reg=nominal&cat=inclusive&var=pt_1&cols=emb');
  assert.deepEqual(viewer.parseHash('#' + hash), params);
  const s = viewer.resolve(gallery('2018', { et: ['m_vis'] }), {});
  assert.equal('era' in viewer.stateToHash(s), false);
  assert.equal(viewer.stateToHash({ ...s, era: '2018' }).era, '2018');
});

test('a selection carried to another era keeps what it has and falls back otherwise', () => {
  const m2018 = gallery('2018', { et: ['m_vis', 'pt_1', 'eta_1'], mt: ['m_vis', 'pt_1'] });
  const m2017 = gallery('2017', { et: ['m_vis', 'pt_1'], tt: ['m_vis'] });
  const kept = viewer.resolve(m2017, { ch: 'et', var: 'pt_1' });
  assert.deepEqual([kept.channel, kept.variable], ['et', 'pt_1']);
  const fallback = viewer.resolve(m2017, { ch: 'mt', var: 'eta_1' });
  assert.deepEqual([fallback.channel, fallback.variable], ['et', 'm_vis']);
  const back = viewer.resolve(m2018, { ch: 'mt', var: 'pt_1' });
  assert.deepEqual([back.channel, back.variable], ['mt', 'pt_1']);
});

test('columns of another era stay wanted while an era without them is shown', () => {
  assert.deepEqual(viewer.keepUnknown(['emb_ff', 'emb', 'classic'], ['emb', 'classic'], ['emb', 'classic']), ['emb_ff', 'emb', 'classic']);
  assert.deepEqual(viewer.keepUnknown(['emb', 'emb_ff', 'classic'], ['classic', 'emb'], ['emb', 'classic']), ['classic', 'emb', 'emb_ff']);
  assert.deepEqual(viewer.keepUnknown(['emb', 'emb_ff'], ['emb'], ['emb', 'emb_ff']), ['emb']); // a known column hidden by the user stays hidden
  assert.deepEqual(viewer.keepUnknown(undefined, ['emb'], ['emb']), ['emb']);
  const m = gallery('2017', { et: ['m_vis'] });
  const s = viewer.resolve(m, { cols: ['emb_ff', 'emb'] });
  assert.deepEqual(s.enabled, ['emb']);
  const wanted = viewer.carry(m, { ch: 'mt', cols: ['emb_ff', 'emb'], order: ['emb_ff', 'emb'] }, s, { var: 'pt_1' });
  assert.deepEqual(wanted, { ch: 'mt', var: 'pt_1', cols: ['emb_ff', 'emb'], order: ['emb_ff', 'emb'] });
});

test('a gallery of a site links to the site page at its selection', () => {
  const m = gallery('2017', { et: ['m_vis'] });
  const s = viewer.resolve(m, { ch: 'et' });
  assert.equal(viewer.siteLink(m, s), '../index.html#era=2017&ch=et&reg=nominal&cat=inclusive&var=m_vis&cols=emb');
  assert.equal(viewer.siteLink({ ...m, site: undefined }, s), null);
});

// ---------------------------------------------------------------- the site page

test('the site page opens the most recent era from file:// with an era-independent title', async () => {
  const p = page(siteDirectory(), 'site', '');
  await settle();
  assert.equal(p.$('title').textContent, 'smrun2 · control plots');
  assert.equal(p.$('era-field').hidden, false);
  assert.deepEqual(p.$('eras').children.map((b) => b.textContent), ['2016preVFP', '2017', '2018']);
  assert.deepEqual(pressed(p, 'eras'), ['2018']);
  assert.ok(p.document.scripts.includes('2018/manifest.js?v=20261019T100000'));
  assert.match(p.location.hash, /^#era=2018&ch=et&/);
  assert.deepEqual(images(p), ['2018/data/emb/et/nominal/inclusive/m_vis.png?v=S2018']); // emb_ff has no et plot
  assert.equal(p.$('description').textContent, 'about 2018');
  assert.equal(p.$('site-link').hidden, true);
  assert.equal(p.$('controls').hidden, false);
  assert.match(p.document.title, /^m_vis · et · 2018 — smrun2 · control plots$/);
});

test('switching the era keeps the selection where the era has it, falls back and comes back', async () => {
  const p = page(siteDirectory(), 'site', 'era=2018&ch=mt&var=pt_1&cols=emb_ff,emb');
  await settle();
  assert.deepEqual(pressed(p, 'channels'), ['mt']);
  assert.deepEqual(images(p), ['2018/data/emb_ff/mt/nominal/inclusive/pt_1.png?v=' + viewer.sourceStamp(gallery('2018', {}), 'emb_ff', 'mt'), '2018/data/emb/mt/nominal/inclusive/pt_1.png?v=S2018']);

  button(p, 'channels', 'et').click(); // et and pt_1 exist in 2017 too
  button(p, 'eras', '2017').click();
  assert.deepEqual(pressed(p, 'eras'), ['2017']); // pressed while the era loads
  await settle();
  assert.equal(p.$('title').textContent, 'smrun2 · control plots');
  assert.match(p.location.hash, /^#era=2017&ch=et&reg=nominal&cat=inclusive&var=pt_1&cols=emb$/); // emb_ff is not in 2017
  assert.deepEqual(images(p), ['2017/data/emb/et/nominal/inclusive/pt_1.png?v=S2017']);

  button(p, 'eras', '2018').click();
  await settle();
  assert.match(p.location.hash, /^#era=2018&ch=et&reg=nominal&cat=inclusive&var=pt_1&cols=emb_ff,emb$/); // emb_ff is back
  button(p, 'channels', 'mt').click();
  button(p, 'eras', '2017').click(); // 2017 has no mt
  await settle();
  assert.deepEqual(pressed(p, 'channels'), ['et']);
  assert.match(p.location.hash, /^#era=2017&ch=et&/);
  button(p, 'eras', '2018').click(); // the wanted channel comes back
  await settle();
  assert.deepEqual(pressed(p, 'channels'), ['mt']);
  assert.match(p.location.hash, /^#era=2018&ch=mt&reg=nominal&cat=inclusive&var=pt_1&/);
});

test('the shown era clicked while another era loads stays shown', async () => {
  const p = page(siteDirectory(), 'site', 'era=2018&ch=et');
  await settle();
  button(p, 'eras', '2017').click();
  button(p, 'eras', '2018').click(); // before 2017 has loaded
  assert.deepEqual(pressed(p, 'eras'), ['2018']);
  await settle();
  assert.deepEqual(pressed(p, 'eras'), ['2018']);
  assert.match(p.location.hash, /^#era=2018&ch=et&/);
  assert.deepEqual(images(p), ['2018/data/emb/et/nominal/inclusive/m_vis.png?v=S2018']);
});

test('a link opens its era and selection; an unknown era falls back to the most recent', async () => {
  const linked = page(siteDirectory(), 'site', 'era=2017&ch=tt&var=m_vis');
  await settle();
  assert.deepEqual(pressed(linked, 'eras'), ['2017']);
  assert.deepEqual(pressed(linked, 'channels'), ['tt']);
  assert.deepEqual(images(linked), ['2017/data/emb/tt/nominal/inclusive/m_vis.png?v=S2017']);
  const unknown = page(siteDirectory(), 'site', 'era=2015&ch=tt');
  await settle();
  assert.deepEqual(pressed(unknown, 'eras'), ['2018']);
  assert.deepEqual(pressed(unknown, 'channels'), ['tt']);
});

test('an era whose gallery is missing says so and keeps the shown era', async () => {
  const p = page(siteDirectory(), 'site', 'era=2017&ch=et&var=pt_1');
  await settle();
  button(p, 'eras', '2016preVFP').click();
  await settle();
  assert.equal(p.$('notice').hidden, false);
  assert.match(p.$('notice').textContent, /era 2016preVFP could not be loaded: 2016preVFP\/manifest\.js/);
  assert.deepEqual(pressed(p, 'eras'), ['2017']);
  assert.match(p.location.hash, /^#era=2017&ch=et&/);
  button(p, 'eras', '2018').click();
  await settle();
  assert.equal(p.$('notice').hidden, true);
  assert.deepEqual(pressed(p, 'eras'), ['2018']);
});

test('a gallery page has no era switch; a gallery of a site links to the site page', async () => {
  const root = siteDirectory();
  const p = page(path.join(root, '2017'), null, 'ch=et&var=pt_1');
  await settle();
  assert.equal(p.$('era-field').hidden, true);
  assert.equal(p.$('title').textContent, 'run 2017');
  assert.equal(p.location.hash, '#ch=et&reg=nominal&cat=inclusive&var=pt_1&cols=emb');
  assert.deepEqual(images(p), ['data/emb/et/nominal/inclusive/pt_1.png?v=S2017']);
  assert.equal(p.$('site-link').hidden, false);
  assert.equal(p.$('site-link').href, '../index.html#era=2017&ch=et&reg=nominal&cat=inclusive&var=pt_1&cols=emb');

  const plain = gallery('2018', { et: ['m_vis'] });
  delete plain.site;
  const directory = fs.mkdtempSync(path.join(os.tmpdir(), 'shapesmith-gallery-'));
  fs.writeFileSync(path.join(directory, 'manifest.js'), 'window.SHAPESMITH_GALLERY = ' + JSON.stringify(plain) + ';\n');
  const old = page(directory, null, '');
  await settle();
  assert.equal(old.$('site-link').hidden, true);
  assert.equal(old.$('era-field').hidden, true);
  assert.equal(old.location.hash, '#ch=et&reg=nominal&cat=inclusive&var=m_vis&cols=emb');
});
