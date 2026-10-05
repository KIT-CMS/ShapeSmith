/* ShapeSmith gallery viewer: one column per variant, the same plot in every column.
   A plain ES2020 script without dependencies that works over HTTP and from file://; the manifest comes
   from manifest.js (window.SHAPESMITH_GALLERY). The pure functions below are exported for the node tests. */
(function () {
  'use strict';

  const THIN = '\u2009';
  const MINUS = '\u2212';
  const DASH = '\u2013';
  const HASH_KEYS = ['ch', 'reg', 'cat', 'var'];
  const THEME_KEY = 'shapesmith-gallery-theme';

  // ---------------------------------------------------------------- pure logic

  function isObject(x) {
    return x !== null && typeof x === 'object' && !Array.isArray(x);
  }

  function lookup(node, path) {
    for (const key of path) {
      if (!isObject(node) || !Object.prototype.hasOwnProperty.call(node, key)) return undefined;
      node = node[key];
    }
    return node;
  }

  function unique(list) {
    return list.filter((x, i) => list.indexOf(x) === i);
  }

  function decode(text) {
    try {
      return decodeURIComponent(text.replace(/\+/g, ' '));
    } catch (e) {
      return null;
    }
  }

  /** "#ch=et&var=m_vis&cols=a,b" -> {ch: "et", var: "m_vis", cols: ["a", "b"]}; unknown or malformed parts are dropped. */
  function parseHash(hash) {
    const out = {};
    for (const part of String(hash || '').replace(/^[#?]/, '').split('&')) {
      const eq = part.indexOf('=');
      if (eq < 1) continue;
      const key = decode(part.slice(0, eq));
      const raw = part.slice(eq + 1);
      if (key === 'cols') out.cols = raw.split(',').map(decode).filter(Boolean);
      else if (HASH_KEYS.includes(key)) {
        const value = decode(raw);
        if (value) out[key] = value;
      }
    }
    return out;
  }

  /** The inverse of parseHash: readable, every value percent-encoded (Unicode-safe), columns joined by commas. */
  function serializeHash(params) {
    const parts = [];
    for (const key of HASH_KEYS) {
      if (params[key] != null && params[key] !== '') parts.push(key + '=' + encodeURIComponent(params[key]));
    }
    if (Array.isArray(params.cols)) parts.push('cols=' + params.cols.map(encodeURIComponent).join(','));
    return parts.join('&');
  }

  function variantKeys(m) {
    return (m.variants || []).map((v) => v.key);
  }

  function variantOf(m, key) {
    return (m.variants || []).find((v) => v.key === key) || { key: key, label: key };
  }

  function labelOf(list, key) {
    const entry = (list || []).find((e) => e.key === key);
    return entry && entry.label ? entry.label : key;
  }

  /** Keys one level below plots[variant][...path], over the given variants in first-seen order. */
  function presentKeys(m, variants, path) {
    const keys = [];
    for (const variant of variants) {
      const node = lookup(m.plots, [variant].concat(path));
      if (isObject(node)) for (const key of Object.keys(node)) if (!keys.includes(key)) keys.push(key);
    }
    return keys;
  }

  /** The manifest entries of the given keys in manifest order, then keys the manifest does not list. */
  function entries(listed, keys) {
    const out = (listed || []).filter((e) => keys.includes(e.key));
    for (const key of keys) if (!out.some((e) => e.key === key)) out.push({ key: key, label: key });
    return out;
  }

  function options(m, list, enabled, path) {
    let keys = presentKeys(m, enabled, path);
    if (!keys.length) keys = presentKeys(m, variantKeys(m), path);
    return keys.length ? entries(m[list], keys) : (m[list] || []).slice();
  }

  function pick(opts, wanted, preferred) {
    const keys = opts.map((o) => o.key);
    if (keys.includes(wanted)) return wanted;
    if (keys.includes(preferred)) return preferred;
    return keys.length ? keys[0] : null;
  }

  /** Column order (all variants) and the enabled ones in that order; unknown keys are dropped, none enabled means all. */
  function resolveColumns(m, cols, order) {
    const all = variantKeys(m);
    const known = unique((order || []).concat(cols || [])).filter((k) => all.includes(k));
    const full = known.concat(all.filter((k) => !known.includes(k)));
    const wanted = (cols || []).filter((k) => all.includes(k));
    return { order: full, enabled: wanted.length ? full.filter((k) => wanted.includes(k)) : full.slice() };
  }

  /** Resolves wanted {ch, reg, cat, var, cols, order} against the manifest; anything unavailable falls back to the first option. */
  function resolve(m, wanted) {
    wanted = wanted || {};
    const columns = resolveColumns(m, wanted.cols, wanted.order);
    const enabled = columns.enabled;
    const channels = entries(m.channels, unique((m.channels || []).map((c) => c.key).concat(presentKeys(m, variantKeys(m), []))));
    const withPlots = presentKeys(m, enabled, []);
    const channel = pick(channels, wanted.ch, channels.map((c) => c.key).find((k) => withPlots.includes(k)));
    const regions = options(m, 'regions', enabled, [channel]);
    const region = pick(regions, wanted.reg, 'nominal');
    const categories = options(m, 'categories', enabled, [channel, region]);
    const category = pick(categories, wanted.cat, 'inclusive');
    const variables = options(m, 'variables', enabled, [channel, region, category]);
    const variable = pick(variables, wanted.var);
    return {
      channel: channel, region: region, category: category, variable: variable,
      order: columns.order, enabled: enabled,
      channels: channels, regions: regions, categories: categories, variables: variables,
      index: variables.findIndex((v) => v.key === variable),
    };
  }

  function stateToHash(s) {
    return { ch: s.channel, reg: s.region, cat: s.category, var: s.variable, cols: s.enabled };
  }

  /** The published formats (e.g. ["png", "pdf"]) of one variant at the selection, or null. */
  function plotFormats(m, variant, sel) {
    const formats = lookup(m.plots, [variant, sel.channel, sel.region, sel.category, sel.variable]);
    return Array.isArray(formats) ? formats : null;
  }

  /** The stamp of the latest source of the variant that published the channel (cache busting per publish). */
  function sourceStamp(m, variant, channel) {
    const sources = variantOf(m, variant).sources || [];
    for (let i = sources.length - 1; i >= 0; i--) {
      if ((sources[i].channels || []).includes(channel) && sources[i].stamp) return String(sources[i].stamp);
    }
    return String(m.updated || '').replace(/[-:]/g, '');
  }

  function imageUrl(m, variant, sel, ext) {
    const path = [variant, sel.channel, sel.region, sel.category, sel.variable].map((k) => encodeURIComponent(k)).join('/');
    const stamp = sourceStamp(m, variant, sel.channel);
    return 'data/' + path + '.' + ext + (stamp ? '?v=' + encodeURIComponent(stamp) : '');
  }

  const AXIS_TITLES = { channels: 'Channel', regions: 'Region', categories: 'Category', variables: 'Variable' };

  /** The title of an axis: the manifest's (e.g. "Process" in a measurement gallery) or the default. */
  function axisTitle(m, axis) {
    const title = lookup(m.axes, [axis]);
    return typeof title === 'string' && title ? title : AXIS_TITLES[axis];
  }

  function plural(word) {
    return /[^aeiou]y$/.test(word) ? word.slice(0, -1) + 'ies' : word + 's';
  }

  /** Why a variant shows no PNG at the selection, naming what it does have; '' if the PNG exists. */
  function describeMissing(m, variant, sel) {
    const steps = [['channels', sel.channel], ['regions', sel.region], ['categories', sel.category]];
    const path = [variant];
    if (!isObject(lookup(m.plots, path))) return 'Nothing is published for this variant.';
    for (const [list, key] of steps) {
      const node = lookup(m.plots, path);
      if (!isObject(lookup(node, [key]))) {
        const available = entries(m[list], Object.keys(node)).map((e) => e.label || e.key);
        return 'Not published for ' + axisTitle(m, list).toLowerCase() + ' ' + labelOf(m[list], key) + (available.length ? '; available: ' + available.join(', ') + '.' : '.');
      }
      path.push(key);
    }
    const formats = lookup(m.plots, path.concat([sel.variable]));
    if (!Array.isArray(formats)) {
      const n = Object.keys(lookup(m.plots, path)).length;
      const noun = axisTitle(m, 'variables').toLowerCase();
      return 'This ' + noun + ' is not published for this selection' + (n ? '; ' + n + ' other ' + (n === 1 ? noun + ' is.' : plural(noun) + ' are.') : '.');
    }
    if (!formats.includes('png')) return formats.length ? 'No PNG published; available as ' + formats.join(', ').toUpperCase() + '.' : 'No image file published.';
    return '';
  }

  function finite(x) {
    return typeof x === 'number' && Number.isFinite(x) ? x : null;
  }

  function ratio(data, prediction) {
    return data == null || prediction == null || prediction === 0 ? null : data / prediction;
  }

  /** The note of a variant's plot at the selection (e.g. the p-value of a measured correction); '' if none. */
  function noteOf(m, variant, sel) {
    const note = lookup(m.notes, [variant, sel.channel, sel.region, sel.category, sel.variable]);
    return typeof note === 'string' ? note : '';
  }

  /** Columns = the facts of the channel (union by key, first seen first), one row per variant with a Map key -> value. */
  function factTable(m, variants, channel) {
    const columns = [];
    const rows = variants.map((key) => {
      const values = new Map();
      const facts = lookup(m.facts, [key, channel]);
      for (const f of Array.isArray(facts) ? facts : []) {
        if (!isObject(f) || f.key == null) continue;
        if (!columns.some((c) => c.key === f.key)) columns.push({ key: f.key, label: f.label || f.key });
        values.set(f.key, String(f.value));
      }
      return { key: key, label: variantOf(m, key).label || key, values: values };
    });
    return { columns: columns, rows: rows };
  }

  /** Whether any variant has yields for any channel, region and category (a measurement gallery has none). */
  function hasYields(m) {
    const deep = (node, depth) => isObject(node) && (depth === 0 || Object.keys(node).some((k) => deep(node[k], depth - 1)));
    return deep(m.yields, 4);
  }

  /** Rows (one per variant) with data, prediction, Data/Pred and a Map of stack group yields; groups = union by key. */
  function yieldTable(m, variants, sel) {
    const groups = [];
    const rows = variants.map((key) => {
      const y = lookup(m.yields, [key, sel.channel, sel.region, sel.category]);
      const row = { key: key, label: variantOf(m, key).label || key, data: null, prediction: null, ratio: null, groups: new Map(), published: isObject(y) };
      if (isObject(y)) {
        row.data = finite(y.data);
        row.prediction = finite(y.prediction);
        for (const g of Array.isArray(y.groups) ? y.groups : []) {
          if (!isObject(g) || g.key == null) continue;
          if (!groups.some((x) => x.key === g.key)) groups.push({ key: g.key, label: g.label || g.key });
          row.groups.set(g.key, finite(g.yield));
        }
      }
      row.ratio = ratio(row.data, row.prediction);
      return row;
    });
    return { groups: groups, rows: rows };
  }

  /** 14934 -> "14 934" with a thin space, 16021.3 -> "16 021", 12.34 -> "12.3", -0.5 -> "-0.50" with a minus sign (U+2212);
      non-numbers -> an en dash. */
  function formatNumber(x, digits) {
    if (typeof x !== 'number' || !Number.isFinite(x)) return DASH;
    const abs = Math.abs(x);
    if (digits == null) digits = Number.isInteger(x) || abs >= 1000 ? 0 : abs >= 10 ? 1 : 2;
    const [whole, frac] = abs.toFixed(digits).split('.');
    const negative = x < 0 && /[1-9]/.test(whole + (frac || ''));
    return (negative ? MINUS : '') + whole.replace(/\B(?=(\d{3})+(?!\d))/g, THIN) + (frac ? '.' + frac : '');
  }

  /** Data/Pred with three decimals; an en dash without data, without prediction or for a zero prediction. */
  function formatRatio(data, prediction) {
    const r = ratio(finite(data), finite(prediction));
    return r == null ? DASH : formatNumber(r, 3);
  }

  function yieldLine(row) {
    if (!row || (row.data == null && row.prediction == null)) return 'No yields published';
    const parts = [];
    if (row.data != null) parts.push('Data ' + formatNumber(row.data));
    if (row.prediction != null) parts.push('Pred ' + formatNumber(row.prediction));
    if (row.ratio != null) parts.push('Data/Pred ' + formatRatio(row.data, row.prediction));
    return parts.join(' · ');
  }

  /** Variables matching every whitespace-separated term in key or label; exact and prefix key matches first. */
  function filterVariables(variables, query) {
    const q = String(query || '').trim().toLowerCase();
    const terms = q.split(/\s+/).filter(Boolean);
    if (!terms.length) return variables.slice();
    const rank = (v) => (v.key.toLowerCase() === q ? 0 : v.key.toLowerCase().startsWith(q) ? 1 : 2);
    return variables
      .filter((v) => {
        const text = (v.key + ' ' + (v.label || '')).toLowerCase();
        return terms.every((t) => text.includes(t));
      })
      .map((v, i) => [rank(v), i, v])
      .sort((a, b) => a[0] - b[0] || a[1] - b[1])
      .map((x) => x[2]);
  }

  function formatTime(iso) {
    return iso ? String(iso).replace('T', ' ').slice(0, 16) : '';
  }

  if (typeof module !== 'undefined' && module.exports) {
    module.exports = {
      parseHash, serializeHash, resolve, resolveColumns, stateToHash, plotFormats, sourceStamp, imageUrl,
      describeMissing, yieldTable, yieldLine, formatNumber, formatRatio, filterVariables, formatTime,
      axisTitle, noteOf, factTable, hasYields,
    };
  }
  if (typeof document === 'undefined') return;

  // ---------------------------------------------------------------- page

  let M = null;
  let S = null; // the resolved selection
  let W = {}; // the wanted one: a choice that is unavailable for now comes back once it is available again
  let lastHash = null;
  let lightbox = null; // index into S.enabled while the lightbox is open
  let lightboxReturn = null;
  let aspect = null; // "w / h" of the last loaded plot, reserves the space of the next ones
  let preloaded = []; // keeps the preloading images of the neighbouring variables referenced
  let suggestions = [];
  let active = -1;
  let query = '';
  let copyTimer = null;

  const $ = (id) => document.getElementById(id);

  function el(tag, attrs, children) {
    const node = document.createElement(tag);
    for (const [key, value] of Object.entries(attrs || {})) {
      if (value == null || value === false) continue;
      if (key === 'text') node.textContent = value;
      else if (key === 'class') node.className = value;
      else if (typeof value === 'function') node.addEventListener(key.replace(/^on/, ''), value);
      else node.setAttribute(key, value === true ? '' : value);
    }
    for (const child of [].concat(children == null ? [] : children)) if (child != null) node.append(child);
    return node;
  }

  function notice(text, error) {
    const box = $('notice');
    box.textContent = text;
    box.classList.toggle('error', !!error);
    box.hidden = false;
  }

  function wide() {
    return window.matchMedia ? window.matchMedia('(min-width: 900px)').matches : true;
  }

  function setTheme(theme) {
    if (theme === 'light' || theme === 'dark') document.documentElement.setAttribute('data-theme', theme);
    else document.documentElement.removeAttribute('data-theme');
  }

  function initTheme() {
    const select = $('theme');
    let theme = 'light';
    try {
      theme = localStorage.getItem(THEME_KEY) || 'light';
    } catch (e) { /* storage blocked: stay light */ }
    if (!['auto', 'light', 'dark'].includes(theme)) theme = 'light';
    select.value = theme;
    setTheme(theme);
    select.addEventListener('change', () => {
      setTheme(select.value);
      try {
        localStorage.setItem(THEME_KEY, select.value);
      } catch (e) { /* not remembered */ }
    });
  }

  function start() {
    initTheme();
    const valid = (m) => isObject(m) && Array.isArray(m.variants);
    const missing = () => notice('No gallery manifest found: manifest.js next to this page is missing or does not define window.SHAPESMITH_GALLERY. Publish a gallery into this directory with ShapeSmith first.', true);
    if (valid(window.SHAPESMITH_GALLERY)) init(window.SHAPESMITH_GALLERY);
    else if (/^https?:$/.test(location.protocol) && window.fetch) {
      fetch('manifest.json', { cache: 'no-cache' })
        .then((r) => (r.ok ? r.json() : null))
        .then((m) => (valid(m) ? init(m) : missing()), missing);
    } else missing();
  }

  function init(m) {
    // channels are shown by their key (et, mt, ...), not by the analysis' mathtext label
    M = Object.assign({}, m, { channels: (m.channels || []).map((c) => Object.assign({}, c, { label: c.key })) });
    m = M;
    const title = m.title || 'ShapeSmith gallery';
    $('title').textContent = title;
    if (m.updated) {
      $('updated').textContent = 'Updated ' + formatTime(m.updated);
      $('updated').setAttribute('datetime', m.updated);
    }
    if (m.generator) $('generator').textContent = (m.updated ? ' · ' : '') + m.generator;
    if (m.description) {
      $('description').textContent = m.description;
      $('about').hidden = false;
    }
    if (m.schema !== 1) notice('This viewer reads manifest schema 1, the manifest has schema ' + m.schema + '; parts of it may not be shown.');
    if (!variantKeys(m).length) return notice('This gallery has no published variants yet.');
    for (const [axis, id] of [['channels', 'channel-label'], ['regions', 'region-label'], ['categories', 'category-label'], ['variables', 'var-label']]) {
      $(id).textContent = axisTitle(m, axis);
    }
    $('controls').hidden = false;
    $('main').hidden = false;
    $('yields-panel').hidden = !hasYields(m);
    $('yields-panel').open = wide();
    bind();
    renderProvenance();
    apply(parseHash(location.hash));
  }

  function apply(wanted) {
    W = wanted;
    S = resolve(M, wanted);
    render();
    writeHash();
    preload();
  }

  function change(patch) {
    apply(Object.assign({}, W, { order: S.order, cols: S.enabled }, patch));
  }

  function step(delta) {
    const n = S.variables.length;
    if (!n) return;
    change({ var: S.variables[S.index < 0 ? 0 : (S.index + delta + n) % n].key });
  }

  function writeHash() {
    const hash = serializeHash(stateToHash(S));
    if (hash === lastHash) return;
    lastHash = hash;
    try {
      history.replaceState(null, '', '#' + hash);
    } catch (e) {
      location.replace('#' + hash);
    }
  }

  function preload() {
    preloaded = [];
    if (typeof Image === 'undefined' || S.index < 0 || S.variables.length < 2) return;
    const n = S.variables.length;
    for (const delta of [1, -1]) {
      const sel = Object.assign({}, S, { variable: S.variables[(S.index + delta + n) % n].key });
      for (const key of S.enabled) {
        if (!(plotFormats(M, key, sel) || []).includes('png')) continue;
        const img = new Image();
        img.src = imageUrl(M, key, sel, 'png');
        preloaded.push(img);
      }
    }
  }

  // ---------------------------------------------------------------- rendering

  function render() {
    const focused = document.activeElement && document.activeElement.dataset ? document.activeElement.dataset.focus : null;
    renderChannels();
    fillSelect($('region'), S.regions, S.region);
    fillSelect($('category'), S.categories, S.category);
    $('category-field').hidden = S.categories.length <= 1 && (M.categories || []).length <= 1; // shown where the gallery has several
    renderVariable();
    renderChips();
    renderFacts();
    renderYields();
    renderGrid();
    if (lightbox != null) renderLightbox();
    if (focused) restoreFocus(focused);
    document.title = (S.variable ? labelOf(S.variables, S.variable) + ' · ' + labelOf(S.channels, S.channel) + ' — ' : '') + (M.title || 'ShapeSmith gallery');
  }

  function restoreFocus(id) {
    const find = (x) => Array.from(document.querySelectorAll('[data-focus]')).find((n) => n.dataset.focus === x);
    let target = find(id);
    if ((!target || target.disabled) && /^(left|right):/.test(id)) target = find('toggle:' + id.replace(/^\w+:/, ''));
    if (target && !target.disabled && document.activeElement !== target) target.focus();
  }

  function selectionText() {
    const parts = [labelOf(S.channels, S.channel), labelOf(S.regions, S.region)];
    if (S.categories.length > 1 || S.category !== 'inclusive') parts.push(labelOf(S.categories, S.category));
    return parts.join(' · ');
  }

  function renderChannels() {
    const withPlots = presentKeys(M, S.enabled, []);
    $('channels').replaceChildren(...S.channels.map((c) => {
      const label = c.label || c.key;
      return el('button', {
        type: 'button', class: 'seg' + (withPlots.includes(c.key) ? '' : ' dim'), 'aria-pressed': String(c.key === S.channel),
        'data-focus': 'ch:' + c.key, title: withPlots.includes(c.key) ? null : 'No shown variant has this channel', onclick: () => change({ ch: c.key }),
      }, [el('span', { text: label }), label !== c.key ? el('code', { class: 'seg-key', text: c.key }) : null]);
    }));
  }

  function fillSelect(select, opts, value) {
    select.replaceChildren(...opts.map((o) => el('option', { value: o.key, text: o.label || o.key })));
    select.value = value == null ? '' : value;
    select.disabled = opts.length <= 1;
  }

  function renderVariable() {
    const input = $('var-input');
    const n = S.variables.length;
    if (document.activeElement !== input) input.value = S.variable ? labelOf(S.variables, S.variable) : '';
    $('counter').textContent = n ? (S.index + 1) + ' / ' + n : '0 / 0';
    $('prev').disabled = $('next').disabled = n <= 1;
    if (!$('var-list').hidden) openList(query, true);
  }

  function renderChips() {
    const n = S.order.length;
    $('variants').replaceChildren(...S.order.map((key, i) => {
      const v = variantOf(M, key);
      const label = v.label || key;
      const on = S.enabled.includes(key);
      return el('div', { class: 'chip' + (on ? ' on' : '') }, [
        el('button', { type: 'button', class: 'chip-move', 'aria-label': 'Move ' + label + ' left', title: 'Move left', disabled: i === 0, 'data-focus': 'left:' + key, onclick: () => move(key, -1) }, '‹'),
        el('button', {
          type: 'button', class: 'chip-toggle', 'aria-pressed': String(on), 'data-focus': 'toggle:' + key,
          title: (v.description ? v.description + '\n' : '') + (on ? 'Hide this variant' : 'Show this variant'), onclick: () => toggle(key),
        }, el('span', { class: 'chip-label', text: label })),
        el('button', { type: 'button', class: 'chip-move', 'aria-label': 'Move ' + label + ' right', title: 'Move right', disabled: i === n - 1, 'data-focus': 'right:' + key, onclick: () => move(key, 1) }, '›'),
      ]);
    }));
  }

  function toggle(key) {
    const on = S.enabled.includes(key);
    if (on && S.enabled.length === 1) return; // one variant always stays shown
    change({ cols: S.order.filter((k) => (k === key ? !on : S.enabled.includes(k))) });
  }

  function move(key, delta) {
    const order = S.order.slice();
    const i = order.indexOf(key);
    const j = i + delta;
    if (i < 0 || j < 0 || j >= order.length) return;
    [order[i], order[j]] = [order[j], order[i]];
    change({ order: order, cols: order.filter((k) => S.enabled.includes(k)) });
  }

  function renderFacts() {
    const table = factTable(M, S.enabled, S.channel);
    $('facts-panel').hidden = !table.columns.length;
    if (!table.columns.length) return;
    $('facts-summary').textContent = 'Summary — ' + labelOf(S.channels, S.channel);
    const head = el('tr', {}, [el('th', { scope: 'col', text: 'Variant' }), ...table.columns.map((c) => el('th', { scope: 'col', title: c.key, text: c.label }))]);
    const rows = table.rows.map((r) => el('tr', {}, [el('th', { scope: 'row', text: r.label }), ...table.columns.map((c) => el('td', { text: r.values.has(c.key) ? r.values.get(c.key) : DASH }))]));
    $('facts').replaceChildren(el('div', { class: 'table-wrap' }, el('table', {}, [el('thead', {}, head), el('tbody', {}, rows)])));
  }

  function renderYields() {
    const table = yieldTable(M, S.enabled, S);
    $('yields-summary').textContent = 'Yields — ' + selectionText();
    if (!table.rows.some((r) => r.published)) return $('yields').replaceChildren(el('p', { class: 'muted', text: 'No yields published for this selection.' }));
    const num = (x) => el('td', { text: formatNumber(x) });
    const head = el('tr', {}, [
      el('th', { scope: 'col', text: 'Variant' }), el('th', { scope: 'col', text: 'Data' }), el('th', { scope: 'col', text: 'Prediction' }),
      el('th', { scope: 'col', text: 'Data/Pred' }),
      ...table.groups.map((g, i) => el('th', { scope: 'col', class: i === 0 ? 'sep' : null, title: g.key, text: g.label })),
    ]);
    const rows = table.rows.map((r) => el('tr', {}, [
      el('th', { scope: 'row', text: r.label }), num(r.data), num(r.prediction),
      el('td', { class: 'ratio', text: formatRatio(r.data, r.prediction) }),
      ...table.groups.map((g, i) => el('td', { class: i === 0 ? 'sep' : null, text: r.groups.has(g.key) ? formatNumber(r.groups.get(g.key)) : DASH })),
    ]));
    $('yields').replaceChildren(el('div', { class: 'table-wrap' }, el('table', {}, [el('thead', {}, head), el('tbody', {}, rows)])));
  }

  function altText(key, sel) {
    return labelOf(S.variables, sel.variable) + ' in ' + selectionText() + ' (' + (variantOf(M, key).label || key) + ')';
  }

  function missingPlate(text) {
    const plate = el('div', { class: 'plate missing' }, el('p', { text: text }));
    if (aspect) plate.style.aspectRatio = aspect;
    return plate;
  }

  function card(key) {
    const v = variantOf(M, key);
    const label = v.label || key;
    const formats = (S.variable && plotFormats(M, key, S)) || [];
    const row = yieldTable(M, [key], S).rows[0];
    const note = S.variable ? noteOf(M, key, S) : '';
    const head = el('header', { class: 'card-head' }, [
      el('h3', { class: 'card-title', title: v.description ? label + ' — ' + v.description : label, text: label }),
      note ? el('span', { class: 'card-ratio', text: note })
        : row.ratio != null ? el('span', { class: 'card-ratio', title: yieldLine(row), text: 'Data/Pred ' + formatRatio(row.data, row.prediction) }) : null,
      // the PNG opens by clicking the plot; the header links only the other formats
      ...formats.filter((ext) => ext !== 'png').map((ext) => el('a', { class: 'card-link', href: imageUrl(M, key, S, ext), target: '_blank', rel: 'noopener', text: ext.toUpperCase() })),
    ]);
    let plate;
    if (formats.includes('png')) {
      const png = imageUrl(M, key, S, 'png');
      const img = el('img', { loading: 'lazy', decoding: 'async', alt: altText(key, S), src: png });
      if (aspect) img.style.aspectRatio = 'auto ' + aspect;
      img.addEventListener('load', () => {
        if (img.naturalWidth && img.naturalHeight) aspect = img.naturalWidth + ' / ' + img.naturalHeight;
      });
      img.addEventListener('error', () => plate.replaceWith(missingPlate('The PNG is listed in the manifest but could not be loaded.')));
      plate = el('a', {
        class: 'plate', href: png, title: 'Enlarge', 'data-focus': 'plate:' + key,
        onclick: (e) => {
          if (e.button !== 0 || e.ctrlKey || e.metaKey || e.shiftKey || e.altKey) return;
          e.preventDefault();
          openLightbox(key);
        },
      }, img);
    } else plate = missingPlate(S.variable ? describeMissing(M, key, S) : 'No plots for this selection.');
    return el('article', { class: 'card', 'data-variant': key }, [head, plate]);
  }

  function renderGrid() {
    $('grid').replaceChildren(...S.enabled.map(card));
  }

  function renderProvenance() {
    const source = (src) => {
      const rows = [
        ['Channels', (src.channels || []).map((k) => { const l = labelOf(M.channels, k); return l === k ? k : l + ' (' + k + ')'; }).join(', ')],
        ['Published', formatTime(src.published)],
        ['Config', src.config],
        ['Overrides', (src.overrides || []).join('\n') || 'none'],
        ['Output', src.output_dir],
      ];
      const dl = el('dl', { class: 'prov-dl' });
      for (const [term, value] of rows) if (value) dl.append(el('dt', { text: term }), el('dd', { text: value }));
      const block = el('div', { class: 'prov-source' }, dl);
      if (src.provenance != null) {
        block.append(el('details', {}, [el('summary', { text: 'Provenance record' }), el('pre', { text: JSON.stringify(src.provenance, null, 2) })]));
      }
      return block;
    };
    $('provenance').replaceChildren(...M.variants.map((v) => el('section', { class: 'prov' }, [
      el('h3', {}, [v.label || v.key, ' ', el('code', { text: v.key })]),
      v.description ? el('p', { class: 'muted', text: v.description }) : null,
      ...(v.sources || []).map(source),
      (v.sources || []).length ? null : el('p', { class: 'muted', text: 'No sources recorded.' }),
    ])));
  }

  // ---------------------------------------------------------------- variable picker

  function openList(text, keepActive) {
    const list = $('var-list');
    query = text;
    const previous = keepActive && suggestions[active] ? suggestions[active].key : null;
    suggestions = filterVariables(S.variables, query);
    active = suggestions.findIndex((v) => v.key === (previous || (query ? null : S.variable)));
    if (active < 0 && query && suggestions.length) active = 0;
    list.replaceChildren(...suggestions.map((v, i) => el('li', {
      id: 'var-opt-' + i, role: 'option', class: 'opt', 'aria-selected': 'false',
      onmousedown: (e) => {
        e.preventDefault();
        choose(i);
      },
    }, [el('span', { class: 'opt-label', text: v.label || v.key }), v.label && v.label !== v.key ? el('code', { class: 'opt-key', text: v.key }) : null])));
    if (!suggestions.length) list.append(el('li', { class: 'opt empty', role: 'presentation', text: 'No matching variable' }));
    list.hidden = false;
    $('var-input').setAttribute('aria-expanded', 'true');
    setActive(active);
  }

  function setActive(i) {
    active = i;
    const input = $('var-input');
    Array.from($('var-list').children).forEach((li, j) => li.setAttribute('aria-selected', String(j === i)));
    const option = i >= 0 ? $('var-opt-' + i) : null;
    if (option) {
      input.setAttribute('aria-activedescendant', option.id);
      if (option.scrollIntoView) option.scrollIntoView({ block: 'nearest' });
    } else input.removeAttribute('aria-activedescendant');
  }

  function closeList() {
    $('var-list').hidden = true;
    $('var-input').setAttribute('aria-expanded', 'false');
    $('var-input').removeAttribute('aria-activedescendant');
  }

  function choose(i) {
    const v = suggestions[i];
    closeList();
    if (v) change({ var: v.key });
    if ($('next').disabled) $('var-input').blur();
    else $('next').focus();
  }

  function onInputKey(e) {
    const list = $('var-list');
    const n = suggestions.length;
    if (e.key === 'ArrowDown' || e.key === 'ArrowUp') {
      e.preventDefault();
      if (list.hidden) return openList('');
      if (n) setActive(e.key === 'ArrowDown' ? (active + 1) % n : (active - 1 + n) % n);
    } else if (e.key === 'Enter') {
      e.preventDefault();
      if (!list.hidden && n) choose(active >= 0 ? active : 0);
    } else if (e.key === 'Escape') {
      if (list.hidden) return e.target.blur();
      closeList();
      e.target.value = S.variable ? labelOf(S.variables, S.variable) : '';
      e.target.select();
    }
  }

  // ---------------------------------------------------------------- lightbox

  function openLightbox(key) {
    lightboxReturn = key;
    lightbox = Math.max(0, S.enabled.indexOf(key));
    $('lightbox').hidden = false;
    document.body.classList.add('lb-open');
    renderLightbox();
    $('lb-close').focus();
  }

  function closeLightbox() {
    const key = S.enabled[lightbox] || lightboxReturn;
    lightbox = null;
    $('lightbox').hidden = true;
    $('lb-img').removeAttribute('src');
    document.body.classList.remove('lb-open');
    const cardNode = Array.from($('grid').children).find((c) => c.dataset.variant === key);
    const plate = cardNode && cardNode.querySelector('a.plate');
    if (plate) plate.focus();
  }

  function shiftVariant(delta) {
    const n = S.enabled.length;
    lightbox = (lightbox + delta + n) % n;
    renderLightbox();
  }

  function renderLightbox() {
    if (lightbox >= S.enabled.length) lightbox = 0;
    const key = S.enabled[lightbox];
    const formats = (S.variable && plotFormats(M, key, S)) || [];
    const img = $('lb-img');
    const missing = $('lb-missing');
    if (formats.includes('png')) {
      img.alt = altText(key, S);
      img.src = imageUrl(M, key, S, 'png');
      img.hidden = false;
      missing.hidden = true;
    } else {
      img.removeAttribute('src');
      img.hidden = true;
      missing.textContent = S.variable ? describeMissing(M, key, S) : 'No plots for this selection.';
      missing.hidden = false;
    }
    $('lb-caption').textContent = (variantOf(M, key).label || key) + ' — ' + (S.variable ? labelOf(S.variables, S.variable) + ' · ' : '') + selectionText();
    $('lb-pos').textContent = 'variant ' + (lightbox + 1) + '/' + S.enabled.length + ' · variable ' + (S.index + 1) + '/' + S.variables.length;
    $('lb-up').disabled = $('lb-down').disabled = S.enabled.length <= 1;
  }

  function trapFocus(e) {
    const buttons = Array.from($('lightbox').querySelectorAll('button:not([disabled])'));
    const i = buttons.indexOf(document.activeElement);
    const j = e.shiftKey ? (i <= 0 ? buttons.length - 1 : i - 1) : (i + 1) % buttons.length;
    e.preventDefault();
    if (buttons[j]) buttons[j].focus();
  }

  // ---------------------------------------------------------------- events

  function typing(target) {
    return !!target && (target.isContentEditable || /^(INPUT|SELECT|TEXTAREA)$/.test(target.tagName));
  }

  function onKey(e) {
    if (e.altKey || e.ctrlKey || e.metaKey) return;
    if (lightbox != null) {
      const actions = { Escape: closeLightbox, ArrowLeft: () => step(-1), ArrowRight: () => step(1), ArrowUp: () => shiftVariant(-1), ArrowDown: () => shiftVariant(1), Tab: () => trapFocus(e) };
      if (actions[e.key]) {
        if (e.key !== 'Tab') e.preventDefault();
        actions[e.key]();
      }
      return;
    }
    if (typing(e.target)) return;
    if (e.key === 'ArrowLeft' || e.key === 'ArrowRight') {
      e.preventDefault();
      step(e.key === 'ArrowLeft' ? -1 : 1);
    } else if (e.key === '/') {
      e.preventDefault();
      $('var-input').focus();
    }
  }

  function legacyCopy(text) {
    const area = el('textarea', { class: 'offscreen', readonly: true, 'aria-hidden': 'true' });
    area.value = text;
    document.body.append(area);
    area.select();
    let ok = false;
    try {
      ok = document.execCommand('copy');
    } catch (e) { /* unsupported */ }
    area.remove();
    return ok;
  }

  function copyLink() {
    const button = $('copy-link');
    const done = (ok) => {
      button.textContent = ok ? 'Copied' : 'Copy failed: use the address bar';
      clearTimeout(copyTimer);
      copyTimer = setTimeout(() => { button.textContent = 'Copy link'; }, 1800);
    };
    const url = location.href;
    if (navigator.clipboard && window.isSecureContext) navigator.clipboard.writeText(url).then(() => done(true), () => done(legacyCopy(url)));
    else done(legacyCopy(url));
  }

  function bind() {
    $('region').addEventListener('change', (e) => change({ reg: e.target.value }));
    $('category').addEventListener('change', (e) => change({ cat: e.target.value }));
    $('prev').addEventListener('click', () => step(-1));
    $('next').addEventListener('click', () => step(1));
    $('copy-link').addEventListener('click', copyLink);
    const input = $('var-input');
    input.addEventListener('focus', () => {
      input.select();
      openList('');
    });
    input.addEventListener('input', () => openList(input.value));
    input.addEventListener('keydown', onInputKey);
    input.addEventListener('blur', () => {
      closeList();
      input.value = S.variable ? labelOf(S.variables, S.variable) : '';
    });
    $('lb-close').addEventListener('click', closeLightbox);
    $('lb-prev').addEventListener('click', () => step(-1));
    $('lb-next').addEventListener('click', () => step(1));
    $('lb-up').addEventListener('click', () => shiftVariant(-1));
    $('lb-down').addEventListener('click', () => shiftVariant(1));
    $('lightbox').addEventListener('click', (e) => {
      if (e.target === $('lightbox') || e.target === $('lb-stage')) closeLightbox();
    });
    document.addEventListener('keydown', onKey);
    window.addEventListener('hashchange', () => {
      const hash = location.hash.replace(/^#/, '');
      if (hash !== lastHash) apply(parseHash(hash));
    });
  }

  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', start);
  else start();
})();
