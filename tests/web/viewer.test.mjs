// Pure logic of the gallery viewer (src/shapesmith/web/static/viewer.js); run with `node --test`.
import assert from 'node:assert/strict';
import { createRequire } from 'node:module';
import test from 'node:test';

const require = createRequire(import.meta.url);
const viewer = require('../../src/shapesmith/web/static/viewer.js');

const both = ['png', 'pdf'];
const manifest = {
  schema: 1,
  generator: 'shapesmith 0.1.2',
  title: 'Test gallery',
  updated: '2026-10-05T01:30:00',
  channels: [{ key: 'et', label: 'eτh' }, { key: 'mt', label: 'μτh' }, { key: 'tt', label: 'τhτh' }],
  regions: [{ key: 'nominal', label: 'nominal' }, { key: 'ss', label: 'same sign' }],
  categories: [{ key: 'inclusive', label: 'inclusive' }],
  variables: [{ key: 'm_vis', label: 'm_vis (GeV)' }, { key: 'pt_1', label: 'pT(τ1) (GeV)' }, { key: 'eta_1', label: 'η(τ1)' }],
  variants: [
    { key: 'classic', label: 'classic', description: 'MC only', sources: [{ channels: ['et', 'mt'], stamp: '20261004T100000' }] },
    { key: 'emb', label: 'emb', sources: [{ channels: ['et'], stamp: '20261004T120000' }, { channels: ['et', 'tt'], stamp: '20261005T013000' }] },
    { key: 'emb_ff', label: 'emb ff', sources: [] },
  ],
  plots: {
    classic: {
      et: { nominal: { inclusive: { m_vis: both, pt_1: both, eta_1: ['pdf'] } }, ss: { inclusive: { m_vis: both } } },
      mt: { nominal: { inclusive: { m_vis: both } } },
    },
    emb: { et: { nominal: { inclusive: { m_vis: both, pt_1: both } } }, tt: { nominal: { inclusive: { m_vis: both } } } },
    emb_ff: { et: { nominal: { inclusive: { m_vis: both, extra: ['png'] } } } },
  },
  yields: {
    classic: { et: { nominal: { inclusive: { data: 14934, prediction: 16021.3, groups: [{ key: 'ztt', label: 'Z→ττ', yield: 9000 }, { key: 'jetFakes', label: 'jet → τh', yield: 3000.25 }] } } } },
    emb: { et: { nominal: { inclusive: { data: 14934, prediction: 15000, groups: [{ key: 'emb', label: 'μ→τ emb.', yield: 8800 }, { key: 'jetFakes', label: 'jet → τh (other)', yield: 3100 }] } } } },
    emb_ff: { et: { nominal: { inclusive: { data: 10, prediction: 0 } } } },
  },
};

test('hash round trip is readable and Unicode-safe', () => {
  const params = { ch: 'et', reg: 'nominal', cat: 'inclusive', var: 'm_vis', cols: ['classic', 'emb', 'emb_ff'] };
  const hash = viewer.serializeHash(params);
  assert.equal(hash, 'ch=et&reg=nominal&cat=inclusive&var=m_vis&cols=classic,emb,emb_ff');
  assert.deepEqual(viewer.parseHash('#' + hash), params);
  const odd = { ch: 'τhτh', var: 'a b&c=d', cols: ['x,y', 'ü'] };
  assert.deepEqual(viewer.parseHash(viewer.serializeHash(odd)), odd);
});

test('hash parsing tolerates junk', () => {
  assert.deepEqual(viewer.parseHash(''), {});
  assert.deepEqual(viewer.parseHash('#%E0%A4%A&ch=et&foo=1&=x&var'), { ch: 'et' });
  assert.deepEqual(viewer.parseHash('#cols=,a,,%ZZ'), { cols: ['a'] });
  assert.deepEqual(viewer.parseHash('#ch=%CF%84%CF%84&var=m+vis'), { ch: 'ττ', var: 'm vis' });
});

test('resolve restores a permalink to a non-first variable and the column order', () => {
  const s = viewer.resolve(manifest, viewer.parseHash('#ch=et&reg=nominal&cat=inclusive&var=pt_1&cols=emb,classic'));
  assert.equal(s.channel, 'et');
  assert.equal(s.variable, 'pt_1');
  assert.equal(s.index, 1);
  assert.deepEqual(s.enabled, ['emb', 'classic']);
  assert.deepEqual(s.order, ['emb', 'classic', 'emb_ff']);
  assert.deepEqual(s.variables.map((v) => v.key), ['m_vis', 'pt_1', 'eta_1']);
  assert.deepEqual(viewer.stateToHash(s), { ch: 'et', reg: 'nominal', cat: 'inclusive', var: 'pt_1', cols: ['emb', 'classic'] });
});

test('resolve falls back gracefully', () => {
  const s = viewer.resolve(manifest, { ch: 'ee', reg: 'nope', var: 'nope', cols: ['gone'] });
  assert.equal(s.channel, 'et');
  assert.equal(s.region, 'nominal');
  assert.equal(s.category, 'inclusive');
  assert.equal(s.variable, 'm_vis');
  assert.deepEqual(s.enabled, ['classic', 'emb', 'emb_ff']);
  // variables unknown to the manifest are appended, union over the enabled variants
  assert.deepEqual(s.variables.map((v) => v.key), ['m_vis', 'pt_1', 'eta_1', 'extra']);
  assert.deepEqual(s.variables[3], { key: 'extra', label: 'extra' });
  // a region only some variants have; a channel without the wanted variable
  assert.equal(viewer.resolve(manifest, { ch: 'et', reg: 'ss', var: 'pt_1' }).variable, 'm_vis');
  const mt = viewer.resolve(manifest, { ch: 'mt', var: 'pt_1', cols: ['emb'] });
  assert.equal(mt.channel, 'mt');
  assert.deepEqual(mt.regions.map((r) => r.key), ['nominal']); // from all variants when no enabled one has the channel
  assert.equal(mt.variable, 'm_vis');
  // the channel prefers one the enabled variants have
  assert.equal(viewer.resolve(manifest, { cols: ['emb'], ch: undefined }).channel, 'et');
  assert.equal(viewer.resolve({ ...manifest, channels: [{ key: 'tt', label: 'τhτh' }] }, { cols: ['classic'] }).channel, 'et');
  // an empty manifest does not throw
  const empty = viewer.resolve({ variants: [] }, {});
  assert.equal(empty.channel, null);
  assert.equal(empty.variable, null);
  assert.equal(empty.index, -1);
});

test('columns keep an explicit order and drop unknown keys', () => {
  assert.deepEqual(viewer.resolveColumns(manifest, ['emb_ff', 'x'], ['emb', 'emb_ff', 'classic']), { order: ['emb', 'emb_ff', 'classic'], enabled: ['emb_ff'] });
  assert.deepEqual(viewer.resolveColumns(manifest, [], null), { order: ['classic', 'emb', 'emb_ff'], enabled: ['classic', 'emb', 'emb_ff'] });
});

test('image URLs carry the stamp of the source that published the channel', () => {
  const sel = { channel: 'et', region: 'nominal', category: 'inclusive', variable: 'm_vis' };
  assert.equal(viewer.imageUrl(manifest, 'emb', sel, 'png'), 'data/emb/et/nominal/inclusive/m_vis.png?v=20261005T013000');
  assert.equal(viewer.imageUrl(manifest, 'classic', sel, 'pdf'), 'data/classic/et/nominal/inclusive/m_vis.pdf?v=20261004T100000');
  assert.equal(viewer.sourceStamp(manifest, 'emb_ff', 'et'), '20261005T013000'); // no source: the gallery stamp
  assert.equal(viewer.imageUrl(manifest, 'emb', { ...sel, channel: 'τ τ' }, 'png'), 'data/emb/%CF%84%20%CF%84/nominal/inclusive/m_vis.png?v=20261005T013000');
  assert.deepEqual(viewer.plotFormats(manifest, 'emb', sel), both);
  assert.equal(viewer.plotFormats(manifest, 'emb', { ...sel, variable: 'eta_1' }), null);
  assert.equal(viewer.plotFormats(manifest, 'emb', { ...sel, variable: 'constructor' }), null);
});

test('missing plots say what exists', () => {
  const sel = { channel: 'et', region: 'nominal', category: 'inclusive', variable: 'm_vis' };
  assert.equal(viewer.describeMissing(manifest, 'emb', sel), '');
  assert.equal(viewer.describeMissing(manifest, 'emb', { ...sel, channel: 'mt' }), 'Not published for channel μτh; available: eτh, τhτh.');
  assert.equal(viewer.describeMissing(manifest, 'emb', { ...sel, region: 'ss' }), 'Not published for region same sign; available: nominal.');
  assert.equal(viewer.describeMissing(manifest, 'emb', { ...sel, category: 'btag' }), 'Not published for category btag; available: inclusive.');
  assert.equal(viewer.describeMissing(manifest, 'emb', { ...sel, variable: 'eta_1' }), 'This variable is not published for this selection; 2 other variables are.');
  assert.equal(viewer.describeMissing(manifest, 'classic', { ...sel, variable: 'eta_1' }), 'No PNG published; available as PDF.');
  assert.equal(viewer.describeMissing(manifest, 'nope', sel), 'Nothing is published for this variant.');
  // keys are looked up as own properties only
  assert.equal(viewer.describeMissing(manifest, 'emb', { ...sel, channel: '__proto__' }), 'Not published for channel __proto__; available: eτh, τhτh.');
  assert.equal(viewer.resolve(manifest, { ch: '__proto__' }).channel, 'et');
});

test('yield table rows and the union of stack groups', () => {
  const sel = { channel: 'et', region: 'nominal', category: 'inclusive' };
  const table = viewer.yieldTable(manifest, ['emb', 'classic', 'emb_ff'], sel);
  assert.deepEqual(table.groups, [{ key: 'emb', label: 'μ→τ emb.' }, { key: 'jetFakes', label: 'jet → τh (other)' }, { key: 'ztt', label: 'Z→ττ' }]);
  const [emb, classic, ff] = table.rows;
  assert.equal(emb.label, 'emb');
  assert.equal(emb.ratio, 14934 / 15000);
  assert.equal(classic.groups.get('jetFakes'), 3000.25);
  assert.equal(classic.groups.has('emb'), false);
  assert.equal(ff.data, 10);
  assert.equal(ff.ratio, null); // zero prediction
  assert.equal(viewer.yieldTable(manifest, ['emb'], { ...sel, channel: 'tt' }).rows[0].published, false);
  assert.equal(viewer.yieldLine(classic), 'Data 14\u2009934 · Pred 16\u2009021 · Data/Pred 0.932');
  assert.equal(viewer.yieldLine(ff), 'Data 10 · Pred 0');
  assert.equal(viewer.yieldLine(viewer.yieldTable(manifest, ['emb'], { ...sel, channel: 'tt' }).rows[0]), 'No yields published');
});

test('number and ratio formatting', () => {
  assert.equal(viewer.formatNumber(14934), '14\u2009934');
  assert.equal(viewer.formatNumber(16021.3), '16\u2009021');
  assert.equal(viewer.formatNumber(1234567.89), '1\u2009234\u2009568');
  assert.equal(viewer.formatNumber(123.45), '123.5');
  assert.equal(viewer.formatNumber(5), '5');
  assert.equal(viewer.formatNumber(0.5), '0.50');
  assert.equal(viewer.formatNumber(-2500.4), '\u22122\u2009500');
  assert.equal(viewer.formatNumber(-0.001), '0.00');
  assert.equal(viewer.formatNumber(12.3456, 3), '12.346');
  for (const bad of [null, undefined, NaN, Infinity, '12']) assert.equal(viewer.formatNumber(bad), '\u2013');
  assert.equal(viewer.formatRatio(14934, 16021.3), '0.932');
  assert.equal(viewer.formatRatio(10, 0), '\u2013');
  assert.equal(viewer.formatRatio(null, 10), '\u2013');
  assert.equal(viewer.formatRatio(5, -10), '\u22120.500');
  assert.equal(viewer.formatRatio(12345, 10), '1\u2009234.500');
});

test('variable filter matches key and label, exact and prefix keys first', () => {
  const variables = [{ key: 'pt_1', label: 'pT(τ1)' }, { key: 'm_vis', label: 'visible mass' }, { key: 'mt_1', label: 'mT' }, { key: 'm', label: 'mass' }];
  assert.deepEqual(viewer.filterVariables(variables, '').map((v) => v.key), ['pt_1', 'm_vis', 'mt_1', 'm']);
  assert.deepEqual(viewer.filterVariables(variables, 'M').map((v) => v.key), ['m', 'm_vis', 'mt_1']);
  assert.deepEqual(viewer.filterVariables(variables, 'mass vis').map((v) => v.key), ['m_vis']);
  assert.deepEqual(viewer.filterVariables(variables, 'τ1').map((v) => v.key), ['pt_1']);
  assert.deepEqual(viewer.filterVariables(variables, 'zzz'), []);
});

test('time formatting', () => {
  assert.equal(viewer.formatTime('2026-10-05T01:30:00'), '2026-10-05 01:30');
  assert.equal(viewer.formatTime(undefined), '');
});

test('measurement galleries: axis titles, notes per plot, facts per channel, no yields', () => {
  const ff = {
    ...manifest,
    kind: 'fake_factors',
    axes: { regions: 'Process', variables: 'Quantity' },
    yields: { v5: { mt: {} } },
    plots: { v5: { mt: { QCD: { n_jets_1p5_up: { fake_factors: ['png'], non_closure_met: ['png'] } } } } },
    notes: { v5: { mt: { QCD: { n_jets_1p5_up: { non_closure_met: 'p = 0.075, set to 1' } } } } },
    facts: { v5: { mt: [{ key: 'ttbar_data_scale', label: 'ttbar data/MC factor', value: '1.145' }] }, v4: { mt: [{ key: 'other', value: 2 }] } },
  };
  assert.equal(viewer.axisTitle(ff, 'regions'), 'Process');
  assert.equal(viewer.axisTitle(ff, 'categories'), 'Category');
  assert.equal(viewer.axisTitle(manifest, 'variables'), 'Variable');
  const sel = { channel: 'mt', region: 'QCD', category: 'n_jets_1p5_up', variable: 'non_closure_met' };
  assert.equal(viewer.noteOf(ff, 'v5', sel), 'p = 0.075, set to 1');
  assert.equal(viewer.noteOf(ff, 'v5', { ...sel, variable: 'fake_factors' }), '');
  assert.equal(viewer.noteOf(manifest, 'emb', sel), '');
  assert.equal(viewer.describeMissing(ff, 'v5', { ...sel, region: 'ttbar' }), 'Not published for process ttbar; available: QCD.');
  assert.equal(viewer.describeMissing(ff, 'v5', { ...sel, variable: 'DR_SR' }), 'This quantity is not published for this selection; 2 other quantities are.');
  const table = viewer.factTable(ff, ['v5', 'v4', 'none'], 'mt');
  assert.deepEqual(table.columns, [{ key: 'ttbar_data_scale', label: 'ttbar data/MC factor' }, { key: 'other', label: 'other' }]);
  assert.deepEqual(table.rows.map((r) => [r.key, Object.fromEntries(r.values)]), [['v5', { ttbar_data_scale: '1.145' }], ['v4', { other: '2' }], ['none', {}]]);
  assert.equal(viewer.factTable(ff, ['v5'], 'et').columns.length, 0);
  assert.equal(viewer.hasYields(ff), false);
  assert.equal(viewer.hasYields(manifest), true);
});
