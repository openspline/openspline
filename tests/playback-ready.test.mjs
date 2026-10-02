import test from 'node:test';
import assert from 'node:assert/strict';
import {waitForPlayback} from '../packages/server/src/openspline_server/static/playback-ready.js';

test('player errors reach the demo immediately', async () => {
  const element = new EventTarget();
  const waiting = waitForPlayback(element);
  element.dispatchEvent(new CustomEvent('avatarerror', {detail: new Error('Connection refused')}));
  await assert.rejects(waiting, /Connection refused/);
});

test('autoplay permission pauses the timeout and waits for real readiness', async t => {
  t.mock.timers.enable({apis: ['setTimeout']});
  const element = new EventTarget();
  let blocked = false, ready = false;
  const waiting = waitForPlayback(element, {timeout: 100, onBlocked: () => blocked = true}).then(() => ready = true);
  element.dispatchEvent(new Event('autoplayblocked'));
  t.mock.timers.tick(1000);
  await Promise.resolve();
  assert.equal(blocked, true);
  assert.equal(ready, false);
  element.dispatchEvent(new CustomEvent('statechange', {detail: 'ready'}));
  await waiting;
  assert.equal(ready, true);
});

test('stopping cancels playback readiness and its timer', async () => {
  const controller = new AbortController();
  const waiting = waitForPlayback(new EventTarget(), {signal: controller.signal});
  controller.abort();
  await assert.rejects(waiting, /Session ended/);
});
