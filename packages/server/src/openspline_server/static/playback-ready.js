// Install listeners before assigning the descriptor, after avatar preparation.
export function waitForPlayback(element, {signal, onBlocked, timeout = 60000} = {}) {
  return new Promise((resolve, reject) => {
    let timer;
    const finish = error => {
      clearTimeout(timer);
      element.removeEventListener('statechange', state);
      element.removeEventListener('avatarerror', failed);
      element.removeEventListener('autoplayblocked', blocked);
      signal?.removeEventListener('abort', aborted);
      error ? reject(error) : resolve();
    };
    const state = e => {
      if (e.detail === 'ready') finish();
      else if (e.detail === 'disconnected') finish(new Error('Playback connection closed.'));
    };
    const failed = e => finish(e.detail instanceof Error ? e.detail : new Error(String(e.detail)));
    const blocked = () => { clearTimeout(timer); onBlocked?.(); };
    const aborted = () => finish(new Error('Session ended.'));
    element.addEventListener('statechange', state);
    element.addEventListener('avatarerror', failed);
    element.addEventListener('autoplayblocked', blocked);
    signal?.addEventListener('abort', aborted, {once: true});
    timer = setTimeout(() => finish(new Error('Playback connection timed out. Check the server connection.')), timeout);
    if (signal?.aborted) aborted();
  });
}
