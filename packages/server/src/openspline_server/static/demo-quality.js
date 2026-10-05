export async function prepareDemoQuality(request, quality, progress, {devices, pause = () => new Promise(resolve => setTimeout(resolve, 750))} = {}) {
  let job = await request('/v1/demo/quality', {
    method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify({quality, devices}),
  });
  const messages = {
    checking: `Checking local ${quality}-quality models…`,
    unloading: 'Releasing the previous model…',
    loading: `Loading ${quality} quality on ${devices?.length > 1 ? `${devices.length} GPUs` : 'your GPU'}…`,
    restoring: 'Could not load the selected quality. Restoring the previous model…',
  };
  while (job.state !== 'ready') {
    if (job.state === 'error') throw new Error(job.error || 'Could not prepare this quality.');
    progress(messages[job.state] || 'Preparing the selected quality…');
    await pause();
    job = await request(`/v1/demo/quality/${job.id}`);
  }
}
