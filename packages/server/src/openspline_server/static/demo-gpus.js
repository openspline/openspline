export async function setupGpuPicker(request, quality, fieldset, list, hint) {
  let editable = false, persistent = false, gpus = [], selected = {}, locked = false;
  function render() {
    list.replaceChildren();
    const high = quality.value === 'high';
    for (const gpu of gpus) {
      const label = document.createElement('label');
      label.className = 'gpu-option';
      const input = document.createElement('input');
      input.type = high ? 'checkbox' : 'radio';
      input.name = 'gpu';
      input.value = String(gpu.id);
      input.checked = (selected[quality.value] || []).includes(gpu.id);
      input.addEventListener('change', () => {
        selected[quality.value] = [...list.querySelectorAll('input:checked')].map(el => Number(el.value));
      });
      const description = document.createElement('span');
      description.textContent = `GPU ${gpu.physical_id} · ${gpu.name}`;
      if (gpu.memory_total_mb) {
        const memory = document.createElement('small');
        memory.textContent = `${Math.round(gpu.memory_total_mb / 1024)} GB total · ${Math.round(gpu.memory_free_mb / 1024)} GB free`;
        description.append(memory);
      }
      label.append(input, description);
      list.append(label);
    }
    fieldset.disabled = locked || !editable;
    if (editable) hint.textContent = `${high ? 'Choose one or more GPUs for this session.' : 'Choose one GPU.'} Applied when you start.${persistent ? ' Saved for restarts.' : ''}`;
  }
  quality.addEventListener('change', render);
  try {
    const data = await request('/v1/demo/gpus');
    ({editable, persistent, gpus, selected} = data);
    hint.textContent = data.message;
    render();
  } catch {
    hint.textContent = 'GPU discovery is unavailable; using the configured GPUs.';
    fieldset.disabled = true;
  }
  return {
    devices() {
      if (!editable) return undefined;
      const devices = selected[quality.value] || [];
      if (!devices.length) throw new Error('Select at least one GPU.');
      return devices;
    },
    lock(value) { locked = value; fieldset.disabled = value || !editable; },
    async refresh() {
      const data = await request('/v1/demo/gpus');
      selected = data.selected;
      gpus = data.gpus;
      render();
    },
  };
}
