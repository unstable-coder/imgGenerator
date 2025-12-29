document.addEventListener('DOMContentLoaded', () => {
  const form = document.getElementById('attrForm');
  const loading = document.getElementById('loading');
  const resultsGrid = document.getElementById('resultsGrid');
  const variationsSlider = document.getElementById('variationsSlider');
  const variationsVal = document.getElementById('variationsVal');

  form.addEventListener('submit', async (e) => {
    e.preventDefault();
    // Read gender and expression, map them to attrs = [Smiling, Male]
    const gender = document.getElementById('genderSelect').value; // 'male'|'female'
    const expression = document.getElementById('expressionSelect').value; // hint
    const male = gender === 'male' ? 1 : 0;
    // Map expressions to Smiling for the model: Happy -> 1, others -> 0
    const smiling = (expression === 'Happy') ? 1 : 0;
    const attrs = [smiling, male];
    const variations = parseInt(variationsSlider.value || '1', 10);

    // UI
    loading.classList.remove('d-none');
    resultsGrid.innerHTML = '';

    try {
      const resp = await fetch('/generate', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ attrs, expression, variations })
      });

      if (!resp.ok) throw new Error('Request failed');

      const data = await resp.json();
      const images = data.images || [];

      if (images.length === 0) {
        resultsGrid.innerHTML = '<div class="text-muted">No images returned.</div>';
      } else {
        images.forEach(b64 => {
          const img = document.createElement('img');
          img.src = 'data:image/png;base64,' + b64;
          img.className = 'img-thumbnail m-1';
          img.style.width = '180px';
          resultsGrid.appendChild(img);
        });
      }
    } catch (err) {
      alert('Error generating images: ' + err.message);
    } finally {
      loading.classList.add('d-none');
    }
  });

  // Update slider display
  if (variationsSlider && variationsVal) {
    variationsSlider.addEventListener('input', () => {
      variationsVal.textContent = variationsSlider.value;
    });
  }
});