document.getElementById('predict-form').addEventListener('submit', async function(e) {
  e.preventDefault();
  const form = e.target;
  const data = new FormData(form);
  const resDiv = document.getElementById('result');
  resDiv.textContent = 'Sending...';

  try {
    const resp = await fetch('/api/predict', { method: 'POST', body: data });
    const j = await resp.json();
    if (!resp.ok) {
      resDiv.textContent = 'Error: ' + (j.message || resp.statusText);
      return;
    }

    const parts = [];
    parts.push('<strong>Fused Label:</strong> ' + (j.fused_label || 'Unknown'));
    parts.push('<strong>FER probs:</strong> ' + JSON.stringify(j.fer_pred));
    parts.push('<strong>Speech probs:</strong> ' + JSON.stringify(j.speech_pred));
    parts.push('<strong>Text probs:</strong> ' + JSON.stringify(j.text_pred));
    parts.push('<strong>Combined:</strong> ' + JSON.stringify(j.combined));

    resDiv.innerHTML = parts.join('<br>');
  } catch (err) {
    resDiv.textContent = 'Request failed: ' + err.message;
  }
});
