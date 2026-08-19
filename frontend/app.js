const API_BASE = window.API_BASE || '';

const form = document.getElementById('predict-form');
const placeholder = document.getElementById('placeholder');
const result = document.getElementById('result');
const errorBox = document.getElementById('error-box');
const probValue = document.getElementById('prob-value');
const probFill = document.getElementById('prob-fill');
const verdict = document.getElementById('verdict');
const rawJson = document.getElementById('raw-json');
const modelNote = document.getElementById('model-note');
const kpiGrid = document.getElementById('kpi-grid');
const invalidTestButton = document.getElementById('invalid-test');

const ETIQUETAS = {
  'admin.': 'Administrativo',
  'blue-collar': 'Obrero',
  entrepreneur: 'Empresario',
  housemaid: 'Trabajo domestico',
  management: 'Gerencia',
  retired: 'Jubilado',
  'self-employed': 'Independiente',
  services: 'Servicios',
  student: 'Estudiante',
  technician: 'Tecnico',
  unemployed: 'Desempleado',
  unknown: 'Sin dato',
  divorced: 'Divorciado',
  married: 'Casado',
  single: 'Soltero',
  primary: 'Primaria',
  secondary: 'Secundaria',
  tertiary: 'Superior',
};

function llenarSelect(id, valores) {
  const select = document.getElementById(id);
  if (!select) return;
  select.innerHTML = valores
    .map((v) => `<option value="${v}">${ETIQUETAS[v] || v}</option>`)
    .join('');
}

async function cargarModelo() {
  try {
    const response = await fetch(`${API_BASE}/model-info`);
    const info = await response.json();

    if (!response.ok) {
      throw new Error(info.detail || 'No se pudo leer la informacion del modelo');
    }

    llenarSelect('job', info.categories.job);
    llenarSelect('marital', info.categories.marital);
    llenarSelect('education', info.categories.education);

    document.getElementById('job').value = 'technician';
    document.getElementById('marital').value = 'married';
    document.getElementById('education').value = 'secondary';

    modelNote.textContent =
      `${info.model} entrenada con ${info.n_train} registros y evaluada con ${info.n_test}. ` +
      `Umbral de decision: ${info.threshold}.`;

    const m = info.metrics;
    kpiGrid.innerHTML = [
      ['Accuracy', m.accuracy],
      ['Precision (yes)', m.precision],
      ['Recall (yes)', m.recall],
      ['F1 (yes)', m.f1],
      ['ROC AUC', m.roc_auc],
    ]
      .map(
        ([nombre, valor]) => `
        <div class="kpi">
          <span class="kpi-label">${nombre}</span>
          <strong>${valor}</strong>
        </div>`,
      )
      .join('');
  } catch (error) {
    modelNote.textContent = `No se pudo contactar la API: ${error.message}`;
  }
}

function mostrarError(mensaje) {
  result.classList.add('hidden');
  placeholder.classList.add('hidden');
  errorBox.classList.remove('hidden');
  errorBox.innerHTML = `<strong>La API rechazo la solicitud</strong><span>${mensaje}</span>`;
}

function mostrarResultado(body) {
  errorBox.classList.add('hidden');
  placeholder.classList.add('hidden');
  result.classList.remove('hidden');

  const porcentaje = Math.round(body.probability * 100);
  probValue.textContent = `${porcentaje}%`;
  probFill.style.width = `${porcentaje}%`;
  probFill.classList.toggle('alta', body.prediction === 'yes');

  verdict.textContent = body.classification;
  verdict.className = `verdict ${body.prediction === 'yes' ? 'positivo' : 'negativo'}`;
}

async function enviar(payload) {
  rawJson.textContent = 'Esperando respuesta...';

  try {
    const response = await fetch(`${API_BASE}/predict`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    });

    const body = await response.json();
    rawJson.textContent = `HTTP ${response.status}\n${JSON.stringify(body, null, 2)}`;

    if (!response.ok) {
      mostrarError(body.detail || body.error || 'Solicitud rechazada');
      return;
    }

    mostrarResultado(body);
  } catch (error) {
    rawJson.textContent = String(error);
    mostrarError(`No se pudo contactar la API. ${error.message}`);
  }
}

function leerFormulario() {
  const data = Object.fromEntries(new FormData(form));
  return {
    age: Number(data.age),
    job: data.job,
    marital: data.marital,
    education: data.education,
    balance: Number(data.balance),
    housing: data.housing,
    loan: data.loan,
    campaign: Number(data.campaign),
  };
}

form.addEventListener('submit', (event) => {
  event.preventDefault();
  enviar(leerFormulario());
});

invalidTestButton.addEventListener('click', () => {
  enviar({ ...leerFormulario(), age: -10 });
});

cargarModelo();
