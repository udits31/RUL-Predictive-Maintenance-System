// API Configuration
const API_BASE = 'http://localhost:5000';
const DASHBOARD_VERSION = '2024-04-13-v2'; // Force cache refresh

// Global state
let modelsData = null;
let shapData = null;
let fleetData = null;
let simulationData = null;
let currentCycleIndex = 0;
let playInterval = null;

console.log('Dashboard JS loaded - Version:', DASHBOARD_VERSION);

// ── Design tokens (match CSS vars) ──────────────────────────────────────────
const T = {
    blue:   '#2d7cf6',
    green:  '#00b37e',
    amber:  '#d4820a',
    red:    '#d94040',
    gray:   '#8b93a1',
    border: 'rgba(255,255,255,0.06)',
    bg1:    '#0d1117',
    bg2:    '#131820',
    bg3:    '#1a2030',
    text0:  '#f0f2f5',
    text1:  '#8b93a1',
    text2:  '#4e5561',
    mono:   "'IBM Plex Mono', monospace",
    sans:   "'IBM Plex Sans', sans-serif",
};

const MODEL_COLORS = ['#2d7cf6', '#00b37e', '#d4820a', '#d94040', '#7c3aed', '#ff2d55'];

// ── Chart.js defaults ────────────────────────────────────────────────────────
Chart.defaults.color = T.text1;
Chart.defaults.borderColor = T.border;
Chart.defaults.font.family = T.mono;
Chart.defaults.font.size = 11;
Chart.defaults.animation.duration = 600;
Chart.defaults.animation.easing = 'easeOutCubic';

// Sensor descriptions
const SENSOR_INFO = {
    's2':  { name: 'T2  — Fan inlet temperature',       type: 'temperature', desc: 'Inlet air temperature, affects engine efficiency and power output.' },
    's3':  { name: 'T30 — HPC outlet temperature',      type: 'temperature', desc: 'High Pressure Compressor outlet temp — key degradation indicator.' },
    's4':  { name: 'P30 — HPC outlet pressure',         type: 'pressure',    desc: 'HPC outlet pressure — indicates compressor health directly.' },
    's7':  { name: 'T50 — LPT outlet temperature',      type: 'temperature', desc: 'Low Pressure Turbine outlet temp — turbine efficiency indicator.' },
    's8':  { name: 'Nf  — Physical fan speed',          type: 'speed',       desc: 'Fan rotational speed — primary thrust generator.' },
    's9':  { name: 'Nc  — Physical core speed',         type: 'speed',       desc: 'Core spool speed — critical for engine health monitoring.' },
    's11': { name: 'Ps30 — Static HPC pressure',        type: 'pressure',    desc: 'Static pressure measurement — compressor performance.' },
    's12': { name: 'phi — Fuel/pressure ratio',         type: 'pressure',    desc: 'Fuel-to-pressure ratio — combustion efficiency metric.' },
    's13': { name: 'NRf — Corrected fan speed',         type: 'speed',       desc: 'Temperature-corrected fan speed — normalized performance.' },
    's14': { name: 'NRc — Corrected core speed',        type: 'speed',       desc: 'Temperature-corrected core speed — strongest RUL predictor.' },
    's15': { name: 'BPR — Bypass ratio',                type: 'pressure',    desc: 'Bypass air ratio — engine efficiency indicator.' },
    's17': { name: 'htBleed — Bleed enthalpy',          type: 'temperature', desc: 'Bleed air energy content — system health indicator.' },
    's20': { name: 'W31 — HPT coolant bleed',           type: 'pressure',    desc: 'High Pressure Turbine cooling flow — thermal management.' },
    's21': { name: 'W32 — LPT coolant bleed',           type: 'pressure',    desc: 'Low Pressure Turbine cooling flow — turbine protection.' },
};

// ── Init ─────────────────────────────────────────────────────────────────────
async function init() {
    setupTabs();
    await loadAllData();
    renderAllTabs();
    animateMetrics();
}

// Count-up animation for numeric metric values
function animateMetrics() {
    document.querySelectorAll('.metric-value').forEach(el => {
        const text = el.textContent.trim();
        if (text === '--' || text === '') return;
        const parsed = parseFloat(text);
        if (isNaN(parsed)) return;

        const duration = 700;
        const steps = 50;
        let step = 0;

        const timer = setInterval(() => {
            step++;
            const progress = step / steps;
            const eased = 1 - Math.pow(1 - progress, 3); // ease-out-cubic
            const current = parsed * eased;

            if (step >= steps) {
                el.textContent = text;
                clearInterval(timer);
            } else {
                el.textContent = text.includes('.')
                    ? current.toFixed(text.split('.')[1]?.length || 2)
                    : Math.floor(current).toString();
            }
        }, duration / steps);
    });
}

// ── Tab switching ─────────────────────────────────────────────────────────────
function setupTabs() {
    document.querySelectorAll('.tab-btn').forEach(btn => {
        btn.addEventListener('click', () => {
            const tabName = btn.dataset.tab;
            document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
            document.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));
            btn.classList.add('active');
            document.getElementById(`${tabName}-tab`).classList.add('active');

            // Animate SHAP bars when tab opens
            if (tabName === 'shap') {
                requestAnimationFrame(() => animateShapBars());
            }
        });
    });
}

// ── Data loading ──────────────────────────────────────────────────────────────
async function loadAllData() {
    try {
        const [models, shap, fleet] = await Promise.all([
            fetch(`${API_BASE}/models`).then(r => r.json()),
            fetch(`${API_BASE}/shap`).then(r => r.json()),
            fetch(`${API_BASE}/fleet`).then(r => r.json()),
        ]);
        modelsData = models.models;
        shapData   = shap;
        fleetData  = fleet;
    } catch (err) {
        console.error('Failed to load data:', err);
        alert('Cannot reach API at ' + API_BASE + '. Ensure the Flask server is running.');
    }
}

function renderAllTabs() {
    try { renderModelsTab(); }    catch (e) { console.error('Models tab error:', e); }
    try { renderShapTab(); }      catch (e) { console.error('SHAP tab error:', e); }
    try { renderFleetTab(); }     catch (e) { console.error('Fleet tab error:', e); }
    try { setupSimulationTab(); } catch (e) { console.error('Sim tab error:', e); }
}

// ── Shared chart options helper ───────────────────────────────────────────────
function baseScales(xTitle = '', yTitle = '') {
    return {
        x: {
            grid:   { color: 'rgba(255,255,255,0.04)', drawBorder: false },
            ticks:  { color: T.text2, font: { family: T.mono, size: 10 } },
            border: { display: false },
            title:  xTitle ? { display: true, text: xTitle, color: T.text2, font: { family: T.mono, size: 10 } } : { display: false },
        },
        y: {
            grid:   { color: 'rgba(255,255,255,0.04)', drawBorder: false },
            ticks:  { color: T.text2, font: { family: T.mono, size: 10 } },
            border: { display: false },
            title:  yTitle ? { display: true, text: yTitle, color: T.text2, font: { family: T.mono, size: 10 } } : { display: false },
        },
    };
}

// ── Tab 1: Model Comparison ──────────────────────────────────────────────────
function renderModelsTab() {
    if (!modelsData) return;

    const models = Object.entries(modelsData);
    const rmseValues = models.map(([, m]) => m.rmse);
    const r2Values   = models.map(([, m]) => m.r2);
    const bestRMSE   = Math.min(...rmseValues);
    const bestR2     = Math.max(...r2Values);
    const bestRMSEName = models.find(([, m]) => m.rmse === bestRMSE)?.[0] ?? '';

    document.getElementById('best-rmse').textContent       = bestRMSE.toFixed(2);
    document.getElementById('best-rmse-model').textContent = bestRMSEName.replace(/_/g, ' ').toUpperCase();
    document.getElementById('best-r2').textContent         = bestR2.toFixed(4);
    document.getElementById('total-features').textContent  = '200+';

    // ── Radar chart ──
    const norm = {
        rmse:  v => Math.max(0, 100 - (v / 50) * 100),
        mae:   v => Math.max(0, 100 - (v / 30) * 100),
        r2:    v => v * 100,
        speed: v => Math.max(0, 100 - (v / 300) * 100),
    };

    new Chart(document.getElementById('radar-chart').getContext('2d'), {
        type: 'radar',
        data: {
            labels: ['RMSE (lower better)', 'MAE (lower better)', 'R² (higher better)', 'Speed (faster better)'],
            datasets: models.map(([name, m], i) => ({
                label: name.replace(/_/g, ' ').toUpperCase(),
                data: [norm.rmse(m.rmse), norm.mae(m.mae), norm.r2(m.r2), norm.speed(m.train_time)],
                backgroundColor: MODEL_COLORS[i] + '18',
                borderColor:     MODEL_COLORS[i],
                borderWidth:     1.5,
                pointBackgroundColor: MODEL_COLORS[i],
                pointBorderColor:     T.bg1,
                pointRadius:     4,
                pointHoverRadius: 5,
            })),
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                r: {
                    beginAtZero: true, max: 100,
                    ticks:      { display: false },
                    grid:       { color: 'rgba(255,255,255,0.05)' },
                    angleLines: { color: 'rgba(255,255,255,0.05)' },
                    pointLabels: { color: T.text1, font: { family: T.mono, size: 14 } },
                },
            },
            plugins: {
                legend: {
                    position: 'bottom',
                    labels: {
                        color: T.text1,
                        padding: 16,
                        usePointStyle: true,
                        pointStyleWidth: 8,
                        font: { family: T.mono, size: 14 },
                    },
                },
            },
        },
    });

    // ── RMSE vs MAE grouped bar ──
    new Chart(document.getElementById('metrics-chart').getContext('2d'), {
        type: 'bar',
        data: {
            labels: models.map(([name]) => name.replace(/_/g, ' ').toUpperCase()),
            datasets: [
                {
                    label: 'RMSE',
                    data: models.map(([, m]) => m.rmse),
                    backgroundColor: T.blue + 'cc',
                    borderRadius: 2,
                    borderSkipped: false,
                },
                {
                    label: 'MAE',
                    data: models.map(([, m]) => m.mae),
                    backgroundColor: T.green + 'cc',
                    borderRadius: 2,
                    borderSkipped: false,
                },
            ],
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    position: 'top',
                    align: 'end',
                    labels: { 
                        color: T.text1, 
                        usePointStyle: true, 
                        pointStyleWidth: 8, 
                        font: { family: T.mono, size: 14 } 
                    },
                },
            },
            scales: {
                ...baseScales('', 'cycles'),
                x: { ...baseScales().x, grid: { display: false } },
            },
            animation: {
                delay: ctx => ctx.dataIndex * 40,
            },
        },
    });

    // ── Table ──
    const tbody = document.querySelector('#models-table tbody');
    tbody.innerHTML = '';
    const bestRmse = Math.min(...models.map(([, m]) => m.rmse));

    models.forEach(([name, m]) => {
        const isBest = m.rmse === bestRmse;
        const row = tbody.insertRow();
        row.innerHTML = `
            <td style="${isBest ? `color:${T.green};font-weight:600;` : ''}">${name.replace(/_/g, ' ').toUpperCase()}${isBest ? ' ✦' : ''}</td>
            <td class="${isBest ? 'best-row' : ''}">${m.rmse.toFixed(2)}</td>
            <td>${m.mae.toFixed(2)}</td>
            <td>${m.r2.toFixed(4)}</td>
            <td style="color:${T.text2}">${m.train_time != null ? m.train_time.toFixed(2) : '—'}</td>
        `;
    });
}

// ── Tab 2: SHAP ──────────────────────────────────────────────────────────────
function renderShapTab() {
    if (!shapData) return;

    const topFeatures = shapData.top_features.slice(0, 10);
    document.getElementById('top-feature').textContent   = topFeatures[0].feature;
    document.getElementById('features-ranked').textContent = shapData.top_features.length;

    const maxVal = topFeatures[0].mean_abs_shap;
    const container = document.getElementById('shap-rows');
    container.innerHTML = '';

    topFeatures.forEach((f, i) => {
        const pct = ((f.mean_abs_shap / maxVal) * 100).toFixed(1);
        const opacity = 1 - (i / topFeatures.length) * 0.55;
        const row = document.createElement('div');
        row.className = 'shap-row';
        row.innerHTML = `
            <span class="shap-rank">${String(i + 1).padStart(2, '0')}</span>
            <span class="shap-name">${f.feature}</span>
            <div class="shap-bar-bg">
                <div class="shap-bar-fill" data-pct="${pct}" style="background:rgba(45,124,246,${opacity});width:${pct}%;"></div>
            </div>
            <span class="shap-val">${f.mean_abs_shap.toFixed(2)}</span>
        `;
        container.appendChild(row);
    });

    // Sensor descriptions
    const sensorList = document.getElementById('sensor-list');
    sensorList.innerHTML = '';
    const seen = new Set();
    topFeatures.forEach(f => {
        const base = f.feature.match(/s\d+/)?.[0];
        if (base && SENSOR_INFO[base] && !seen.has(base)) {
            seen.add(base);
            const info = SENSOR_INFO[base];
            const div = document.createElement('div');
            div.className = `sensor-item ${info.type}`;
            div.innerHTML = `
                <div class="sensor-name">${info.name}</div>
                <div class="sensor-desc">${info.desc}</div>
            `;
            sensorList.appendChild(div);
        }
    });
}

function animateShapBars() {
    document.querySelectorAll('.shap-bar-fill').forEach((bar, i) => {
        bar.style.transform = 'scaleX(0)';
        setTimeout(() => {
            bar.style.transition = `transform 0.55s cubic-bezier(0.16,1,0.3,1) ${i * 40}ms`;
            bar.style.transform  = 'scaleX(1)';
        }, 20);
    });
}

// ── Tab 3: Simulation ─────────────────────────────────────────────────────────
async function setupSimulationTab() {
    if (!fleetData) return;

    const select = document.getElementById('engine-select');
    fleetData.engines.forEach(engine => {
        const opt = document.createElement('option');
        opt.value = engine.engine_id;
        opt.textContent = `Engine ${engine.engine_id}`;
        select.appendChild(opt);
    });

    await loadSimulation(fleetData.engines[0].engine_id);

    select.addEventListener('change', async e => {
        if (playInterval) { 
            clearInterval(playInterval); 
            playInterval = null; 
        }
        document.getElementById('play-btn').textContent = '▶ Play';
        await loadSimulation(parseInt(e.target.value));
    });

    document.getElementById('prev-btn').addEventListener('click', () => {
        if (currentCycleIndex > 0) { 
            currentCycleIndex--; 
            updateSimulationDisplay(); 
        }
    });

    document.getElementById('next-btn').addEventListener('click', () => {
        if (currentCycleIndex < simulationData.simulation.length - 1) {
            currentCycleIndex++;
            updateSimulationDisplay();
        }
    });

    document.getElementById('play-btn').addEventListener('click', togglePlay);

    document.getElementById('cycle-slider').addEventListener('input', e => {
        currentCycleIndex = parseInt(e.target.value);
        updateSimulationDisplay();
    });
}

async function loadSimulation(engineId) {
    try {
        simulationData = await fetch(`${API_BASE}/simulate/${engineId}`).then(r => r.json());
        document.getElementById('cycle-slider').max = simulationData.simulation.length - 1;
        renderSimulationChart();
        resetSimulation();
    } catch (err) {
        console.error('Simulation load error:', err);
    }
}

function resetSimulation() {
    currentCycleIndex = 0;

    if (window.simulationChart) {
        window.simulationChart.data.datasets[0].data = [];
        window.simulationChart.update('none');
    }

    if (!simulationData || !simulationData.simulation || simulationData.simulation.length === 0) {
        console.error('resetSimulation: Invalid simulation data');
        return;
    }

    const currentCycleEl = document.getElementById('current-cycle');
    const predictedRulEl = document.getElementById('predicted-rul');
    const actualRulEl = document.getElementById('actual-rul');
    const sliderEl = document.getElementById('cycle-slider');
    const progressEl = document.getElementById('cycle-progress');
    const bannerEl = document.getElementById('status-banner');
    const statusTextEl = bannerEl?.querySelector('.status-text');
    const statusDetailEl = document.getElementById('status-detail');

    if (!currentCycleEl || !predictedRulEl || !actualRulEl || !sliderEl || !progressEl || !bannerEl || !statusTextEl || !statusDetailEl) {
        console.error('resetSimulation: Required DOM elements not found');
        return;
    }

    currentCycleEl.textContent = simulationData.simulation[0].cycle;
    predictedRulEl.textContent = '—';
    actualRulEl.textContent = simulationData.simulation[0].actual_rul.toFixed(1);
    sliderEl.value = 0;
    progressEl.textContent = '0%';

    bannerEl.className = 'status-banner';
    statusTextEl.textContent = 'READY';
    statusDetailEl.textContent = 'Press play to begin simulation';
}

function renderSimulationChart() {
    const cycles    = simulationData.simulation.map(d => d.cycle);
    const actualRUL = simulationData.simulation.map(d => d.actual_rul);

    if (window.simulationChart) window.simulationChart.destroy();

    const ctx = document.getElementById('simulation-chart').getContext('2d');

    window.simulationChart = new Chart(ctx, {
        type: 'line',
        data: {
            labels: cycles,
            datasets: [
                {
                    label: 'Predicted RUL',
                    data: [],
                    borderColor:     T.blue,
                    backgroundColor: T.blue + '18',
                    borderWidth: 2,
                    tension: 0.3,
                    fill: false,
                    pointRadius: 0,
                    pointHoverRadius: 4,
                    pointHoverBackgroundColor: T.blue,
                },
                {
                    label: 'Actual RUL',
                    data: actualRUL,
                    borderColor:     T.text2,
                    backgroundColor: 'transparent',
                    borderWidth: 1.5,
                    tension: 0.3,
                    fill: false,
                    pointRadius: 0,
                    pointHoverRadius: 0,
                },
                {
                    label: 'Alert Threshold',
                    data: Array(cycles.length).fill(30),
                    borderColor:     T.red + '60',
                    borderWidth: 1,
                    borderDash: [4, 4],
                    fill: { target: 'origin', above: T.red + '0d', below: 'transparent' },
                    pointRadius: 0,
                    tension: 0,
                },
            ],
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            interaction: { mode: 'index', intersect: false },
            plugins: {
                legend: { display: false },
                tooltip: {
                    backgroundColor: T.bg3,
                    borderColor: 'rgba(255,255,255,0.08)',
                    borderWidth: 1,
                    titleColor: T.text0,
                    bodyColor:  T.text1,
                    titleFont:  { family: T.mono, size: 11 },
                    bodyFont:   { family: T.mono, size: 10 },
                    padding:    10,
                    callbacks: {
                        title:  items => `Cycle ${items[0].label}`,
                        label:  item => ` ${item.dataset.label}: ${item.parsed.y?.toFixed(1) ?? '—'}`,
                    },
                },
            },
            scales: {
                ...baseScales('Cycle', 'RUL (cycles)'),
            },
            animation: { duration: 0 },
        },
    });
}

function updateSimulationDisplay() {
    if (!simulationData || !simulationData.simulation) {
        console.error('updateSimulationDisplay: Invalid simulation data');
        return;
    }
    
    const d = simulationData.simulation[currentCycleIndex];
    if (!d) {
        console.error('updateSimulationDisplay: No data at index', currentCycleIndex);
        return;
    }

    const currentCycleEl = document.getElementById('current-cycle');
    const predictedRulEl = document.getElementById('predicted-rul');
    const actualRulEl = document.getElementById('actual-rul');
    const sliderEl = document.getElementById('cycle-slider');
    const progressEl = document.getElementById('cycle-progress');
    const bannerEl = document.getElementById('status-banner');
    const statusTextEl = bannerEl?.querySelector('.status-text');
    const statusDetailEl = document.getElementById('status-detail');

    if (!currentCycleEl || !predictedRulEl || !actualRulEl || !sliderEl || !progressEl || !bannerEl || !statusTextEl || !statusDetailEl) {
        console.error('updateSimulationDisplay: Required DOM elements not found');
        return;
    }

    currentCycleEl.textContent = d.cycle;
    predictedRulEl.textContent = d.predicted_rul.toFixed(1);
    actualRulEl.textContent = d.actual_rul.toFixed(1);
    sliderEl.value = currentCycleIndex;

    const pct = ((currentCycleIndex / (simulationData.simulation.length - 1)) * 100).toFixed(0);
    progressEl.textContent = `${pct}%`;

    // Status banner
    const risk = d.risk_level.toLowerCase();
    bannerEl.className = `status-banner ${risk}`;
    statusTextEl.textContent = d.risk_level;

    if (risk === 'critical') {
        statusDetailEl.textContent = `${d.predicted_rul.toFixed(0)} cycles to failure — schedule maintenance`;
    } else if (risk === 'warning' || risk === 'caution') {
        statusDetailEl.textContent = `${d.predicted_rul.toFixed(0)} cycles remaining — monitor closely`;
    } else {
        statusDetailEl.textContent = `${d.predicted_rul.toFixed(0)} cycles remaining — nominal operation`;
    }

    // Update chart with predicted data up to current index
    if (window.simulationChart) {
        const predictedData = simulationData.simulation
            .slice(0, currentCycleIndex + 1)
            .map(d => d.predicted_rul);
        
        window.simulationChart.data.datasets[0].data = predictedData;
        window.simulationChart.update('none');
    }
}

function togglePlay() {
    if (!simulationData || !simulationData.simulation) {
        console.error('togglePlay: No simulation data loaded');
        return;
    }

    const btn = document.getElementById('play-btn');
    if (!btn) {
        console.error('togglePlay: Play button not found');
        return;
    }

    // Restart if at end
    if (currentCycleIndex >= simulationData.simulation.length - 1) {
        currentCycleIndex = 0;
        if (window.simulationChart) {
            window.simulationChart.data.datasets[0].data = [];
            window.simulationChart.update('none');
        }
    }

    if (playInterval) {
        // Pause
        clearInterval(playInterval);
        playInterval = null;
        btn.textContent = '▶ Play';
    } else {
        // Play
        btn.textContent = '⏸ Pause';
        
        // Update immediately to show current point
        updateSimulationDisplay();
        
        // Start interval
        playInterval = setInterval(() => {
            if (currentCycleIndex >= simulationData.simulation.length - 1) {
                clearInterval(playInterval);
                playInterval = null;
                btn.textContent = '▶ Play';
                return;
            }
            currentCycleIndex++;
            updateSimulationDisplay();
        }, 80);
    }
}

// ── Tab 4: Fleet Health ───────────────────────────────────────────────────────
function renderFleetTab() {
    if (!fleetData) return;

    document.getElementById('total-fleet').textContent   = fleetData.total_engines;
    document.getElementById('critical-count').textContent = fleetData.risk_summary.critical;
    document.getElementById('warning-count').textContent  = fleetData.risk_summary.warning;
    document.getElementById('healthy-count').textContent  =
        (fleetData.risk_summary.healthy ?? 0) + (fleetData.risk_summary.caution ?? 0);

    // ── Histogram ──
    const rulValues = fleetData.engines.map(e => e.predicted_rul);
    const bins = [0, 20, 40, 60, 80, 100, 130];
    const counts = new Array(bins.length - 1).fill(0);
    rulValues.forEach(rul => {
        for (let i = 0; i < bins.length - 1; i++) {
            if (rul >= bins[i] && rul < bins[i + 1]) { counts[i]++; break; }
        }
    });

    const histColors = [T.red, T.red + 'bb', T.amber, T.green + '88', T.green + 'bb', T.green];

    new Chart(document.getElementById('rul-histogram').getContext('2d'), {
        type: 'bar',
        data: {
            labels: ['0–20', '20–40', '40–60', '60–80', '80–100', '100+'],
            datasets: [{
                label: 'Engine Count',
                data: counts,
                backgroundColor: histColors,
                borderRadius: 2,
                borderSkipped: false,
            }],
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: { legend: { display: false } },
            scales: {
                ...baseScales('RUL Range (cycles)', 'Count'),
                x: { ...baseScales().x, grid: { display: false } },
            },
            animation: { delay: ctx => ctx.dataIndex * 60 },
        },
    });

    // ── Scatter ──
    const scatterColors = fleetData.engines.map(e => {
        const r = e.risk_level.toLowerCase();
        if (r === 'critical')              return T.red   + 'cc';
        if (r === 'warning' || r === 'caution') return T.amber + 'cc';
        return T.green + 'cc';
    });

    new Chart(document.getElementById('scatter-chart').getContext('2d'), {
        type: 'scatter',
        data: {
            datasets: [
                {
                    label: 'Engines',
                    data: fleetData.engines.map(e => ({ x: e.actual_rul, y: e.predicted_rul })),
                    backgroundColor: scatterColors,
                    pointRadius: 4,
                    pointHoverRadius: 6,
                },
                {
                    label: 'Perfect Prediction',
                    data: [{ x: 0, y: 0 }, { x: 130, y: 130 }],
                    type: 'line',
                    borderColor: T.text2,
                    borderDash: [4, 4],
                    borderWidth: 1,
                    pointRadius: 0,
                },
            ],
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    display: true,
                    labels: { color: T.text1, usePointStyle: true, pointStyleWidth: 8, font: { family: T.mono, size: 10 } },
                },
                tooltip: {
                    backgroundColor: T.bg3,
                    borderColor: 'rgba(255,255,255,0.08)',
                    borderWidth: 1,
                    titleFont: { family: T.mono, size: 11 },
                    bodyFont:  { family: T.mono, size: 10 },
                    titleColor: T.text0,
                    bodyColor:  T.text1,
                },
            },
            scales: baseScales('Actual RUL', 'Predicted RUL'),
        },
    });

    // ── Fleet grid with compact/expanded toggle ──
    let fleetCompactMode = false;
    renderFleetGrid(fleetData.engines, fleetCompactMode);

    document.getElementById('sort-select').addEventListener('change', e => {
        const by  = e.target.value;
        let sorted = [...fleetData.engines];
        const riskOrder = { CRITICAL: 0, WARNING: 1, CAUTION: 2, HEALTHY: 3 };

        if      (by === 'rul-asc')  sorted.sort((a, b) => a.predicted_rul - b.predicted_rul);
        else if (by === 'rul-desc') sorted.sort((a, b) => b.predicted_rul - a.predicted_rul);
        else if (by === 'risk')     sorted.sort((a, b) => (riskOrder[a.risk_level] ?? 4) - (riskOrder[b.risk_level] ?? 4));
        else sorted.sort((a, b) => a.engine_id - b.engine_id);

        renderFleetGrid(sorted, fleetCompactMode);
    });

    // Compact/Expanded toggle functionality
    const toggleBtn = document.getElementById('fleet-toggle');
    if (toggleBtn) {
        toggleBtn.addEventListener('click', () => {
            fleetCompactMode = !fleetCompactMode;
            toggleBtn.textContent = fleetCompactMode ? 'Expanded View' : 'Compact View';
            
            // Re-render with current sort order
            const sortBy = document.getElementById('sort-select').value;
            let sorted = [...fleetData.engines];
            const riskOrder = { CRITICAL: 0, WARNING: 1, CAUTION: 2, HEALTHY: 3 };

            if      (sortBy === 'rul-asc')  sorted.sort((a, b) => a.predicted_rul - b.predicted_rul);
            else if (sortBy === 'rul-desc') sorted.sort((a, b) => b.predicted_rul - a.predicted_rul);
            else if (sortBy === 'risk')     sorted.sort((a, b) => (riskOrder[a.risk_level] ?? 4) - (riskOrder[b.risk_level] ?? 4));
            else sorted.sort((a, b) => a.engine_id - b.engine_id);

            renderFleetGrid(sorted, fleetCompactMode);
        });
    }
}

function renderFleetGrid(engines, compact = false) {
    const grid = document.getElementById('fleet-grid');
    grid.innerHTML = '';
    
    // Toggle compact class on grid
    if (compact) {
        grid.classList.add('compact');
    } else {
        grid.classList.remove('compact');
    }

    engines.forEach(e => {
        const tile = document.createElement('div');
        tile.className = `engine-tile ${e.risk_level.toLowerCase()}`;
        tile.title = `Engine ${e.engine_id} — Predicted RUL: ${e.predicted_rul.toFixed(0)} cycles`;
        
        if (compact) {
            // Compact mode: just show RUL number
            tile.innerHTML = `<div class="engine-rul">${e.predicted_rul.toFixed(0)}</div>`;
        } else {
            // Expanded mode: show full details
            tile.innerHTML = `
                <div class="engine-id">ENG ${String(e.engine_id).padStart(3, '0')}</div>
                <div class="engine-rul">${e.predicted_rul.toFixed(0)}</div>
                <div class="engine-label">cycles</div>
            `;
        }
        
        grid.appendChild(tile);
    });
}

// ── Bootstrap ─────────────────────────────────────────────────────────────────
document.addEventListener('DOMContentLoaded', () => {
    console.log('Dashboard initializing...');
    init();
});