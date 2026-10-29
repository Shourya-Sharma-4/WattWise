/**
 * WattWise — ML-Based Electricity Consumption Anomaly Detection System
 * Frontend Controller: script.js
 * 
 * Interacts with FastAPI backend endpoints (/api/summary, /api/readings, /api/predict).
 * Renders interactive Chart.js visualizations, manages live prediction form,
 * and maintains telemetry data pagination and filtering.
 */

// State variables
let timeseriesChartInstance = null;
let applianceChartInstance = null;

let currentReadingsLimit = 100;
let currentTableOffset = 0;
let currentTableLimit = 25;
let currentFilterMode = 'all'; // 'all', 'anomalies', 'normal'
let currentSearchQuery = '';
let totalFilteredRecords = 0;

// Presets data for quick demonstration
const DEMO_PRESETS = {
    'preset-normal-hvac': {
        appliance: 'HVAC',
        hour: 14,
        day_of_week: 2,
        power: 1850.0,
        baseline: 1800.0
    },
    'preset-hvac-anomaly': {
        appliance: 'HVAC',
        hour: 3,
        day_of_week: 1,
        power: 3900.0,
        baseline: 450.0
    },
    'preset-fridge-leak': {
        appliance: 'Refrigeration',
        hour: 2,
        day_of_week: 0,
        power: 680.0,
        baseline: 110.0
    },
    'preset-wh-stuck': {
        appliance: 'Water Heater',
        hour: 2,
        day_of_week: 3,
        power: 4400.0,
        baseline: 80.0
    }
};

const DAY_NAMES = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'];

// DOM Content Loaded Handler
document.addEventListener('DOMContentLoaded', () => {
    initNavigation();
    initHourSlider();
    initPresets();
    initPredictionForm();
    initTableControls();
    initRefreshButton();

    // Load initial dashboard data
    loadDashboardSummary();
    loadTimeseriesData(currentReadingsLimit);
    loadTableReadings();
});

/**
 * Smooth scrolling and active navigation menu highlights
 */
function initNavigation() {
    const navItems = document.querySelectorAll('.nav-item');
    navItems.forEach(item => {
        item.addEventListener('click', (e) => {
            const href = item.getAttribute('href');
            if (href.startsWith('#')) {
                navItems.forEach(n => n.classList.remove('active'));
                item.classList.add('active');
            }
        });
    });
}

/**
 * Synchronizes hour range slider, number box, and 12/24 hour display
 */
function initHourSlider() {
    const slider = document.getElementById('slider-hour');
    const numberInput = document.getElementById('input-hour');
    const preview = document.getElementById('hour-time-preview');

    function updateHour(val) {
        const h = parseInt(val, 10);
        slider.value = h;
        numberInput.value = h;
        const ampm = h === 0 ? '12 AM' : h === 12 ? '12 PM' : h < 12 ? `${h} AM` : `${h - 12} PM`;
        preview.textContent = `${String(h).padStart(2, '0')}:00 (${ampm})`;
    }

    slider.addEventListener('input', (e) => updateHour(e.target.value));
    numberInput.addEventListener('input', (e) => {
        let val = parseInt(e.target.value, 10);
        if (isNaN(val)) val = 0;
        if (val < 0) val = 0;
        if (val > 23) val = 23;
        updateHour(val);
    });

    updateHour(slider.value);
}

/**
 * Initializes quick demo preset scenario buttons
 */
function initPresets() {
    Object.keys(DEMO_PRESETS).forEach(id => {
        const btn = document.getElementById(id);
        if (btn) {
            btn.addEventListener('click', () => {
                const p = DEMO_PRESETS[id];
                document.getElementById('input-appliance').value = p.appliance;
                document.getElementById('slider-hour').value = p.hour;
                document.getElementById('input-hour').value = p.hour;
                document.getElementById('input-day').value = p.day_of_week;
                document.getElementById('input-power').value = p.power;
                document.getElementById('input-baseline').value = p.baseline;

                // Trigger hour preview update
                const ampm = p.hour === 0 ? '12 AM' : p.hour === 12 ? '12 PM' : p.hour < 12 ? `${p.hour} AM` : `${p.hour - 12} PM`;
                document.getElementById('hour-time-preview').textContent = `${String(p.hour).padStart(2, '0')}:00 (${ampm})`;

                // Immediately trigger prediction for smooth viva demonstration
                triggerPrediction();
            });
        }
    });
}

/**
 * Refreshes summary metrics and data charts
 */
function initRefreshButton() {
    const btn = document.getElementById('btn-refresh-all');
    if (btn) {
        btn.addEventListener('click', async () => {
            btn.classList.add('loading');
            await Promise.all([
                loadDashboardSummary(),
                loadTimeseriesData(currentReadingsLimit),
                loadTableReadings()
            ]);
            setTimeout(() => btn.classList.remove('loading'), 400);
        });
    }
}

/**
 * Fetches and displays aggregated summary statistics from /api/summary
 */
async function loadDashboardSummary() {
    try {
        const response = await fetch('/api/summary');
        if (!response.ok) throw new Error(`HTTP error ${response.status}`);
        const data = await response.json();

        // Populate KPI cards
        document.getElementById('val-total-readings').textContent = Number(data.total_readings).toLocaleString();
        document.getElementById('val-anomalies').textContent = Number(data.anomaly_count).toLocaleString();
        document.getElementById('val-anomaly-count-pill').textContent = `${data.anomaly_count} flagged`;
        document.getElementById('val-anomaly-rate').textContent = `${data.anomaly_percentage}%`;
        document.getElementById('val-avg-consumption').textContent = `${Math.round(data.average_consumption_watts)} W`;
        document.getElementById('val-baseline-comparison').textContent = `vs ${Math.round(data.average_baseline_watts)} W expected baseline`;

        // Update sidebar F1 if available
        if (data.model_metrics && data.model_metrics.f1_score) {
            document.getElementById('badge-f1').textContent = data.model_metrics.f1_score.toFixed(2);
        }

        // Render Appliance breakdown chart
        if (data.by_appliance) {
            renderApplianceChart(data.by_appliance);
            renderApplianceStatsList(data.by_appliance);
        }

        document.getElementById('backend-status-text').textContent = 'FastAPI Engine Online';
    } catch (err) {
        console.error('Failed to load summary metrics:', err);
        document.getElementById('backend-status-text').textContent = 'Backend Reconnecting...';
    }
}

/**
 * Fetches recent readings and renders the interactive Time-Series Chart
 */
async function loadTimeseriesData(limit = 100) {
    try {
        const response = await fetch(`/api/readings?limit=${limit}&offset=0`);
        if (!response.ok) throw new Error(`HTTP error ${response.status}`);
        const data = await response.json();

        renderTimeseriesChart(data.readings);
    } catch (err) {
        console.error('Failed to load readings chart data:', err);
    }
}

/**
 * Builds or updates the Chart.js line chart for time-series readings
 */
function renderTimeseriesChart(readings) {
    const ctx = document.getElementById('timeseriesChart').getContext('2d');

    const labels = readings.map((r, idx) => {
        const dateStr = r.timestamp ? r.timestamp.split(' ')[0].slice(5) : `#${idx}`;
        return `${dateStr} ${String(r.hour).padStart(2, '0')}:00`;
    });

    const powerData = readings.map(r => r.power_consumption);
    const baselineData = readings.map(r => r.baseline_consumption);

    // Dynamic point styling: anomalous points highlighted bright red
    const pointRadii = readings.map(r => r.is_anomaly ? 6 : 0);
    const pointHoverRadii = readings.map(r => r.is_anomaly ? 9 : 4);
    const pointBackgroundColors = readings.map(r => r.is_anomaly ? '#f43f5e' : '#10b981');
    const pointBorderColors = readings.map(r => r.is_anomaly ? '#ffffff' : '#10b981');
    const pointBorderWidths = readings.map(r => r.is_anomaly ? 2 : 1);

    if (timeseriesChartInstance) {
        timeseriesChartInstance.destroy();
    }

    timeseriesChartInstance = new Chart(ctx, {
        type: 'line',
        data: {
            labels: labels,
            datasets: [
                {
                    label: 'Observed Power (Watts)',
                    data: powerData,
                    borderColor: '#10b981',
                    backgroundColor: 'rgba(16, 185, 129, 0.08)',
                    borderWidth: 2,
                    fill: true,
                    tension: 0.25,
                    pointRadius: pointRadii,
                    pointHoverRadius: pointHoverRadii,
                    pointBackgroundColor: pointBackgroundColors,
                    pointBorderColor: pointBorderColors,
                    pointBorderWidth: pointBorderWidths,
                },
                {
                    label: 'Baseline Nominal (Watts)',
                    data: baselineData,
                    borderColor: '#64748b',
                    borderWidth: 1.5,
                    borderDash: [5, 5],
                    fill: false,
                    tension: 0.1,
                    pointRadius: 0,
                    pointHoverRadius: 0,
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            interaction: {
                mode: 'index',
                intersect: false,
            },
            plugins: {
                legend: {
                    display: false,
                },
                tooltip: {
                    backgroundColor: 'rgba(15, 23, 42, 0.95)',
                    titleColor: '#f8fafc',
                    bodyColor: '#cbd5e1',
                    borderColor: 'rgba(255, 255, 255, 0.15)',
                    borderWidth: 1,
                    padding: 12,
                    callbacks: {
                        afterTitle: function(context) {
                            const idx = context[0].dataIndex;
                            const r = readings[idx];
                            return `${r.appliance} | Day: ${DAY_NAMES[r.day_of_week] || r.day_of_week}`;
                        },
                        afterBody: function(context) {
                            const idx = context[0].dataIndex;
                            const r = readings[idx];
                            if (r.is_anomaly) {
                                return `\n[ANOMALY FLAGGED]\nRatio: ${r.consumption_ratio}x baseline\nNote: ${r.anomaly_reason}`;
                            }
                            return `\nRatio: ${r.consumption_ratio}x (Normal)`;
                        }
                    }
                }
            },
            scales: {
                x: {
                    grid: { color: 'rgba(255, 255, 255, 0.04)' },
                    ticks: {
                        color: '#64748b',
                        maxTicksLimit: 12,
                        font: { size: 10 }
                    }
                },
                y: {
                    grid: { color: 'rgba(255, 255, 255, 0.06)' },
                    ticks: {
                        color: '#94a3b8',
                        font: { size: 11 },
                        callback: function(value) { return value + ' W'; }
                    }
                }
            }
        }
    });

    // Time filter pills
    const filterPills = document.querySelectorAll('.chart-controls .filter-pill');
    filterPills.forEach(pill => {
        pill.addEventListener('click', () => {
            filterPills.forEach(p => p.classList.remove('active'));
            pill.classList.add('active');
            const newLimit = parseInt(pill.getAttribute('data-filter'), 10);
            currentReadingsLimit = newLimit;
            loadTimeseriesData(newLimit);
        });
    });
}

/**
 * Builds or updates the Doughnut chart for anomalies by appliance
 */
function renderApplianceChart(applianceData) {
    const ctx = document.getElementById('applianceChart').getContext('2d');

    const labels = applianceData.map(a => a.appliance);
    const anomalies = applianceData.map(a => a.anomalies);

    const colors = [
        '#f43f5e', // Rose
        '#38bdf8', // Sky Blue
        '#f59e0b', // Amber
        '#10b981', // Emerald
        '#a855f7', // Purple
        '#fb923c'  // Orange
    ];

    if (applianceChartInstance) {
        applianceChartInstance.destroy();
    }

    applianceChartInstance = new Chart(ctx, {
        type: 'doughnut',
        data: {
            labels: labels,
            datasets: [{
                data: anomalies,
                backgroundColor: colors,
                borderColor: '#111a2e',
                borderWidth: 3,
                hoverOffset: 6
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            cutout: '68%',
            plugins: {
                legend: {
                    position: 'bottom',
                    labels: {
                        color: '#94a3b8',
                        font: { size: 10, family: 'Inter' },
                        boxWidth: 10,
                        padding: 10
                    }
                },
                tooltip: {
                    backgroundColor: 'rgba(15, 23, 42, 0.95)',
                    titleColor: '#f8fafc',
                    bodyColor: '#cbd5e1',
                    callbacks: {
                        label: function(context) {
                            const val = context.raw;
                            const total = anomalies.reduce((a, b) => a + b, 0);
                            const pct = total > 0 ? ((val / total) * 100).toFixed(1) : 0;
                            return ` ${context.label}: ${val} anomalies (${pct}%)`;
                        }
                    }
                }
            }
        }
    });
}

/**
 * Populates the side stat list for appliance breakdown
 */
function renderApplianceStatsList(applianceData) {
    const container = document.getElementById('appliance-breakdown-list');
    if (!container) return;

    container.innerHTML = '';
    applianceData.forEach(item => {
        const row = document.createElement('div');
        row.className = 'appliance-stat-row';
        row.innerHTML = `
            <span class="app-name">${item.appliance}</span>
            <span class="app-stat">${item.anomalies} anomalies (${item.anomaly_rate}%)</span>
        `;
        container.appendChild(row);
    });
}

/**
 * Initializes prediction form events and live inference submission
 */
function initPredictionForm() {
    const form = document.getElementById('prediction-form');
    if (form) {
        form.addEventListener('submit', (e) => {
            e.preventDefault();
            triggerPrediction();
        });
    }
}

/**
 * Executes POST /api/predict with the current form values
 */
async function triggerPrediction() {
    const btn = document.getElementById('btn-submit-predict');
    const btnText = document.getElementById('btn-text');
    const spinner = document.getElementById('predict-spinner');

    const appliance = document.getElementById('input-appliance').value;
    const hour = parseInt(document.getElementById('input-hour').value, 10);
    const dayOfWeek = parseInt(document.getElementById('input-day').value, 10);
    const power = parseFloat(document.getElementById('input-power').value);
    const baseline = parseFloat(document.getElementById('input-baseline').value);

    // Disable button & show spinner
    btn.disabled = true;
    spinner.classList.remove('hidden');
    btnText.textContent = 'Analyzing Feature Vectors...';

    const payload = {
        appliance: appliance,
        hour: hour,
        day_of_week: dayOfWeek,
        power_consumption: power,
        baseline_consumption: baseline
    };

    try {
        const response = await fetch('/api/predict', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload)
        });

        if (!response.ok) {
            const err = await response.json();
            throw new Error(err.detail || `Server returned ${response.status}`);
        }

        const result = await response.json();
        renderPredictionResult(result);
    } catch (err) {
        console.error('Prediction failed:', err);
        alert(`Prediction Error: ${err.message}`);
    } finally {
        btn.disabled = false;
        spinner.classList.add('hidden');
        btnText.textContent = 'Execute ML Anomaly Inference';
    }
}

/**
 * Updates result panel with animated meter, status badges, and explanation
 */
function renderPredictionResult(res) {
    const placeholder = document.getElementById('result-placeholder');
    const card = document.getElementById('result-card');

    placeholder.classList.add('hidden');
    card.classList.remove('hidden');

    const statusPill = document.getElementById('res-status-pill');
    const statusTitle = document.getElementById('res-status-title');
    const riskTag = document.getElementById('res-risk-tag');
    const scoreNum = document.getElementById('res-score-number');
    const meterBar = document.getElementById('res-meter-bar');

    // Values
    document.getElementById('res-power-val').textContent = `${res.features.power_consumption_watts.toLocaleString()} W`;
    document.getElementById('res-baseline-val').textContent = `${res.features.baseline_consumption_watts.toLocaleString()} W`;
    document.getElementById('res-ratio-val').textContent = `${res.features.consumption_ratio}x`;
    const diff = res.features.consumption_diff_watts;
    document.getElementById('res-diff-val').textContent = `${diff >= 0 ? '+' : ''}${diff.toLocaleString()} W`;

    // Status styling
    if (res.is_anomaly) {
        statusPill.className = 'status-pill status-anomaly';
        statusTitle.textContent = 'ANOMALY DETECTED';
        riskTag.className = 'risk-tag risk-high';
        riskTag.textContent = `${res.risk_level.toUpperCase()} RISK`;
    } else {
        statusPill.className = 'status-pill status-normal';
        statusTitle.textContent = 'NORMAL PROFILE';
        riskTag.className = 'risk-tag risk-low';
        riskTag.textContent = `${res.risk_level.toUpperCase()} RISK`;
    }

    // Meter bar color and width
    const scorePct = res.anomaly_score_percentage;
    scoreNum.textContent = `${scorePct}%`;
    meterBar.style.width = `${Math.min(100, Math.max(2, scorePct))}%`;

    if (scorePct > 55) {
        meterBar.style.backgroundColor = '#f43f5e'; // Rose
    } else if (scorePct > 35) {
        meterBar.style.backgroundColor = '#f59e0b'; // Amber
    } else {
        meterBar.style.backgroundColor = '#10b981'; // Emerald
    }

    // Diagnostic text
    const diagBox = document.querySelector('.diagnostic-box');
    if (res.is_anomaly) {
        diagBox.classList.add('anomaly-diag');
    } else {
        diagBox.classList.remove('anomaly-diag');
    }

    document.getElementById('res-explanation-text').textContent = res.explanation;
    document.getElementById('res-disclaimer').textContent = res.disclaimer;
}

/**
 * Initializes telemetry table filters, search, and pagination
 */
function initTableControls() {
    const btnAll = document.getElementById('btn-filter-all');
    const btnAnomalies = document.getElementById('btn-filter-anomalies');
    const btnNormal = document.getElementById('btn-filter-normal');
    const searchInput = document.getElementById('table-search');
    const btnPrev = document.getElementById('btn-page-prev');
    const btnNext = document.getElementById('btn-page-next');

    btnAll.addEventListener('click', () => {
        setActiveFilterPill(btnAll);
        currentFilterMode = 'all';
        currentTableOffset = 0;
        loadTableReadings();
    });

    btnAnomalies.addEventListener('click', () => {
        setActiveFilterPill(btnAnomalies);
        currentFilterMode = 'anomalies';
        currentTableOffset = 0;
        loadTableReadings();
    });

    btnNormal.addEventListener('click', () => {
        setActiveFilterPill(btnNormal);
        currentFilterMode = 'normal';
        currentTableOffset = 0;
        loadTableReadings();
    });

    let searchDebounce;
    searchInput.addEventListener('input', (e) => {
        clearTimeout(searchDebounce);
        searchDebounce = setTimeout(() => {
            currentSearchQuery = e.target.value.trim();
            currentTableOffset = 0;
            loadTableReadings();
        }, 300);
    });

    btnPrev.addEventListener('click', () => {
        if (currentTableOffset > 0) {
            currentTableOffset = Math.max(0, currentTableOffset - currentTableLimit);
            loadTableReadings();
        }
    });

    btnNext.addEventListener('click', () => {
        if (currentTableOffset + currentTableLimit < totalFilteredRecords) {
            currentTableOffset += currentTableLimit;
            loadTableReadings();
        }
    });
}

function setActiveFilterPill(activeBtn) {
    document.querySelectorAll('.table-filter-pills .t-pill').forEach(btn => btn.classList.remove('active'));
    activeBtn.classList.add('active');
}

/**
 * Loads telemetry table records from /api/readings with pagination and filters
 */
async function loadTableReadings() {
    const tbody = document.getElementById('readings-table-body');
    const paginationInfo = document.getElementById('pagination-info');
    const btnPrev = document.getElementById('btn-page-prev');
    const btnNext = document.getElementById('btn-page-next');

    let url = `/api/readings?limit=${currentTableLimit}&offset=${currentTableOffset}`;

    if (currentFilterMode === 'anomalies') {
        url += '&anomaly_only=true';
    }

    if (currentSearchQuery) {
        url += `&appliance=${encodeURIComponent(currentSearchQuery)}`;
    }

    try {
        const response = await fetch(url);
        if (!response.ok) throw new Error(`HTTP error ${response.status}`);
        const data = await response.json();

        totalFilteredRecords = data.total;
        let readings = data.readings;

        // In-memory filter for 'normal' if needed
        if (currentFilterMode === 'normal') {
            readings = readings.filter(r => !r.is_anomaly);
        }

        tbody.innerHTML = '';

        if (readings.length === 0) {
            tbody.innerHTML = `<tr><td colspan="8" class="loading-cell" style="text-align: center; color: var(--text-muted);">No readings matching filter.</td></tr>`;
            paginationInfo.textContent = 'Showing 0 records';
            btnPrev.disabled = true;
            btnNext.disabled = true;
            return;
        }

        readings.forEach(r => {
            const tr = document.createElement('tr');
            const dayName = DAY_NAMES[r.day_of_week] || `Day ${r.day_of_week}`;
            const badgeClass = r.is_anomaly ? 'badge-anomaly' : 'badge-normal';
            const badgeText = r.is_anomaly ? 'ANOMALOUS' : 'NORMAL';

            tr.innerHTML = `
                <td><strong>${r.timestamp}</strong></td>
                <td>${String(r.hour).padStart(2, '0')}:00 (${dayName})</td>
                <td>${r.appliance}</td>
                <td><strong>${r.power_consumption.toLocaleString()} W</strong></td>
                <td>${r.baseline_consumption.toLocaleString()} W</td>
                <td>${r.consumption_ratio}x</td>
                <td><span class="badge-status ${badgeClass}">${badgeText}</span></td>
                <td style="font-size: 0.76rem; max-width: 260px; color: ${r.is_anomaly ? '#fb7185' : 'var(--text-muted)'};">${r.anomaly_reason}</td>
            `;
            tbody.appendChild(tr);
        });

        // Pagination controls
        const startRecord = currentTableOffset + 1;
        const endRecord = Math.min(totalFilteredRecords, currentTableOffset + currentTableLimit);
        paginationInfo.textContent = `Showing records ${startRecord} - ${endRecord} of ${totalFilteredRecords.toLocaleString()}`;

        btnPrev.disabled = currentTableOffset === 0;
        btnNext.disabled = currentTableOffset + currentTableLimit >= totalFilteredRecords;

    } catch (err) {
        console.error('Failed to load table readings:', err);
        tbody.innerHTML = `<tr><td colspan="8" class="loading-cell" style="color: var(--accent-rose); text-align: center;">Error connecting to API. Please ensure FastAPI server is running.</td></tr>`;
    }
}
