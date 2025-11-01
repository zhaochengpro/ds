const DASHBOARD_STREAM_PATH = '/ws/dashboard';
const STREAM_RECONNECT_BASE_MS = 1500;
const STREAM_RECONNECT_MAX_MS = 15000;
const STREAM_QUEUE_LIMIT = 32;
const EQUITY_RANGES = ['day', 'week', 'month', 'year'];
const EQUITY_DEFAULT_RANGE = 'day';
const EQUITY_COLORS = {
  day: {
    line: 'rgba(167, 130, 255, 0.9)',
    fill: 'rgba(167, 130, 255, 0.22)',
    solid: '#a782ff',
  },
  week: {
    line: 'rgba(40, 199, 111, 0.9)',
    fill: 'rgba(40, 199, 111, 0.18)',
    solid: '#28c76f',
  },
  month: {
    line: 'rgba(64, 156, 255, 0.9)',
    fill: 'rgba(64, 156, 255, 0.2)',
    solid: '#409cff',
  },
  year: {
    line: 'rgba(255, 159, 67, 0.9)',
    fill: 'rgba(255, 159, 67, 0.22)',
    solid: '#ff9f43',
  },
};

const PLACEHOLDER_EQUITY_BASELINE = 10000;
const PLACEHOLDER_RANGE_CONFIG = {
  day: { segments: 6, stepMs: 4 * 60 * 60 * 1000 },
  week: { segments: 7, stepMs: 24 * 60 * 60 * 1000 },
  month: { segments: 6, stepMs: 5 * 24 * 60 * 60 * 1000 },
  year: { segments: 6, stepMs: 60 * 24 * 60 * 60 * 1000 },
};

function adjustAlpha(color, alpha) {
  if (!color || typeof color !== 'string') {
    return color;
  }
  if (color.startsWith('rgba')) {
    return color.replace(/rgba\(([^,]+),([^,]+),([^,]+),[^)]+\)/, (_, r, g, b) => `rgba(${r.trim()}, ${g.trim()}, ${b.trim()}, ${alpha})`);
  }
  if (color.startsWith('rgb')) {
    return color.replace(/rgb\(([^)]+)\)/, (_, channels) => `rgba(${channels}, ${alpha})`);
  }
  return color;
}

const CONFIDENCE_KEYWORDS = {
  HIGH: 82,
  MEDIUM: 55,
  LOW: 28,
};

const currencyFormatter = new Intl.NumberFormat('zh-CN', {
  style: 'currency',
  currency: 'USD',
  maximumFractionDigits: 2,
});

const numberFormatter = new Intl.NumberFormat('zh-CN', {
  maximumFractionDigits: 4,
});

let previousState = null;
let previousAccountSnapshot = null;
let previousStrategySummary = null;
let previousRuntimeSummary = null;
let equitySeries = {
  day: [],
  week: [],
  month: [],
  year: [],
};
let currentEquityRange = EQUITY_DEFAULT_RANGE;
let equityChart = null;
let equityTooltip = null;
let latestDataTimestamp = null;
let equityChartState = {
  points: [],
  palette: getEquityColors('day'),
  placeholder: true,
  yMin: null,
  yMax: null,
};
let dashboardSocket = null;
let streamMessageQueue = [];
let streamReconnectTimer = null;
let streamReconnectAttempt = 0;
let streamReceivedSnapshot = false;

function toNumber(value) {
  const parsed = Number(value);
  return Number.isFinite(parsed) ? parsed : null;
}

function normalizeSymbol(symbol) {
  return (symbol || '').toUpperCase();
}

function normalizeAction(action) {
  if (!action && action !== 0) {
    return '';
  }
  return String(action).trim().toUpperCase();
}

function formatCurrency(value) {
  const num = toNumber(value);
  return num === null ? '--' : currencyFormatter.format(num);
}

function formatNumber(value, fraction = 4) {
  const num = toNumber(value);
  if (num === null) {
    return '--';
  }
  return Number(num).toFixed(fraction).replace(/\.0+$/, '');
}

function formatPercent(value) {
  const num = toNumber(value);
  if (num === null) {
    return '--';
  }
  return `${num.toFixed(2)}%`;
}

function formatDuration(seconds) {
  const total = toNumber(seconds);
  if (total === null || total < 0) {
    return '--';
  }
  const hours = Math.floor(total / 3600);
  const minutes = Math.floor((total % 3600) / 60);
  const secs = Math.floor(total % 60);
  if (hours > 0) {
    return `${hours}小时${minutes.toString().padStart(2, '0')}分`;
  }
  if (minutes > 0) {
    return `${minutes}分${secs.toString().padStart(2, '0')}秒`;
  }
  return `${secs}秒`;
}

function formatTimestamp(value) {
  if (!value) {
    return '--';
  }
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) {
    return value;
  }
  return date.toLocaleString('zh-CN', {
    hour12: false,
  });
}

function toDate(value) {
  if (!value) {
    return null;
  }
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) {
    return null;
  }
  return date;
}

function updateLastUpdatedDisplay(date) {
  if (!date) {
    return;
  }
  const updatedEl = document.getElementById('last-updated');
  if (updatedEl) {
    updatedEl.textContent = formatTimestamp(date.toISOString());
  }
}

function resetLastUpdatedFromSources(timestamps) {
  if (!Array.isArray(timestamps) || timestamps.length === 0) {
    return;
  }
  const latest = timestamps
    .map(toDate)
    .filter(Boolean)
    .sort((a, b) => b.getTime() - a.getTime())[0];
  if (!latest) {
    return;
  }
  latestDataTimestamp = latest;
  updateLastUpdatedDisplay(latest);
}

function updateRuntimeSummary(summary) {
  const uptimeEl = document.getElementById('runtime-uptime');
  const iterationsEl = document.getElementById('runtime-iterations');
  if (!uptimeEl || !iterationsEl) {
    return;
  }
  if (!summary || typeof summary !== 'object') {
    uptimeEl.textContent = '--';
    iterationsEl.textContent = '--';
    uptimeEl.removeAttribute('title');
    iterationsEl.removeAttribute('title');
    return;
  }
  const uptimeSeconds = toNumber(summary.uptime_seconds);
  uptimeEl.textContent = formatDuration(uptimeSeconds);
  if (summary.started_at) {
    uptimeEl.title = `启动时间：${formatTimestamp(summary.started_at)}`;
  } else {
    uptimeEl.removeAttribute('title');
  }
  const iterations = toNumber(summary.total_iterations);
  iterationsEl.textContent = Number.isFinite(iterations) ? iterations : '--';
  if (Array.isArray(summary.symbols) && summary.symbols.length > 0) {
    iterationsEl.title = `交易币种：${summary.symbols.join(', ')}`;
  } else {
    iterationsEl.removeAttribute('title');
  }
  previousRuntimeSummary = summary;
}

function applyLastUpdated(timestamp) {
  const parsed = toDate(timestamp);
  if (!parsed) {
    return;
  }
  if (!latestDataTimestamp || parsed.getTime() > latestDataTimestamp.getTime()) {
    latestDataTimestamp = parsed;
    updateLastUpdatedDisplay(parsed);
  }
}

function truncateText(value, maxLength = 80) {
  if (!value) {
    return '--';
  }
  const text = String(value).trim();
  if (text.length <= maxLength) {
    return text;
  }
  return `${text.slice(0, maxLength - 1)}…`;
}

function triggerFlash(element, delta) {
  if (!element || !Number.isFinite(delta) || delta === 0) {
    return;
  }
  element.classList.remove('flash-up', 'flash-down');
  // 强制重绘以重新触发动画
  void element.offsetWidth;
  const cls = delta > 0 ? 'flash-up' : 'flash-down';
  element.classList.add(cls);
  window.setTimeout(() => {
    element.classList.remove(cls);
  }, 700);
}

function flashNumericChange(element, newValue, oldValue) {
  if (!element) {
    return;
  }
  const next = toNumber(newValue);
  const prev = toNumber(oldValue);
  if (next === null || prev === null || next === prev) {
    return;
  }
  triggerFlash(element, next - prev);
}

function cloneState(state) {
  if (!state) {
    return null;
  }
  if (typeof structuredClone === 'function') {
    try {
      return structuredClone(state);
    } catch (error) {
      // fall back to JSON
    }
  }
  try {
    return JSON.parse(JSON.stringify(state));
  } catch (error) {
    return null;
  }
}

function formatChartLabel(timestamp, range) {
  const date = new Date(timestamp);
  if (Number.isNaN(date.getTime())) {
    return timestamp || '--';
  }
  const month = String(date.getMonth() + 1).padStart(2, '0');
  const day = String(date.getDate()).padStart(2, '0');
  const hours = String(date.getHours()).padStart(2, '0');
  const minutes = String(date.getMinutes()).padStart(2, '0');
  if (range === 'day') {
    return `${hours}:${minutes}`;
  }
  if (range === 'week') {
    return `${month}/${day} ${hours}:${minutes}`;
  }
  if (range === 'month') {
    return `${month}/${day}`;
  }
  return `${date.getFullYear()}/${month}`;
}

function getEquityColors(range) {
  return EQUITY_COLORS[range] || EQUITY_COLORS.week;
}

function ensureEquityChart() {
  if (equityChart) {
    return equityChart;
  }
  const canvas = document.getElementById('equity-chart');
  if (!canvas) {
    return null;
  }
  const context = canvas.getContext('2d');
  if (!context) {
    return null;
  }
  const container = canvas.parentElement;
  if (container && !equityTooltip) {
    equityTooltip = document.createElement('div');
    equityTooltip.className = 'chart-tooltip';
    container.appendChild(equityTooltip);
  }
  equityChart = {
    canvas,
    context,
    hoverIndex: null,
    pointerActive: false,
    mappedPoints: [],
  };
  canvas.addEventListener('mousemove', handleEquityPointerMove);
  canvas.addEventListener('mouseleave', handleEquityPointerLeave);
  window.addEventListener('resize', () => {
    drawEquityChart();
  });
  return equityChart;
}

function hideEquityTooltip() {
  if (equityTooltip) {
    equityTooltip.classList.remove('is-visible');
  }
}

function updateEquityTooltip(targetPoint, chart) {
  if (!targetPoint || !chart) {
    hideEquityTooltip();
    return;
  }
  const { canvas } = chart;
  const parent = canvas.parentElement;
  if (!parent) {
    hideEquityTooltip();
    return;
  }
  if (!equityTooltip) {
    equityTooltip = document.createElement('div');
    equityTooltip.className = 'chart-tooltip';
    parent.appendChild(equityTooltip);
  }
  const tooltip = equityTooltip;
  const timestampLabel = formatTimestamp(targetPoint.original.timestamp);
  const valueLabel = formatCurrency(targetPoint.original.value).replace('US$', '$');
  tooltip.innerHTML = `
    <div class="chart-tooltip-time">${timestampLabel}</div>
    <div class="chart-tooltip-value">${valueLabel}</div>
  `;
  tooltip.classList.add('is-visible');
  const parentRect = parent.getBoundingClientRect();
  const canvasRect = canvas.getBoundingClientRect();
  const anchorLeft = canvasRect.left + targetPoint.x - parentRect.left;
  const anchorTop = canvasRect.top + targetPoint.y - parentRect.top;
  tooltip.style.left = `${anchorLeft}px`;
  tooltip.style.top = `${anchorTop}px`;
}

function handleEquityPointerMove(event) {
  const chart = ensureEquityChart();
  if (!chart) {
    return;
  }
  const { canvas, mappedPoints } = chart;
  if (equityChartState.placeholder || !Array.isArray(mappedPoints) || mappedPoints.length === 0) {
    hideEquityTooltip();
    chart.hoverIndex = null;
    chart.pointerActive = false;
    return;
  }
  chart.pointerActive = true;
  const rect = canvas.getBoundingClientRect();
  const cursorX = event.clientX - rect.left;
  let nearestIndex = 0;
  let minDistance = Number.POSITIVE_INFINITY;
  mappedPoints.forEach((point, index) => {
    const distance = Math.abs(point.x - cursorX);
    if (distance < minDistance) {
      minDistance = distance;
      nearestIndex = index;
    }
  });
  if (chart.hoverIndex !== nearestIndex) {
    chart.hoverIndex = nearestIndex;
    drawEquityChart();
  }
  const refreshedPoints = chart.mappedPoints || mappedPoints;
  const targetPoint = refreshedPoints[nearestIndex];
  if (targetPoint) {
    updateEquityTooltip(targetPoint, chart);
  } else {
    hideEquityTooltip();
  }
}

function handleEquityPointerLeave() {
  const chart = equityChart;
  if (!chart) {
    hideEquityTooltip();
    return;
  }
  chart.pointerActive = false;
  if (chart.hoverIndex !== null) {
    chart.hoverIndex = null;
    drawEquityChart();
  }
  hideEquityTooltip();
}

function resolveWebSocketUrl(path) {
  const normalized = path.startsWith('/') ? path : `/${path}`;
  const protocol = window.location.protocol === 'https:' ? 'wss' : 'ws';
  return `${protocol}://${window.location.host}${normalized}`;
}

function flushStreamQueue() {
  if (!dashboardSocket || dashboardSocket.readyState !== WebSocket.OPEN) {
    return;
  }
  while (streamMessageQueue.length > 0) {
    const message = streamMessageQueue.shift();
    try {
      dashboardSocket.send(message);
    } catch (error) {
      console.error('发送排队消息失败', error);
      break;
    }
  }
}

function sendStreamMessage(message) {
  if (!message || typeof message !== 'object') {
    return;
  }
  let serialized;
  try {
    serialized = JSON.stringify(message);
  } catch (error) {
    console.error('序列化推送消息失败', error, message);
    return;
  }
  if (dashboardSocket && dashboardSocket.readyState === WebSocket.OPEN) {
    try {
      dashboardSocket.send(serialized);
    } catch (error) {
      console.error('发送推送消息失败', error);
    }
    return;
  }
  if (streamMessageQueue.length >= STREAM_QUEUE_LIMIT) {
    streamMessageQueue.shift();
  }
  streamMessageQueue.push(serialized);
}

function handleEquityStreamPayload(payload) {
  if (!payload || typeof payload !== 'object') {
    return;
  }

  if (payload.timeframes && typeof payload.timeframes === 'object') {
    Object.entries(payload.timeframes).forEach(([range, points]) => {
      setEquitySeries(range, Array.isArray(points) ? points : []);
    });
  }

  if (Array.isArray(payload.points)) {
    const targetRange = EQUITY_RANGES.includes(payload.range) ? payload.range : currentEquityRange;
    setEquitySeries(targetRange, payload.points);
    if (targetRange === currentEquityRange) {
      renderEquityChart();
    }
  } else if (payload.timeframes && payload.timeframes[currentEquityRange]) {
    renderEquityChart();
  }

  if (payload.latest_timestamp) {
    applyLastUpdated(payload.latest_timestamp);
  }
}

function handleDashboardStreamMessage(message) {
  if (!message || typeof message !== 'object') {
    return;
  }
  const { type } = message;
  if (!type || typeof type !== 'string') {
    return;
  }
  const normalized = type.toLowerCase();

  if (normalized === 'state') {
    const payload = message.payload && typeof message.payload === 'object' ? message.payload : {};
    if (payload.equity && typeof payload.equity === 'object') {
      Object.entries(payload.equity).forEach(([range, points]) => {
        setEquitySeries(range, Array.isArray(points) ? points : []);
      });
      if (payload.equity[currentEquityRange]) {
        renderEquityChart();
      }
    }
    renderState(payload);
    streamReceivedSnapshot = true;
    setStatus('online');
    return;
  }

  switch (normalized) {
    case 'account':
      handleAccountUpdate(message.payload || {});
      break;
    case 'equity':
      handleEquityStreamPayload(message);
      break;
    case 'runtime':
      updateRuntimeSummary(message.payload || {});
      break;
    case 'heartbeat':
    case 'pong':
      break;
    default:
      break;
  }
}

function scheduleStreamReconnect() {
  if (streamReconnectTimer) {
    window.clearTimeout(streamReconnectTimer);
  }
  const attempt = streamReconnectAttempt + 1;
  streamReconnectAttempt = attempt;
  const delay = Math.min(
    STREAM_RECONNECT_MAX_MS,
    STREAM_RECONNECT_BASE_MS * 2 ** (attempt - 1),
  );
  streamReconnectTimer = window.setTimeout(() => {
    streamReconnectTimer = null;
    connectDashboardStream();
  }, delay);
}

function connectDashboardStream() {
  if (
    dashboardSocket
    && (dashboardSocket.readyState === WebSocket.OPEN || dashboardSocket.readyState === WebSocket.CONNECTING)
  ) {
    return;
  }

  if (streamReconnectTimer) {
    window.clearTimeout(streamReconnectTimer);
    streamReconnectTimer = null;
  }

  const url = resolveWebSocketUrl(DASHBOARD_STREAM_PATH);
  const socket = new WebSocket(url);
  dashboardSocket = socket;
  streamReceivedSnapshot = false;
  setStatus('pending');

  socket.addEventListener('open', () => {
    streamReconnectAttempt = 0;
    flushStreamQueue();
    sendStreamMessage({ type: 'request_snapshot' });
    sendStreamMessage({ type: 'set_equity_range', range: currentEquityRange, include_timeframes: true });
  });

  socket.addEventListener('message', (event) => {
    try {
      const data = JSON.parse(event.data);
      handleDashboardStreamMessage(data);
    } catch (error) {
      console.error('解析推送数据失败', error);
    }
  });

  socket.addEventListener('close', () => {
    if (dashboardSocket === socket) {
      dashboardSocket = null;
    }
    setStatus('offline');
    streamReceivedSnapshot = false;
    hideEquityTooltip();
    scheduleStreamReconnect();
  });

  socket.addEventListener('error', () => {
    socket.close();
  });
}

function prepareEquityCanvas(chart) {
  const { canvas, context } = chart;
  const parent = canvas.parentElement;
  if (!parent) {
    return { width: 0, height: 0, context };
  }
  const width = parent.clientWidth || canvas.clientWidth || 0;
  const height = parent.clientHeight || canvas.clientHeight || 0;
  const dpr = window.devicePixelRatio || 1;
  const scaledWidth = Math.max(1, Math.floor(width * dpr));
  const scaledHeight = Math.max(1, Math.floor(height * dpr));
  if (canvas.width !== scaledWidth || canvas.height !== scaledHeight) {
    canvas.width = scaledWidth;
    canvas.height = scaledHeight;
  }
  context.setTransform(dpr, 0, 0, dpr, 0, 0);
  context.clearRect(0, 0, Math.max(width, 1), Math.max(height, 1));
  return { width: Math.max(width, 1), height: Math.max(height, 1), context };
}

function drawEquityChart() {
  const chart = ensureEquityChart();
  if (!chart) {
    return;
  }
  const { width, height, context } = prepareEquityCanvas(chart);
  if (width <= 0 || height <= 0) {
    return;
  }

  const { points, palette, placeholder, yMin, yMax } = equityChartState;
  if (!points || points.length === 0) {
    chart.mappedPoints = [];
    chart.hoverIndex = null;
    return;
  }

  const padding = {
    top: 18,
    right: 32,
    bottom: 36,
    left: 72,
  };

  const plotWidth = Math.max(width - padding.left - padding.right, 8);
  const plotHeight = Math.max(height - padding.top - padding.bottom, 8);

  const minTime = points[0].timestampMs;
  const maxTime = points[points.length - 1].timestampMs;
  const horizontalSpan = Math.max(maxTime - minTime, 1);

  const valueList = points.map((p) => p.value);
  const computedMax = valueList.length > 0 ? Math.max(...valueList) : 1;

  let minValue = Number.isFinite(yMin) ? yMin : 0;
  let maxValue = Number.isFinite(yMax) ? yMax : computedMax;

  if (!Number.isFinite(minValue)) {
    minValue = 0;
  }
  if (!Number.isFinite(maxValue)) {
    maxValue = 1;
  }

  if (!Number.isFinite(yMin)) {
    minValue = 0;
  }

  if (!Number.isFinite(yMax)) {
    maxValue = Math.max(computedMax, minValue + 1);
  }

  if (maxValue <= minValue) {
    maxValue = minValue + 1;
  }

  const verticalSpan = Math.max(maxValue - minValue, 1e-6);

  const mapX = (timestamp) => {
    if (horizontalSpan === 0) {
      return padding.left + plotWidth / 2;
    }
    return (
      padding.left + ((timestamp - minTime) / horizontalSpan) * plotWidth
    );
  };

  const mapY = (value) => {
    if (verticalSpan === 0) {
      return padding.top + plotHeight / 2;
    }
    return (
      padding.top + (1 - (value - minValue) / verticalSpan) * plotHeight
    );
  };

  const mappedPoints = points.map((point) => ({
    x: mapX(point.timestampMs),
    y: mapY(point.value),
    original: point,
  }));
  chart.mappedPoints = mappedPoints;
  chart.placeholder = placeholder;
  if (placeholder) {
    chart.hoverIndex = null;
    hideEquityTooltip();
  }

  context.lineJoin = 'round';
  context.lineCap = 'round';

  context.lineWidth = 1;
  context.strokeStyle = 'rgba(255, 255, 255, 0.08)';
  const horizontalLines = 4;
  for (let i = 0; i <= horizontalLines; i += 1) {
    const y = padding.top + (plotHeight / horizontalLines) * i;
    context.beginPath();
    context.moveTo(padding.left, y);
    context.lineTo(width - padding.right, y);
    context.stroke();
  }

  context.strokeStyle = 'rgba(255, 255, 255, 0.16)';
  context.beginPath();
  context.moveTo(padding.left, padding.top);
  context.lineTo(padding.left, padding.top + plotHeight);
  context.lineTo(width - padding.right, padding.top + plotHeight);
  context.stroke();

  context.fillStyle = 'rgba(255, 255, 255, 0.6)';
  context.font = '12px/1.4 "Inter", "PingFang SC", "Helvetica Neue", Arial, sans-serif';
  context.textBaseline = 'middle';
  const yTicks = 4;
  for (let i = 0; i <= yTicks; i += 1) {
    const value = minValue + (verticalSpan * i) / yTicks;
    const y = padding.top + plotHeight - (plotHeight * i) / yTicks;
    const label = formatCurrency(value).replace('US$', '$');
    context.fillText(label, 12, y);
  }

  const xTickCount = Math.min(points.length - 1, 4);
  context.textBaseline = 'alphabetic';
  for (let i = 0; i <= xTickCount; i += 1) {
    const ratio = xTickCount === 0 ? 0 : i / xTickCount;
    const index = Math.min(
      points.length - 1,
      Math.round(ratio * (points.length - 1)),
    );
    const point = points[index];
    const x = mapX(point.timestampMs);
    const label = formatChartLabel(point.timestampMs, currentEquityRange);
    const textWidth = context.measureText(label).width;
    context.fillText(label, x - textWidth / 2, height - 8);
  }

  context.beginPath();
  mappedPoints.forEach((point, index) => {
    if (index === 0) {
      context.moveTo(point.x, point.y);
    } else {
      context.lineTo(point.x, point.y);
    }
  });
  context.lineTo(mappedPoints[mappedPoints.length - 1].x, padding.top + plotHeight);
  context.lineTo(mappedPoints[0].x, padding.top + plotHeight);
  context.closePath();
  const gradient = context.createLinearGradient(0, padding.top, 0, padding.top + plotHeight);
  const baseFill = palette.fill || 'rgba(64, 156, 255, 0.2)';
  if (placeholder) {
    gradient.addColorStop(0, adjustAlpha(baseFill, 0.2));
    gradient.addColorStop(1, adjustAlpha(baseFill, 0.05));
  } else {
    gradient.addColorStop(0, baseFill);
    gradient.addColorStop(1, 'rgba(0, 0, 0, 0)');
  }
  context.fillStyle = gradient;
  context.fill();

  context.beginPath();
  mappedPoints.forEach((point, index) => {
    if (index === 0) {
      context.moveTo(point.x, point.y);
    } else {
      context.lineTo(point.x, point.y);
    }
  });
  const baseLine = palette.line || 'rgba(64, 156, 255, 0.85)';
  context.strokeStyle = placeholder ? adjustAlpha(baseLine, 0.6) : baseLine;
  context.lineWidth = 2;
  context.stroke();

  const hoverIndex = !placeholder && Number.isInteger(chart.hoverIndex)
    ? Math.min(Math.max(chart.hoverIndex, 0), mappedPoints.length - 1)
    : null;
  if (hoverIndex !== chart.hoverIndex) {
    chart.hoverIndex = hoverIndex;
  }
  const hoverPoint = hoverIndex !== null ? mappedPoints[hoverIndex] : null;

  if (hoverPoint) {
    context.save();
    context.setLineDash([4, 4]);
    context.strokeStyle = adjustAlpha(palette.line || '#ffffff', 0.45);
    context.lineWidth = 1;
    context.beginPath();
    context.moveTo(hoverPoint.x, padding.top);
    context.lineTo(hoverPoint.x, padding.top + plotHeight);
    context.stroke();
    context.restore();

    context.save();
    context.fillStyle = '#0b101a';
    context.strokeStyle = palette.solid || '#ffffff';
    context.lineWidth = 2;
    context.beginPath();
    context.arc(hoverPoint.x, hoverPoint.y, 4, 0, Math.PI * 2);
    context.fill();
    context.stroke();
    context.restore();
  } else if (placeholder) {
    hideEquityTooltip();
  }
}

function resolveBaselineValue() {
  const candidates = [
    previousAccountSnapshot?.account_value,
    previousState?.account?.account_value,
  ];

  for (const candidate of candidates) {
    const numeric = toNumber(candidate);
    if (Number.isFinite(numeric) && numeric > 0) {
      return numeric;
    }
  }

  return PLACEHOLDER_EQUITY_BASELINE;
}

function buildPlaceholderSeries(range) {
  const baseline = Math.max(resolveBaselineValue(), 1);
  const config = PLACEHOLDER_RANGE_CONFIG[range] || PLACEHOLDER_RANGE_CONFIG.day;
  const now = Date.now();
  const points = [];

  for (let i = config.segments; i >= 0; i -= 1) {
    const timestamp = now - config.stepMs * i;
    points.push({
      timestamp: new Date(timestamp).toISOString(),
      timestampMs: timestamp,
      value: baseline,
    });
  }

  const padding = Math.max(baseline * 0.15, baseline > 1 ? baseline * 0.1 : 1);
  const min = Math.max(baseline - padding, 0);
  const max = baseline + padding;

  return {
    points,
    suggestedMin: min,
    suggestedMax: max > min ? max : min + 1,
  };
}

function renderEquityChart() {
  const activeData = equitySeries[currentEquityRange] || [];
  const emptyEl = document.getElementById('equity-empty');
  const palette = getEquityColors(currentEquityRange);

  if (!activeData || activeData.length === 0) {
    if (emptyEl) {
      emptyEl.style.display = 'flex';
    }
    const placeholder = buildPlaceholderSeries(currentEquityRange);
    equityChartState = {
      points: placeholder.points.map((point) => ({
        timestamp: point.timestamp,
        timestampMs: point.timestampMs ?? point.timestamp,
        value: point.value,
      })),
      palette,
      placeholder: true,
      yMin: placeholder.suggestedMin,
      yMax: placeholder.suggestedMax,
    };
    if (equityChart) {
      equityChart.hoverIndex = null;
      equityChart.pointerActive = false;
    }
    hideEquityTooltip();
    drawEquityChart();
    return;
  }

  if (emptyEl) {
    emptyEl.style.display = 'none';
  }

  const normalizedPoints = activeData
    .map((point) => {
      if (!point || !point.timestamp) {
        return null;
      }
      const timestampMs = Number.isFinite(point.timestampMs)
        ? point.timestampMs
        : toDate(point.timestamp)?.getTime();
      if (!Number.isFinite(timestampMs)) {
        return null;
      }
      const numericValue = toNumber(point.value);
      if (!Number.isFinite(numericValue)) {
        return null;
      }
      return {
        timestamp: point.timestamp,
        timestampMs,
        value: numericValue,
      };
    })
    .filter(Boolean)
    .sort((a, b) => a.timestampMs - b.timestampMs);

  const values = normalizedPoints.map((point) => point.value);
  const minValue = values.length > 0 ? Math.min(...values) : null;
  const suggestedMin = Number.isFinite(minValue) ? minValue - 1 : 0;

  equityChartState = {
    points: normalizedPoints,
    palette,
    placeholder: false,
    yMin: suggestedMin,
    yMax: null,
  };
  drawEquityChart();

  if (
    equityChart
    && equityChart.pointerActive
    && Number.isInteger(equityChart.hoverIndex)
    && Array.isArray(equityChart.mappedPoints)
    && equityChart.mappedPoints[equityChart.hoverIndex]
  ) {
    updateEquityTooltip(
      equityChart.mappedPoints[equityChart.hoverIndex],
      equityChart,
    );
  } else {
    hideEquityTooltip();
  }
}

function applyRangeTheme(range) {
  const palette = getEquityColors(range);
  document.documentElement.style.setProperty('--equity-accent', palette.solid);
  equityChartState = {
    ...equityChartState,
    palette,
  };
  drawEquityChart();
}

function setActiveRange(range, options = {}) {
  const normalized = EQUITY_RANGES.includes(range) ? range : EQUITY_DEFAULT_RANGE;
  const previousRange = currentEquityRange;
  currentEquityRange = normalized;
  document.querySelectorAll('.range-button').forEach((button) => {
    button.classList.toggle('is-active', button.dataset.range === normalized);
  });
  applyRangeTheme(normalized);
  const hasData = Array.isArray(equitySeries[normalized]) && equitySeries[normalized].length > 0;
  renderEquityChart();
  const shouldForce = options.force === true || !hasData;
  const messageType = normalized !== previousRange ? 'set_equity_range' : 'request_equity';
  sendStreamMessage({
    type: messageType,
    range: normalized,
    include_timeframes: messageType === 'set_equity_range' || shouldForce,
  });
}

function setEquitySeries(range, points) {
  const normalizedRange = EQUITY_RANGES.includes(range) ? range : EQUITY_DEFAULT_RANGE;
  const list = Array.isArray(points) ? points : [];
  equitySeries[normalizedRange] = list
    .map((point) => {
      if (!point || !point.timestamp) {
        return null;
      }
      const numericValue = toNumber(point.value);
      if (numericValue === null) {
        return null;
      }
      const timestampMs = toDate(point.timestamp)?.getTime();
      if (!Number.isFinite(timestampMs)) {
        return null;
      }
      return {
        timestamp: point.timestamp,
        timestampMs,
        value: numericValue,
      };
    })
    .filter(Boolean)
    .sort((a, b) => a.timestampMs - b.timestampMs);
}

function setStatus(state) {
  const el = document.getElementById('connection-status');
  if (!el) {
    return;
  }
  el.classList.remove('status-online', 'status-offline', 'status-pending');
  if (state === 'online') {
    el.textContent = '在线';
    el.classList.add('status-online');
  } else if (state === 'offline') {
    el.textContent = '离线';
    el.classList.add('status-offline');
  } else {
    el.textContent = '刷新中';
    el.classList.add('status-pending');
  }
}

function updateAccountCard(account, prevAccount = {}) {
  const valueEl = document.getElementById('account-value');
  const cashEl = document.getElementById('account-cash');
  const returnEl = document.getElementById('account-return');
  const sharpeEl = document.getElementById('account-sharpe');

  if (!valueEl || !cashEl || !returnEl || !sharpeEl) {
    return;
  }

  valueEl.textContent = formatCurrency(account?.account_value);
  cashEl.textContent = formatCurrency(account?.available_cash);
  returnEl.textContent = formatPercent(account?.return_pct);
  sharpeEl.textContent = formatNumber(account?.sharpe_ratio, 2);

  returnEl.classList.remove('positive', 'negative');
  const returnValue = toNumber(account?.return_pct);
  if (returnValue !== null) {
    returnEl.classList.add(returnValue >= 0 ? 'positive' : 'negative');
  }

  flashNumericChange(valueEl, account?.account_value, prevAccount?.account_value);
  flashNumericChange(cashEl, account?.available_cash, prevAccount?.available_cash);
  flashNumericChange(returnEl, account?.return_pct, prevAccount?.return_pct);
  flashNumericChange(sharpeEl, account?.sharpe_ratio, prevAccount?.sharpe_ratio);
}

function handleAccountUpdate(account = {}) {
  const safeAccount = account && typeof account === 'object' ? { ...account } : {};
  const prevAccount = previousAccountSnapshot || {};
  updateAccountCard(safeAccount, prevAccount);
  previousAccountSnapshot = { ...safeAccount };
  if (previousState && typeof previousState === 'object') {
    previousState.account = { ...previousAccountSnapshot };
  }
  applyLastUpdated(safeAccount.timestamp);
}

function updatePositionsTable(positions, prevPositions = {}) {
  const tbody = document.getElementById('positions-body');
  if (!tbody) {
    return;
  }
  tbody.innerHTML = '';

  const items = Array.isArray(positions?.items) ? positions.items : [];
  const prevItems = Array.isArray(prevPositions?.items) ? prevPositions.items : [];
  const prevMap = new Map();
  prevItems.forEach((item) => {
    const key = normalizeSymbol(item?.symbol);
    if (!prevMap.has(key)) {
      prevMap.set(key, item);
    }
  });

  if (items.length === 0) {
    const row = document.createElement('tr');
    row.innerHTML = '<td class="empty" data-label="状态" colspan="7">暂无持仓数据</td>';
    tbody.appendChild(row);
    return;
  }

  items.forEach((item) => {
    const tr = document.createElement('tr');
    const key = normalizeSymbol(item?.symbol || item?.symbol_code);
    const previous = prevMap.get(key);
    if (previous) {
      prevMap.delete(key);
    }

    const columns = [
      { label: '合约', key: 'symbol', display: item.symbol || '--', className: 'cell-symbol' },
      { label: '数量', key: 'quantity', display: formatNumber(item.quantity), numeric: true },
      { label: '入场价', key: 'entry_price', display: formatNumber(item.entry_price, 2), numeric: true },
      { label: '当前价', key: 'current_price', display: formatNumber(item.current_price, 2), numeric: true },
      {
        label: '未实现盈亏',
        key: 'unrealized_pnl',
        display: formatCurrency(item.unrealized_pnl),
        numeric: true,
        positiveNegative: true,
        className: 'cell-pnl',
      },
      {
        label: '杠杆',
        key: 'leverage',
        display: `${formatNumber(item.leverage, 2)}x`,
        numeric: true,
      },
      { label: '风险 (USD)', key: 'risk_usd', display: formatCurrency(item.risk_usd), numeric: true },
    ];

    columns.forEach((column) => {
      const td = document.createElement('td');
      td.dataset.label = column.label;
      td.textContent = column.display;

      if (column.numeric) {
        td.classList.add('cell-numeric');
      }

      if (column.className) {
        td.classList.add(column.className);
      }

      if (column.positiveNegative) {
        const numericValue = toNumber(item[column.key]);
        if (numericValue === null) {
          td.classList.add('positive');
        } else {
          td.classList.add(numericValue >= 0 ? 'positive' : 'negative');
        }
      }

      if (column.numeric) {
        const previousValue = previous ? previous[column.key] : undefined;
        flashNumericChange(td, item[column.key], previousValue);
      }

      tr.appendChild(td);
    });

    tbody.appendChild(tr);
  });
}

function extractConfidencePercent(signal) {
  if (!signal) {
    return null;
  }
  const clamp = (value) => {
    if (!Number.isFinite(value)) {
      return null;
    }
    return Math.min(Math.max(value, 0), 100);
  };

  const rawValue = signal.confidence_score ?? signal.confidence;

  if (typeof rawValue === 'number') {
    const numeric = rawValue > 1 ? rawValue : rawValue * 100;
    return clamp(numeric);
  }

  if (typeof rawValue === 'string') {
    const trimmed = rawValue.trim();
    const keyword = CONFIDENCE_KEYWORDS[trimmed.toUpperCase()];
    if (keyword !== undefined) {
      return keyword;
    }
    const match = trimmed.match(/-?\d+(?:\.\d+)?/);
    if (match) {
      const parsed = Number(match[0]);
      if (trimmed.includes('%') || parsed > 1) {
        return clamp(parsed);
      }
      return clamp(parsed * 100);
    }
  }

  if (typeof signal.confidence === 'string') {
    const keyword = CONFIDENCE_KEYWORDS[signal.confidence.toUpperCase()];
    if (keyword !== undefined) {
      return keyword;
    }
  }

  return null;
}

function resolveConfidence(signal) {
  if (!signal) {
    return '--';
  }
  if (typeof signal.confidence === 'string') {
    return signal.confidence.toUpperCase();
  }
  const score = signal.confidence_score ?? signal.confidence;
  const num = toNumber(score);
  if (num === null) {
    return '--';
  }
  return `${(num * 100).toFixed(0)}%`;
}

function setStrategyActionBadge(element, action) {
  if (!element) {
    return;
  }
  const classes = ['is-buy', 'is-sell', 'is-short', 'is-hold'];
  element.classList.remove(...classes);
  const normalized = normalizeAction(action);
  if (!normalized) {
    element.textContent = '--';
    return;
  }
  element.textContent = normalized;
  if (normalized === 'BUY' || normalized === 'LONG') {
    element.classList.add('is-buy');
  } else if (normalized === 'SELL') {
    element.classList.add('is-sell');
  } else if (normalized === 'SHORT') {
    element.classList.add('is-short');
  } else if (normalized === 'HOLD' || normalized === 'NEUTRAL') {
    element.classList.add('is-hold');
  }
}

function getStrategyNotional(signal) {
  if (!signal || typeof signal !== 'object') {
    return null;
  }
  const candidates = [
    signal.usdt_amount,
    signal.position_size,
    signal.notional,
    signal.size_usd,
    signal.notional_value,
  ];
  for (const value of candidates) {
    const numeric = toNumber(value);
    if (numeric !== null) {
      return numeric;
    }
  }
  return null;
}

function getLatestStrategySignal(strategy) {
  if (!strategy || typeof strategy !== 'object') {
    return null;
  }
  const signals = strategy.signals || {};
  let latest = null;

  Object.entries(signals).forEach(([coin, records]) => {
    if (!Array.isArray(records) || records.length === 0) {
      return;
    }
    const candidate = records[records.length - 1];
    const timestamp = candidate.timestamp || candidate.created_at || candidate.time;
    const parsed = toDate(timestamp);
    if (!parsed) {
      return;
    }
    if (!latest || parsed.getTime() > latest.timestampMs) {
      latest = {
        record: {
          ...candidate,
          coin: candidate.coin || candidate.symbol || coin,
          timestamp,
        },
        timestampMs: parsed.getTime(),
      };
    }
  });

  return latest ? latest.record : null;
}

function updateStrategyOverview(strategy) {
  const prevSummary = previousStrategySummary || {};
  const symbolEl = document.getElementById('strategy-symbol');
  const actionEl = document.getElementById('strategy-action');
  const confidenceValueEl = document.getElementById('strategy-confidence-value');
  const confidenceBarEl = document.getElementById('strategy-confidence-bar');
  const statusEl = document.getElementById('strategy-status');
  const nameEl = document.getElementById('strategy-name');
  const leverageEl = document.getElementById('strategy-leverage');
  const sizeEl = document.getElementById('strategy-size');
  const updatedEl = document.getElementById('strategy-updated');
  const noteEl = document.getElementById('strategy-note');

  if (!symbolEl || !actionEl || !confidenceValueEl || !confidenceBarEl || !statusEl) {
    return;
  }

  const latest = getLatestStrategySignal(strategy);
  const defaultName = '智能量化监控';

  // if (!latest) {
  //   symbolEl.textContent = '--';
  //   setStrategyActionBadge(actionEl, '');
  //   confidenceValueEl.textContent = '--';
  //   confidenceBarEl.style.width = '0%';
  //   if (statusEl) {
  //     statusEl.textContent = '待机';
  //     statusEl.classList.remove('is-online');
  //   }
  //   if (nameEl) {
  //     nameEl.textContent = defaultName;
  //   }
  //   if (leverageEl) {
  //     leverageEl.textContent = '--';
  //   }
  //   if (sizeEl) {
  //     sizeEl.textContent = '--';
  //   }
  //   if (updatedEl) {
  //     updatedEl.textContent = '--';
  //   }
  //   if (noteEl) {
  //     noteEl.textContent = '--';
  //   }
  //   previousStrategySummary = null;
  //   return;
  // }

  const normalizedCoin = normalizeSymbol(latest.coin || latest.symbol);
  symbolEl.textContent = normalizedCoin || '--';

  const action = latest.action || latest.signal || latest.decision;
  setStrategyActionBadge(actionEl, action);

  const confidencePercent = extractConfidencePercent(latest);
  confidenceBarEl.style.width = confidencePercent !== null ? `${confidencePercent}%` : '0%';
  confidenceValueEl.textContent = resolveConfidence(latest);
  if (confidencePercent !== null) {
    flashNumericChange(
      confidenceValueEl,
      confidencePercent,
      prevSummary.confidencePercent ?? undefined,
    );
  }

  if (statusEl) {
    statusEl.classList.remove('is-online');
    statusEl.textContent = '运行中';
    statusEl.classList.add('is-online');
  }

  if (nameEl) {
    nameEl.textContent = latest.strategy_name || latest.strategy || defaultName;
  }

  if (leverageEl) {
    const leverageValue = toNumber(latest.leverage);
    leverageEl.textContent = leverageValue !== null ? `${formatNumber(leverageValue, 2)}x` : '--';
    flashNumericChange(
      leverageEl,
      leverageValue ?? undefined,
      prevSummary.leverage ?? undefined,
    );
  }

  if (sizeEl) {
    const sizeValue = getStrategyNotional(latest);
    sizeEl.textContent = sizeValue !== null ? formatCurrency(sizeValue) : '--';
    flashNumericChange(
      sizeEl,
      sizeValue ?? undefined,
      prevSummary.size ?? undefined,
    );
  }

  if (updatedEl) {
    updatedEl.textContent = latest.timestamp ? formatTimestamp(latest.timestamp) : '--';
  }

  if (noteEl) {
    const noteValue = latest.reason || latest.note || latest.summary || latest.comment;
    noteEl.textContent = noteValue ? truncateText(noteValue, 90) : '--';
  }

  applyLastUpdated(latest.timestamp);

  previousStrategySummary = {
    action: normalizeAction(action),
    confidencePercent,
    leverage: toNumber(latest.leverage),
    size: getStrategyNotional(latest),
    timestamp: latest.timestamp,
  };
}

function updateStrategySignals(strategy, prevStrategy = {}) {
  const container = document.getElementById('strategy-signals');
  if (!container) {
    return;
  }
  container.innerHTML = '';

  const signals = strategy?.signals || {};
  const prevSignals = prevStrategy?.signals || {};
  const entries = Object.entries(signals).filter(([, value]) => Array.isArray(value) && value.length > 0);

  if (entries.length === 0) {
    container.innerHTML = '<p class="empty">暂无信号数据</p>';
    return;
  }

  entries
    .sort(([a], [b]) => a.localeCompare(b))
    .forEach(([coin, records]) => {
      const latest = records[records.length - 1] || {};
      const prevRecords = prevSignals[coin] || prevSignals[normalizeSymbol(coin)] || [];
      const prevLatest = prevRecords.length > 0 ? prevRecords[prevRecords.length - 1] : null;
      const card = document.createElement('article');
      card.className = 'signal-card';

      const action = (latest.signal || 'HOLD').toUpperCase();
      const actionClass = `signal-action signal-${action.toLowerCase()}`;

      card.innerHTML = `
        <header class="signal-header">
          <span class="signal-coin">${coin}</span>
          <span class="${actionClass}">${action}</span>
        </header>
        <dl class="signal-meta">
          <div><dt>信心</dt><dd>${resolveConfidence(latest)}</dd></div>
          <div><dt>杠杆</dt><dd>${formatNumber(latest.leverage, 2)}x</dd></div>
          <div><dt>仓位规模</dt><dd>${formatNumber(latest.amount, 3)} 张</dd></div>
          <div><dt>目标 / 止损</dt><dd>${formatNumber(latest.take_profit, 2)} / ${formatNumber(latest.stop_loss, 2)}</dd></div>
        </dl>
        <p class="signal-reason">${latest.reason || latest.justification || '—'}</p>
        <footer class="signal-footer">
          <span>${formatTimestamp(latest.timestamp)}</span>
          <span>风险 USD：${formatCurrency(latest.risk_usd)}</span>
        </footer>
      `;

      container.appendChild(card);

      const latestNotional = toNumber(latest.usdt_amount);
      const prevNotional = toNumber(prevLatest?.usdt_amount);
      let delta = null;
      if (prevNotional !== null && latestNotional !== null) {
        delta = latestNotional - prevNotional;
      }
      if ((delta === null || delta === 0) && prevLatest) {
        const prevSignal = (prevLatest.signal || '').toUpperCase();
        if (action !== prevSignal) {
          delta = ['SELL', 'SHORT'].includes(action) ? -1 : 1;
        }
      }
      if (!prevLatest) {
        delta = ['SELL', 'SHORT'].includes(action) ? -1 : 1;
      }
      if (Number.isFinite(delta) && delta !== 0) {
        triggerFlash(card, delta);
      }
    });
}

function updateStrategyBatches(batches, prevBatches = []) {
  const container = document.getElementById('batch-list');
  if (!container) {
    return;
  }
  container.innerHTML = '';

  const items = Array.isArray(batches) ? batches.slice() : [];
  const prevMap = new Map();
  (Array.isArray(prevBatches) ? prevBatches : []).forEach((batch) => {
    if (batch && batch.timestamp && !prevMap.has(batch.timestamp)) {
      prevMap.set(batch.timestamp, batch);
    }
  });
  if (items.length === 0) {
    container.innerHTML = '<p class="empty">尚未记录交易批次</p>';
    return;
  }

  items
    .sort((a, b) => new Date(b.timestamp || 0) - new Date(a.timestamp || 0))
    .slice(0, 5)
    .forEach((batch) => {
      const card = document.createElement('article');
      card.className = 'batch-card';

      const signals = Array.isArray(batch.signals) ? batch.signals : [];
      const rows = signals
        .slice(0, 5)
        .map((signal) => {
          const action = (signal.signal || 'HOLD').toUpperCase();
          const actionClass = `batch-action batch-${action.toLowerCase()}`;
          return `
            <li>
              <span class="batch-coin">${(signal.coin || '').toUpperCase()}</span>
              <span class="${actionClass}">${action}</span>
              <span class="batch-notional">规模：${formatCurrency(signal.usdt_amount)}</span>
              <span>杠杆：${formatNumber(signal.leverage, 2)}x</span>
              <span>理由：${signal.reason || signal.justification || '—'}</span>
            </li>
          `;
        })
        .join('');

      card.innerHTML = `
        <header>
          <h3>${formatTimestamp(batch.timestamp)}</h3>
        </header>
        <ul class="batch-signals">${rows || '<li>无有效信号</li>'}</ul>
      `;

      container.appendChild(card);

      const prev = prevMap.get(batch.timestamp);
      const totalNotional = signals.reduce((sum, signal) => sum + (toNumber(signal.usdt_amount) || 0), 0);
      const prevTotal = prev && Array.isArray(prev.signals)
        ? prev.signals.reduce((sum, signal) => sum + (toNumber(signal.usdt_amount) || 0), 0)
        : null;
      let delta = null;
      if (prevTotal !== null) {
        delta = totalNotional - prevTotal;
      }
      if ((delta === null || delta === 0) && prev) {
        const diffCount = signals.length - (Array.isArray(prev.signals) ? prev.signals.length : 0);
        if (diffCount !== 0) {
          delta = diffCount;
        }
      }
      if (!prev) {
        delta = totalNotional || signals.length || 1;
      }
      if (Number.isFinite(delta) && delta !== 0) {
        triggerFlash(card, delta);
      }
    });
}

function renderState(state) {
  const account = state?.account || {};
  const positions = state?.positions || {};
  const strategy = state?.strategy || {};
  const prevPositions = previousState?.positions || {};
  const prevStrategy = previousState?.strategy || {};

  handleAccountUpdate(account);
  updatePositionsTable(positions, prevPositions);
  // updateStrategyOverview(strategy);
  updateStrategySignals(strategy, prevStrategy);
  updateStrategyBatches(strategy.batches, prevStrategy?.batches || []);

  const timestamps = [account.timestamp, positions.timestamp];
  if (Array.isArray(strategy?.batches) && strategy.batches.length > 0) {
    const latestBatch = strategy.batches[strategy.batches.length - 1];
    timestamps.push(latestBatch.timestamp);
  }
  resetLastUpdatedFromSources(timestamps);

  previousState = cloneState(state) || state;
  if (previousState && typeof previousState === 'object') {
    previousState.account = { ...(previousAccountSnapshot || account) };
  }
}

document.addEventListener('DOMContentLoaded', () => {
  document.querySelectorAll('.range-button').forEach((button) => {
    button.addEventListener('click', () => {
      if (button.dataset.range) {
        setActiveRange(button.dataset.range, { force: true });
      }
    });
  });

  currentEquityRange = EQUITY_DEFAULT_RANGE;
  setActiveRange(EQUITY_DEFAULT_RANGE, { force: true });
  connectDashboardStream();
});
