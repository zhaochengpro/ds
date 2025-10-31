const API_ENDPOINT = '/api/state';
const ACCOUNT_ENDPOINT = '/api/account';
const REFRESH_INTERVAL_MS = 5000;
const ACCOUNT_REFRESH_INTERVAL_MS = 2000;
const EQUITY_RANGES = ['day', 'week', 'month', 'year'];
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
let equitySeries = {
  day: [],
  week: [],
  month: [],
  year: [],
};
let currentEquityRange = 'day';
let equityChart = null;
let latestDataTimestamp = null;

function toNumber(value) {
  const parsed = Number(value);
  return Number.isFinite(parsed) ? parsed : null;
}

function normalizeSymbol(symbol) {
  return (symbol || '').toUpperCase();
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
  if (typeof Chart === 'undefined') {
    return null;
  }
  const canvas = document.getElementById('equity-chart');
  if (!canvas) {
    return null;
  }
  const context = canvas.getContext('2d');
  equityChart = new Chart(context, {
    type: 'line',
    data: {
      labels: [],
      datasets: [
        {
          label: '账户权益',
          data: [],
          borderColor: 'rgba(40, 199, 111, 0.85)',
          backgroundColor: 'rgba(40, 199, 111, 0.15)',
          tension: 0.35,
          fill: true,
          borderWidth: 2,
          pointRadius: 0,
          pointHoverRadius: 4,
          cubicInterpolationMode: 'monotone',
        },
      ],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: { display: false },
        tooltip: {
          mode: 'index',
          intersect: false,
          callbacks: {
            label: (ctx) => `账户权益：${formatCurrency(ctx.parsed.y)}`,
          },
        },
      },
      interaction: {
        mode: 'index',
        intersect: false,
      },
      scales: {
        x: {
          grid: {
            color: 'rgba(255, 255, 255, 0.08)',
          },
          ticks: {
            color: 'rgba(255, 255, 255, 0.6)',
            maxRotation: 0,
            autoSkip: true,
            maxTicksLimit: 6,
          },
        },
        y: {
          grid: {
            color: 'rgba(255, 255, 255, 0.08)',
          },
          ticks: {
            color: 'rgba(255, 255, 255, 0.6)',
            callback: (value) => formatCurrency(value).replace('US$', '$'),
          },
        },
      },
    },
  });
  return equityChart;
}

function renderEquityChart() {
  const activeData = equitySeries[currentEquityRange] || [];
  const emptyEl = document.getElementById('equity-empty');
  if (!activeData || activeData.length === 0) {
    if (emptyEl) {
      emptyEl.style.display = 'flex';
    }
    if (equityChart) {
      equityChart.data.labels = [];
      equityChart.data.datasets[0].data = [];
      equityChart.update('none');
    }
    return;
  }

  if (emptyEl) {
    emptyEl.style.display = 'none';
  }

  const chart = ensureEquityChart();
  if (!chart) {
    return;
  }

  const palette = getEquityColors(currentEquityRange);
  chart.data.datasets[0].borderColor = palette.line;
  chart.data.datasets[0].backgroundColor = palette.fill;
  chart.data.labels = activeData.map((point) => formatChartLabel(point.timestamp, currentEquityRange));
  chart.data.datasets[0].data = activeData.map((point) => Number(point.value) || 0);
  chart.update('none');
}

function applyRangeTheme(range) {
  const palette = getEquityColors(range);
  document.documentElement.style.setProperty('--equity-accent', palette.solid);
  if (equityChart) {
    equityChart.data.datasets[0].borderColor = palette.line;
    equityChart.data.datasets[0].backgroundColor = palette.fill;
    equityChart.update('none');
  }
}

function setActiveRange(range) {
  if (!EQUITY_RANGES.includes(range)) {
    return;
  }
  currentEquityRange = range;
  document.querySelectorAll('.range-button').forEach((button) => {
    button.classList.toggle('is-active', button.dataset.range === range);
  });
  applyRangeTheme(range);
  renderEquityChart();
}

function updateEquitySeries(series) {
  if (!series || typeof series !== 'object') {
    return;
  }

  EQUITY_RANGES.forEach((range) => {
    const points = Array.isArray(series[range]) ? series[range] : [];
    equitySeries[range] = points.map((point) => ({
      timestamp: point.timestamp,
      value: Number(point.value) || 0,
    }));
  });

  if (!EQUITY_RANGES.includes(currentEquityRange)) {
    currentEquityRange = 'day';
  }

  applyRangeTheme(currentEquityRange);
  renderEquityChart();
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
  updateStrategySignals(strategy, prevStrategy);
  updateStrategyBatches(strategy.batches, prevStrategy?.batches || []);
  updateEquitySeries(state?.equity);

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

async function fetchState() {
  setStatus('pending');
  try {
    const response = await fetch(API_ENDPOINT, { cache: 'no-store' });
    if (!response.ok) {
      throw new Error(`请求失败：${response.status}`);
    }
    const data = await response.json();
    renderState(data);
    setStatus('online');
  } catch (error) {
    console.error('获取仪表盘数据失败', error);
    setStatus('offline');
  }
}

async function fetchAccount() {
  try {
    const response = await fetch(ACCOUNT_ENDPOINT, { cache: 'no-store' });
    if (!response.ok) {
      throw new Error(`请求失败：${response.status}`);
    }
    const account = await response.json();
    handleAccountUpdate(account);
  } catch (error) {
    console.error('获取账户数据失败', error);
  }
}

document.addEventListener('DOMContentLoaded', () => {
  document.querySelectorAll('.range-button').forEach((button) => {
    button.addEventListener('click', () => {
      if (button.dataset.range) {
        setActiveRange(button.dataset.range);
      }
    });
  });

  setActiveRange(currentEquityRange);
  fetchState();
  fetchAccount();
  setInterval(fetchState, REFRESH_INTERVAL_MS);
  setInterval(fetchAccount, ACCOUNT_REFRESH_INTERVAL_MS);
});
