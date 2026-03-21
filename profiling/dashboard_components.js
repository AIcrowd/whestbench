// profiling/dashboard_components.js
// All React components for the profiling dashboard.
// Uses React.createElement (no JSX/Babel). Data from window.__PROFILING_DATA__.
'use strict';

var h = React.createElement;
var useState = React.useState;
var useEffect = React.useEffect;

var BACKEND_COLORS = {
  numpy: '#4A90D9', pytorch: '#e65100', jax: '#334155',
  cython: '#23B761', numba: '#6a1b9a', scipy: '#00838f'
};
var BACKEND_ORDER = ['numpy', 'pytorch', 'jax', 'cython', 'numba', 'scipy'];

function getSpeedupColor(s) {
  if (s == null) return 'transparent';
  if (s >= 0.9 && s <= 1.1) return '#F8F9F9';
  if (s > 1.1) {
    var t = Math.min((s - 1.1) / 3.0, 1.0);
    var r = Math.round(35 + (1 - t) * 220);
    var g = Math.round(183 + (1 - t) * 72);
    var b = Math.round(97 + (1 - t) * 152);
    return 'rgb(' + r + ',' + g + ',' + b + ')';
  }
  var t2 = Math.min((1.0 - s) / 0.5, 1.0);
  var r2 = Math.round(240 + (1 - t2) * 8);
  var g2 = Math.round(82 + (1 - t2) * 165);
  var b2 = Math.round(77 + (1 - t2) * 172);
  return 'rgb(' + r2 + ',' + g2 + ',' + b2 + ')';
}

function formatTime(s) {
  if (s < 0.001) return (s * 1e6).toFixed(0) + 'µs';
  if (s < 1) return (s * 1000).toFixed(1) + 'ms';
  return s.toFixed(3) + 's';
}

function uniqueVals(timing, key) {
  var s = {};
  timing.forEach(function(t) { s[t[key]] = true; });
  return Object.keys(s).map(function(v) { return isNaN(v) ? v : Number(v); }).sort(function(a, b) { return a - b; });
}

function getUnique(data, key) {
  var all = [];
  Object.values(data.configs).forEach(function(c) { all = all.concat(c.timing); });
  return uniqueVals(all, key);
}

function App() {
  var data = window.__PROFILING_DATA__;
  var operations = getUnique(data, 'operation');
  var widths = getUnique(data, 'width');
  var depths = getUnique(data, 'depth');
  var nSamples = getUnique(data, 'n_samples');
  var isMulti = Object.keys(data.configs).length > 1;

  var s = useState({
    operation: operations[0] || 'run_mlp',
    width: widths[widths.length - 1] || 256,
    depth: depths[depths.length - 1] || 4,
    nSamples: nSamples[nSamples.length - 1] || 10000,
    modal: null
  });
  var filters = s[0]; var setFilters = s[1];

  function onFilterChange(key, val) {
    setFilters(function(prev) {
      var n = {}; for (var k in prev) n[k] = prev[k];
      n[key] = isNaN(val) ? val : Number(val);
      return n;
    });
  }

  return h('div', {className: 'dashboard'},
    h(Header, {data: data}),
    h(SpeedupHeatmap, {data: data, filters: filters, onFilterChange: onFilterChange,
      onCellClick: function(m) { onFilterChange('modal', m); }}),
    isMulti ? h(CPUScalingChart, {data: data, filters: filters, onFilterChange: onFilterChange}) : null,
    h(DataTable, {data: data}),
    filters.modal ? h(CellDetailModal, {data: data, config: filters.modal.config,
      backend: filters.modal.backend, filters: filters, onFilterChange: onFilterChange,
      onClose: function() { onFilterChange('modal', null); }}) : null
  );
}

function Header(props) {
  var d = props.data;
  return h('header', {className: 'app-header'},
    h('h1', null, 'nestim Profiling Dashboard'),
    h('div', {style: {display: 'flex', alignItems: 'center', gap: '12px',
      fontSize: '12px', color: '#6B7280'}},
      d.run_id ? h('span', null, d.run_id) : null,
      d.git_commit ? h('span', {style: {fontFamily: 'var(--font-mono)'}},
        d.git_commit.substring(0, 7),
        d.git_dirty ? h('span', {className: 'badge-slower',
          style: {marginLeft: '4px', fontSize: '9px', padding: '1px 6px'}}, 'dirty') : null
      ) : null,
      d.collected_at ? h('span', null, new Date(d.collected_at).toLocaleString()) : null
    )
  );
}

function SpeedupHeatmap(props) {
  var data = props.data, f = props.filters;
  var configs = Object.keys(data.configs);
  var backends = BACKEND_ORDER.filter(function(b) {
    return configs.some(function(c) {
      return data.configs[c].timing.some(function(t) { return t.backend === b; }) ||
        (data.configs[c].skipped_backends && data.configs[c].skipped_backends[b]);
    });
  });

  function getCell(config, backend) {
    var cfg = data.configs[config];
    if (cfg.skipped_backends && cfg.skipped_backends[backend]) return {type: 'skipped', reason: cfg.skipped_backends[backend]};
    var entry = cfg.timing.find(function(t) {
      return t.backend === backend && t.operation === f.operation &&
        t.width === f.width && t.depth === f.depth && t.n_samples === f.nSamples;
    });
    if (!entry) return {type: 'missing'};
    var corr = cfg.correctness.find(function(c) { return c.backend === backend; });
    return {type: 'data', speedup: entry.speedup_vs_numpy,
      correctnessFailed: corr && !corr.passed, correctnessError: corr ? corr.error : ''};
  }

  var operations = getUnique(data, 'operation');
  var widths = getUnique(data, 'width');
  var depths = getUnique(data, 'depth');
  var nSamples = getUnique(data, 'n_samples');

  function sel(label, key, opts, val) {
    return h('div', null,
      h('div', {className: 'filter-label'}, label),
      h('select', {className: 'filter-select', value: val,
        onChange: function(e) { props.onFilterChange(key, e.target.value); }},
        opts.map(function(v) { return h('option', {key: v, value: v}, String(v)); })
      )
    );
  }

  return h('section', {className: 'heatmap-section'},
    h('div', {className: 'section-header'}, 'SPEEDUP VS NUMPY'),
    h('div', {className: 'heatmap-filters'},
      sel('Operation', 'operation', operations, f.operation),
      sel('Width', 'width', widths, f.width),
      sel('Depth', 'depth', depths, f.depth),
      sel('N Samples', 'nSamples', nSamples, f.nSamples)
    ),
    h('table', {className: 'heatmap-table'},
      h('thead', null, h('tr', null,
        h('th', null, 'Config'),
        backends.map(function(b) { return h('th', {key: b}, b); })
      )),
      h('tbody', null, configs.map(function(config) {
        var hw = data.configs[config].hardware;
        var ramGB = Math.round((hw.ram_total_bytes || 0) / 1073741824);
        var cells = backends.map(function(b) { return getCell(config, b); });
        var bestIdx = -1; var bestVal = -Infinity;
        cells.forEach(function(c, i) {
          if (c.type === 'data' && c.speedup > bestVal) { bestVal = c.speedup; bestIdx = i; }
        });

        return h('tr', {key: config},
          h('td', null, config, ' ', h('span', {className: 'config-hw'},
            '(' + (hw.cpu_count_logical || '?') + 'c / ' + ramGB + 'G)')),
          cells.map(function(c, i) {
            if (c.type === 'skipped') return h('td', {key: backends[i],
              style: {color: '#9CA3AF', fontStyle: 'italic'}, title: c.reason}, 'skip');
            if (c.type === 'missing') return h('td', {key: backends[i],
              style: {color: '#D1D5DB'}}, '—');
            return h('td', {key: backends[i],
              className: i === bestIdx ? 'best' : '',
              style: {background: getSpeedupColor(c.speedup), cursor: 'pointer'},
              onClick: function() { props.onCellClick({config: config, backend: backends[i]}); }},
              c.speedup.toFixed(2),
              c.correctnessFailed ? h('span', {className: 'cell-warning',
                title: c.correctnessError}, ' ⚠') : null
            );
          })
        );
      }),
      configs.every(function(c) {
        return backends.every(function(b) { return getCell(c, b).type !== 'data'; });
      }) ? h('tr', null, h('td', {colSpan: backends.length + 1,
        style: {textAlign: 'center', color: '#9CA3AF', padding: '20px'}},
        'No data for this combination')) : null
      )
    )
  );
}

function CellDetailModal(props) {
  var data = props.data, f = props.filters;
  var config = props.config, backend = props.backend;

  useEffect(function() {
    function onKey(e) { if (e.key === 'Escape') props.onClose(); }
    document.addEventListener('keydown', onKey);
    return function() { document.removeEventListener('keydown', onKey); };
  }, []);

  var cfg = data.configs[config];
  var entry = cfg.timing.find(function(t) {
    return t.backend === backend && t.operation === f.operation &&
      t.width === f.width && t.depth === f.depth && t.n_samples === f.nSamples;
  });

  var profiled = cfg.timing.find(function(t) {
    return t.backend === backend && t.operation === 'run_mlp_profiled' &&
      t.width === f.width && t.depth === f.depth && t.n_samples === f.nSamples;
  });

  var npEntry = cfg.timing.find(function(t) {
    return t.backend === 'numpy' && t.operation === f.operation &&
      t.width === f.width && t.depth === f.depth && t.n_samples === f.nSamples;
  });

  var speedup = entry ? entry.speedup_vs_numpy : null;
  var configs = Object.keys(data.configs);
  var widths = getUnique(data, 'width');
  var depths = getUnique(data, 'depth');
  var nSamples = getUnique(data, 'n_samples');

  function sel(label, key, opts, val) {
    return h('div', null,
      h('div', {className: 'filter-label'}, label),
      h('select', {className: 'filter-select', value: val,
        onChange: function(e) { props.onFilterChange(key, e.target.value); }},
        opts.map(function(v) { return h('option', {key: v, value: v}, String(v)); })
      )
    );
  }

  return h('div', {className: 'modal-overlay', onClick: function(e) {
      if (e.target.className === 'modal-overlay') props.onClose(); }},
    h('div', {className: 'modal'},
      h('div', {className: 'modal-header'},
        h('div', {style: {display: 'flex', alignItems: 'center', gap: '10px'}},
          h('span', {className: 'modal-title'}, backend),
          speedup != null ? h('span', {className: speedup >= 1 ? 'badge-faster' : 'badge-slower'},
            speedup.toFixed(1) + 'x ' + (speedup >= 1 ? 'faster' : 'slower')) : null
        ),
        h('button', {className: 'modal-close', onClick: props.onClose}, '✕')
      ),
      h('div', {className: 'modal-filters'},
        configs.length > 1 ? sel('Config', 'modal', configs.map(function(c) { return c; }), config) : null,
        sel('Width', 'width', widths, f.width),
        sel('Depth', 'depth', depths, f.depth),
        sel('N Samples', 'nSamples', nSamples, f.nSamples)
      ),
      h('div', {className: 'modal-body'},
        h('div', null,
          entry ? [
            h('div', {key: 'big', className: 'timing-big'},
              formatTime(entry.median_time), h('span', {className: 'timing-unit'}, '')),
            h('div', {key: 'sub', className: 'timing-sub'},
              'median of ' + (entry.times || []).length + ' runs'),
            (entry.times || []).map(function(t, i) {
              var maxT = Math.max.apply(null, entry.times);
              var pct = (t / maxT * 100).toFixed(0);
              var isMedian = Math.abs(t - entry.median_time) < 1e-9;
              return h('div', {key: i, className: 'run-bar-row'},
                h('span', {className: 'run-bar-label'}, '#' + (i + 1)),
                h('div', {className: 'run-bar-track'},
                  h('div', {className: 'run-bar-fill', style: {width: pct + '%',
                    background: isMedian ? 'var(--success)' : 'var(--chart-3)'}},
                    h('span', {className: 'run-bar-value'}, formatTime(t))
                  )
                )
              );
            }),
            (function() {
              var times = entry.times || [];
              if (times.length < 2) return null;
              var mean = times.reduce(function(a, b) { return a + b; }, 0) / times.length;
              var variance = times.reduce(function(a, t) { return a + (t - mean) * (t - mean); }, 0) / (times.length - 1);
              var std = Math.sqrt(variance);
              var cv = (std / mean * 100).toFixed(1);
              return h('div', {key: 'stats', className: 'timing-stats'},
                'σ = ' + formatTime(std) + '  CV = ' + cv + '%');
            })()
          ] : h('div', {style: {color: '#9CA3AF'}}, 'No timing data')
        ),
        h('div', null,
          profiled && profiled.breakdown ? [
            h('div', {key: 'label', className: 'section-header'}, 'OPERATION BREAKDOWN'),
            h('div', {key: 'bar', className: 'breakdown-bar'},
              h('span', {style: {width: profiled.breakdown.matmul_pct + '%', background: '#334155'}},
                profiled.breakdown.matmul_pct > 15 ? profiled.breakdown.matmul_pct.toFixed(0) + '%' : ''),
              h('span', {style: {width: profiled.breakdown.relu_pct + '%', background: '#F0524D'}},
                profiled.breakdown.relu_pct > 15 ? profiled.breakdown.relu_pct.toFixed(0) + '%' : ''),
              h('span', {style: {width: profiled.breakdown.overhead_pct + '%', background: '#E5E7EB'}}, '')
            ),
            h('div', {key: 'legend', className: 'breakdown-legend'},
              h('div', {className: 'breakdown-row'},
                h('span', null, h('span', {className: 'breakdown-swatch', style: {background: '#334155'}}), 'MatMul'),
                h('span', null, h('span', {className: 'breakdown-value'}, formatTime(profiled.breakdown.total_matmul)),
                  h('span', {className: 'breakdown-pct'}, profiled.breakdown.matmul_pct.toFixed(1) + '%'))
              ),
              h('div', {className: 'breakdown-row'},
                h('span', null, h('span', {className: 'breakdown-swatch', style: {background: '#F0524D'}}), 'ReLU'),
                h('span', null, h('span', {className: 'breakdown-value'}, formatTime(profiled.breakdown.total_relu)),
                  h('span', {className: 'breakdown-pct'}, profiled.breakdown.relu_pct.toFixed(1) + '%'))
              ),
              h('div', {className: 'breakdown-row'},
                h('span', null, h('span', {className: 'breakdown-swatch', style: {background: '#E5E7EB'}}), 'Overhead'),
                h('span', null, h('span', {className: 'breakdown-value'}, formatTime(profiled.breakdown.overhead)),
                  h('span', {className: 'breakdown-pct'}, profiled.breakdown.overhead_pct.toFixed(1) + '%'))
              )
            ),
            profiled.breakdown.matmul_per_layer ? h('div', {key: 'spark'},
              h('div', {className: 'section-header', style: {marginTop: '14px'}}, 'MATMUL PER LAYER'),
              h('div', {className: 'sparkline'},
                profiled.breakdown.matmul_per_layer.map(function(v, i) {
                  var maxV = Math.max.apply(null, profiled.breakdown.matmul_per_layer);
                  return h('div', {key: i, className: 'sparkline-bar',
                    style: {height: (v / maxV * 100) + '%'}, title: 'Layer ' + (i+1) + ': ' + formatTime(v)});
                })
              ),
              h('div', {className: 'sparkline-labels'},
                h('span', null, '1'),
                h('span', null, String(profiled.breakdown.matmul_per_layer.length))
              )
            ) : null
          ] : h('div', {style: {color: '#9CA3AF', padding: '20px 0'}},
            'No profiled data for this configuration')
        )
      ),
      entry && npEntry ? h('div', {className: 'modal-footer'},
        h('span', {className: 'compare-label'}, 'numpy'),
        h('div', {className: 'compare-bar'},
          h('div', {className: 'compare-fill', style: {
            width: '100%', background: BACKEND_COLORS.numpy}},
            h('span', null, formatTime(npEntry.median_time)))
        ),
        h('span', {className: 'compare-label'}, backend),
        h('div', {className: 'compare-bar'},
          h('div', {className: 'compare-fill', style: {
            width: Math.min(entry.median_time / npEntry.median_time * 100, 100) + '%',
            background: BACKEND_COLORS[backend] || '#666'}},
            h('span', null, formatTime(entry.median_time)))
        )
      ) : null
    )
  );
}

function CPUScalingChart(props) {
  var data = props.data, f = props.filters;
  var configs = Object.keys(data.configs);
  if (configs.length <= 1) return null;

  var LC = Recharts.LineChart, L = Recharts.Line, XA = Recharts.XAxis,
      YA = Recharts.YAxis, TT = Recharts.Tooltip, Lg = Recharts.Legend,
      RC = Recharts.ResponsiveContainer;

  var fs = useState('all');
  var family = fs[0]; var setFamily = fs[1];

  var sorted = configs.slice().sort(function(a, b) {
    return (data.configs[a].hardware.cpu_count_logical || 0) -
           (data.configs[b].hardware.cpu_count_logical || 0);
  }).filter(function(c) {
    if (family === 'all') return true;
    return c.indexOf(family) === 0;
  });

  var backends = {};
  sorted.forEach(function(c) {
    data.configs[c].timing.forEach(function(t) {
      if (t.operation === f.operation && t.width === f.width &&
          t.depth === f.depth && t.n_samples === f.nSamples) {
        backends[t.backend] = true;
      }
    });
  });
  var backendList = Object.keys(backends);

  var chartData = sorted.map(function(c) {
    var hw = data.configs[c].hardware;
    var point = {name: c + ' (' + hw.cpu_count_logical + 'c)'};
    data.configs[c].timing.forEach(function(t) {
      if (t.operation === f.operation && t.width === f.width &&
          t.depth === f.depth && t.n_samples === f.nSamples) {
        point[t.backend] = t.median_time;
      }
    });
    return point;
  });

  return h('section', {className: 'scaling-section'},
    h('div', {className: 'section-header'}, 'CPU SCALING'),
    h('div', {className: 'heatmap-filters'},
      h('div', null,
        h('div', {className: 'filter-label'}, 'Config Family'),
        h('select', {className: 'filter-select', value: family,
          onChange: function(e) { setFamily(e.target.value); }},
          h('option', {value: 'all'}, 'All'),
          h('option', {value: 'compute'}, 'Compute'),
          h('option', {value: 'general'}, 'General')
        )
      )
    ),
    h('div', {className: 'scaling-chart-container'},
      h(RC, {width: '100%', height: 300},
        h(LC, {data: chartData, margin: {top: 5, right: 30, left: 20, bottom: 5}},
          h(XA, {dataKey: 'name', fontSize: 10}),
          h(YA, {fontSize: 10, label: {value: 'Time (s)', angle: -90, position: 'insideLeft'}}),
          h(TT, null),
          h(Lg, null),
          backendList.map(function(b) {
            return h(L, {key: b, type: 'monotone', dataKey: b,
              stroke: BACKEND_COLORS[b] || '#666', strokeWidth: 2, dot: {r: 4}});
          })
        )
      )
    )
  );
}

function DataTable(props) {
  var data = props.data;
  var st = useState({sortKey: 'config', sortDir: 'asc', filters: {}});
  var state = st[0]; var setState = st[1];

  var rows = [];
  Object.keys(data.configs).forEach(function(config) {
    var cfg = data.configs[config];
    cfg.timing.forEach(function(t) {
      rows.push({config: config, backend: t.backend, operation: t.operation,
        width: t.width, depth: t.depth, n_samples: t.n_samples,
        median_time: t.median_time, speedup: t.speedup_vs_numpy,
        hardware: cfg.hardware, backend_versions: cfg.backend_versions});
    });
  });

  var filtered = rows.filter(function(r) {
    for (var k in state.filters) {
      if (state.filters[k] && String(r[k]) !== String(state.filters[k])) return false;
    }
    return true;
  });

  filtered.sort(function(a, b) {
    var av = a[state.sortKey], bv = b[state.sortKey];
    if (av < bv) return state.sortDir === 'asc' ? -1 : 1;
    if (av > bv) return state.sortDir === 'asc' ? 1 : -1;
    return 0;
  });

  function toggleSort(key) {
    setState(function(prev) {
      return {sortKey: key, sortDir: prev.sortKey === key && prev.sortDir === 'asc' ? 'desc' : 'asc',
        filters: prev.filters};
    });
  }

  function setFilter(key, val) {
    setState(function(prev) {
      var f = {}; for (var k in prev.filters) f[k] = prev.filters[k];
      f[key] = val || '';
      return {sortKey: prev.sortKey, sortDir: prev.sortDir, filters: f};
    });
  }

  var cols = [
    {key: 'config', label: 'Config'}, {key: 'backend', label: 'Backend'},
    {key: 'operation', label: 'Operation'}, {key: 'width', label: 'Width'},
    {key: 'depth', label: 'Depth'}, {key: 'n_samples', label: 'N Samples'},
    {key: 'median_time', label: 'Median Time'}, {key: 'speedup', label: 'Speedup'}
  ];

  var expanded = useState({});
  var exp = expanded[0]; var setExp = expanded[1];

  return h('section', {className: 'datatable-section'},
    h('div', {className: 'section-header'}, 'RAW DATA'),
    h('table', {className: 'datatable'},
      h('thead', null,
        h('tr', null, cols.map(function(c) {
          var arrow = state.sortKey === c.key ? (state.sortDir === 'asc' ? ' ▲' : ' ▼') : '';
          return h('th', {key: c.key, onClick: function() { toggleSort(c.key); }},
            c.label, h('span', {className: 'sort-arrow'}, arrow));
        })),
        h('tr', null, cols.map(function(c) {
          var vals = uniqueVals(rows, c.key);
          return h('th', {key: c.key + '-f', style: {padding: '4px'}},
            vals.length <= 20 ? h('select', {className: 'filter-select',
              style: {width: '100%', fontSize: '9px'},
              value: state.filters[c.key] || '',
              onChange: function(e) { setFilter(c.key, e.target.value); }},
              h('option', {value: ''}, 'All'),
              vals.map(function(v) { return h('option', {key: v, value: v}, String(v)); })
            ) : null
          );
        }))
      ),
      h('tbody', null, filtered.map(function(r, i) {
        var rowKey = r.config + '-' + r.backend + '-' + r.operation + '-' + r.width + '-' + r.depth + '-' + r.n_samples;
        var isExp = exp[rowKey];
        return [
          h('tr', {key: rowKey, onClick: function() {
            setExp(function(prev) { var n = {}; for (var k in prev) n[k] = prev[k]; n[rowKey] = !prev[rowKey]; return n; });
          }, style: {cursor: 'pointer'}},
            h('td', {style: {fontFamily: 'var(--font-sans)'}}, r.config),
            h('td', null, r.backend),
            h('td', null, r.operation),
            h('td', null, r.width),
            h('td', null, r.depth),
            h('td', null, r.n_samples),
            h('td', null, formatTime(r.median_time)),
            h('td', {style: {fontWeight: r.speedup > 1.1 ? '600' : '400',
              color: r.speedup > 1 ? 'var(--success)' : 'var(--coral)'}}, r.speedup.toFixed(2) + 'x')
          ),
          isExp ? h('tr', {key: rowKey + '-detail'},
            h('td', {colSpan: 8, style: {background: 'var(--gray-50)', fontSize: '10px',
              fontFamily: 'var(--font-mono)', padding: '8px 12px'}},
              'CPU: ' + (r.hardware.cpu_count_logical || '?') +
              ' | RAM: ' + Math.round((r.hardware.ram_total_bytes || 0) / 1073741824) + 'GB' +
              ' | Platform: ' + (r.hardware.platform || '?') +
              (r.backend_versions ? ' | Versions: ' + JSON.stringify(r.backend_versions) : '')
            )
          ) : null
        ];
      }))
    )
  );
}

ReactDOM.render(h(App, null), document.getElementById('root'));
