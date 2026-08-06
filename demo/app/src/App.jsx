import React, { useEffect, useMemo, useRef, useState } from 'react'

/* ================================================================================
   SkimSearchAgent — the live playground.
   The agent is a little magnifying-glass critter that hops between tool stations
   (Search → Read → Answer) on a stage; every hop, packet and lit-up tile is driven
   by the REAL agent loop's SSE stream. Strategy identities (validated palette on
   the paper surface): Sieve = teal, Search-Visit = violet.
   ================================================================================ */

const STRATS = {
  sieve: { label: 'Sieve', sub: 'Boolean search · fetch sections', color: '#0e9384', soft: '#e2f1ee' },
  search_visit: { label: 'Search-Visit', sub: 'BM25 · read whole documents', color: '#7c3aed', soft: '#efe9fb' },
}
const MODES = [
  { id: 'sieve', name: 'Sieve' },
  { id: 'search_visit', name: 'Search-Visit' },
  { id: 'both', name: '⚔ Race both' },
]
const fmtTok = n => (n >= 10000 ? `${(n / 1000).toFixed(1)}K` : (n || 0).toLocaleString())

/* ---------- SSE client ---------- */

async function streamRun(body, onEvent, onFail) {
  try {
    const res = await fetch('/api/run', {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    })
    if (!res.ok || !res.body) { onFail(`server error (${res.status})`); return }
    const reader = res.body.getReader()
    const dec = new TextDecoder()
    let buf = ''
    for (;;) {
      const { value, done } = await reader.read()
      if (done) break
      buf += dec.decode(value, { stream: true })
      const frames = buf.split('\n\n')
      buf = frames.pop()
      for (const f of frames) {
        const data = f.split('\n').find(l => l.startsWith('data: '))
        if (data) onEvent(JSON.parse(data.slice(6)))
      }
    }
  } catch {
    onFail('offline')
  }
}

/* one incoming step -> { station, action, feed items } */
function interpretStep(step, n) {
  if (step.type === 'search') {
    return {
      station: 'search', action: `search: ${step.query}`,
      items: [
        { kind: 'act', step: n, op: 'search', text: step.query },
        ...step.hits.map(h => ({ kind: 'hit', step: n, hit: h, query: step.query })),
        ...(step.hits.length === 0 ? [{ kind: 'zero', step: n, text: step.status || 'no matches' }] : []),
      ],
      seen: step.hits.map(h => h.id),
    }
  }
  if (step.type === 'fetch') {
    return {
      station: 'read', action: `fetch: ${step.ask || `${step.doc} § ${step.section}`}`,
      items: [{ kind: 'act', step: n, op: 'fetch', text: step.ask },
              { kind: 'read', step: n, doc: step.doc, part: `§ ${step.section}`,
                text: step.text, error: step.error, whole: false }],
      read: step.error ? [] : [step.doc],
    }
  }
  if (step.type === 'visit') {
    return {
      station: 'read', action: `read: ${step.doc} (whole document)`,
      items: [{ kind: 'act', step: n, op: 'read', text: `${step.doc} — whole document` },
              { kind: 'read', step: n, doc: step.doc, part: step.title,
                text: step.text, error: step.error, whole: true }],
      read: step.error ? [] : [step.doc],
    }
  }
  if (['answer', 'submit', 'stop'].includes(step.name)) {
    return { station: 'answer', action: 'writing the answer…', items: [] }
  }
  if (step.name === 'budget') {
    return { station: null, action: 'step budget reached — committing to an answer',
             items: [{ kind: 'note', step: n, text: 'step budget reached — forcing a final answer' }] }
  }
  return { station: null, action: null,
           items: [{ kind: 'note', step: n, text: `${step.name}: ${String(step.observation || '').slice(0, 200)}` }] }
}

/* ---------- imperative stage effects (packets, tile flights, confetti) ---------- */

function spawnPackets(strategy, station, color) {
  const stage = document.getElementById(`stage-${strategy}`)
  const st = document.getElementById(`st-${strategy}-${station}`)
  if (!stage || !st) return
  const base = st.offsetLeft + st.offsetWidth / 2 - 6
  for (let i = 0; i < 5; i++) {
    const p = document.createElement('div')
    p.className = 'packet'
    p.style.background = color
    p.style.left = `${base}px`
    p.style.top = '86px'
    p.style.setProperty('--dx', `${(i - 2) * 26}px`)
    stage.appendChild(p)
    setTimeout(() => p.classList.add('fly'), i * 80)
    setTimeout(() => p.remove(), 1100 + i * 80)
  }
}

function flyTile(strategy, docId) {
  const cell = document.getElementById(`cell-${docId}`)
  const st = document.getElementById(`st-${strategy}-read`)
  if (!cell || !st) return
  const a = cell.getBoundingClientRect(), b = st.getBoundingClientRect()
  const t = document.createElement('div')
  t.className = 'flytile'
  t.style.left = `${a.left}px`; t.style.top = `${a.top}px`
  document.body.appendChild(t)
  requestAnimationFrame(() => {
    t.style.left = `${b.left + b.width / 2}px`; t.style.top = `${b.top + 14}px`
    t.style.transform = 'scale(1.7) rotate(170deg)'; t.style.opacity = '0'
  })
  setTimeout(() => t.remove(), 950)
}

function confetti(strategy) {
  const stage = document.getElementById(`stage-${strategy}`)
  const st = document.getElementById(`st-${strategy}-answer`)
  if (!stage || !st) return
  const colors = ['#0e9384', '#7c3aed', '#d97706', '#317a31']
  const base = st.offsetLeft + st.offsetWidth / 2
  for (let i = 0; i < 14; i++) {
    const p = document.createElement('div')
    p.className = 'conf'
    p.style.background = colors[i % 4]
    p.style.left = `${base}px`; p.style.top = '46px'
    p.style.setProperty('--cx', `${Math.random() * 130 - 65}px`)
    p.style.setProperty('--cy', `${Math.random() * -75 - 18}px`)
    stage.appendChild(p)
    setTimeout(() => p.classList.add('go'), i * 30)
    setTimeout(() => p.remove(), 1300)
  }
}

/* ---------- the logo (custom mark: lens + skim line) ---------- */

function Logo() {
  return (
    <svg className="logo-mark" viewBox="0 0 26 26" width="22" height="22" aria-hidden="true">
      <circle cx="11" cy="11" r="7.2" fill="none" stroke="#0e9384" strokeWidth="2.6" />
      <line x1="16.5" y1="16.5" x2="23" y2="23" stroke="#0e9384" strokeWidth="3" strokeLinecap="round" />
      <line x1="7.4" y1="9.4" x2="14.6" y2="9.4" stroke="#0e9384" strokeWidth="1.7" strokeLinecap="round" opacity=".85" />
      <line x1="7.4" y1="12.8" x2="12.2" y2="12.8" stroke="#0e9384" strokeWidth="1.7" strokeLinecap="round" opacity=".45" />
    </svg>
  )
}

/* ---------- the mascot ---------- */

function Mascot({ color, thinking, sad, happy }) {
  return (
    <div className={`critter${thinking ? ' thinking' : ''}${sad ? ' sad' : ''}${happy ? ' happy' : ''}`}
         style={{ '--ac': color }}>
      <div className="lens">
        <span className="eye l" /><span className="eye r" />
        <span className="mouth" />
        <div className="handle" />
      </div>
      <span className="bubble">{sad ? '💧' : '💭'}</span>
    </div>
  )
}

/* ---------- meter (stat strip) ---------- */

function Meter({ usage, t0, done, color }) {
  const [, tick] = useState(0)
  useEffect(() => {
    if (done) return
    const t = setInterval(() => tick(x => x + 1), 250)
    return () => clearInterval(t)
  }, [done])
  const secs = ((done || Date.now()) - t0) / 1000
  const tok = (usage.prompt_tokens || 0) + (usage.completion_tokens || 0)
  return (
    <div className="meter">
      <b style={{ color }}>${(usage.cost_usd || 0).toFixed(5)}</b>
      <span>{fmtTok(tok)} tok</span>
      <span>{usage.steps || 0} steps</span>
      <span>{secs.toFixed(1)}s</span>
    </div>
  )
}

/* ---------- the stage ---------- */

const STATIONS = [
  { id: 'search', icon: '🔎', name: 'Search', left: '12%' },
  { id: 'read', icon: '📖', name: 'Read', left: '44%' },
  { id: 'answer', icon: '✍️', name: 'Answer', left: '76%' },
]
const AGENT_POS = { idle: '1%', search: '13%', read: '45%', answer: '77%' }

function Stage({ strategy, col, question, mini }) {
  const s = STRATS[strategy]
  const station = col ? col.station : 'idle'
  const [landing, setLanding] = useState(false)
  const prev = useRef(station)
  useEffect(() => {
    if (prev.current !== station) {
      prev.current = station
      const t = setTimeout(() => { setLanding(true); setTimeout(() => setLanding(false), 500) }, 720)
      return () => clearTimeout(t)
    }
  }, [station])
  // pupils track the cursor a little
  const stageRef = useRef(null)
  const onMove = e => {
    const r = stageRef.current?.getBoundingClientRect()
    if (!r) return
    stageRef.current.style.setProperty('--ex', `${((e.clientX - r.left) / r.width - .5) * 4}px`)
    stageRef.current.style.setProperty('--ey', `${((e.clientY - r.top) / r.height - .5) * 3}px`)
  }
  const running = col && !col.doneAt && !col.error
  const action = col?.action ||
    (col ? 'warming up…' : 'give me a question and a key, then press ↑')
  return (
    <div className={`stage-card${mini ? ' mini' : ''}`} style={{ '--ac': s.color, '--ac-soft': s.soft }}>
      <div className="stage-head">
        <div className="stage-who">
          <span className="who-dot" />
          <span className="who-name">{s.label}</span>
          <span className="who-sub">{s.sub}</span>
        </div>
        {col && <Meter usage={col.usage} t0={col.t0} done={col.doneAt} color={s.color} />}
      </div>
      <div className={`ticker${running ? ' live' : ''}`}>{action}</div>
      <div className="stage" id={`stage-${strategy}`} ref={stageRef} onMouseMove={onMove}>
        <div className="floor" />
        {STATIONS.map(st => (
          <div key={st.id} id={`st-${strategy}-${st.id}`}
               className={`station${station === st.id ? ' active' : ''}`}
               style={{ left: st.left }}>
            <div className="st-pad">{st.icon}</div>
            <div className="st-name">{st.name}</div>
          </div>
        ))}
        <div className="agent" style={{ left: AGENT_POS[station] || AGENT_POS.idle }}>
          <div className={landing ? 'land' : ''}>
            <Mascot color={s.color}
                    thinking={running && station !== 'answer'}
                    sad={!!col?.error}
                    happy={!!col?.doneAt && !col?.error} />
          </div>
        </div>
      </div>
    </div>
  )
}

/* ---------- feed cards ---------- */

function Snip({ text, query }) {
  const terms = useMemo(
    () => new Set((query.match(/[A-Za-z][A-Za-z-]{3,}/g) || []).map(t => t.toLowerCase())),
    [query])
  const parts = text.split(/([A-Za-z][A-Za-z-]{3,})/g)
  return (
    <div className="snip">
      {parts.map((p, i) => terms.has(p.toLowerCase()) ? <mark key={i}>{p}</mark> : p)}
    </div>
  )
}

function HitCard({ hit, query }) {
  return (
    <div className="hit">
      <div className="hit-t">
        <span className="rank">{hit.rank}</span>
        <b>{hit.title}</b>
        <span className="hid">{hit.id}</span>
        {hit.matched && <span className="matched">matched: {hit.matched}</span>}
      </div>
      {hit.sections.length > 0 && (
        <div className="secs">{hit.sections.map(x => <span className="sec" key={x}>§ {x}</span>)}</div>
      )}
      {hit.snippet && <Snip text={hit.snippet} query={query} />}
    </div>
  )
}

function ReadCard({ doc, part, text, error, whole }) {
  const [open, setOpen] = useState(false)
  const long = text.length > 550
  const shown = open || !long ? text : text.slice(0, 550) + ' …'
  return (
    <div className={`read-card${error ? ' err' : ''}`}
         onClick={() => long && setOpen(!open)} style={long ? { cursor: 'pointer' } : null}>
      <div className="read-h">
        <span>{error ? '⚠' : '📖'}</span> <b>{doc}{part ? ` · ${part}` : ''}</b>
        {whole && !error && <span className="whole">whole document · {fmtTok(text.length)} chars</span>}
      </div>
      <div className="read-b">{shown}</div>
      {long && <div className="more">{open ? 'collapse ▴' : 'expand ▾'}</div>}
    </div>
  )
}

function Feed({ col, gold }) {
  return (
    <div className="feed">
      {col.events.map((e, i) => {
        if (e.kind === 'act') return (
          <div className="act-line" key={i}>
            <span className="n">{e.step}</span>
            <b>{e.op}:</b>&nbsp;<span className="at">{e.text}</span>
          </div>)
        if (e.kind === 'hit') return <HitCard hit={e.hit} query={e.query} key={i} />
        if (e.kind === 'read') return <ReadCard {...e} key={i} />
        if (e.kind === 'zero') return <div className="note" key={i}>{e.text}</div>
        if (e.kind === 'note') return <div className="note" key={i}>{e.text}</div>
        return null
      })}
      {col.error && <div className="err-banner">⚠ {col.error}</div>}
      {col.answer != null && <AnswerCard answer={col.answer} gold={gold} usage={col.usage}
                                         secs={((col.doneAt || Date.now()) - col.t0) / 1000} />}
    </div>
  )
}

function AnswerCard({ answer, gold, usage, secs }) {
  const ok = gold != null && answer && answer.toLowerCase().includes(gold.toLowerCase())
  return (
    <div className={`answer${gold != null ? (ok ? ' good' : ' miss') : ''}`}>
      <div className="k">final answer</div>
      <div className="t">{answer || 'no answer given'}</div>
      <div className="g">
        {gold != null && (ok
          ? <span className="ok">✓ matches the gold answer</span>
          : <span className="no">✗ gold: <i>{gold}</i></span>)}
        <span className="stats"> ${(usage.cost_usd || 0).toFixed(5)} · {usage.steps || 0} steps · {secs.toFixed(1)}s</span>
      </div>
    </div>
  )
}

/* ---------- head-to-head ---------- */

function Compare({ cols }) {
  const metrics = [
    ['tokens', c => (c.usage.prompt_tokens || 0) + (c.usage.completion_tokens || 0), fmtTok],
    ['cost', c => c.usage.cost_usd || 0, v => `$${v.toFixed(5)}`],
    ['steps', c => c.usage.steps || 0, v => `${v}`],
    ['time', c => ((c.doneAt || Date.now()) - c.t0) / 1000, v => `${v.toFixed(1)}s`],
  ]
  return (
    <div className="compare">
      <div className="cmp-head">
        <span className="cmp-title">Head to head</span>
        <span className="cmp-legend">
          {cols.map(c => (
            <span key={c.strategy}>
              <span className="dot" style={{ background: STRATS[c.strategy].color }} />
              {STRATS[c.strategy].label}
            </span>))}
        </span>
      </div>
      {metrics.map(([label, f, fmt]) => {
        const vals = cols.map(f); const max = Math.max(...vals, 1e-9)
        return (
          <div className="cmp-metric" key={label}>
            <span className="cmp-label">{label}</span>
            <div className="cmp-bars">
              {cols.map((c, i) => (
                <div className="cmp-row" key={c.strategy}>
                  <div className="track">
                    <div className="fill" style={{
                      width: `${Math.max((vals[i] / max) * 100, 2.5)}%`,
                      background: STRATS[c.strategy].color,
                    }} />
                  </div>
                  <span className="val">{fmt(vals[i])}</span>
                </div>))}
            </div>
          </div>)
      })}
    </div>
  )
}

/* ---------- the collection wall ---------- */

function Wall({ corpus, seen, read }) {
  const reading = [...read].slice(-1)[0]
  const now = corpus.find(d => d.id === reading)
  return (
    <aside className="shelf">
      <div className="sh-h">The collection <span>{corpus.length} docs</span></div>
      <div className="sh-hint">watch it light up as the agent works</div>
      <div className="wall">
        {corpus.map(d => (
          <div key={d.id} id={`cell-${d.id}`} title={d.title}
               className={`cell${read.has(d.id) ? ' read' : seen.has(d.id) ? ' seen' : ''}`} />
        ))}
      </div>
      <div className="legend">
        <span><i className="sw seen" />surfaced</span>
        <span><i className="sw read" />read</span>
      </div>
      <div className="sh-now">
        {now ? <><b>last read</b>{now.title}</>
             : seen.size > 0 ? <><b>{seen.size} docs surfaced</b>skimming the cards…</> : null}
      </div>
    </aside>
  )
}

/* ---------- the app ---------- */

export default function App() {
  const [meta, setMeta] = useState(null)          // null = loading, false = server offline
  const [question, setQuestion] = useState('')
  const [apiKey, setApiKey] = useState(() => sessionStorage.getItem('demo_key') || '')
  const [mode, setMode] = useState('sieve')
  const [cols, setCols] = useState(null)
  const [running, setRunning] = useState(false)
  const [fail, setFail] = useState('')
  const askedQuestion = useRef('')

  useEffect(() => {
    fetch('/api/meta').then(r => r.json()).then(setMeta).catch(() => setMeta(false))
  }, [])

  const start = () => {
    const strategies = mode === 'both' ? ['sieve', 'search_visit'] : [mode]
    if (!question.trim()) { setFail('type a question — or tap an example below'); return }
    if (!apiKey.trim()) { setFail('paste your OpenAI API key — it never leaves your machine except to call OpenAI'); return }
    sessionStorage.setItem('demo_key', apiKey)
    setFail(''); setRunning(true)
    askedQuestion.current = question
    const t0 = Date.now()
    setCols(strategies.map(s => ({
      strategy: s, events: [], usage: {}, steps: 0, answer: null, error: null,
      t0, doneAt: null, station: 'idle', action: null,
    })))
    const upd = (strategy, f) => setCols(cs =>
      cs.map(c => (c.strategy === strategy ? f({ ...c }) : c)))
    streamRun({ question, api_key: apiKey, model: 'gpt-4o-mini', strategies }, msg => {
      if (msg.event === 'step') {
        upd(msg.strategy, c => {
          c.steps += 1
          const it = interpretStep(msg.step, c.steps)
          c.events = [...c.events, ...it.items]
          if (it.station) c.station = it.station
          if (it.action) c.action = it.action
          c.usage = msg.usage
          if (it.station === 'search') {
            setTimeout(() => spawnPackets(msg.strategy, 'search', STRATS[msg.strategy].color), 780)
          }
          for (const d of it.read || []) setTimeout(() => flyTile(msg.strategy, d), 780)
          return c
        })
      }
      if (msg.event === 'done') upd(msg.strategy, c => {
        c.answer = msg.answer; c.usage = msg.usage; c.doneAt = Date.now()
        c.station = 'answer'; c.action = 'done'
        setTimeout(() => confetti(msg.strategy), 760)
        return c
      })
      if (msg.event === 'error') upd(msg.strategy, c => {
        c.error = msg.message; c.doneAt = Date.now(); c.action = 'something went wrong'
        return c
      })
    }, m => {
      setFail(m === 'offline'
        ? 'lost the connection to the local server — is it still running?' : m)
      setRunning(false)
    }).then(() => setRunning(false))
  }

  const { seen, read } = useMemo(() => {
    const seen = new Set(), read = new Set()
    for (const c of cols || []) for (const e of c.events) {
      if (e.kind === 'hit') seen.add(e.hit.id)
      if (e.kind === 'read' && !e.error) read.add(e.doc)
    }
    return { seen, read }
  }, [cols])

  const gold = useMemo(() => {
    if (!meta || !meta.questions) return null
    const hit = meta.questions.find(x => x.question === askedQuestion.current)
    return hit ? hit.gold : null
  }, [meta, cols])

  const ran = cols != null
  const duo = ran && cols.length === 2

  const composer = meta && (
    <div className="composer">
      <textarea
        className="q-input" rows={2} value={question}
        placeholder="Ask the agent anything about the collection…"
        onChange={e => setQuestion(e.target.value)}
        onKeyDown={e => {
          if (e.key === 'Enter' && (e.metaKey || e.ctrlKey)) { e.preventDefault(); start() }
        }}
      />
      <div className="c-row">
        <div className="pills">
          {MODES.map(m => (
            <button key={m.id} className={`pill${mode === m.id ? ' on' : ''}${m.id === 'both' ? ' vs' : ''}`}
                    onClick={() => setMode(m.id)}>{m.name}</button>))}
        </div>
        <label className="key">🔑
          <input type="password" value={apiKey} placeholder="OpenAI key (sk-…)"
                 onChange={e => setApiKey(e.target.value)} />
        </label>
        <button className="send" disabled={running} onClick={start} title="run (⌘↩)">
          {running ? '…' : '↑'}
        </button>
      </div>
    </div>
  )

  return (
    <div className="page">
      <nav className="topbar">
        <div className="logo"><Logo /> SkimSearch<b>Agent</b></div>
        <div className="top-links">
          <span className="tag">Sieve · live demo</span>
          <a href="https://github.com/ielab/skim-search-agent" target="_blank" rel="noreferrer">GitHub ↗</a>
        </div>
      </nav>

      <div className={`layout${meta ? ' with-shelf' : ''}`}>
        <main className="content">
          {!ran && (
            <header className="hero">
              <h1>What should the agent <span className="hi">find</span> for you?</h1>
              <p>{meta ? `${meta.corpus.length} real documents` : '…'} · the real research loop · every token on the bill</p>
            </header>
          )}

          {meta === false && (
            <div className="offline">
              This page needs its local server.<br />
              Start it with <code>python demo/server.py</code> and reload.
            </div>
          )}

          {!ran && composer}
          {!ran && meta && (
            <div className="examples">
              {meta.questions.map(x => (
                <button className="ex" key={x.question} onClick={() => setQuestion(x.question)}>
                  <b>{x.label}</b>
                  {x.question.length > 150 ? x.question.slice(0, 150) + '…' : x.question}
                </button>))}
            </div>
          )}
          {!ran && meta && (
            <div className="idle-stage"><Stage strategy={mode === 'both' ? 'sieve' : mode} col={null} /></div>
          )}
          {fail && <div className="err-banner center">⚠ {fail}</div>}

          {ran && (
            <div className="asked">
              <span className="asked-k">Q</span>{askedQuestion.current}
            </div>
          )}
          {ran && (
            <div className={`arena${duo ? ' duo' : ''}`}>
              {cols.map(c => (
                <div className="lane" key={c.strategy}>
                  <Stage strategy={c.strategy} col={c} mini={duo} />
                  <Feed col={c} gold={gold} />
                </div>))}
            </div>
          )}
          {duo && cols.every(c => c.doneAt) && <Compare cols={cols} />}

          <footer className="foot">
            every run on this page is the real <code>agent_search</code> loop — nothing is recorded
            or faked · “Search, Inspect, Fetch” (Sieve)
          </footer>
        </main>

        {meta && <Wall corpus={meta.corpus} seen={seen} read={read} />}
      </div>

      {ran && <div className="dock">{composer}</div>}
    </div>
  )
}
