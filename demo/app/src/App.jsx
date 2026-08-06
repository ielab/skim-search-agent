import React, { useEffect, useMemo, useRef, useState } from 'react'
import DATA from './data.json'

/* ---------- playback engine: expands each episode into timed events ---------- */

function toEvents(ep) {
  const ev = []
  let step = 0
  for (const s of ep.steps) {
    step += 1
    if (s.type === 'search') {
      ev.push({ kind: 'agent', step, op: 'search', text: s.query, dwell: 250 })
      ev.push({ kind: 'results', step, status: s.status, hits: [], dwell: 150 })
      s.hits.forEach((h, i) =>
        ev.push({ kind: 'hit', step, hit: h, query: s.query, dwell: 300 }))
      ev.push({ kind: 'pause', dwell: 600 })
    } else if (s.type === 'fetch') {
      ev.push({ kind: 'agent', step, op: 'fetch', text: s.ask, dwell: 250 })
      ev.push({ kind: 'section', step, doc: s.doc, section: s.section, text: s.text, error: s.error, dwell: 850 })
    }
  }
  ev.push({ kind: 'answer', dwell: 0 })
  return ev
}

/* ---------- live mode: SSE client + step -> event expansion ---------- */

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
  } catch (e) {
    onFail('cannot reach the live server — start it with: python demo/live/server.py')
  }
}

// one incoming step -> the same event kinds the replay's toEvents emits (appended live,
// no dwell pacing: real network/API latency provides the rhythm)
function liveStepToEvents(step, n) {
  if (step.type === 'search') {
    return [
      { kind: 'agent', step: n, op: 'search', text: step.query },
      { kind: 'results', step: n, status: step.status, hits: [] },
      ...step.hits.map(h => ({ kind: 'hit', step: n, hit: h, query: step.query })),
    ]
  }
  if (step.type === 'fetch') {
    return [{ kind: 'agent', step: n, op: 'fetch', text: step.ask },
            { kind: 'section', step: n, doc: step.doc, section: step.section,
              text: step.text, error: step.error }]
  }
  if (step.type === 'visit') {
    return [{ kind: 'agent', step: n, op: 'visit', text: `${step.doc} (whole document)` },
            { kind: 'visit', step: n, doc: step.doc, title: step.title,
              text: step.text, error: step.error }]
  }
  return [{ kind: 'generic', step: n, name: step.name, text: step.observation }]
}

function VisitCard({ doc, title, text, error }) {
  const [open, setOpen] = useState(false)
  const shown = open || text.length <= 900 ? text : text.slice(0, 900) + ' …'
  return (
    <div className={`sect visit${error ? ' err' : ''}`} onClick={() => setOpen(!open)}>
      <div className="from">{error ? '⚠' : '⤓'} {doc} · {title} · whole document</div>
      <div className="body">{shown}</div>
      {text.length > 900 && <div className="more">{open ? 'collapse' : 'expand'}</div>}
    </div>
  )
}

function CostMeter({ usage, t0, done }) {
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
      <b>${(usage.cost_usd || 0).toFixed(5)}</b> · {tok.toLocaleString()} tok
      · {usage.steps || 0} steps · {secs.toFixed(1)}s
    </div>
  )
}

const STRAT_LABEL = { sieve: 'Sieve (search + fetch sections)',
                      search_visit: 'Search-Visit (BM25 + whole docs)' }

function LiveColumn({ col }) {
  return (
    <div className="livecol">
      <div className="colhead">
        <span className="colname">{STRAT_LABEL[col.strategy]}</span>
        <CostMeter usage={col.usage} t0={col.t0} done={col.doneAt} />
      </div>
      {col.events.map((e, i) => {
        if (e.kind === 'agent') return (
          <div className="row" key={i}>
            <div className="who agent"><span className="dot" />AGENT · step {e.step}</div>
            <div className="say"><span className={e.op === 'search' ? 'op' : 'fop'}>
              {e.op}:</span> {e.text}</div>
          </div>)
        if (e.kind === 'results') return (
          <div className="row" key={i}>
            <div className="who tool"><span className="dot" />SEARCH RESULTS</div>
            {e.status && <div className="status">{e.status}</div>}
          </div>)
        if (e.kind === 'hit') return (
          <div className="row" key={i}><div className="cards">
            <HitCard hit={e.hit} query={e.query} /></div></div>)
        if (e.kind === 'section') return (
          <div className="row" key={i}><SectionCard {...e} /></div>)
        if (e.kind === 'visit') return (
          <div className="row" key={i}><VisitCard {...e} /></div>)
        if (e.kind === 'generic') return (
          <div className="row" key={i}>
            <div className="who tool"><span className="dot" />{e.name.toUpperCase()}</div>
            {e.text && <div className="status">{String(e.text).slice(0, 400)}</div>}
          </div>)
        return null
      })}
      {col.error && <div className="liveerr">⚠ {col.error}</div>}
      {col.answer != null && (
        <div className="ans"><div className="k">FINAL ANSWER</div>
          <div className="t">{col.answer || '(no answer given)'}</div></div>)}
    </div>
  )
}

function CompareBar({ cols }) {
  const rows = [
    ['tokens', c => (c.usage.prompt_tokens || 0) + (c.usage.completion_tokens || 0)],
    ['cost $', c => c.usage.cost_usd || 0],
    ['steps', c => c.usage.steps || 0],
    ['seconds', c => ((c.doneAt || Date.now()) - c.t0) / 1000],
  ]
  return (
    <div className="cmp">
      <div className="k">HEAD TO HEAD</div>
      {rows.map(([label, f]) => {
        const vals = cols.map(f); const max = Math.max(...vals, 1e-9)
        return (
          <div className="cmprow" key={label}>
            <span className="cmplabel">{label}</span>
            {cols.map((c, i) => (
              <span className="cmpcell" key={c.strategy}>
                <span className="cmpbar" style={{ width: `${(vals[i] / max) * 100}%` }} />
                <span className="cmpval">{label === 'cost $'
                  ? vals[i].toFixed(5) : Math.round(vals[i] * 10) / 10}</span>
              </span>))}
          </div>)
      })}
    </div>
  )
}

function LiveTab({ examples }) {
  const [question, setQuestion] = useState('')
  const [apiKey, setApiKey] = useState(() => sessionStorage.getItem('demo_key') || '')
  const [strats, setStrats] = useState({ sieve: true, search_visit: false })
  const [cols, setCols] = useState(null)
  const [running, setRunning] = useState(false)
  const [fail, setFail] = useState('')

  const start = () => {
    const strategies = Object.keys(strats).filter(s => strats[s])
    if (!question.trim() || !apiKey.trim() || strategies.length === 0) {
      setFail('need a question, an API key, and at least one strategy'); return
    }
    sessionStorage.setItem('demo_key', apiKey)
    setFail(''); setRunning(true)
    const t0 = Date.now()
    const init = strategies.map(s => ({
      strategy: s, events: [], usage: {}, steps: 0, answer: null, error: null, t0, doneAt: null,
    }))
    setCols(init)
    const upd = (strategy, f) => setCols(cs =>
      cs.map(c => (c.strategy === strategy ? f({ ...c }) : c)))
    streamRun({ question, api_key: apiKey, model: 'gpt-4o-mini', strategies }, msg => {
      if (msg.event === 'step') upd(msg.strategy, c => {
        c.steps += 1
        c.events = [...c.events, ...liveStepToEvents(msg.step, c.steps)]
        c.usage = msg.usage; return c
      })
      if (msg.event === 'done') upd(msg.strategy, c => {
        c.answer = msg.answer; c.usage = msg.usage; c.doneAt = Date.now(); return c
      })
      if (msg.event === 'error') upd(msg.strategy, c => {
        c.error = msg.message; c.doneAt = Date.now(); return c
      })
    }, m => { setFail(m); setRunning(false) }).then(() => setRunning(false))
  }

  const allDone = cols && cols.every(c => c.doneAt)
  return (
    <div className="live">
      <div className="liveform">
        <textarea className="qbox" rows={2} value={question} placeholder="Ask anything about the collection…"
                  onChange={e => setQuestion(e.target.value)} />
        <div className="chipsrow">try:{examples.map(q => (
          <button className="exchip" key={q} onClick={() => setQuestion(q)}
                  title={q}>{q.slice(0, 70)}…</button>))}</div>
        <div className="formrow">
          <input className="keybox" type="password" value={apiKey} placeholder="OpenAI API key (sk-…)"
                 onChange={e => setApiKey(e.target.value)} />
          <span className="modeltag">gpt-4o-mini</span>
          {Object.keys(STRAT_LABEL).map(s => (
            <label className="stratpick" key={s}>
              <input type="checkbox" checked={strats[s]}
                     onChange={e => setStrats({ ...strats, [s]: e.target.checked })} />
              {STRAT_LABEL[s]}
            </label>))}
          <button className="btn primary" disabled={running} onClick={start}>
            {running ? '⏳ running' : '▶ run live'}</button>
        </div>
        <div className="keynote">Your key goes browser → this local server → OpenAI. Never stored or logged.</div>
        {fail && <div className="liveerr">⚠ {fail}</div>}
      </div>
      {cols && (
        <div className={`livecols${cols.length === 2 ? ' two' : ''}`}>
          {cols.map(c => <LiveColumn col={c} key={c.strategy} />)}
        </div>)}
      {allDone && cols.length === 2 && <CompareBar cols={cols} />}
    </div>
  )
}

/* ---------- leaf components ---------- */

function Typewriter({ prefix, prefixClass, text, cps = 40, speed, onDone }) {
  const [n, setN] = useState(0)
  const done = n >= text.length
  useEffect(() => {
    if (done) { onDone && onDone(); return }
    const t = setTimeout(() => setN(n + 1), 1000 / (cps * speed))
    return () => clearTimeout(t)
  }, [n, done, cps, speed])
  return (
    <div className="say">
      <span className={prefixClass}>{prefix}</span> {text.slice(0, n)}
      {!done && <span className="cur" />}
    </div>
  )
}

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
    <div className="card">
      <div className="top">
        <span className="rank">#{hit.rank}</span>
        <span className="title">{hit.title}</span>
        <span className="docid">{hit.id}</span>
        {hit.matched && <span className="match">matched: {hit.matched}</span>}
      </div>
      <div className="chips">{hit.sections.map(s => <span className="chip" key={s}>§ {s}</span>)}</div>
      <Snip text={hit.snippet} query={query} />
    </div>
  )
}

function SectionCard({ doc, section, text, error }) {
  return (
    <div className={`sect${error ? ' err' : ''}`}>
      <div className="from">{error ? '⚠' : '§'} {doc} · {section}</div>
      <div className="body">{text}</div>
    </div>
  )
}

function EvidenceCard({ e }) {
  const [open, setOpen] = useState(false)
  return (
    <div className={`ecard${open ? ' open' : ''}`} onClick={() => setOpen(!open)}>
      <div className="h">{e.doc} · §{e.section}</div>
      <div className="b">{e.text}</div>
    </div>
  )
}

/* ---------- the app ---------- */

export default function App() {
  const [ep, setEp] = useState(0)
  const [cursor, setCursor] = useState(0)         // how many events are visible
  const [playing, setPlaying] = useState(true)
  const [speed, setSpeed] = useState(1)
  const [typing, setTyping] = useState(false)     // gate: wait for typewriter
  const [liveMode, setLiveMode] = useState(false)
  const streamEnd = useRef(null)

  const examples = DATA.episodes.map(e => e.question)
  const episode = DATA.episodes[ep]
  const events = useMemo(() => toEvents(episode), [episode])
  const visible = events.slice(0, cursor)

  // advance the cursor on a timer (paused while a typewriter is running)
  useEffect(() => {
    if (!playing || typing || cursor >= events.length) return
    const cur = events[cursor - 1]
    const t = setTimeout(() => {
      const nxt = events[cursor]
      if (nxt && nxt.kind === 'agent') setTyping(true)
      setCursor(cursor + 1)
    }, ((cur && cur.dwell) || 300) / speed)
    return () => clearTimeout(t)
  }, [playing, typing, cursor, events, speed])

  useEffect(() => { streamEnd.current?.scrollIntoView({ behavior: 'smooth' }) }, [cursor, typing])
  useEffect(() => {
    const h = e => { if (e.code === 'Space') { e.preventDefault(); setPlaying(p => !p) } }
    window.addEventListener('keydown', h); return () => window.removeEventListener('keydown', h)
  }, [])

  const pick = i => { setEp(i); setCursor(0); setTyping(false); setPlaying(true) }
  const restart = () => { setCursor(0); setTyping(false); setPlaying(true) }

  // derived state for sidebar + counters
  const evidence = [], seen = new Set(), fetched = new Set()
  let stepNow = 0, calls = 0
  for (const e of visible) {
    if (e.kind === 'agent') { stepNow = e.step; calls += 1 }
    if (e.kind === 'hit') seen.add(e.hit.id)
    if (e.kind === 'section' && !e.error) { evidence.push(e); fetched.add(e.doc) }
  }
  const finished = cursor >= events.length

  return (
    <div className="app">
      <header>
        <div className="logo">⌕ SkimSearch<span>Agent</span></div>
        <div className="tabs">
          {DATA.episodes.map((e, i) => (
            <button key={i} className={`tab${!liveMode && i === ep ? ' on' : ''}`}
                    onClick={() => { setLiveMode(false); pick(i) }}>
              Q{i + 1} · {e.question.slice(0, 44)}{e.question.length > 44 ? '…' : ''}
            </button>
          ))}
          <button className={`tab${liveMode ? ' on' : ''}`}
                  onClick={() => setLiveMode(true)}>▶ Live</button>
        </div>
        <div className="controls">
          <span className="counters">step <b>{stepNow}</b> · calls <b>{calls}</b></span>
          <input className="speed" type="range" min="0.4" max="3" step="0.2"
                 value={speed} onChange={e => setSpeed(parseFloat(e.target.value))} title="speed" />
          <button className="btn" onClick={restart} title="restart">↺</button>
          <button className="btn primary" onClick={() => finished ? restart() : setPlaying(!playing)}>
            {finished ? '▶ replay' : playing ? '⏸ pause' : '▶ play'}
          </button>
        </div>
      </header>
      {liveMode ? (
        <main className="livemain"><LiveTab examples={examples} /></main>
      ) : (
      <main>
        <div className="stream">
          <div className="q">
            <div className="k">QUESTION</div>
            <div className="t">{episode.question}</div>
            {episode.settings && Object.keys(episode.settings).length > 0 &&
              <div className="settings">
                {Object.entries(episode.settings).map(([k, v]) =>
                  <span className="set" key={k}><b>{k}</b> {String(v)}</span>)}
              </div>}
          </div>
          {visible.map((e, i) => {
            if (e.kind === 'agent') return (
              <div className="row" key={i}>
                <div className="who agent"><span className="dot" />AGENT · step {e.step}</div>
                <Typewriter prefix={`${e.op}:`} prefixClass={e.op === 'search' ? 'op' : 'fop'}
                            text={e.text} speed={speed}
                            onDone={() => setTyping(false)} />
              </div>)
            if (e.kind === 'results') return (
              <div className="row" key={i}>
                <div className="who tool"><span className="dot" />SEARCH RESULTS</div>
                {e.status && <div className="status">{e.status}</div>}
              </div>)
            if (e.kind === 'hit') return (
              <div className="row" key={i}><div className="cards"><HitCard hit={e.hit} query={e.query} /></div></div>)
            if (e.kind === 'section') return (
              <div className="row" key={i}><SectionCard {...e} /></div>)
            if (e.kind === 'answer') return (
              <div className="ans" key={i}>
                <div className="k">FINAL ANSWER</div>
                <div className="t">{episode.answer}</div>
                <div className="g">gold: <b>{episode.gold}</b> · {episode.correct
                  ? <span className="ok">✓ correct</span> : '✗'} · {episode.calls} model calls</div>
              </div>)
            return null
          })}
          <div ref={streamEnd} />
        </div>
        <aside className="side">
          <h3>EVIDENCE COLLECTED</h3>
          <div className="evi">
            {evidence.length === 0 && <div className="none">nothing fetched yet…</div>}
            {evidence.map((e, i) => <EvidenceCard e={e} key={i} />)}
          </div>
          <h3>THE COLLECTION</h3>
          <div className="shelf">
            {DATA.corpus.map(d => (
              <div key={d.id}
                   className={`doc${fetched.has(d.id) ? ' fetched' : seen.has(d.id) ? ' seen' : ''}`}>
                <span className="lamp" />{d.title}<span className="secs">{d.sections.length}§</span>
              </div>
            ))}
          </div>
        </aside>
      </main>
      )}
    </div>
  )
}
