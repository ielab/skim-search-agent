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
  const streamEnd = useRef(null)

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
            <button key={i} className={`tab${i === ep ? ' on' : ''}`} onClick={() => pick(i)}>
              Q{i + 1} · {e.question.slice(0, 44)}{e.question.length > 44 ? '…' : ''}
            </button>
          ))}
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
    </div>
  )
}
