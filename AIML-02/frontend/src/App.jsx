import React, { useEffect, useState } from 'react';
import { Activity, AlertTriangle, CheckCircle2, Clock3, FileImage, HeartPulse, RefreshCw, ShieldCheck, UploadCloud, XCircle } from 'lucide-react';

const API = import.meta.env.VITE_API_BASE || 'http://localhost:8000';

function StatusPill({ children, kind = 'neutral' }) { return <span className={`pill ${kind}`}>{children}</span>; }

export default function App() {
  const [health, setHealth] = useState(null);
  const [cases, setCases] = useState([]);
  const [selected, setSelected] = useState(null);
  const [file, setFile] = useState(null);
  const [preview, setPreview] = useState('');
  const [busy, setBusy] = useState(false);
  const [message, setMessage] = useState('');
  const [reviewer, setReviewer] = useState('reviewer-01');
  const [note, setNote] = useState('');

  async function refresh() {
    try {
      const [h, c] = await Promise.all([fetch(`${API}/health`), fetch(`${API}/cases`)]);
      if (!h.ok || !c.ok) throw new Error('API unavailable');
      setHealth(await h.json());
      const nextCases = (await c.json()).items || [];
      setCases(nextCases);
      if (selected) setSelected(nextCases.find((x) => x.case_id === selected.case_id) || selected);
    } catch { setHealth({ status: 'unreachable', model_available: false }); }
  }
  useEffect(() => { refresh(); const timer = setInterval(refresh, 8000); return () => clearInterval(timer); }, []);

  function onFileChange(event) {
    const next = event.target.files?.[0] || null;
    setFile(next); setMessage(''); setSelected(null);
    if (preview) URL.revokeObjectURL(preview);
    setPreview(next ? URL.createObjectURL(next) : '');
  }

  async function analyze(event) {
    event.preventDefault();
    if (!file) { setMessage('Choose a PNG or JPEG image export first.'); return; }
    setBusy(true); setMessage('');
    try {
      const form = new FormData(); form.append('file', file);
      const response = await fetch(`${API}/analyze`, { method: 'POST', body: form });
      const body = await response.json();
      if (!response.ok) throw new Error(body.detail || 'Analysis request failed.');
      setSelected(body); setMessage('Analysis returned. This output is research-only and requires independent review.');
      await refresh();
    } catch (error) {
      setMessage(error.message || 'Request failed. No diagnosis was generated.');
    } finally { setBusy(false); }
  }

  async function review(decision) {
    if (!selected) return;
    setBusy(true); setMessage('');
    try {
      const response = await fetch(`${API}/cases/${selected.case_id}/review`, {
        method: 'PATCH', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ decision, reviewer_id: reviewer, note: note || null }),
      });
      const body = await response.json();
      if (!response.ok) throw new Error(body.detail || 'Review could not be saved.');
      setSelected(body); setMessage(`Review saved: ${decision}.`); await refresh();
    } catch (error) { setMessage(error.message); }
    finally { setBusy(false); }
  }

  const modelReady = Boolean(health?.model_available);
  const awaiting = cases.filter((c) => c.status === 'awaiting_review').length;
  const resolved = cases.filter((c) => c.status === 'approved' || c.status === 'rejected').length;

  return <div className="app-shell">
    <header className="topbar">
      <div className="brand"><div className="brand-mark"><HeartPulse size={23}/></div><div><div className="brand-name">MedVision <span>Review</span></div><div className="brand-sub">AI-assisted imaging workflow · research prototype</div></div></div>
      <div className="top-status"><span className={`dot ${health?.status === 'ok' ? 'green' : 'red'}`}></span><span>API {health?.status === 'ok' ? 'connected' : 'offline'}</span><StatusPill kind={modelReady ? 'green' : 'amber'}>{modelReady ? 'Model loaded' : 'Model not configured'}</StatusPill></div>
    </header>
    <main className="content">
      <section className="notice"><AlertTriangle size={19}/><div><strong>For education and research only — not for clinical use.</strong><p>Do not use this system to diagnose, triage, or treat a patient. AI outputs can be wrong or biased and must not replace a qualified clinician.</p></div></section>
      <section className="page-heading"><div><div className="eyebrow">RADIOLOGY WORKSPACE</div><h1>Imaging review console</h1><p>Upload a de-identified chest X-ray export, inspect model evidence when a trained checkpoint is available, and record a human review.</p></div><button className="secondary refresh" onClick={refresh}><RefreshCw size={16}/> Refresh</button></section>
      <section className="stats-grid">
        <div className="stat-card"><div className="stat-icon blue"><Activity size={19}/></div><div><div className="stat-label">AI model</div><div className="stat-value small">{modelReady ? 'Ready' : 'Unavailable'}</div><div className="stat-foot">Inference fails closed</div></div></div>
        <div className="stat-card"><div className="stat-icon amber"><Clock3 size={19}/></div><div><div className="stat-label">Awaiting review</div><div className="stat-value">{awaiting}</div><div className="stat-foot">Cases not yet reviewed</div></div></div>
        <div className="stat-card"><div className="stat-icon green"><ShieldCheck size={19}/></div><div><div className="stat-label">Reviewed cases</div><div className="stat-value">{resolved}</div><div className="stat-foot">Decision audit recorded</div></div></div>
      </section>
      <div className="workspace">
        <section className="panel upload-panel">
          <div className="panel-head"><div><h2>New study</h2><p>Supported in this prototype: PNG/JPEG exports</p></div><FileImage size={19}/></div>
          <form onSubmit={analyze}>
            <label className={`dropzone ${preview ? 'has-file' : ''}`}>
              {preview ? <img src={preview} alt="Selected upload preview"/> : <><div className="upload-icon"><UploadCloud size={25}/></div><strong>Choose an image</strong><span>PNG or JPEG · Maximum 15 MB</span></>}
              <input type="file" accept="image/png,image/jpeg" onChange={onFileChange} hidden />
            </label>
            {file && <div className="file-row"><FileImage size={16}/><span>{file.name}</span><span>{(file.size / 1024).toFixed(0)} KB</span></div>}
            <button className="primary full" type="submit" disabled={busy || !file}>{busy ? 'Working…' : 'Analyze image'}</button>
          </form>
          {!modelReady && <div className="inline-warning"><AlertTriangle size={17}/><span>No trained clinical research checkpoint is installed. Upload validation works, but prediction is disabled until trained weights are added.</span></div>}
          {message && <div className="message" role="status">{message}</div>}
          <div className="small-note">For DICOM studies, use the local Orthanc instance. Never upload identifiable patient information into this prototype.</div>
        </section>
        <section className="panel results-panel">
          <div className="panel-head"><div><h2>AI findings & evidence</h2><p>Prediction is not a diagnosis</p></div><Activity size={19}/></div>
          {selected ? <>
            <div className="result-meta"><span>Case {selected.case_id.slice(0, 8)}</span><StatusPill kind={selected.status === 'awaiting_review' ? 'amber' : 'green'}>{selected.status.replaceAll('_', ' ')}</StatusPill></div>
            {selected.heatmap_data_url && <div className="heatmap-wrap"><img src={selected.heatmap_data_url} alt="Grad-CAM activation overlay"/><span>Grad-CAM activation overlay — not a lesion mask</span></div>}
            <div className="finding-list">{(selected.findings || []).slice(0, 5).map((f) => <div className="finding" key={f.label}><div><span>{f.label.replaceAll('_', ' ')}</span><div className="score-track"><div style={{ width: `${Math.max(0, Math.min(100, f.score * 100))}%` }} /></div></div><strong>{(f.score * 100).toFixed(1)}%</strong></div>)}</div>
            <div className="small-note">Scores are model outputs and are not necessarily calibrated probabilities. Verify the image and findings independently.</div>
            {selected.status === 'awaiting_review' && <div className="review-form"><h3>Clinician review log</h3><label>Reviewer identifier<input value={reviewer} onChange={(e) => setReviewer(e.target.value)} maxLength={100}/></label><label>Review note (optional)<textarea value={note} onChange={(e) => setNote(e.target.value)} maxLength={1000} rows={2}/></label><div className="review-actions"><button className="approve" onClick={() => review('approved')} disabled={busy || !reviewer.trim()}><CheckCircle2 size={16}/> Approve</button><button className="reject" onClick={() => review('rejected')} disabled={busy || !reviewer.trim()}><XCircle size={16}/> Reject</button></div><div className="small-note">For this prototype, “approve” means accepting this record for workflow demonstration, not endorsing a diagnosis.</div></div>}
            {selected.review && <div className="review-saved"><CheckCircle2 size={17}/> Reviewed by {selected.review.reviewer_id} · {selected.review.decision}</div>}
          </> : <div className="empty-state"><div className="empty-icon"><Activity size={28}/></div><h3>No study selected</h3><p>Upload an image. Findings and Grad-CAM appear here only when a trained checkpoint is available.</p><div className="no-fake">No sample diagnoses are fabricated in this build.</div></div>}
        </section>
      </div>
      <section className="panel queue-panel"><div className="panel-head"><div><h2>Review queue</h2><p>Cases generated by the configured research model</p></div><span className="queue-count">{cases.length} total</span></div>{cases.length ? <div className="table-wrap"><table><thead><tr><th>Case</th><th>Created (UTC)</th><th>Top finding</th><th>Review status</th><th>Action</th></tr></thead><tbody>{cases.map((item) => <tr key={item.case_id}><td>#{item.case_id.slice(0, 8)}</td><td>{new Date(item.created_at_utc).toLocaleString()}</td><td>{item.top_finding?.replaceAll('_', ' ') || '—'}</td><td><StatusPill kind={item.status === 'awaiting_review' ? 'amber' : 'green'}>{item.status.replaceAll('_', ' ')}</StatusPill></td><td><button className="text-button" onClick={() => { setSelected(item); setMessage(''); }}>Open case</button></td></tr>)}</tbody></table></div> : <div className="empty-queue">No cases yet. A new case appears after an inference request succeeds.</div>}</section>
      <footer><span>MedVision Review · Prototype v0.1.0</span><span>Educational use only · Not validated for clinical use</span></footer>
    </main>
  </div>;
}
